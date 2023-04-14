import PIL
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import imageio
import cv2
import mmcv
from PIL import Image

from .motion_visualisation import plot_instance_map, visualise_output, make_contour, generate_instance_colours, \
    plot_motion_prediction, plot_motion_label, plot_motion_prediction_with_map, plot_motion_label_with_map
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
import math
import pdb


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(
        figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np


def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)

def flip_rotate_image_ori(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)
    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    return np.array(pil_img)

class Visualizer(object):
    def __init__(self, grid_conf=None, coordinate_system='lidar', out_dir='visualize'):

        self.coordinate_system = coordinate_system
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        if grid_conf is None:
            grid_conf = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }

        # bev grid
        self.grid_conf = grid_conf
        xbound, ybound, zbound = grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        self.bev_resolution = np.array([xbound[2], ybound[2], zbound[2]])
        self.bev_start = np.array(
            [xbound[0] + xbound[2] / 2.0,
             ybound[0] + ybound[2] / 2.0,
             zbound[0] + zbound[2] / 2.0])

        canvas_h = int((ybound[1] - ybound[0]) / ybound[2])
        canvas_w = int((xbound[1] - xbound[0]) / xbound[2])
        self.canvas_size = (canvas_h, canvas_w)
        self.ignore_index = 255

    def make_bev_canvas(self, channel=3):
        if channel == 1:
            canvas_size = self.canvas_size
        else:
            canvas_size = (self.canvas_size[0], self.canvas_size[1], 3)

        return np.zeros(canvas_size)

    # plot bev-segmentation for lidar_boxes
    def lidar_boxes_to_binary_segmentation(self, lidar_boxes):
        bev_canvas = self.make_bev_canvas(channel=1)

        # when number of objects > 0
        if lidar_boxes.tensor.shape[0] > 0:
            # get corners of boxes
            bottom_centers = lidar_boxes.corners[:, [0, 3, 7, 4], :]
            bottom_centers = bottom_centers[..., :2]
            # map corners to pixels on bev grid
            bev_start_pos = bottom_centers.new_tensor(
                self.bev_start[:2]).view(1, 1, -1)
            bev_reso = bottom_centers.new_tensor(
                self.bev_resolution[:2]).view(1, 1, -1)
            bottom_centers = torch.round(
                (bottom_centers - bev_start_pos + bev_reso / 2.0) / bev_reso)
            bottom_centers = bottom_centers.numpy().astype(np.int)

            for k in range(bottom_centers.shape[0]):
                cv2.fillPoly(bev_canvas, [bottom_centers[k]], 1.0)

        return bev_canvas

    def draw_hdmap(self, semantic_map):
        color_semantic_map = semantic_map.permute(1, 2, 0) * 255
        color_semantic_map = color_semantic_map.numpy().astype(np.uint8)
        bg_mask = color_semantic_map.sum(axis=2) == 0
        color_semantic_map[bg_mask] = np.array([255, 255, 255])

        return color_semantic_map

    def visualize_input(self, img_metas=None):
        save_dir = os.path.join(self.out_dir, 'input_imgs')
        temporal_img_infos = img_metas['img_info']
        for index, img_infos in enumerate(temporal_img_infos):
            this_save_dir = os.path.join(save_dir, 'frame_{}'.format(index))
            os.makedirs(this_save_dir, exist_ok=True)
            for cam_name, img_info in img_infos.items():
                # read image
                img_filename = img_info['data_path']
                img = mmcv.imread(img_filename)
                # save image
                mmcv.imwrite(img, os.path.join(
                    this_save_dir, '{}.png'.format(cam_name)))


    def visualize_deepaccident(self, img_metas, bbox_results, gt_bboxes_3d, gt_labels_3d, motion_labels, motion_preds, vis_thresh=0.25,
                          save_path=None):
        save_path_input = save_path
        save_path = os.path.join(self.out_dir, 'visualization', save_path)
        assert save_path is not None
        os.makedirs(save_path, exist_ok=True)

        # ground-truth objects
        gt_lidar_boxes = gt_bboxes_3d.data[0][0]
        gt_labels = gt_labels_3d.data[0][0]
        img_infos = img_metas['img_info'][-1]

        # predictions
        bbox_results = bbox_results["pts_bbox"]
        pred_lidar_boxes = bbox_results["boxes_3d"]
        pred_labels = bbox_results['labels_3d']
        pred_scores_3d = bbox_results["scores_3d"]
        pred_score_mask = pred_scores_3d > vis_thresh
        pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
        pred_labels = pred_labels[pred_score_mask]

        pred_bbox_color = (241, 101, 72)
        gt_bbox_color = (61, 102, 255)

        img_all_cams = {}
        for cam_type, img_info in img_infos.items():
            img_filename = img_info['image_path']
            img = imageio.imread(img_filename)

            img_resize = cv2.resize(img, (620, 350), interpolation=cv2.INTER_AREA)
            img_all_cams[cam_type] = img_resize

            lidar2cam_rt = img_info['lidar_to_camera_matrix']

            R = np.dot(img_metas['ego_to_world_matrix'], img_metas['lidar_to_ego_matrix'])
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            lidar_pitch_radian = math.atan2(-R[2, 0], sy)
            lidar_pitch_radian_tmp = lidar_pitch_radian

            ego2lidar_rt = np.linalg.inv(img_metas['lidar_to_ego_matrix'])
            ego2cam_rt = lidar2cam_rt @ ego2lidar_rt

            intrinsic = img_info['camera_intrinsic_matrix']


            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0],
                    :intrinsic.shape[1]] = intrinsic

            lidar2img = (viewpad @ lidar2cam_rt)
            ego2img = (viewpad @ ego2cam_rt)


            if self.coordinate_system == 'lidar':
                img_with_gt = draw_lidar_bbox3d_on_img(
                    gt_lidar_boxes, img, lidar2img, None, color=gt_bbox_color, thickness=2,
                lidar_pitch_radian=lidar_pitch_radian_tmp)
            else:
                img_with_gt = draw_lidar_bbox3d_on_img(
                    gt_lidar_boxes, img, ego2img, None, color=gt_bbox_color, thickness=2,
                    lidar_pitch_radian=lidar_pitch_radian_tmp)
            img_with_pred = draw_lidar_bbox3d_on_img(
                pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color, thickness=2,
                lidar_pitch_radian=lidar_pitch_radian_tmp)

            if False:
                img_show = cv2.cvtColor(img_with_gt, cv2.COLOR_RGB2BGR)
                img_show2 = cv2.cvtColor(img_with_pred, cv2.COLOR_RGB2BGR)
                pdb.set_trace()
                cv2.imshow('pred', img_show2)
                cv2.imshow('gt', img_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            imageio.imwrite(
                '{}/det_gt_{}.png'.format(save_path, cam_type), img_with_gt)
            imageio.imwrite(
                '{}/det_pred_{}.png'.format(save_path, cam_type), img_with_pred)

        folder_list = img_infos['Camera_Front']['image_path'].split('/')

        folder_list[5] = 'BEV_instance_camera'
        folder_list[-1] = folder_list[-1].split('.')[0] + '.npz'

        map_file = '/'.join(folder_list)

        map_img = self.plot_map(map_file)

        figure_motion_label = plot_motion_label_with_map(motion_labels, map_img)


        transformed_motion_img = flip_rotate_image(figure_motion_label)
        imageio.imwrite('{}/motion_gt.png'.format(save_path),
                        transformed_motion_img)

        img_all = np.zeros((700, 2560, 3), dtype=np.uint8)
        res_h, res_w, _ = img_all_cams['Camera_Front'].shape
        img_all[:res_h, :res_w] = img_all_cams['Camera_FrontLeft']
        img_all[:res_h, res_w:2*res_w] = img_all_cams['Camera_Front']
        img_all[:res_h, 2*res_w:3*res_w] = img_all_cams['Camera_FrontRight']
        img_all[res_h:2*res_h, :res_w] = img_all_cams['Camera_BackRight']
        img_all[res_h:2*res_h, res_w:2*res_w] = img_all_cams['Camera_Back']
        img_all[res_h:2*res_h, 2*res_w:3*res_w] = img_all_cams['Camera_BackLeft']

        transformed_motion_img_resize = cv2.resize(transformed_motion_img, (700, 700), interpolation=cv2.INTER_AREA)
        img_all[:700, 3 * res_w:3 * res_w+700] = transformed_motion_img_resize
        # pdb.set_trace()
        frame = '_'.join(save_path.split('/')[-1].split('_')[4:])
        video_path = '/'.join(save_path.split('/')[:-1])
        imageio.imwrite('{}/motion_video_{}.png'.format(video_path, frame), img_all)

        if len(motion_preds) > 1:
            for pred_idx, motion_pred in enumerate(motion_preds):
                figure_motion_pred = plot_motion_prediction_with_map(motion_pred, map_img)
                imageio.imwrite('{}/motion_pred_{}.png'.format(save_path, pred_idx+1),
                                flip_rotate_image(figure_motion_pred))
        else:
            figure_motion_pred = plot_motion_prediction_with_map(motion_preds, map_img)
            imageio.imwrite('{}/motion_pred.png'.format(save_path),
                            flip_rotate_image(figure_motion_pred))

    def plot_map(self, map_file):
        file = np.load(map_file)
        data_raw = file['data']

        width = data_raw.shape[0]
        cropped_width = width / 60 * 50

        start_idx = int(width // 2 - cropped_width // 2)
        end_idx = int(width // 2 + cropped_width // 2)

        data_raw = data_raw[start_idx:end_idx, start_idx:end_idx]


        classes2 = {}
        for elem in data_raw[:, :, 2].flatten():
            if elem not in classes2.keys():
                classes2[elem] = 1
            else:
                classes2[elem] += 1


        color_mapping = {}

        for semantic_class in classes2.keys():
            color_mapping[semantic_class] = [255 / 255, 255 / 255, 255 / 255]
            # color_mapping[semantic_class] = [128 / 255, 64 / 255, 128 / 255]

        color_mapping[4] = [60 / 255, 20 / 255, 220 / 255]

        # color_mapping[6] = [50/255, 234/255, 157/255]
        color_mapping[6] = [40 / 255, 40 / 255, 40 / 255]

        # color_mapping[6] = [0, 0, 0]
        # color_mapping[7] = [128/255, 64/255, 128/255]
        color_mapping[7] = [200 / 255, 200 / 255, 200 / 255]

        # color_mapping[8] = [232/255, 35/255, 244/255]
        color_mapping[8] = [0 / 255, 0 / 255, 200 / 255]

        color_mapping[10] = [142 / 255, 0 / 255, 0 / 255]

        color_mapping[4] = color_mapping[7]
        color_mapping[10] = color_mapping[7]
        color_mapping[12] = color_mapping[7]
        color_mapping[18] = color_mapping[7]
        color_mapping[20] = color_mapping[7]

        valid_class = [6, 7, 10]

        # valid_class = [8]
        for semantic_class in classes2.keys():
            if semantic_class not in valid_class:
                color_mapping[semantic_class] = [255 / 255, 255 / 255, 255 / 255]

        # color_mapping[6] = color_mapping[7]
        # color_mapping[8] = color_mapping[7]

        img_show = np.zeros([data_raw.shape[0], data_raw.shape[1], data_raw.shape[2]])

        for i in range(data_raw.shape[0]):
            for j in range(data_raw.shape[1]):
                img_show[i, j] = color_mapping[data_raw[:, :, 2][i][j]]

        img_show = cv2.resize(img_show, (200, 200), interpolation=cv2.INTER_AREA)

        img_show = cv2.rotate(img_show, cv2.ROTATE_90_CLOCKWISE)
        img_show = (img_show * 255).astype(np.uint8)

        return img_show


    def visualize_deepaccident_motion(self, motion_labels, motion_preds, save_path=None):
        save_path = os.path.join(self.out_dir, 'visualization', save_path)
        assert save_path is not None
        os.makedirs(save_path, exist_ok=True)

        figure_motion_label = plot_motion_prediction(motion_labels)
        imageio.imwrite('{}/motion_gt.png'.format(save_path),
                        flip_rotate_image(figure_motion_label))
        if len(motion_preds) > 1:
            for pred_idx, motion_pred in enumerate(motion_preds):
                figure_motion_pred = plot_motion_prediction(motion_pred)
                imageio.imwrite('{}/motion_pred_{}.png'.format(save_path, pred_idx+1),
                                flip_rotate_image(figure_motion_pred))
        else:
            figure_motion_pred = plot_motion_prediction(motion_preds)
            imageio.imwrite('{}/motion_pred.png'.format(save_path),
                            flip_rotate_image(figure_motion_pred))

    def visualize_motion(self, labels=None, output=None, fps=1):
        # labels require ['instance', 'segmentation', 'flow', 'centerness', 'offset']
        # output require ['segmentation', 'instance_flow', 'instance_center', 'instance_offset']
        # [B, T, C, H, W]
        video = visualise_output(labels, output)[0]
        save_dir = os.path.join(self.out_dir, 'motion')

        gifs = []
        for index in range(video.shape[0]):
            image = video[index].transpose((1, 2, 0))
            gifs.append(image)

        os.makedirs(save_dir, exist_ok=True)
        imageio.mimsave("{}/motion.gif".format(save_dir), gifs, fps=fps)

    def visualize_detection(self, img_metas, bbox_results, gt_bboxes_3d=None,
                            gt_labels_3d=None, vis_thresh=0.25):

        save_dir = os.path.join(self.out_dir, 'det')
        os.makedirs(save_dir, exist_ok=True)

        img_infos = img_metas['img_info'][-1]

        # ground-truth objects
        gt_lidar_boxes = gt_bboxes_3d.data[0][0]
        gt_labels = gt_labels_3d.data[0][0]

        # predictions
        bbox_results = bbox_results["pts_bbox"]
        pred_lidar_boxes = bbox_results["boxes_3d"]
        pred_labels = bbox_results['labels_3d']
        pred_scores_3d = bbox_results["scores_3d"]

        # score-filtering predictions
        pred_score_mask = pred_scores_3d > vis_thresh
        pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
        pred_labels = pred_labels[pred_score_mask]

        gt_imgs = {}
        pred_imgs = {}

        gt_bbox_color = (61, 102, 255)
        pred_bbox_color = (241, 101, 72)

        for cam_type, img_info in img_infos.items():
            img_filename = img_info['data_path']
            img = imageio.imread(img_filename)
            file_name = os.path.split(img_filename)[-1].split(".")[0]

            cam2lidar_rt = np.eye(4)
            cam2lidar_rt[:3, :3] = img_info['sensor2lidar_rotation']
            cam2lidar_rt[:3, -1] = img_info['sensor2lidar_translation']
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)

            lidar2ego_rt = np.eye(4)
            lidar2ego_rt[:3, :3] = img_metas['lidar2ego_rots']
            lidar2ego_rt[:3, -1] = img_metas['lidar2ego_trans']
            ego2lidar_rt = np.linalg.inv(lidar2ego_rt)

            ego2cam_rt = lidar2cam_rt @ ego2lidar_rt

            intrinsic = img_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0],
                    :intrinsic.shape[1]] = intrinsic

            lidar2img = (viewpad @ lidar2cam_rt)
            ego2img = (viewpad @ ego2cam_rt)

            # draw
            if self.coordinate_system == 'lidar':
                img_with_gt = draw_lidar_bbox3d_on_img(
                    gt_lidar_boxes, img, lidar2img, None, color=gt_bbox_color)
            else:
                img_with_gt = draw_lidar_bbox3d_on_img(
                    gt_lidar_boxes, img, ego2img, None, color=gt_bbox_color)

            img_with_pred = draw_lidar_bbox3d_on_img(
                pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color)
            gt_imgs[cam_type] = img_with_gt
            pred_imgs[cam_type] = img_with_pred

        val_w = 6.4
        val_h = val_w / 16 * 9
        fig = plt.figure(figsize=(3 * val_w, 4 * val_h))
        width_ratios = (val_w, val_w, val_w)
        gs = mpl.gridspec.GridSpec(4, 3, width_ratios=width_ratios)
        # gs.update(wspace=0.01, hspace=0.1, left=0,
        #           right=1.0, top=1.0, bottom=0.1)

        vis_orders = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        label_font_size = 16
        for img_index, vis_cam_type in enumerate(vis_orders):
            vis_gt_img = gt_imgs[vis_cam_type]
            vis_pred_img = pred_imgs[vis_cam_type]

            # prediction
            ax = plt.subplot(gs[(img_index // 3) * 2, img_index % 3])
            # plt.annotate(vis_cam_type.replace('_', ' ').replace(
            #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
            # plt.annotate(vis_cam_type.replace('_', ' ').replace(
            #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
            plt.imshow(vis_pred_img)
            # if img_index % 3 == 0:
            plt.title(vis_cam_type, fontsize=label_font_size)
            plt.axis('off')
            ax.set_ylabel("Prediction", fontsize=label_font_size)
            plt.draw()

            # ground-truth
            ax = plt.subplot(gs[(img_index // 3) * 2 + 1, img_index % 3])
            # if img_index % 3 == 0:
            plt.imshow(vis_gt_img)
            plt.axis('off')
            ax.set_ylabel("Ground-truth", fontsize=label_font_size)
            plt.draw()

        plt.tight_layout()
        plt.savefig(save_dir + '/det.png')
        figure_numpy = convert_figure_numpy(fig)
        plt.close()

        return figure_numpy

    def visualize_test_detection(self, img_metas, bbox_results, vis_thresh=0.25):
        save_dir = os.path.join(self.out_dir, 'det')
        os.makedirs(save_dir, exist_ok=True)

        img_infos = img_metas['img_info'][-1]

        # predictions
        bbox_results = bbox_results["pts_bbox"]
        pred_lidar_boxes = bbox_results["boxes_3d"]
        pred_labels = bbox_results['labels_3d']
        pred_scores_3d = bbox_results["scores_3d"]

        # score-filtering predictions
        pred_score_mask = pred_scores_3d > vis_thresh
        pred_lidar_boxes = pred_lidar_boxes[pred_score_mask]
        pred_labels = pred_labels[pred_score_mask]

        pred_imgs = {}
        pred_bbox_color = (241, 101, 72)

        for cam_type, img_info in img_infos.items():
            img_filename = img_info['data_path']

            # img = mmcv.imread(img_filename)
            img = imageio.imread(img_filename)
            file_name = os.path.split(img_filename)[-1].split(".")[0]

            cam2lidar_rt = np.eye(4)
            cam2lidar_rt[:3, :3] = img_info['sensor2lidar_rotation']
            cam2lidar_rt[:3, -1] = img_info['sensor2lidar_translation']
            lidar2cam_rt = np.linalg.inv(cam2lidar_rt)

            lidar2ego_rt = np.eye(4)
            lidar2ego_rt[:3, :3] = img_metas['lidar2ego_rots']
            lidar2ego_rt[:3, -1] = img_metas['lidar2ego_trans']
            ego2lidar_rt = np.linalg.inv(lidar2ego_rt)

            ego2cam_rt = lidar2cam_rt @ ego2lidar_rt

            intrinsic = img_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0],
                    :intrinsic.shape[1]] = intrinsic

            lidar2img = (viewpad @ lidar2cam_rt)
            ego2img = (viewpad @ ego2cam_rt)

            img_with_pred = draw_lidar_bbox3d_on_img(
                pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color)
            pred_imgs[cam_type] = img_with_pred

        val_w = 6.4
        val_h = val_w / 16 * 9
        fig = plt.figure(figsize=(3 * val_w, 2 * val_h))
        width_ratios = (val_w, val_w, val_w)
        gs = mpl.gridspec.GridSpec(2, 3, width_ratios=width_ratios)

        vis_orders = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        label_font_size = 16
        for img_index, vis_cam_type in enumerate(vis_orders):
            vis_pred_img = pred_imgs[vis_cam_type]

            # prediction
            ax = plt.subplot(gs[(img_index // 3), img_index % 3])
            # plt.annotate(vis_cam_type.replace('_', ' ').replace(
            #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
            # plt.annotate(vis_cam_type.replace('_', ' ').replace(
            #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
            plt.imshow(vis_pred_img)
            # if img_index % 3 == 0:
            plt.title(vis_cam_type, fontsize=label_font_size)
            plt.axis('off')
            ax.set_ylabel("Prediction", fontsize=label_font_size)
            plt.draw()

        plt.tight_layout()
        plt.savefig(save_dir + '/det.png')
        plt.close()

    def convert_color_map(self, map_lables, bg_color=(169, 169, 169)):
        map_lables = torch.nn.functional.one_hot(map_lables[0], 4)[..., 1:]
        color_map = map_lables * 255
        color_map = color_map.numpy().astype(np.uint8)

        # make backgorund
        bg_mask = color_map.sum(axis=2) == 0
        color_map[bg_mask] = np.array(bg_color)

        # make contour
        color_map = make_contour(color_map)

        return color_map

    def visualize_map(self, labels=None, output=None, save_file=None):
        # input of shape (b, h, w)

        if labels is not None:
            labels = labels.cpu()
            color_map_label = self.convert_color_map(labels)
        else:
            color_map_label = np.zeros((*output.shape[1:], 3), dtype=np.uint8)

        if output is not None:
            output = output.cpu()
            color_map_output = self.convert_color_map(output)
        else:
            color_map_output = np.zeros((*labels.shape[1:], 3), dtype=np.uint8)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(color_map_output)
        plt.axis('off')
        plt.title('Prediction')
        plt.subplot(122)
        plt.imshow(color_map_label)
        plt.axis('off')
        plt.title('Ground-truth')
        plt.tight_layout()

        save_dir = os.path.join(self.out_dir, 'map')
        os.makedirs(save_dir, exist_ok=True)

        if save_file is None:
            save_file = 'map.png'
        plt.savefig('{}/{}'.format(save_dir, save_file))
        figure_numpy = convert_figure_numpy(fig)
        plt.close()

        return figure_numpy

    def plot_temporal_instances(self, sample):
        temporal_instances = sample['motion_instance']
        num_frame = temporal_instances.shape[0]

        instance_ids = np.unique(temporal_instances)[1:]
        instance_ids = instance_ids[instance_ids != self.ignore_index]
        instance_map = dict(zip(instance_ids, instance_ids))

        plt.figure(0, figsize=(18, 4))
        for i in range(num_frame):
            color_instance_i = plot_instance_map(
                temporal_instances[i], instance_map)
            plt.subplot(1, num_frame, i + 1)
            plt.imshow(color_instance_i)
            plt.axis('off')

        plt.savefig('{}/temporal_instances.png'.format(self.out_dir))
        plt.close()

        # plot instance centerness, instance offset, instance flow

    def visualize_bev(self, sample):
        semantic_map = sample['semantic_map']
        color_semantic_map = self.draw_hdmap(semantic_map)

        det_instance = self.lidar_boxes_to_binary_segmentation(
            sample['gt_bboxes_3d']._data)

        det_instance = det_instance.astype(np.int)
        instance_ids = np.unique(det_instance)[1:]
        instance_map = dict(zip(instance_ids, instance_ids))
        color_det_instance = plot_instance_map(det_instance, instance_map)

        plt.figure(0, figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(color_semantic_map)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(color_det_instance)
        plt.axis('off')
        plt.savefig('{}/bev_label.png'.format(self.out_dir))
        plt.close()
