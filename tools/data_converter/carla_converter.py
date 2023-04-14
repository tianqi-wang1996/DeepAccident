# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os

from os import path as osp
import glob
import pdb

from collections import OrderedDict
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import random

from mmdet3d.core.bbox.box_np_ops import points_cam2img

carla_categories = ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian')

def create_carla_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          info_save_path=None):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    test = 'test' in version
    train_list = []
    val_list = []
    if not test:
        with open(osp.join(root_path, 'train.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
        with open(osp.join(root_path, 'val.txt'), 'r') as f:
            # val_list = [line.rstrip() for line in f]
            val_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
        print('train scene: {}, val scene: {}'.format(
            len(train_list), len(val_list)))
    else:
        with open(osp.join(root_path, 'test.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
        print('test scene: {}'.format(len(train_list)))


    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        root_path, train_list, val_list, test)

    metadata = dict(version=version)

    if not test:
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(info_save_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)

        data = dict(infos=val_nusc_infos, metadata=metadata)
        info_path = osp.join(info_save_path,
                             '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(info_save_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)

def create_carla_infos_mini(root_path,
                          info_prefix,
                          version='v1.0-mini',
                          info_save_path=None):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    test = False

    with open(osp.join(root_path, 'train_mini.txt'), 'r') as f:
        train_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
    with open(osp.join(root_path, 'val_mini.txt'), 'r') as f:
        # val_list = [line.rstrip() for line in f]
        val_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
    print('train scene: {}, val scene: {}'.format(
        len(train_list), len(val_list)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        root_path, train_list, val_list, test)

    metadata = dict(version=version)

    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(info_save_path,
                         '{}_infos_train_mini.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)

    data = dict(infos=val_nusc_infos, metadata=metadata)
    info_path = osp.join(info_save_path,
                         '{}_infos_val_mini.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)


def _fill_trainval_infos(root_path, train_list, val_list, test):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    # test = False

    for (scenario_type, file_prefix) in mmcv.track_iter_progress(train_list):

        file_list = glob.glob(osp.join(root_path, scenario_type,
                                        'ego_vehicle', 'label', file_prefix)+'/*')
        frame_list = []
        for file_path in file_list:
            frame_list.append(file_path.split('/')[-1].split('.')[0])
        frame_list_last_frame = max([int(frame_list[i].split('_')[-1]) for i in range(len(frame_list))])
        frame_list_filtered = []
        for frame_name in frame_list:
            frame_number = int(frame_name.split('_')[-1])
            if (frame_list_last_frame - frame_number) % 5 == 0:
            # if True:
                frame_list_filtered.append(frame_name)

        vehicle_names = ['ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure']

        for key in vehicle_names:
            for frame_name in frame_list_filtered:
                lidar_path = osp.join(root_path, scenario_type, key, 'lidar01', file_prefix, frame_name + '.npz')
                bev_path = osp.join(root_path, scenario_type, key, 'BEV_instance_camera', file_prefix,
                                    frame_name + '.npz')

                calib_path = osp.join(root_path, scenario_type, key, 'calib', file_prefix, frame_name + '.pkl')

                calib_dict = mmcv.load(calib_path)
                scene_name = scenario_type + '_' + file_prefix.split('_')[0] + '_' + file_prefix.split('_')[-1]

                time_stamp = int(frame_name.split('_')[-1])

                # lidar_prefix = osp.join(key, 'lidar01', file_prefix)
                lidar_prefix = scenario_type + '_' + key + '_' + frame_name

                # reverse the y axis
                y_reverse_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]
                                             ]).astype(np.float)
                # calib_dict['ego_to_world'] = y_reverse_matrix @ calib_dict['ego_to_world']
                # calib_dict['lidar_to_ego'] = y_reverse_matrix @ calib_dict['lidar_to_ego']
                info = {
                    'scenario_type': scenario_type,
                    'vehicle_name': key,
                    'scene_name': scene_name,
                    'lidar_prefix': lidar_prefix,
                    'lidar_path': lidar_path,
                    'bev_path': bev_path,
                    'timestamp': time_stamp,
                    'scenario_length': frame_list_last_frame,
                    'cams': dict(),
                    'lidar_to_ego_matrix': calib_dict['lidar_to_ego'],
                    'ego_to_world_matrix': calib_dict['ego_to_world']
                }

                camera_types = [
                    'Camera_FrontLeft',
                    'Camera_Front',
                    'Camera_FrontRight',
                    'Camera_BackLeft',
                    'Camera_Back',
                    'Camera_BackRight',
                ]

                for cam in camera_types:
                    cam_info = dict()
                    cam_path = osp.join(root_path, scenario_type, key, cam, file_prefix, frame_name + '.jpg')
                    cam_info['image_path'] = cam_path

                    # # reverse the y axis
                    # calib_dict['lidar_to_'+cam] = y_reverse_matrix @ calib_dict['lidar_to_'+cam]

                    cam_info['lidar_to_camera_matrix'] = calib_dict['lidar_to_'+cam]
                    intrinsic = calib_dict['intrinsic_' + cam]
                    # reverse_y = np.array([[1.0, 0.0, 0.0],
                    #                         [0.0, -1.0, 0.0],
                    #                         [0.0, 0.0, 1.0]
                    #                          ]).astype(np.float)
                    # intrinsic = reverse_y @ intrinsic
                    cam_info['camera_intrinsic_matrix'] = intrinsic
                    cam_info['timestamp'] = time_stamp
                    info['cams'].update({cam: cam_info})

                if not test:
                    bev_path = osp.join(root_path, scenario_type, key, 'BEV_instance_camera', file_prefix,
                                        frame_name + '.npz')
                    info.update(bev_path=bev_path)

                    label_path = osp.join(root_path, scenario_type, key, 'label', file_prefix, frame_name + '.txt')

                    with open(label_path) as f:
                        lines = [line.rstrip('\n') for line in f]

                    info.update(vehicle_speed_x=float(lines[0].split(' ')[0]))
                    info.update(vehicle_speed_y=float(lines[0].split(' ')[1]))

                    bbox_list = []
                    for line in lines[1:]:
                        if len(line.split(' ')) <= 1:
                            continue
                        # num_pts_list.append((vehicle_class, center_point[0], center_point[1], center_point[2], l, w, h, yaw,
                        #                              vel_other_vec_transformed[0], vel_other_vec_transformed[1], veh_id))
                        cls_label = str(line.split(' ')[0])
                        bbox = line.split(' ')[1:8]

                        bbox = list(map(float, bbox))
                        # original order: h w l x y z yaw
                        # transformed order: x y z l w h -yaw

                        # # for carla mini
                        # bbox = [bbox[3], bbox[4], bbox[5], bbox[2], bbox[1],
                        #                     bbox[0], -bbox[6]]


                        # reverse the y axis
                        bbox[6] = -bbox[6]
                        # bbox[1] = -bbox[1]
                        # if bbox[6] > 0:
                        #     bbox[6] = 3.1415927 - bbox[6]
                        # else:
                        #     bbox[6] = -3.1415927 - bbox[6]

                        # bbox[6] = -bbox[6]

                        # for carla full
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]]


                        vel = line.split(' ')[8:10]
                        vel = list(map(float, vel))
                        # reverse the y axis
                        vel = [vel[0], -vel[1]]
                        # vel = [0, 0]

                        vehicle_id = int(line.split(' ')[-3])
                        num_lidar_pts = int(line.split(' ')[-2])
                        if line.split(' ')[-1] == 'True':
                            camera_visibility = 1
                        else:
                            camera_visibility = 0
                        bbox_list.append((cls_label, bbox, vel, vehicle_id, num_lidar_pts, camera_visibility))

                    label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
                    bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)
                    vel_array = np.array([bbox[2] for bbox in bbox_list]).reshape(-1, 2)
                    vehicle_id_array = np.array([bbox[3] for bbox in bbox_list]).reshape(-1)
                    num_lidar_pts_array = np.array([bbox[4] for bbox in bbox_list]).reshape(-1)
                    camera_visibility_array = np.array([bbox[5] for bbox in bbox_list]).reshape(-1)

                    info['gt_names'] = label_array
                    # h w l x y z yaw
                    info['gt_boxes'] = bbox_array
                    info['gt_velocity'] = vel_array
                    info['vehicle_id'] = vehicle_id_array
                    info['num_lidar_pts'] = num_lidar_pts_array
                    info['camera_visibility'] = camera_visibility_array

                train_nusc_infos.append(info)

    if not test:
        for (scenario_type, file_prefix) in mmcv.track_iter_progress(val_list):
            file_list = glob.glob(osp.join(root_path, scenario_type,
                                           'ego_vehicle', 'label', file_prefix) + '/*')
            frame_list = []
            for file_path in file_list:
                frame_list.append(file_path.split('/')[-1].split('.')[0])

            frame_list_last_frame = max([int(frame_list[i].split('_')[-1]) for i in range(len(frame_list))])
            frame_list_filtered = []
            for frame_name in frame_list:
                frame_number = int(frame_name.split('_')[-1])
                if (frame_list_last_frame - frame_number) % 5 == 0:
                # if True:
                    frame_list_filtered.append(frame_name)

            vehicle_names = ['ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure']

            for key in vehicle_names:
                for frame_name in frame_list_filtered:
                    lidar_path = osp.join(root_path, scenario_type, key, 'lidar01', file_prefix,
                                          frame_name + '.npz')
                    bev_path = osp.join(root_path, scenario_type, key, 'BEV_instance_camera', file_prefix,
                                        frame_name + '.npz')

                    calib_path = osp.join(root_path, scenario_type, key, 'calib', file_prefix, frame_name + '.pkl')

                    calib_dict = mmcv.load(calib_path)
                    scene_name = scenario_type + '_' + file_prefix.split('_')[0] + '_' + file_prefix.split('_')[-1]
                    time_stamp = int(frame_name.split('_')[-1])

                    # lidar_prefix = osp.join(key, 'lidar01', file_prefix)
                    lidar_prefix = scenario_type + '_' + key + '_' + frame_name

                    # reverse the y axis
                    y_reverse_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, -1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0, 1.0]
                                                 ]).astype(np.float)
                    # calib_dict['ego_to_world'] = y_reverse_matrix @ calib_dict['ego_to_world']
                    # calib_dict['lidar_to_ego'] = y_reverse_matrix @ calib_dict['lidar_to_ego']
                    info = {
                        'scenario_type': scenario_type,
                        'vehicle_name': key,
                        'scene_name': scene_name,
                        'lidar_prefix': lidar_prefix,
                        'lidar_path': lidar_path,
                        'bev_path': bev_path,
                        'timestamp': time_stamp,
                        'scenario_length': frame_list_last_frame,
                        'cams': dict(),
                        'lidar_to_ego_matrix': calib_dict['lidar_to_ego'],
                        'ego_to_world_matrix': calib_dict['ego_to_world']
                    }

                    camera_types = [
                        'Camera_FrontLeft',
                        'Camera_Front',
                        'Camera_FrontRight',
                        'Camera_BackLeft',
                        'Camera_Back',
                        'Camera_BackRight',
                    ]

                    for cam in camera_types:
                        cam_info = dict()
                        cam_path = osp.join(root_path, scenario_type, key, cam, file_prefix, frame_name + '.jpg')
                        cam_info['image_path'] = cam_path

                        # # reverse the y axis
                        # calib_dict['lidar_to_' + cam] = y_reverse_matrix @ calib_dict['lidar_to_' + cam]

                        cam_info['lidar_to_camera_matrix'] = calib_dict['lidar_to_' + cam]
                        intrinsic = calib_dict['intrinsic_' + cam]
                        # reverse_y = np.array([[1.0, 0.0, 0.0],
                        #                       [0.0, 1.0, 0.0],
                        #                       [0.0, 0.0, -1.0]
                        #                       ]).astype(np.float)
                        # intrinsic = reverse_y @ intrinsic
                        cam_info['camera_intrinsic_matrix'] = intrinsic
                        cam_info['timestamp'] = time_stamp
                        info['cams'].update({cam: cam_info})

                    if not test:
                        bev_path = osp.join(root_path, scenario_type, key, 'BEV_instance_camera', file_prefix,
                                            frame_name + '.npz')
                        info.update(bev_path=bev_path)

                        label_path = osp.join(root_path, scenario_type, key, 'label', file_prefix,
                                              frame_name + '.txt')

                        with open(label_path) as f:
                            lines = [line.rstrip('\n') for line in f]
                        info.update(vehicle_speed_x=float(lines[0].split(' ')[0]))
                        info.update(vehicle_speed_y=float(lines[0].split(' ')[1]))

                        bbox_list = []
                        for line in lines[1:]:
                            if len(line.split(' ')) <= 1:
                                continue
                            # num_pts_list.append((vehicle_class, center_point[0], center_point[1], center_point[2], l, w, h, yaw,
                            #                              vel_other_vec_transformed[0], vel_other_vec_transformed[1], veh_id))
                            cls_label = str(line.split(' ')[0])
                            bbox = line.split(' ')[1:8]

                            bbox = list(map(float, bbox))
                            # original order: h w l x y z yaw
                            # transformed order: x y z l w h -yaw

                            # # for carla mini
                            # bbox = [bbox[3], bbox[4], bbox[5], bbox[2], bbox[1],
                            #                     bbox[0], -bbox[6]]

                            # # for carla full
                            # bbox = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], -bbox[6]]

                            # reverse the y axis
                            bbox[6] = -bbox[6]
                            # bbox[1] = -bbox[1]
                            # if bbox[6] > 0:
                            #     bbox[6] = 3.1415927 - bbox[6]
                            # else:
                            #     bbox[6] = -3.1415927 - bbox[6]
                            # bbox[6] = -bbox[6]

                            # for carla full
                            bbox = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]]


                            vel = line.split(' ')[8:10]
                            vel = list(map(float, vel))
                            # reverse the y axis
                            vel = [vel[0], -vel[1]]
                            # vel = [0, 0]

                            vehicle_id = int(line.split(' ')[-3])
                            num_lidar_pts = int(line.split(' ')[-2])
                            if line.split(' ')[-1] == 'True':
                                camera_visibility = 1
                            else:
                                camera_visibility = 0
                            bbox_list.append((cls_label, bbox, vel, vehicle_id, num_lidar_pts, camera_visibility))

                        label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
                        bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)
                        vel_array = np.array([bbox[2] for bbox in bbox_list]).reshape(-1, 2)
                        vehicle_id_array = np.array([bbox[3] for bbox in bbox_list]).reshape(-1)
                        num_lidar_pts_array = np.array([bbox[4] for bbox in bbox_list]).reshape(-1)
                        camera_visibility_array = np.array([bbox[5] for bbox in bbox_list]).reshape(-1)

                        info['gt_names'] = label_array
                        # h w l x y z yaw
                        info['gt_boxes'] = bbox_array
                        info['gt_velocity'] = vel_array
                        info['vehicle_id'] = vehicle_id_array
                        info['num_lidar_pts'] = num_lidar_pts_array
                        info['camera_visibility'] = camera_visibility_array

                    val_nusc_infos.append(info)
    if len(train_nusc_infos) > 0:
        random.shuffle(train_nusc_infos)
    if len(val_nusc_infos) > 0:
        random.shuffle(val_nusc_infos)
    return train_nusc_infos, val_nusc_infos


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'Camera_FrontLeft',
        'Camera_Front',
        'Camera_FrontRight',
        'Camera_BackLeft',
        'Camera_Back',
        'Camera_BackRight',
    ]
    carla_infos = mmcv.load(info_path)['infos']
    # nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [
        dict(id=carla_categories.index(cat_name), name=cat_name)
        for cat_name in carla_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    # pdb.set_trace()
    for info in mmcv.track_iter_progress(carla_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['image_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['image_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec