import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

from ..datasets.utils.instance import predict_instance_segmentation_and_trajectories, predict_instance_segmentation_and_trajectories_accident
import math
from pprint import pprint
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours, approximate_polygon
import matplotlib.pyplot as plt
import shapely.geometry as geo
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sympy import Point, Polygon
import numpy as np
import pdb


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    # multi-task settings
    test_mode = data_loader.dataset.test_submission

    task_enable = model.module.pts_bbox_head.task_enbale
    det_enable = task_enable.get('3dod', False)
    map_enable = task_enable.get('map', False)
    motion_enable = task_enable.get('motion', False)
    det_results = []

    if test_mode:
        map_enable = False
        motion_enable = False

    # define metrics
    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

    if motion_enable:
        # evaluate motion in (short, long) ranges
        EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        num_motion_class = 2

        motion_panoptic_metrics = {}
        motion_iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            motion_panoptic_metrics[key] = PanopticMetric(
                n_classes=num_motion_class, temporally_consistent=True).cuda()
            motion_iou_metrics[key] = IntersectionOverUnion(
                num_motion_class).cuda()
        motion_prediction_metrics = {"true_positive": [], "false_positive": [], "false_negative": [],
                                     "minADE": [], "pADE": [], "brierADE": [], "minFDE": [], "pFDE": [],
                                     "brierFDE": [], "MR": [], "p-MR": [], "true_positive_calculation": []}
        accident_metrics = {"true_positive": [], "false_positive": [], "false_negative": [], "id_error_sum": [],
                            "position_error_sum": [], "time_error_sum": []}

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    coordinate_system = dataset.coordinate_system
    if show:
        if test_mode:
            default_out_dir = 'test_visualize'
        else:
            default_out_dir = 'eval_visualize'

        out_dir = out_dir or default_out_dir
        visualizer = Visualizer(
            out_dir=out_dir, coordinate_system=coordinate_system)

    time.sleep(2)  # This line can prevent deadlock problem in some cases.


    for i, data in enumerate(data_loader):
        data_prefix = data['img_metas'][0].data[0][0]['sample_idx']
        town_name = data_prefix.split('_type')[0].split('_')[-1]
        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                img_metas=data['img_metas'],
                img_inputs=data['img_inputs'],
                future_egomotions=data['future_egomotions'],
                img_is_valid=data['img_is_valid'][0],
            )
            # probs = np.zeros(len(result['motion_predictions']))
            # for i, sig in enumerate(range(-2, 3)):
            #     probs[i] = scipy.stats.norm(0, 1).pdf(sig)
            # # probs = np.exp(probs)/sum(np.exp(probs)) #softmax probability
            # probs /= probs[len(result['motion_predictions']) // 2]  # let mean trajectory prob=1
            probs = np.ones(1)

        if det_enable:
            det_results.extend(result['bbox_results'])

        if map_enable:
            pred_semantic_indices = result['pred_semantic_indices']
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        if motion_enable:
            motion_segmentation, motion_instance = result['motion_segmentation'], result['motion_instance']
            has_invalid_frame = data['has_invalid_frame'][0]
            # valid future frames < n_future_frame, skip the evaluation
            if not has_invalid_frame.item():

                # generate targets
                motion_targets = {
                    'motion_segmentation': data['motion_segmentation'][0],
                    'motion_instance': data['motion_instance'][0],
                    'instance_centerness': data['instance_centerness'][0],
                    'instance_offset': data['instance_offset'][0],
                    'instance_flow': data['instance_flow'][0],
                    'future_egomotion': data['future_egomotions'][0],
                }
                motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                    motion_targets, center_sampling_mode='bilinear')

                for key, grid in EVALUATION_RANGES.items():
                    limits = slice(grid[0], grid[1])
                    motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous(
                    ), motion_labels['instance'][..., limits, limits].contiguous().cuda())

                    motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                    ), motion_labels['segmentation'][..., limits, limits].contiguous().cuda())

        if show:
            # target_semantic_indices = data['semantic_indices'][0].cuda()
            # motion_targets = {
            #     'motion_segmentation': data['motion_segmentation'][0],
            #     'motion_instance': data['motion_instance'][0],
            #     'instance_centerness': data['instance_centerness'][0],
            #     'instance_offset': data['instance_offset'][0],
            #     'instance_flow': data['instance_flow'][0],
            #     'future_egomotion': data['future_egomotions'][0],
            # }
            # motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
            #     motion_targets, center_sampling_mode='bilinear')

            # visualizer.visualize_beverse(
            #     img_metas=data['img_metas'][0].data[0][0],
            #     bbox_results=result['bbox_results'][0],
            #     gt_bboxes_3d=data['gt_bboxes_3d'][0],
            #     gt_labels_3d=data['gt_labels_3d'][0],
            #     map_labels=target_semantic_indices,
            #     map_preds=result['pred_semantic_indices'],
            #     motion_labels=motion_labels,
            #     motion_preds=result['motion_predictions'],
            #     save_path='beverse_demo_visualize_v2/{}'.format(
            #         data['img_metas'][0].data[0][0]['sample_idx'])
            # )

            motion_targets = {
                'motion_segmentation': data['motion_segmentation'][0],
                'motion_instance': data['motion_instance'][0],
                'instance_centerness': data['instance_centerness'][0],
                'instance_offset': data['instance_offset'][0],
                'instance_flow': data['instance_flow'][0],
                'future_egomotion': data['future_egomotions'][0],
            }

            # motion_targets = {
            #     'motion_segmentation': data['motion_segmentation_vis'][0],
            #     'motion_instance': data['motion_instance_vis'][0],
            #     'instance_centerness': data['instance_centerness_vis'][0],
            #     'instance_offset': data['instance_offset_vis'][0],
            #     'instance_flow': data['instance_flow_vis'][0],
            #     'future_egomotion': data['future_egomotions'][0],
            # }
            motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                motion_targets, center_sampling_mode='bilinear')

            img_metas = data['img_metas'][0].data[0][0]
            visualizer.visualize_beverse(
                img_metas=img_metas,
                bbox_results=result['bbox_results'][0],
                gt_bboxes_3d=data['gt_bboxes_3d'][0],
                gt_labels_3d=data['gt_labels_3d'][0],
                motion_labels=motion_labels,
                motion_preds=result['motion_predictions'],
                # save_path='{:04d}'.format(i+1),
                save_path='{:04d}_'.format(i + 1) + img_metas['sample_idx']
            )

        motion_targets = {
            'motion_segmentation': data['motion_segmentation'][0],
            'motion_instance': data['motion_instance'][0],
            'instance_centerness': data['instance_centerness'][0],
            'instance_offset': data['instance_offset'][0],
            'instance_flow': data['instance_flow'][0],
            'future_egomotion': data['future_egomotions'][0],
        }
        motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
            motion_targets, center_sampling_mode='bilinear')

        matchestraj = list()

        # visualize BEV instance trajectory
        segmentation_binary = motion_labels['segmentation']
        segmentation = segmentation_binary.new_zeros(
            segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)
        motion_labels['segmentation'] = segmentation.float() * 10
        motion_labels['instance_center'] = motion_labels['centerness']
        motion_labels['instance_offset'] = motion_labels['offset']
        motion_labels['instance_flow'] = motion_labels['flow']

        # segmentation instance_center instance_offset instance_flow
        consistent_instance_seg_label, matched_centers_label, segmentpixelscnt_label = predict_instance_segmentation_and_trajectories_accident(
            motion_labels, compute_matched_centers=True)

        # if len(result['motion_predictions']) > 1:
        #     motion_preds_result = result['motion_predictions'][-1]
        # else:
        #     motion_preds_result = result['motion_predictions']
        # consistent_instance_seg_pred, matched_centers_pred, segmentpixelscnt_pred = predict_instance_segmentation_and_trajectories_accident(
        #     motion_preds_result, compute_matched_centers=True)

        consistent_instance_seg_pred_list, matched_centers_pred_list, segmentpixelscnt_pred_list = [], [], []
        for i in range(len(result['motion_predictions'])):
            consistent_instance_seg_pred, matched_centers_pred, segmentpixelscnt_pred = predict_instance_segmentation_and_trajectories_accident(
                result['motion_predictions'][i], compute_matched_centers=True)
            consistent_instance_seg_pred_list.append(consistent_instance_seg_pred)
            matched_centers_pred_list.append(matched_centers_pred)
            segmentpixelscnt_pred_list.append(segmentpixelscnt_pred)

        # matches, distances, unmpred, unmgt = traj_mapping(matched_centers_pred, matched_centers_label,
        #                                                   consistent_instance_seg_pred,
        #                                                   consistent_instance_seg_label)

        for i in range(len(result['motion_predictions'])):
            matches, distances, unmpred, unmgt = traj_mapping(matched_centers_pred_list[i], matched_centers_label,
                                                              consistent_instance_seg_pred_list[i],
                                                              consistent_instance_seg_label)

            matchestraj.append(matches)


        mappingall = dict()
        for gid in matched_centers_label.keys():
            # mappingall[gid] = [-1] * len(result['motion_predictions'])
            mappingall[gid] = [-1]
            for idx, m in enumerate(matchestraj):
                for p, g in m:
                    if g == gid:
                        mappingall[gid][idx] = p
                        break

        metrics = [len(matches), len(unmpred), len(unmgt)]
        # pdb.set_trace()
        metrics += eval_metrics(matched_centers_pred, matched_centers_label, mappingall, probs)
        # metrics += eval_metrics(matched_centers_pred_list, matched_centers_label, mappingall, probs)

        for val, metric in zip(metrics, motion_prediction_metrics.keys()):
            motion_prediction_metrics[metric].append(val)

        # eval_accidents(matched_centers_pred, matched_centers_label, consistent_instance_seg_pred, consistent_instance_seg_label)

        accident_metric_single = \
            eval_accidents(matched_centers_pred, matched_centers_label, consistent_instance_seg_pred,
                                      consistent_instance_seg_label, matchestraj)
        for val, metric in zip(accident_metric_single, accident_metrics.keys()):
            accident_metrics[metric].append(val)


        if rank == 0:
            for _ in range(data_loader.batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if det_enable:
        if gpu_collect:
            det_results = collect_results_gpu(det_results, len(dataset))
        else:
            det_results = collect_results_cpu(
                det_results, len(dataset), tmpdir)

    if map_enable:
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        if rank == 0:
            print('\n[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
                len(dataset), len(dataset), scores, mIoU,
            ))

    if motion_enable:
        if rank == 0:
            print(
                '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(len(dataset), len(dataset)))

        for key, grid in EVALUATION_RANGES.items():
            results_str = 'grid = {}: '.format(key)

            panoptic_scores = motion_panoptic_metrics[key].compute()
            iou_scores = motion_iou_metrics[key].compute()

            # logging
            if rank == 0:
                results_str += 'iou = {:.3f}, '.format(
                    iou_scores[1].item() * 100)
                for panoptic_key, value in panoptic_scores.items():
                    results_str += '{} = {:.3f}, '.format(
                        panoptic_key, value[1].item() * 100)
                print(results_str)

        print("Evaluation metrics for motion prediction:")
        total_tp = sum(motion_prediction_metrics['true_positive'])
        total_fp = sum(motion_prediction_metrics['false_positive'])
        total_fn = sum(motion_prediction_metrics['false_negative'])
        motion_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        print("motion total tp: %d, total fn: %d, total total_fp: %d" % (total_tp, total_fn, total_fp))
        print("motion_accuracy: %.4f" % (motion_accuracy))

        total_tp_cal = sum(motion_prediction_metrics['true_positive_calculation'])

        if total_tp > 0:
            avg_Ade_error = sum(motion_prediction_metrics['minADE']) / total_tp_cal
            avg_pAde_error = sum(motion_prediction_metrics['pADE']) / total_tp_cal
            avg_brierAde_error = sum(motion_prediction_metrics['brierADE']) / total_tp_cal
            avg_Fde_error = sum(motion_prediction_metrics['minFDE']) / total_tp_cal
            avg_pFde_error = sum(motion_prediction_metrics['pFDE']) / total_tp_cal
            avg_brierFde_error = sum(motion_prediction_metrics['brierFDE']) / total_tp_cal
            avg_MR_error = sum(motion_prediction_metrics['MR']) / total_tp_cal
            avg_pMR_error = sum(motion_prediction_metrics['p-MR']) / total_tp_cal
        else:
            avg_Ade_error = 10000
            avg_pAde_error = 10000
            avg_brierAde_error = 10000
            avg_Fde_error = 10000
            avg_pFde_error = 10000
            avg_brierFde_error = 10000
            avg_MR_error = 1
            avg_pMR_error = 1

        print("avg_Ade_error: %.4f" % (avg_Ade_error))
        print("avg_pAde_error: %.4f" % (avg_pAde_error))
        print("avg_brierAde_error: %.4f" % (avg_brierAde_error))
        print("avg_Fde_error: %.4f" % (avg_Fde_error))
        print("avg_pFde_error: %.4f" % (avg_pFde_error))
        print("avg_brierFde_error: %.4f" % (avg_brierFde_error))
        print("avg_MR_error: %.4f" % (avg_MR_error))
        print("avg_pMR_error: %.4f \n" % (avg_pMR_error))

        print("Evaluation metrics for accident prediction:")
        total_tp = sum(accident_metrics['true_positive'])
        total_fp = sum(accident_metrics['false_positive'])
        total_fn = sum(accident_metrics['false_negative'])
        accident_accuracy = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        print("total tp: %d, total fn: %d, total total_fp: %d"%(total_tp, total_fn, total_fp))
        print("accident_accuracy: %.4f"%(accident_accuracy))

        if total_tp > 0:
            avg_id_error = sum(accident_metrics['id_error_sum']) / total_tp
            avg_position_error = sum(accident_metrics['position_error_sum']) / total_tp
            avg_time_error = sum(accident_metrics['time_error_sum']) / total_tp
        else:
            avg_id_error = 1000
            avg_position_error = 1000
            avg_time_error = 1000

        print("avg_id_error: %.4f"%(avg_id_error))
        print("avg_position_error: %.4f" % (avg_position_error))
        print("avg_time_error: %.4f \n" % (avg_time_error))

    return det_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def poly_distance(id_A, id_B, t, instance_giou):
    instance_id1 = np.where(instance_giou[t]==id_A, id_A, 0)
    if not np.any(instance_id1):
        return 10e10
    contours = find_contours(instance_id1, 0.1)
    pred_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]
    if len(pred_contours[0]) < 3:
        return 10e10
    predpolygon = geo.Polygon(pred_contours[0])

    instance_id2 = np.where(instance_giou[t]==id_B, id_B, 0)
    if not np.any(instance_id2):
        return 10e10
    contours = find_contours(instance_id2, 0.1)
    gt_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]
    if len(gt_contours[0]) < 3:
        return 10e10
    gtpolygon = geo.Polygon(gt_contours[0])

    return gtpolygon.distance(predpolygon)

# Helper function: whether agent pairs b/w gt and pred are matching
def matchingpair(predagents, gtagents, matchestraj):
    match = 0
    for pa in predagents:
        for ga in gtagents:
            if [pa, ga] in matchestraj:
                match += 1
                break

    return True if match == 2 else False

# Evaluate accident prediction: slide 9 of the shared powerpoint
def eval_accidents(pred_trajs_list, gt_trajs, consistent_instance_seg_pred_list, consistent_instance_seg_label, matchestraj):

    pred_instance_giou_list = []
    number_of_predictions = len(pred_trajs_list)

    for i in range(number_of_predictions):
        pred_instance_giou1 = consistent_instance_seg_pred_list[i][0].clone().detach().cpu().numpy()
        pred_instance_giou_list.append(pred_instance_giou1)

    gt_instance_giou = consistent_instance_seg_label[0].clone().detach().cpu().numpy()
    sequence_length = max(len(t) for t in gt_trajs.values())
    # dist_threshold = 1  # pixel distance for accident
    # dist_threshold = 0.001  # pixel distance for accident

    dist_threshold_gt = 0.001 # pixel distance for accident
    dist_threshold_pred = 5  # pixel distance for accident

    # dist_threshold_gt = 0.5  # pixel distance for accident
    # dist_threshold_pred = dist_threshold_gt  # pixel distance for accident

    bev_resolution = 0.5 # m/pixel
    seconds_per_sample = 0.5 # seconds

    # tp_position_threshold = 20 # 10 meters per agent

    # tp_position_thresholds = [5, 10, 15, 20]

    tp_position_thresholds = [5, 10, 15]

    num_threshold = len(tp_position_thresholds)
    true_positive = [0] * num_threshold
    false_positive = [0] * num_threshold
    false_negative = [0] * num_threshold

    id_error_sum = [0] * num_threshold
    position_error_sum = [0] * num_threshold
    time_error_sum = [0] * num_threshold

    max_time = 3 * seconds_per_sample


    for t in range(sequence_length):
        gt_accident = list()
        pred_accident_total = list()
        for idx, (gt_id_A, gt_traj_A) in enumerate(gt_trajs.items()):
            for gt_id_B, gt_traj_B in list(gt_trajs.items())[idx + 1:]:
                gtdist = poly_distance(gt_id_A, gt_id_B, t, gt_instance_giou)
                if gtdist < dist_threshold_gt:
                    gt_accident.append((t, gt_id_A, gt_id_B, gt_trajs[gt_id_A][t], gt_trajs[gt_id_B][t], gtdist))

        # gt_accident.sort(key=lambda x: x[-1])
        # print(gt_accident[0])
        # pdb.set_trace()

        # if there is more than two pairs with accident scenario, choose the one with less distance
    if len(gt_accident) > 1:
        gt_accident.sort(key=lambda x: x[-1])


    if len(pred_trajs_list[0].values()) == 0:
        if len(gt_accident) > 0:
            false_negative = 1
        else:
            false_negative = 0
        return [0] * num_threshold, [0] * num_threshold, [false_negative] * num_threshold, [0] * num_threshold, [0] * num_threshold, [0] * num_threshold

    for pred_trajs, pred_instance_giou in zip(pred_trajs_list, pred_instance_giou_list):
        pred_accident = list()
        for idx, (pred_id_A, pred_traj_A) in enumerate(pred_trajs.items()):
            for pred_id_B, pred_traj_B in list(pred_trajs.items())[idx + 1:]:
                preddist = poly_distance(pred_id_A, pred_id_B, t, pred_instance_giou)
                if preddist < dist_threshold_pred:
                    pred_accident.append(
                        (t, pred_id_A, pred_id_B, pred_trajs[pred_id_A][t], pred_trajs[pred_id_B][t], preddist))
        if len(pred_accident) > 1:
            pred_accident.sort(key=lambda x: x[-1])
        if len(pred_accident) > 0:
            pred_accident = pred_accident[0]
        pred_accident_total.append(pred_accident)

    if len(gt_accident) > 0:
        gt_accident = gt_accident[0]
        # print('\n gt init: ', gt_accident)
        # print('pred init total: ', pred_accident_total)

    # if len(gt_accident) > 0 and len(pred_accident) > 0:
    #     print('gt: ', gt_accident)
    #     print('pred: ', pred_accident)
    #     pdb.set_trace()

    pred_accident_total_valid = []
    for i in range(number_of_predictions):
        if len(pred_accident_total[i]) > 0:
            pred_accident_total_valid.append(i)

    # pdb.set_trace()

    for thre_idx, tp_position_threshold in enumerate(tp_position_thresholds):
        max_position = tp_position_threshold
        if len(gt_accident) > 0:
            if len(pred_accident_total_valid) == 0:
                false_negative[thre_idx] += 1
            else:
                best_position_err = float("inf")
                best_pred_idx = -1

                for i in pred_accident_total_valid:
                    pred_accident = pred_accident_total[i]
                    position_err1 = np.sqrt(
                        (pred_accident[3][0] - gt_accident[3][0]) ** 2 + (pred_accident[3][1] - gt_accident[3][1]) ** 2) + \
                                    np.sqrt((pred_accident[4][0] - gt_accident[4][0]) ** 2 + (
                                            pred_accident[4][1] - gt_accident[4][1]) ** 2)
                    position_err2 = np.sqrt(
                        (pred_accident[3][0] - gt_accident[4][0]) ** 2 + (
                                pred_accident[3][1] - gt_accident[4][1]) ** 2) + \
                                    np.sqrt((pred_accident[4][0] - gt_accident[3][0]) ** 2 + (
                                            pred_accident[4][1] - gt_accident[3][1]) ** 2)
                    position_err = min(position_err1, position_err2) * bev_resolution
                    if position_err < best_position_err:
                        best_position_err = position_err
                        best_pred_idx = i

                if best_position_err <= tp_position_threshold:
                    true_positive[thre_idx] += 1
                    pred_accident = pred_accident_total[best_pred_idx]
                    position_error_sum[thre_idx] += min(best_position_err, max_position)
                    time_error_sum[thre_idx] += min(abs(pred_accident[0] - gt_accident[0]) * seconds_per_sample, max_time)
                    if not matchingpair(pred_accident[1:3], gt_accident[1:3], matchestraj[best_pred_idx]):
                        id_error_sum[thre_idx] += 1
                else:
                    false_positive[thre_idx] += 1
                    # false_negative[thre_idx] += 1
        else:
            # if len(pred_accident_total_valid) != 0:
            if len(pred_accident_total[-1]) != 0:
                false_positive[thre_idx] += 1

    # return accident_acc / sequence_length
    return true_positive, false_positive, false_negative, id_error_sum, position_error_sum, time_error_sum


def eval_metrics(pred_trajs, gt_traj, mapping, prob):
    # https://github.com/argoai/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py
    sequence_length = max(len(t) for t in gt_traj.values())
    # pred_trajs, gt_traj = interpolate_centers(pred_trajs, gt_traj, sequence_length)
    if len(pred_trajs[0].values()) == 0:
        return 0, 0, 0, \
               0, 0, 0, \
               0, 0, 0

    for i in range(len(pred_trajs)):
        pred_trajs[i] = interpolate_centers(pred_trajs[i], sequence_length)

    gt_traj = interpolate_centers(gt_traj, sequence_length)
    miss_threshold = 5.0  # 2mx
    max_ade = 8  # 8 meter
    max_fde = 10  # 10 meter

    minADEs, prob_min_ade, brier_min_ade = list(), list(), list()
    minFDEs, prob_min_fde, brier_min_fde = list(), list(), list()
    n_misses, prob_n_misses = list(), list()

    true_positive = 0

    for gid in gt_traj.keys():
        # if sum(mapping[gid]) == -5:  # no trajectories predicted in all 5 scenarios
        #     continue  # TODO: maybe giving penalty?
        minFDE, minidx, minpid = float("inf"), -1, -1

        if sum(mapping[gid])/ len(mapping[gid]) == -1:  # no trajectories predicted in all 5 scenarios

            continue  # TODO: maybe giving penalty?

        true_positive += 1
        for sig, pid in enumerate(mapping[gid]):
            if pid == -1:
                continue
            FDE = math.sqrt(
                (pred_trajs[sig][pid][-1][0] - gt_traj[gid][-1][0]) ** 2
                + (pred_trajs[sig][pid][-1][1] - gt_traj[gid][-1][1]) ** 2
            )

            if FDE < minFDE:
                minFDE = FDE
                # minpid = pid
                minidx, minpid = sig, pid

        minADE = float(
            sum(
                math.sqrt(
                    (pred_trajs[minidx][minpid][i][0] - gt_traj[gid][i][0]) ** 2
                    + (pred_trajs[minidx][minpid][i][1] - gt_traj[gid][i][1]) ** 2
                )
                for i in range(sequence_length)
            )
            / sequence_length
        )

        minADE *= 0.5  # convert from bev pixel value to real meter distance
        minFDE *= 0.5  # convert from bev pixel value to real meter distance
        minADE = min(minADE, max_ade)
        minFDE = min(minFDE, max_fde)

        n_misses.append(minFDE > miss_threshold)
        prob_n_misses.append(1.0 if minFDE > miss_threshold else (1.0 - prob[minidx]))
        minADEs.append(minADE)
        prob_min_ade.append(min(-np.log(prob[minidx]), -np.log(0.05)) + minADE)
        brier_min_ade.append((1 - prob[minidx]) ** 2 + minADE)
        minFDEs.append(minFDE)
        prob_min_fde.append(min(-np.log(prob[minidx]), -np.log(0.05)) + minFDE)
        brier_min_fde.append((1 - prob[minidx]) ** 2 + minFDE)


    return sum(minADEs), sum(prob_min_ade), sum(brier_min_ade), \
           sum(minFDEs), sum(prob_min_fde), sum(brier_min_fde), \
           sum(n_misses), sum(prob_n_misses), true_positive

# Interpolate centres of all instances' trajectories
# Example: {1:[[120,60], [121,61]]} => {1:[[120,60], [121,61], [122,62], [123,63], [124,64]]}
def interpolate_centers(trajs, sequence_length):
    for vid, traj in trajs.items():
        if len(traj) != sequence_length:
            if len(traj) == 1:
                trajs[vid] = [traj[0] for _ in range(sequence_length)]
            else:
                # dx, dy = [], []
                #
                # for t in range(len(traj) - 1):
                #     dx.append(traj[t + 1][0] - traj[t][0])
                #     dy.append(traj[t + 1][1] - traj[t][1])
                # dxmean, dymean = sum(dx) / len(dx), sum(dy) / len(dy)

                dxmean, dymean = traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1]

                tointerpolate = sequence_length - len(traj)
                for index in range(tointerpolate):
                    trajs[vid].append([trajs[vid][-1][0] + (index+1) * dxmean, trajs[vid][-1][1] + (index+1) * dymean])

    return trajs

def traj_mapping(pred_trajs, gt_trajs, pred_seg, gt_seg):
    distance_matrix = np.zeros((len(pred_trajs) + 1, len(gt_trajs) + 1))
    matched_indices = list()
    # sequence_length_pred = max(len(t) for t in pred_trajs.values())
    # if len(gt_trajs.keys())==0:
    #     pdb.set_trace()
    sequence_length = max(len(t) for t in gt_trajs.values())
    # sequence_length = max(sequence_length, sequence_length_pred)

    # pred_trajs, gt_trajs = interpolate_centers(pred_trajs, gt_trajs, sequence_length)
    gt_trajs = interpolate_centers(gt_trajs, sequence_length)

    if len(pred_trajs.values()) == 0:
        unmatched_gts = list()
        for gid in gt_trajs.keys():
            unmatched_gts.append(gid)
        return [], [], [], np.array(unmatched_gts)

    pred_trajs = interpolate_centers(pred_trajs, sequence_length)
    matching = "L2"
    bev_resolution = 0.5 # 0.5m per bev grid

    if matching == "L2":
        for pid, ptraj in pred_trajs.items():
            for gid, gtraj in gt_trajs.items():
                for t in range(sequence_length):
                    diff = (np.array(ptraj[t]) - np.array(gtraj[t])) * bev_resolution
                    distance_matrix[pid, gid] += np.sqrt(np.dot(diff.T, diff))

        # distance_matrix = distance_matrix[1:,1:]
        num_preds, num_gts = distance_matrix.shape
        dist_threshold_per_frame = 4 # meter
        dist_threshold = dist_threshold_per_frame * sequence_length

        # association in the greedy manner
        # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
        distance_1d = distance_matrix.reshape(-1)
        index_1d = np.argsort(distance_1d)
        index_2d = np.stack([index_1d // num_gts, index_1d % num_gts], axis=1)
        pred_id_matches_to_gt_id = [-1] * num_preds
        gt_id_matches_to_pred_id = [-1] * num_gts
        for sort_i in range(index_2d.shape[0]):
            pred_id = int(index_2d[sort_i][0])
            gt_id = int(index_2d[sort_i][1])
            if gt_id_matches_to_pred_id[gt_id] == -1 and pred_id_matches_to_gt_id[pred_id] == -1:
                gt_id_matches_to_pred_id[gt_id] = pred_id
                pred_id_matches_to_gt_id[pred_id] = gt_id
                matched_indices.append([pred_id, gt_id])
        if len(matched_indices) == 0:
            matched_indices = np.empty((0, 2))
        else:
            matched_indices = np.asarray(matched_indices)
        # matched_indices += 1

        unmatched_preds = list()
        for pid in pred_trajs.keys():
            if pid not in matched_indices[:, 0]:
                unmatched_preds.append(pid)

        unmatched_gts = list()
        for gid in gt_trajs.keys():
            if gid not in matched_indices[:, 1]:
                unmatched_gts.append(gid)

        matches = list()
        distances = list()
        for m in matched_indices:
            if distance_matrix[m[0], m[1]] > dist_threshold:
                unmatched_preds.append(m[0])
                unmatched_gts.append(m[1])
            else:
                distances.append((m.reshape(2).tolist(), distance_matrix[m[0], m[1]].tolist()))
                matches.append(m.reshape(2).tolist())

    elif matching == "GIOU":
        pred_instance_giou = pred_seg[0].clone().detach().cpu().numpy()
        gt_instance_giou = gt_seg[0].clone().detach().cpu().numpy()
        giou_dist_threshold = 1.5

        giou_matrix = np.zeros((len(pred_trajs) + 1, len(gt_trajs) + 1))
        for t in range(sequence_length):
            for pred_id in range(len(pred_trajs) + 1):
                for gt_id in range(len(gt_trajs) + 1):
                    pred_instance_id = np.where(pred_instance_giou[t] == pred_id, pred_id, 0)
                    gt_instance_id = np.where(gt_instance_giou[t] == gt_id, gt_id, 0)
                    giou_matrix[pred_id, gt_id] += giou2d(pred_instance_id, gt_instance_id)

        giou_matrix /= 5
        distance_matrix = 1 - giou_matrix
        row_ind, col_ind = linear_sum_assignment(distance_matrix[1:, 1:])
        matched_indices = np.stack([row_ind + 1, col_ind + 1], axis=1)

        unmatched_preds = list()
        for pid in pred_trajs.keys():
            if pid not in matched_indices[:, 0]:
                unmatched_preds.append(pid)

        unmatched_gts = list()
        for gid in gt_trajs.keys():
            if gid not in matched_indices[:, 1]:
                unmatched_gts.append(gid)

        gioumatches = list()
        for m in matched_indices:
            if distance_matrix[m[0], m[1]] > giou_dist_threshold:
                unmatched_preds.append(m[0])
                unmatched_gts.append(m[1])
            else:
                gioumatches.append(m.reshape(2).tolist())

    # return sorted(l2matches)[1:], sorted(gioumatches)
    return matches[1:], distances, np.array(unmatched_preds), np.array(unmatched_gts)


def create_mapping(pred_instance, gt_instance, save_path):
    batch_size, sequence_length = gt_instance.shape[:2]
    # Process labels
    assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
    pred_segmentation = (pred_instance > 0).long()
    gt_segmentation = (gt_instance > 0).long()

    for b in range(batch_size):
        unique_id_mapping = {}
        ungt_dict, unpred_dict, gtuq_dict, preduq_dict = {}, {}, {}, {}
        evaluation = "GIOU"

        # print(torch.unique(pred_instance).detach().cpu().numpy()[1:])
        # print(torch.unique(gt_instance).detach().cpu().numpy()[1:])
        # exit()

        result, mapping, ungt, unpred, gt_unique, pred_unique = hota_mapping(
            pred_segmentation[b].detach(),
            pred_instance[b].detach(),
            gt_segmentation[b],
            gt_instance[b],
            unique_id_mapping,
            evaluation,
            save_path,
        )

    # prediction center tracking
    matched_centers = dict()
    segmentpixelscnt = dict()
    _, seq_len, h, w = pred_instance.shape
    grid = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=pred_instance.device),
        torch.arange(w, dtype=torch.float, device=pred_instance.device),
        indexing='ij',
    ))

    # for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
    for t in range(seq_len):
        max_id = torch.unique(pred_instance[0, t])[1:].cpu().numpy().max()
        segmentpixelscnt[t] = [-1 for _ in range(max_id + 1)]
        matched_centers[t] = [-1 for _ in range(max_id + 1)]
        for instance_id in torch.unique(pred_instance[0, t])[1:].cpu().numpy():
            instance_mask = pred_instance[0, t] == instance_id
            segmentpixelscnt[t][instance_id] = instance_mask.sum().detach().cpu().item()
            if instance_mask.sum() > 0:
                matched_centers[t][instance_id] = grid[:, instance_mask].mean(dim=-1).detach().cpu().numpy().tolist()

    # ground truth center tracking
    gt_matched_centers = dict()
    gt_segmentpixelscnt = dict()
    _, seq_len, h, w = gt_instance.shape
    gt_grid = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=gt_instance.device),
        torch.arange(w, dtype=torch.float, device=gt_instance.device),
        indexing='ij',
    ))

    # for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
    for t in range(seq_len):
        max_id = torch.unique(gt_instance[0, t])[1:].cpu().numpy().max()
        gt_segmentpixelscnt[t] = [-1 for _ in range(max_id + 1)]
        gt_matched_centers[t] = [-1 for _ in range(max_id + 1)]
        for instance_id in torch.unique(gt_instance[0, t])[1:].cpu().numpy():
            instance_mask = gt_instance[0, t] == instance_id
            gt_segmentpixelscnt[t][instance_id] = instance_mask.sum().detach().cpu().item()
            if instance_mask.sum() > 0:
                gt_matched_centers[t][instance_id] = gt_grid[:, instance_mask].mean(
                    dim=-1).detach().cpu().numpy().tolist()

    return gt_segmentpixelscnt, gt_matched_centers, segmentpixelscnt, matched_centers, unique_id_mapping, ungt_dict, unpred_dict, gtuq_dict, preduq_dict


def _mapping(pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping, evaluation, save_path):
    """
    Computes panoptic quality metric components.

    Parameters
    ----------
        pred_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
        pred_instance: [H, W] range {0, ..., n_instances} (zero means background)
        gt_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
        gt_instance: [H, W] range {0, ..., n_instances} (zero means background)
        unique_id_mapping: instance id mapping to check consistency
    """
    assert pred_instance.shape == gt_instance.shape

    n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())

    if evaluation == "GIOU":
        pred_instance_giou = pred_instance.clone().detach().cpu().numpy()
        gt_instance_giou = gt_instance.clone().detach().cpu().numpy()

        giou_matrix = np.zeros((int(gt_instance.max().item()) + 1, int(pred_instance.max().item()) + 1))
        for gt_id in torch.unique(gt_instance).detach().cpu().numpy()[1:]:
            for pred_id in torch.unique(pred_instance).detach().cpu().numpy()[1:]:
                gt_instance_id = np.where(gt_instance_giou == gt_id, gt_id, 0)
                pred_instance_id = np.where(pred_instance_giou == pred_id, pred_id, 0)
                giou_matrix[gt_id, pred_id] = giou2d(gt_instance_id, pred_instance_id)

        dist_matrix = 1 - giou_matrix
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)

        for v_id in torch.unique(pred_instance).detach().cpu().numpy()[1:]:
            pred_instance_per_id = np.where(pred_instance_giou == v_id, v_id, 0)
            contours = find_contours(pred_instance_per_id, 0.1)
            simple_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]

            # visualization (debugging purpose)
            fig, ax = plt.subplots()
            for contour in simple_contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.imshow(pred_instance.detach().cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(f'{save_path}/pred{v_id}.png')

        for v_id in torch.unique(gt_instance).detach().cpu().numpy()[1:]:
            gt_instance_per_id = np.where(gt_instance_giou == v_id, v_id, 0)
            contours = find_contours(gt_instance_per_id, 0.1)
            simple_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]

            # visualization (debugging purpose)
            fig, ax = plt.subplots()
            for contour in simple_contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.imshow(gt_instance.detach().cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(f'{save_path}/gt{v_id}.png')


    elif evaluation == "IOU":
        pred_instance = pred_instance.clone().view(-1)
        gt_instance = gt_instance.clone().view(-1)
        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = pred_instance + (n_instances + 1) * gt_instance
        bincount_2d = torch.bincount(
            x.long(), minlength=(n_instances + 1) ** 2)
        if bincount_2d.shape[0] != (n_instances + 1) ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_instances + 1, n_instances + 1))
        # Drop void class
        # conf = conf[1:, 1:]

        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) /
                          (union.float() + 1e-9), torch.zeros_like(union).float())

        dist_matrix = 1 - iou
        row_ind, col_ind = linear_sum_assignment(dist_matrix.detach().cpu())
        matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_gts = list()
    for d in range(int(gt_instance.max().item())):
        if d not in matched_indices[:, 0]:
            unmatched_gts.append(d)

    unmatched_preds = list()
    for t in range(int(pred_instance.max().item())):
        if t not in matched_indices[:, 1]:
            unmatched_preds.append(t)

    matches = list()
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > 1.5:  # threshold filtering
            unmatched_gts.append(m[0])
            unmatched_preds.append(m[1])
        else:
            if m[0] in torch.unique(gt_instance).detach().cpu().numpy()[1:] and m[1] in torch.unique(
                    pred_instance).detach().cpu().numpy()[1:]:
                matches.append(m.reshape(2).tolist())

    unmatched_gts, unmatched_preds = np.array(unmatched_gts), np.array(unmatched_preds)

    gt_filter = list()
    for i, gt in enumerate(unmatched_gts):
        if gt not in torch.unique(gt_instance).detach().cpu().numpy()[1:]:
            gt_filter.append(i)

    unmatched_gts = np.delete(unmatched_gts, gt_filter)

    pred_filter = list()
    for i, pred in enumerate(unmatched_preds):
        if pred not in torch.unique(pred_instance).detach().cpu().numpy()[1:]:
            pred_filter.append(i)
    unmatched_preds = np.delete(unmatched_preds, pred_filter)

    gt_unique = torch.unique(gt_instance).detach().cpu().numpy()[1:]
    pred_unique = torch.unique(pred_instance).detach().cpu().numpy()[1:]
    return None, matches, unmatched_gts, unmatched_preds, gt_unique, pred_unique

    # In the iou matrix, first dimension is target idx, second dimension is pred idx.
    # Mapping will contain a tuple that maps prediction idx to target idx for segments matched by iou.
    mapping = (iou > 0.1).nonzero(as_tuple=False)

    return None, mapping

    ###

    # Check that classes match.
    is_matching = pred_to_cls[mapping[:, 1]
                  ] == target_to_cls[mapping[:, 0]]
    mapping = mapping[is_matching]
    tp_mask = torch.zeros_like(conf, dtype=torch.bool)
    tp_mask[mapping[:, 0], mapping[:, 1]] = True

    # First ids correspond to "stuff" i.e. semantic seg.
    # Instance ids are offset accordingly
    for target_id, pred_id in mapping:
        cls_id = pred_to_cls[pred_id]

        if self.temporally_consistent and cls_id == self.vehicles_id:
            if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                # Not temporally consistent
                result['false_negative'][target_to_cls[target_id]] += 1
                result['false_positive'][pred_to_cls[pred_id]] += 1
                unique_id_mapping[target_id.item()] = pred_id.item()
                continue

        result['true_positive'][cls_id] += 1
        result['iou'][cls_id] += iou[target_id][pred_id]
        unique_id_mapping[target_id.item()] = pred_id.item()

    for target_id in range(n_classes, n_all_things):
        # If this is a true positive do nothing.
        if tp_mask[target_id, n_classes:].any():
            continue
        # If this target instance didn't match with any predictions and was present set it as false negative.
        if target_to_cls[target_id] != -1:
            result['false_negative'][target_to_cls[target_id]] += 1

    for pred_id in range(n_classes, n_all_things):
        # If this is a true positive do nothing.
        if tp_mask[n_classes:, pred_id].any():
            continue
        # If this predicted instance didn't match with any prediction, set that predictions as false positive.
        if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
            result['false_positive'][pred_to_cls[pred_id]] += 1

    return result, unique_id_mapping


def giou2d(gt_instance_id, pred_instance_id):
    # boxa_corners = np.array(BBox.box2corners2d(box_a))
    # boxb_corners = np.array(BBox.box2corners2d(box_b))
    contours = find_contours(gt_instance_id, 0.1)
    gt_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]

    contours = find_contours(pred_instance_id, 0.1)
    pred_contours = [approximate_polygon(p, tolerance=0.5) for p in contours]

    if len(gt_contours) == 0 or len(pred_contours) == 0:
        return 0
    else:
        # print(gt_contours, pred_contours)
        reca, recb = Polygon(gt_contours[0]), Polygon(pred_contours[0])

        # compute intersection and union
        I = reca.intersection(recb).area
        # U = box_a.w * box_a.l + box_b.w * box_b.l - I
        U = reca.union(recb).area

        # compute the convex area
        all_corners = np.vstack((gt_contours[0], pred_contours[0]))
        C = ConvexHull(all_corners)
        convex_corners = all_corners[C.vertices]
        convex_area = PolyArea2D(convex_corners)
        C = convex_area

        # compute giou
        return I / U - (C - U) / C


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def flip_rotate_image(image):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.transpose(Image.ROTATE_90)

    return np.array(pil_img)
