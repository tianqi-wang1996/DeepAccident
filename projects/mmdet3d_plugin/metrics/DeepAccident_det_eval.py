# Copyright (c) OpenMMLab. All rights reserved.
import gc
import io as sysio
import numba
import numpy as np
import pdb
import math

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=50):
    scores.sort()
    scores = scores[::-1]
    current_recall = 1.0 / num_sample_pts
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        while (i + 1) / num_gt > current_recall:
            thresholds.append(score)
            current_recall += 1.0 / (num_sample_pts)
    return thresholds

@numba.jit
def get_thresholds_recall(scores: np.ndarray, num_gt, num_sample_pts=50):
    scores.sort()
    scores = scores[::-1]
    current_recall = 1.0 / num_sample_pts
    thresholds = []
    recall = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall

        while (i+1)/num_gt > current_recall:
            # print((i + 1) / num_gt, current_recall)
            thresholds.append(score)
            recall.append(current_recall)
            current_recall += 1.0 / (num_sample_pts)
    return thresholds, recall


def clean_data(gt_anno, dt_anno, current_class, difficulty=None):
    # pdb.set_trace()
    # CLASS_NAMES = ['car', 'motorcycle', 'pedestrian']
    # CLASS_NAMES = ['car', 'motorcycle', 'pedestrian', 'cyclist', 'van', 'truck']
    CLASS_NAMES = [
        'car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian',
        'invalid1', 'invalid2', 'invalid3', 'invalid4'
    ]


    # [0, 30m), [30m, 50m), [50m, +inf)
    VALID_DISTANCE_RANGE = [[0, 30], [30, 50], [50, 120]]

    if difficulty is None:
        VALID_DISTANCE_RANGE = [[0, 120]]
        difficulty = 0

    # MIN_HEIGHT = [40, 25, 25]
    # MAX_OCCLUSION = [0, 1, 2]
    # MAX_TRUNCATION = [0.15, 0.3, 0.5]

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    # pdb.set_trace()
    for i in range(num_gt):
        gt_name = gt_anno['name'][i].lower()

        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        # elif (current_cls_name == 'Pedestrian'.lower()
        #       and 'Person_sitting'.lower() == gt_name):
        #     valid_class = 0
        # elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
        #     valid_class = 0
        else:
            valid_class = -1
            # valid_class = 0
        # if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
        #         or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
        #         or (height <= MIN_HEIGHT[difficulty])):
        #     ignore = True

        ignore = False
        distance = np.sqrt(gt_anno['location'][i][0]**2 + gt_anno['location'][i][0]**2)
        if distance < VALID_DISTANCE_RANGE[difficulty][0] or distance >= VALID_DISTANCE_RANGE[difficulty][1]:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        # if ignore:
        #     ignored_dt.append(1)
        # elif valid_class == 1:
        #     ignored_dt.append(0)
        # else:
        #     ignored_dt.append(-1)

    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        distance = np.sqrt(dt_anno['location'][i][0] ** 2 + dt_anno['location'][i][0] ** 2)
        if distance < VALID_DISTANCE_RANGE[difficulty][0] or distance >= VALID_DISTANCE_RANGE[difficulty][1]:
            ignore = True
        # height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if ignore:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))

                # iw = (
                #     min(boxes[i, 1], qboxes[j, 1]) -
                #     max(boxes[i, 1] - boxes[i, 4],
                #         qboxes[j, 1] - qboxes[j, 4]))

                iw = (
                        min(boxes[i, 2], qboxes[j, 2]) -
                        max(boxes[i, 2] - boxes[i, 5],
                            qboxes[j, 2] - qboxes[j, 5]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    # pdb.set_trace()
    from .rotate_iou import rotate_iou_gpu_eval
    # rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
    #                            qboxes[:, [0, 2, 3, 5, 6]], 2)

    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)

    # pdb.set_trace()

    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    # print(rinc[np.where(rinc>0.01)])
    return rinc

@numba.jit(nopython=True, parallel=True)
def box_distance(gt_boxes, dt_boxes):
    num_gt = gt_boxes.shape[0]
    num_dt = dt_boxes.shape[0]

    distance = 1000 * np.ones((num_gt, num_dt))
    for i in range(num_gt):
        for j in range(num_dt):
            gt_box = gt_boxes[i]
            dt_box = dt_boxes[j]
            distance[i, j] = np.sqrt((gt_box[0] - dt_box[0])**2 + (gt_box[1] - dt_box[1])**2 +
                                     (gt_box[2] - dt_box[2])**2)

    return distance






@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas
    dt_scores = dt_datas

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0

        fp -= nstuff

    return tp, fp, fn, similarity, thresholds[:thresh_idx]

# @numba.jit(nopython=True)
def compute_other_tp_metric(matched_gt, matched_dt):
    # translation_error, scale_error, orientation_error, velocity_error
    # pdb.set_trace()
    difference = matched_gt - matched_dt
    translation_error = np.sqrt((difference[:, 0]) ** 2 + (difference[:, 1]) ** 2
     + (difference[:, 2]) ** 2)
    velocity_error = np.sqrt((difference[:, 7]) ** 2 + (difference[:, 8]) ** 2)
    orientation_error = np.abs(difference[:, 6])
    scale_error = (1 - d3_box_overlap(matched_gt, matched_dt)).diagonal()

    return translation_error[..., np.newaxis], scale_error[..., np.newaxis], orientation_error[..., np.newaxis], \
           velocity_error[..., np.newaxis]

@numba.jit(nopython=True)
def compute_statistics_distance_jit(distances,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           max_distance,
                           thresh=0,
                           compute_fp=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]

    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))

    # calculated_metric = []
    # calculated_metric = 0

    thresh_idx = 0

    # matched_gt_list = []
    # matched_dt_list = []
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        min_distance = 10000
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            distance = distances[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (distance < max_distance)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (distance < max_distance)
                  and (distance < min_distance or assigned_ignored_det)
                  and ignored_det[j] == 0):
                min_distance = distance
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (distance < max_distance)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]

            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0

        fp -= nstuff

    return tp, fp, fn, similarity, thresholds[:thresh_idx]

def compute_statistics_distance_tp(distances,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           max_distance,
                           thresh=0,
                           compute_fp=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]

    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))

    err_trans = np.zeros((gt_size, ))
    err_scale = np.zeros((gt_size,))
    err_orient = np.zeros((gt_size,))
    err_vel = np.zeros((gt_size,))

    # calculated_metric = []
    # calculated_metric = 0

    thresh_idx = 0

    # matched_gt_list = []
    # matched_dt_list = []
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        min_distance = 10000
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            distance = distances[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (distance < max_distance)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (distance < max_distance)
                  and (distance < min_distance or assigned_ignored_det)
                  and ignored_det[j] == 0):
                min_distance = distance
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (distance < max_distance)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]

            err_trans[thresh_idx], err_scale[thresh_idx], err_orient[thresh_idx], err_vel[
                thresh_idx] = calcualte_tp_error(gt_datas[i], dt_datas[det_idx][:-1])

            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0

        fp -= nstuff

    return tp, fp, fn, similarity, thresholds[:thresh_idx], err_trans[:thresh_idx], err_scale[:thresh_idx], \
           err_orient[:thresh_idx], err_vel[:thresh_idx]

# @numba.jit(nopython=True)
def calcualte_tp_error(gt_data, dt_data):
    # dt_data[6] = dt_data[6] + 3.1415926 / 2
    # print(gt_data[6], dt_data[6])
    gt_data[6] = gt_data[6] % (2 * 3.1415926)
    dt_data[6] = dt_data[6] % (2 * 3.1415926)
    difference = gt_data - dt_data
    translation_error = np.sqrt((difference[0]) ** 2 + (difference[1]) ** 2
                                + (difference[2]) ** 2)
    velocity_error = np.sqrt((difference[7]) ** 2 + (difference[8]) ** 2)
    orientation_error = abs(difference[6])
    if math.isnan(orientation_error):
        pdb.set_trace()

    # scale_error = (1 - d3_box_overlap(gt_data[np.newaxis, ...], dt_data[np.newaxis, ...])).diagonal()
    scale_error = (1 - d3_box_overlap(gt_data[np.newaxis, ...], dt_data[np.newaxis, ...]))[0][0]

    return translation_error, scale_error, orientation_error, \
           velocity_error



def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[i]

            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]

            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]

@numba.jit(nopython=True)
def fused_compute_distance_statistics(distances,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             max_distance,
                             thresholds):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            distance = distances[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            # gt_data = gt_datas[i]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]

            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]

            tp, fp, fn, similarity, threshold = compute_statistics_distance_jit(
                distance,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                max_distance=max_distance,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn

            # if len(calculated_metric) > 0:
            #     error_list.append(calculated_metric)
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    # pdb.set_trace()
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
        dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1)
        loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
        dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
        dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1)
        # pdb.set_trace()
        overlap_part = d3_box_overlap(gt_boxes,
                                      dt_boxes).astype(np.float64)

        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def calculate_distance_partly(gt_annos, dt_annos, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    # pdb.set_trace()
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    parted_distance = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        gt_boxes = np.concatenate([a['location'] for a in gt_annos_part], 0)
        dt_boxes = np.concatenate([a['location'] for a in dt_annos_part], 0)

        distance_part = box_distance(gt_boxes, dt_boxes).astype(np.float64)
        parted_distance.append(distance_part)
        example_idx += num_part
    distance = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            distance.append(
                parted_distance[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return distance, parted_distance, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_num_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0

    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_num = gt_annos[i]['name'].shape[0]
        # gt_datas = np.concatenate(
        #     [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        # dt_datas = np.concatenate([
        #     dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
        #     dt_annos[i]['score'][..., np.newaxis]
        # ], 1)
        dt_datas = dt_annos[i]['score']
        gt_num_list.append(gt_num)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_num_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def _prepare_data_distance_matching(gt_annos, dt_annos, current_class):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0

    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt


        # pdb.set_trace()
        gt_datas = np.concatenate(
            [gt_annos[i]['location'], gt_annos[i]['dimensions'], gt_annos[i]['rotation_y'][..., np.newaxis],
             gt_annos[i]['velocity']], 1)
        dt_datas = np.concatenate(
            [dt_annos[i]['location'], dt_annos[i]['dimensions'], dt_annos[i]['rotation_y'][..., np.newaxis],
             dt_annos[i]['velocity'], dt_annos[i]['score'][..., np.newaxis]], 1)

        # gt_num = gt_annos[i]['name'].shape[0]
        # dt_datas = dt_annos[i]['score']

        # gt_datas = np.concatenate(
        #     [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        # dt_datas = np.concatenate([
        #     dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
        #     dt_annos[i]['score'][..., np.newaxis]
        # ], 1)

        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               eval_types,
               min_overlaps,
               distance_threshold,
               num_parts=200):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    # pdb.set_trace()

    N_SAMPLE_PTS = 40

    num_class = len(current_classes)
    mATE = np.zeros([num_class])
    mASE = np.zeros([num_class])
    mAOE = np.zeros([num_class])
    mAVE = np.zeros([num_class])


    if eval_types == 'iou_mAP':
        rets = calculate_iou_partly(dt_annos, gt_annos, num_parts)

        overlaps, parted_overlaps, total_dt_num, total_gt_num = rets


        num_minoverlap = len(min_overlaps)
        num_class = len(current_classes)
        num_difficulty = len(difficultys)
        precision = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        recall = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

        # pdb.set_trace()
        for m, current_class in enumerate(current_classes):
            for idx_l, difficulty in enumerate(difficultys):
                # if m == 1:
                #     pdb.set_trace()
                rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
                # pdb.set_trace()
                (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                 dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(min_overlaps[:, m]):
                    thresholdss = []
                    # pdb.set_trace()
                    for i in range(len(gt_annos)):
                        rets = compute_statistics_jit(
                            overlaps[i],
                            gt_datas_list[i],
                            dt_datas_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            dontcares[i],
                            min_overlap=min_overlap,
                            thresh=0.0,
                            compute_fp=False)
                        tp, fp, fn, similarity, thresholds = rets
                        thresholdss += thresholds.tolist()
                        # thresholdss += thresholds
                    thresholdss = np.array(thresholdss)
                    thresholds = get_thresholds(thresholdss, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS)
                    thresholds = np.array(thresholds)
                    pr = np.zeros([len(thresholds), 4])
                    idx = 0
                    for j, num_part in enumerate(split_parts):
                        # pdb.set_trace()
                        # gt_datas_part = np.concatenate(
                        #     gt_datas_list[idx:idx + num_part], 0)
                        gt_datas_part = np.array(
                            gt_datas_list[idx:idx + num_part])
                        dt_datas_part = np.concatenate(
                            dt_datas_list[idx:idx + num_part], 0)
                        dc_datas_part = np.concatenate(
                            dontcares[idx:idx + num_part], 0)
                        ignored_dets_part = np.concatenate(
                            ignored_dets[idx:idx + num_part], 0)
                        ignored_gts_part = np.concatenate(
                            ignored_gts[idx:idx + num_part], 0)
                        # pdb.set_trace()
                        fused_compute_statistics(
                            parted_overlaps[j],
                            pr,
                            total_gt_num[idx:idx + num_part],
                            total_dt_num[idx:idx + num_part],
                            total_dc_num[idx:idx + num_part],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            min_overlap=min_overlap,
                            thresholds=thresholds)
                        idx += num_part

                    for i in range(len(thresholds)):
                        recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[m, idx_l, k, i] = pr[i, 0] / (
                            pr[i, 0] + pr[i, 1])

                    # for i in range(len(thresholds)):
                    #     precision[m, idx_l, k, i] = np.max(
                    #         precision[m, idx_l, k, i:], axis=-1)
                    #     recall[m, idx_l, k, i] = np.max(
                    #         recall[m, idx_l, k, i:], axis=-1)

        # clean temp variables
        del overlaps
        del parted_overlaps

        gc.collect()

    else:
        # pdb.set_trace()
        rets = calculate_distance_partly(dt_annos, gt_annos, num_parts)

        distance, parted_distance, total_dt_num, total_gt_num = rets

        num_class = len(current_classes)
        num_difficulty = len(difficultys)
        num_distance_threshold = len(distance_threshold)

        # precision = np.zeros(
        #     [num_class, num_difficulty, num_distance_threshold, N_SAMPLE_PTS])
        # recall = np.zeros(
        #     [num_class, num_difficulty, num_distance_threshold, N_SAMPLE_PTS])

        precision = np.zeros(
            [num_class, num_distance_threshold, N_SAMPLE_PTS])
        recall = np.zeros(
            [num_class, num_distance_threshold, N_SAMPLE_PTS])



        # pdb.set_trace()
        for m, current_class in enumerate(current_classes):
            # if m == 1:
            #     pdb.set_trace()
            # pdb.set_trace()
            rets = _prepare_data_distance_matching(gt_annos, dt_annos, current_class)
            # pdb.set_trace()
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, distance_threshold_single in enumerate(distance_threshold):
                thresholdss = []
                # error_list = []
                # pdb.set_trace()
                # calculated_errors_list = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_distance_jit(
                        distance[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        max_distance=distance_threshold_single,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()

                    # if len(calculated_errors) > 0:
                    #     calculated_errors_list.append(calculated_errors)
                    # thresholdss += thresholds
                # pdb.set_trace()

                # calculated_errors_array = np.concatenate((calculated_errors_list), axis=0)
                # tp_mean = calculated_errors_array.mean(axis=0)
                # pdb.set_trace()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    # pdb.set_trace()
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    # gt_datas_part = np.array(
                    #     gt_datas_list[idx:idx + num_part])
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    # pdb.set_trace()
                    fused_compute_distance_statistics(
                        parted_distance[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        max_distance=distance_threshold_single,
                        thresholds=thresholds)
                    idx += num_part
                # pdb.set_trace()
                # error_array = np.concatenate((error_list), axis=0)
                # error_mean = error_array.mean(axis=0)
                for i in range(len(thresholds)):
                    recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, k, i] = pr[i, 0] / (
                            pr[i, 0] + pr[i, 1])

                # pdb.set_trace()
                # for i in range(len(thresholds)):
                #     precision[m, k, i] = np.max(
                #         precision[m, k, i:], axis=-1)
                #     recall[m, k, i] = np.max(
                #         recall[m, k, i:], axis=-1)


        # precision_tp_err = np.zeros(
        #     [num_class, N_SAMPLE_PTS])
        # recall_tp_err = np.zeros(
        #     [num_class, N_SAMPLE_PTS])

        # pdb.set_trace()
        distance_threshold_tp = 2
        for m, current_class in enumerate(current_classes):
            # if m == 1:
            #     pdb.set_trace()
            # pdb.set_trace()
            rets = _prepare_data_distance_matching(gt_annos, dt_annos, current_class)
            # pdb.set_trace()
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets

            thresholdss = []
            error_list = []
            # pdb.set_trace()
            # calculated_errors_list = []
            for i in range(len(gt_annos)):
                rets = compute_statistics_distance_tp(
                    distance[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dontcares[i],
                    max_distance=distance_threshold_tp,
                    thresh=0.0,
                    compute_fp=False)
                tp, fp, fn, similarity, thresholds, err_trans, err_scale, err_orient, err_vel = rets
                # pdb.set_trace()
                error_single = np.stack([err_trans, err_scale, err_orient, err_vel], axis=1)
                error_list.append(error_single)
                thresholdss += thresholds.tolist()

                # if len(calculated_errors) > 0:
                #     calculated_errors_list.append(calculated_errors)
                # thresholdss += thresholds
            # pdb.set_trace()

            # calculated_errors_array = np.concatenate((calculated_errors_list), axis=0)
            # tp_mean = calculated_errors_array.mean(axis=0)
            # pdb.set_trace()

            error_list = np.concatenate(error_list, axis=0)
            thresholds_arr = np.array(thresholdss)
            thresholds_arr_ori = np.array(thresholdss)

            # pdb.set_trace()
            thresholds, recall_test = get_thresholds_recall(thresholds_arr, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS)

            # pdb.set_trace()
            thresholds = np.array(thresholds)
            # recall_test = np.array(recall_test)

            mATE[m], mASE[m], mAOE[m], mAVE[m] = accumulate_tp_error(error_list, thresholds, thresholds_arr_ori,
                                                                     num_sample_pts=N_SAMPLE_PTS)


        # clean temp variables
        del distance
        del parted_distance

        gc.collect()

    # pdb.set_trace()

    ret_dict = {
        'recall': recall,
        'precision': precision,
        'mATE': mATE,
        'mASE': mASE,
        'mAOE': mAOE,
        'mAVE': mAVE,
    }
    return ret_dict

def accumulate_tp_error(error_list, thresholds, scores, num_sample_pts=50):
    starting_idx = int(0.1 * num_sample_pts)
    ATE, ASE, AOE, AVE = [], [], [], []
    for i in range(starting_idx, num_sample_pts):
        if i < len(thresholds):
            threshold_single = thresholds[i]
            # pdb.set_trace()
            mask = scores >= threshold_single
            err_result = np.mean(error_list[mask], axis=0)
            if np.isnan(err_result.any()):
                # print('nan')
                # pdb.set_trace()
                ATE.append(1)
                ASE.append(1)
                AOE.append(1)
                AVE.append(1)
            # ATE.append(min(max(err_result[0], 0), 1))
            # ASE.append(min(max(err_result[1], 0), 1))
            # AOE.append(min(max(err_result[2], 0), 1))
            # AVE.append(min(max(err_result[3], 0), 1))
            else:
                ATE.append(err_result[0])
                ASE.append(err_result[1])
                AOE.append(err_result[2])
                AVE.append(err_result[3])
        else:
            ATE.append(1)
            ASE.append(1)
            AOE.append(1)
            AVE.append(1)

    # pdb.set_trace()
    ATE = sum(ATE) / len(ATE)
    ASE = sum(ASE) / len(ASE)
    AOE = sum(AOE) / len(AOE)
    AVE = sum(AVE) / len(AVE)

    return ATE, ASE, AOE, AVE




# def get_mAP(prec):
#     # pdb.set_trace()
#     sums = 0
#     for i in range(0, prec.shape[-1], 4):
#         sums = sums + prec[..., i]
#     return sums / 11 * 100

def get_mAP(prec):
    pr_length = prec.shape[-1]
    count = 0
    sums = 0
    for i in range(int(0.1*pr_length), pr_length):
        sums = sums + prec[..., i]
        count += 1
    return sums / count * 100

def get_mAP_40_points(prec):
    # pdb.set_trace()
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100

def get_mAP_50_points(prec):
    # pdb.set_trace()
    sums = 0
    for i in range(prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 50 * 100

def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            distance_threshold,
            eval_types='iou_mAP'):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, eval_types,
                     min_overlaps, distance_threshold)
    # pdb.set_trace()
    # mAP_3d = get_mAP_50_points(ret['precision'])
    mAP_3d = get_mAP(ret['precision'])

    mATE, mASE, mAOE, mAVE = ret['mATE'], ret['mASE'], ret['mAOE'], ret['mAVE']

    return mAP_3d, mATE, mASE, mAOE, mAVE


def DeepAccident_det_eval(gt_annos,
               dt_annos,
               current_classes,
               eval_types='iou_mAP'):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    # pdb.set_trace()
    # current_classes = [2]

    assert len(eval_types) > 0, 'must contain at least one evaluation type'

    # pdb.set_trace()
    # # car motorcycle pedestrian cyclist van truck
    # # 0.7, 0.5, 0.3, 0.5, 0.7, 0.7
    # # 0.5, 0.3, 0.15, 0.3, 0.5, 0.5
    # overlap_0_7 = np.array([0.7, 0.5, 0.3, 0.5, 0.7, 0.7])  # only this line for 3d bbox (strict)
    # overlap_0_5 = np.array([0.5, 0.3, 0.15, 0.3, 0.5, 0.5])  # only this line for 3d bbox (loose)

    # 'car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian', 'invalid1', 'invalid2', 'invalid3', 'invalid4'
    # 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3,
    # 0.5, 0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15,
    # overlap_0_7 = np.array([0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])  # only this line for 3d bbox (strict)
    # overlap_0_5 = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15])  # only this line for 3d bbox (loose)
    overlap_0_7 = np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])  # only this line for 3d bbox (strict)
    overlap_0_5 = np.array(
        [0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])  # only this line for 3d bbox (loose)

    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]

    # distance_threshold1 = np.array([1, 1, 1, 1, 1])
    # distance_threshold2 = np.array([2, 2, 2, 2, 2])
    # distance_threshold3 = np.array([4, 4, 4, 4, 4])
    #
    # distance_threshold = np.stack([distance_threshold1, distance_threshold2, distance_threshold3], axis=0)

    distance_threshold = np.array([0.5, 1, 2, 4])
    


    # ['car', 'motorcycle', 'pedestrian', 'cyclist', 'van', 'truck']
    # ['car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian',
    #         'invalid1', 'invalid2', 'invalid3', 'invalid4']

    class_to_name = {
        0: 'car',
        1: 'truck',
        2: 'van',
        3: 'cyclist',
        4: 'motorcycle',
        5: 'pedestrian',
        6: 'invalid1',
        7: 'invalid2',
        8: 'invalid3',
        9: 'invalid4',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []

    current_classes_filter = []
    for class_single in current_classes:
        if 'invalid' not in class_single:
            current_classes_filter.append(class_single)
    # current_classes = [if 'invalid' not in class_single for class_single in current_classes]
    current_classes = current_classes_filter

    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, current_classes]

    result = ''

    mAP3d, mATE, mASE, mAOE, mAVE = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, distance_threshold, eval_types)
    if eval_types == 'iou_mAP':
        ret_dict = {}
        difficulty = ['easy', 'moderate', 'hard']
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            result += '\n'
            curcls_name = class_to_name[curcls]
            for i in range(min_overlaps.shape[0]):
                # prepare results for print
                result += ('{} IOU matching threshold: {:.2f}\n'.format(
                    curcls_name, min_overlaps[i, j]))
                # result += ('{} difficulty@{}, {}, {}:\n'.format(
                #     curcls_name, *difficulty))
                result += ("%-10s\t%-10s\t%-10s\n"%("0-30m","30-50m",">50m") + "-" * 40 + "\n")

                # print("%-10s\t%-10s\t%-10s\n" % ("easy", "", "email") + "-" * 50)

                if mAP3d is not None:
                    easy_mAP = int(mAP3d[j, 0, i] * 100) / 100
                    moderate_mAP = int(mAP3d[j, 1, i] * 100) / 100
                    hard_mAP = int(mAP3d[j, 2, i] * 100) / 100
                    result += ("%-10s\t%-10s\t%-10s\n" % (easy_mAP, moderate_mAP, hard_mAP))

                # prepare results for logger
                for idx in range(3):
                    if i == 0:
                        postfix = f'{difficulty[idx]}_strict'
                    else:
                        postfix = f'{difficulty[idx]}_loose'
                    prefix = f'CARLA/{curcls_name}'
                    if mAP3d is not None:
                        ret_dict[f'{prefix}_3D_{postfix}'] = mAP3d[j, idx, i]

                result += '\n'

        # # calculate mAP over all classes if there are multiple classes
        # if len(current_classes) > 1:
        # prepare results for print

        result += '\nmAP_by_category@\n'
        mAP3d_cls = mAP3d.mean(axis=-1)
        mAP3d_cls = mAP3d_cls.mean(axis=-1)
        for i, class_int in enumerate(current_classes):
            result += '{}: {:.4f}\n'.format(class_to_name[class_int], mAP3d_cls[i])

        # prepare results for logger
        for i, class_int in enumerate(current_classes):
            postfix = f'{class_to_name[class_int]}'
            if mAP3d is not None:
                ret_dict[f'CARLA/Overall_3D_{postfix}'] = mAP3d_cls[i]

        result += ('\nOverall AP@{}, {}, {}:\n'.format(*difficulty))
        if mAP3d is not None:
            mAP3d_distance = mAP3d.mean(axis=0)
            mAP3d_distance = mAP3d_distance.mean(axis=-1)
            result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP3d_distance)

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP3d is not None:
                ret_dict[f'CARLA/Overall_3D_{postfix}'] = mAP3d_distance[idx]

        mAP3d_all = mAP3d.mean(axis=0)
        mAP3d_all = mAP3d_all.mean(axis=0)
        mAP3d_all = mAP3d_all.mean(axis=0)
        result += '\nOverall AP@: %.4f\n' %(mAP3d_all)
        ret_dict[f'CARLA/Overall_3D'] = mAP3d_all

    else:

        mAP3d = np.mean(mAP3d, axis=-1)

        ret_dict = {}
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_minoverlap]
            result += '\n'
            curcls_name = class_to_name[curcls]
            # for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} average mAP over distance matching threshold: (0.5, 1, 2, 4) meter\n'.format(
                curcls_name))

            if mAP3d is not None:
                avg_mAP = int(mAP3d[j] * 100) / 100
                result += ("%-10s\n" % (avg_mAP))

                postfix = f'average_distance_mAP'
                prefix = f'CARLA/{curcls_name}'
                ret_dict[f'{prefix}_3D_{postfix}'] = mAP3d[j]

                result += ("%-10s\t%-10s\t%-10s\t%-10s\n" % ("mATE", "mASE", "mAOE", "mAVE") + "-" * 40 + "\n")
                result += ("%-10s\t%-10s\t%-10s\t%-10s\n" % (mATE[j], mASE[j], mAOE[j], mAVE[j]))

                prefix = f'CARLA/{curcls_name}'
                ret_dict[f'{prefix}_mATE'] = mATE[j]
                ret_dict[f'{prefix}_mASE'] = mASE[j]
                ret_dict[f'{prefix}_mAOE'] = mAOE[j]
                ret_dict[f'{prefix}_mAVE'] = mAVE[j]

            result += '\n'

        # calculate mAP over all classes if there are multiple classes
        if len(current_classes) > 1:
            # prepare results for print
            result += '\nOverall mAP:\n'
            if mAP3d is not None:
                mAP3d = mAP3d.mean(axis=0)
                result += '{:.4f}\n'.format(mAP3d)
                if mAP3d is not None:
                    ret_dict[f'CARLA/Overall_3D_mAP'] = mAP3d


    return result, ret_dict

