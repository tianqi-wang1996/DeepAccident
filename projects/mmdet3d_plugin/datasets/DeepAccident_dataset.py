# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset, output_to_nusc_box, lidar_nusc_box_to_global
from mmdet3d.datasets.pipelines import Compose
from .utils import VectorizedLocalMap, preprocess_map
from nuscenes.nuscenes import NuScenes
from .utils.geometry import invert_matrix_egopose_numpy, mat2pose_vec
import random
import numpy.random as random_py
import numba

import pdb
import time
from mmcv.utils import print_log


@DATASETS.register_module()
class DeepAccidentDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    # CLASSES = ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian')
    CLASSES = [
        'car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian',
        'invalid1', 'invalid2', 'invalid3', 'invalid4'
    ]
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 coordinate_system='ego',
                 filter_empty_gt=True,
                 test_mode=False,
                 grid_conf=None,
                 map_grid_conf=None,
                 receptive_field=1,
                 future_frames=0,
                 filter_invalid_sample=False):

        self.receptive_field = receptive_field
        self.n_future = future_frames
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.data_infos = self.data_infos[::load_interval]


        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        # whether test-set
        self.test_submission = 'test' in self.ann_file
        # self.test_submission = False

        # temporal settings
        self.sequence_length = receptive_field + future_frames
        self.filter_invalid_sample = filter_invalid_sample
        self.coordinate_system = coordinate_system

        # default, we use the LiDAR coordinate system as the BEV system
        assert self.coordinate_system in ['lidar', 'ego']

        # for vector maps
        self.map_dataroot = self.data_root

        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)

        # pdb.set_trace()
        # # hdmap settings
        # self.map_max_channel = 3
        # self.map_thickness = 5
        # self.map_angle_class = 36
        # self.vector_map = VectorizedLocalMap(
        #     dataroot=self.map_dataroot,
        #     patch_size=self.map_patch_size,
        #     canvas_size=self.map_canvas_size,
        # )

        # process infos so that they are sorted w.r.t. scenes & time_stamp
        # self.data_infos.sort(key=lambda x: (x['scene_name'], x['vehicle_name'], x['timestamp']))
        self._set_group_flag()



    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data_raw = mmcv.load(ann_file)
        self.data_info_all = {}
        data_infos = []
        if not self.test_mode:
            for data in data_raw['infos']:
                key_name = data['scene_name'] + '_' + data['vehicle_name'] + '_' + str(data['timestamp'])
                self.data_info_all[key_name] = data
                if (data['scenario_length'] - data['timestamp']) % 10 == 0:
                    data_infos.append(data)

        else:
            for data in data_raw['infos']:
                key_name = data['scene_name'] + '_' + data['vehicle_name'] + '_' + str(data['timestamp'])
                self.data_info_all[key_name] = data

                if ((data['scenario_length'] - data['timestamp']) <= 4 * 5) and (
                        (data['scenario_length'] - data['timestamp']) > 0) \
                        and 'accident' in data['scene_name']:
                    data_infos.append(data)

            data_infos = list(sorted(data_infos, key=lambda x: (x['scene_name'], x['vehicle_name'], x['timestamp']), reverse=False))
            # selected_index = ['type1_subtype2_accident_ego_vehicle_behind_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_infrastructure_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_other_vehicle_behind_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_ego_vehicle_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_other_vehicle_Town04_type001_subtype0002_scenario00021_024']
            # data_infos = [data_single for data_single in data_infos if data_single['lidar_prefix'] in selected_index]
        # data_infos = data_infos[:50]
        self.metadata = data_raw['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_temporal_indices(self, index):
        current_scene_name = self.data_infos[index]['scene_name']
        current_vehicle_name = self.data_infos[index]['vehicle_name']

        # generate the past
        previous_indices = []
        for t in range(- self.receptive_field + 1, 0):
            # index_t = index + t
            retrive_timestamp = self.data_infos[index]['timestamp'] + 5 * t
            key_name = self.data_infos[index]['scene_name'] + '_' + self.data_infos[index]['vehicle_name'] \
                       + '_' + str(retrive_timestamp)
            if retrive_timestamp > 0 and self.data_info_all[key_name]['scene_name'] == current_scene_name\
                and self.data_info_all[key_name]['vehicle_name'] == current_vehicle_name:
                previous_indices.append(key_name)
            else:
                previous_indices.append('invalid')  # for invalid indices

        # generate the future
        future_indices = []
        for t in range(1, self.n_future + 1):
            retrive_timestamp = self.data_infos[index]['timestamp'] + 5 * t
            key_name = self.data_infos[index]['scene_name'] + '_' + self.data_infos[index]['vehicle_name']\
                        + '_' + str(retrive_timestamp)
            if retrive_timestamp <= self.data_infos[index]['scenario_length'] and self.data_info_all[key_name]['scene_name'] == current_scene_name\
                and self.data_info_all[key_name]['vehicle_name'] == current_vehicle_name:
                future_indices.append(key_name)
            else:
                future_indices.append('invalid')

        return previous_indices, future_indices

    @staticmethod
    def get_egopose_from_info(info):
        # ego2global transformation (lidar_ego)
        e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
        e2g_rot = info['ego2global_rotation']
        e2g_trans = info['ego2global_translation']
        e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
            e2g_rot).rotation_matrix
        e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
        e2g_trans_matrix[3, 3] = 1.0

        return e2g_trans_matrix

    def get_egomotions(self, key_names):
        # pdb.set_trace()
        # get ego_motion for each frame
        future_egomotions = []
        for key_name in key_names:
            ego_motion = np.eye(4, dtype=np.float32)
            if key_name != 'invalid':
                cur_info = self.data_info_all[key_name]

                # next_key_name = index + 1
                retrive_timestamp = self.data_info_all[key_name]['timestamp'] + 5 * 1
                next_key_name = self.data_info_all[key_name]['scene_name'] + '_' + self.data_info_all[key_name]['vehicle_name'] \
                           + '_' + str(retrive_timestamp)

                # 如何处理 invalid frame
                if retrive_timestamp <= self.data_info_all[key_name]['scenario_length'] \
                        and self.data_info_all[next_key_name]['scene_name'] == cur_info['scene_name']\
                        and self.data_info_all[next_key_name]['vehicle_name'] == cur_info['vehicle_name']:
                    next_info = self.data_info_all[next_key_name]
                    # get ego2global transformation matrices

                    cur_egopose = cur_info['ego_to_world_matrix'].astype(dtype=np.float32)
                    next_egopose = next_info['ego_to_world_matrix'].astype(dtype=np.float32)

                    # trans from next to current frame
                    ego_motion = invert_matrix_egopose_numpy(
                        next_egopose).dot(cur_egopose)
                    ego_motion[3, :3] = 0.0
                    ego_motion[3, 3] = 1.0

            # transformation between adjacent frames (from next to current)
            ego_motion = torch.Tensor(ego_motion).float()
            ego_motion = mat2pose_vec(ego_motion)
            future_egomotions.append(ego_motion)

        return torch.stack(future_egomotions, dim=0)

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # when the labels for future frames are not complete, skip the sample
        if self.filter_invalid_sample and input_dict['has_invalid_frame'] is True:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or
                                     ~(example['gt_labels_3d']._data != -1).any()):
            return None

        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            scenario_length=info['scenario_length'],
            sample_idx=info['lidar_prefix'],
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'],
            data_root=self.data_root,
            bev_path=info['bev_path'],
            lidar_to_ego_matrix=info['lidar_to_ego_matrix'],
            ego_to_world_matrix=info['ego_to_world_matrix'],
        )

        if self.test_mode:
            lidar_path = info['lidar_path']
            prefix = lidar_path.split('/')[:4]
            prefix.append('meta')
            meta_file_name = lidar_path.split('/')[-2] + '.txt'
            prefix.append(meta_file_name)
            meta_path = '/'.join(prefix)
            with open(meta_path, 'r') as f:
                meta_info = [(line.rstrip()) for line in f]

            town_name = info['scene_name'].split('_')[3]
            weather_and_time = meta_info[0].split(' ')[0]
            if 'Noon' in weather_and_time:
                time_of_the_day = 'Noon'
                weather = weather_and_time.replace('Noon', '')
            elif 'Night' in weather_and_time:
                time_of_the_day = 'Night'
                weather = weather_and_time.replace('Night', '')
            elif 'Sunset' in weather_and_time:
                time_of_the_day = 'Sunset'
                weather = weather_and_time.replace('Sunset', '')

            if time_of_the_day == 'Night':
                time_of_the_day = 'Night'
            else:
                time_of_the_day = 'Daytime'

            if 'Rain' in weather:
                weather = 'Rainy'
            elif 'Cloud' in weather:
                weather = 'Cloudy'
            else:
                weather = 'Clear'

            collision_id1, collision_id2 = meta_info[0].split(' ')[2], meta_info[0].split(' ')[4]
            if collision_id1 == '-1' or collision_id2 == '-1':
                collision_status = 'no_collision'
            elif collision_id1 == 'pedestrian' or collision_id2 == 'pedestrian':
                collision_status = 'collides_with_pedestrian'
            else:
                collision_status = 'collides_between_vehicles'
            junction_type = meta_info[3].split(': ')[1]

            self_direction = meta_info[-2].split(': ')[-1]
            other_direction = meta_info[-1].split(': ')[-1]

            if junction_type == 'four-way junction' and self_direction == 'straight' and other_direction == 'straight':
                trajectory_type = 'straight_straight'
            else:
                trajectory_type = 'one_of_the_agents_turning'

            if junction_type == 'four-way junction':
                junction_type = 'four_way_junction'
            else:
                junction_type = 'three_way_junction'

            input_dict['scenario_type'] = info['scenario_type']
            input_dict['town_name'] = town_name
            input_dict['weather'] = weather
            input_dict['time_of_the_day'] = time_of_the_day
            # actually accident happens or not
            input_dict['collision_status'] = collision_status
            input_dict['junction_type'] = junction_type
            input_dict['trajectory_type'] = trajectory_type

        # get temporal indices
        prev_indices, future_indices = self.get_temporal_indices(index)

        # ego motions are needed for all frames
        retrive_timestamp = self.data_infos[index]['timestamp']
        key_name = self.data_infos[index]['scene_name'] + '_' + self.data_infos[index]['vehicle_name'] \
                   + '_' + str(retrive_timestamp)
        all_frames = prev_indices + [key_name] + future_indices

        # [num_seq, 6 DoF]
        future_egomotions = self.get_egomotions(all_frames)

        input_dict['future_egomotions'] = future_egomotions

        # whether invalid frame is present
        has_invalid_frame = 'invalid' in all_frames
        input_dict['has_invalid_frame'] = has_invalid_frame

        input_dict['img_is_valid'] = np.array(all_frames) != 'invalid'

        # for past frames, we need images, camera paramters, depth(optional)
        img_infos = []
        for prev_key_name in prev_indices:
            if prev_key_name != 'invalid':
                img_infos.append(self.data_info_all[prev_key_name]['cams'])
            else:
                # get the information of current frame for invalid frames
                img_infos.append(info['cams'])

        # current frame
        img_infos.append(info['cams'])
        input_dict['img_info'] = img_infos

        lidar_to_ego_matrix = info['lidar_to_ego_matrix'].astype(np.float32)
        lidar2ego_translation = lidar_to_ego_matrix[:3, 3]
        lidar2ego_rotation = lidar_to_ego_matrix[:3, :3]

        input_dict['lidar2ego_rots'] = torch.tensor(lidar2ego_rotation)
        input_dict['lidar2ego_trans'] = torch.tensor(lidar2ego_translation)


        # for future frames, we need detection labels
        if not self.test_submission:
            # generate detection labels for current + future frames
            label_frames = [key_name] + future_indices
            detection_ann_infos = []
            for label_frame in label_frames:
                if label_frame != 'invalid':
                    detection_ann_infos.append(
                        self.get_detection_ann_info_v2x(label_frame))
                else:
                    detection_ann_infos.append(None)
            if self.test_mode:
                past_current_frames = prev_indices + [key_name]
                accident_visibility_past_current = []
                for checking_visibility_frame in past_current_frames:
                    accident_visibility_past_current.append(self.check_accident_vehicles_visibility(checking_visibility_frame))

                # vehicles behind steer curtains are not visible for annotations, which is a bug in CARLA
                exlude_invisible_scenarios = ['Town05_type001_subtype0002_scenario00035',
                                              'Town05_type001_subtype0002_scenario00004']

                accident_visibility_list = np.array(accident_visibility_past_current)
                accident_invisibility = (accident_visibility_list == False).sum() > len(
                    accident_visibility_list) / 2.0 and (accident_visibility_list[-1] == False)
                scenario_and_timestep = input_dict['pts_filename'].split('/')[-1].split('.')[0]
                scenario_name_split = scenario_and_timestep.split('_')[:-1]
                scenario_name = '_'.join(scenario_name_split)

                accident_invisibility *= (scenario_name not in exlude_invisible_scenarios)
                accident_invisibility *= (input_dict['collision_status'] != 'no_collision')

                accident_visibility = ~accident_invisibility
                input_dict['accident_visibility'] = accident_visibility

            input_dict['ann_info'] = detection_ann_infos
            input_dict['vectors'] = []

        return input_dict

    def get_detection_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """

        info = self.data_info_all[index]
        # pdb.set_trace()
        # filter out bbox containing no points

        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_instance_tokens = info["vehicle_id"]

        gt_valid_flag = np.ones(len(info["gt_boxes"])).astype(np.int8)

        close_objects_flag = []

        if info['vehicle_name'] == 'infrastructure':
            close_objects_thre = 15
        else:
            close_objects_thre = 10

        for box in gt_bboxes_3d:
            if np.sqrt(box[0] ** 2 + box[1] ** 2) < close_objects_thre:
                close_objects_flag.append(1)
            else:
                close_objects_flag.append(0)

        for box_index in range(len(gt_bboxes_3d)):
            if gt_names_3d[box_index] == 'pedestrian' or gt_names_3d[box_index] == 'motorcycle' \
                    or gt_names_3d[box_index] == 'cyclist':
                if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(
                        info["num_lidar_pts"][box_index] >= 1)
                elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 2) or int(
                        info["camera_visibility"][box_index])
                else:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(
                        info["num_lidar_pts"][box_index] >= 1)
            else:
                if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(
                        info["num_lidar_pts"][box_index] >= 1)
                elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 5) or int(
                        info["camera_visibility"][box_index])
                else:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(
                        info["num_lidar_pts"][box_index] >= 2)

        gt_vis_tokens = 3 * (
                gt_valid_flag.astype(np.int) * (info["vehicle_id"] != -1) * (info["vehicle_id"] < 2000000)) + 1

        gt_valid_flag = gt_valid_flag.astype(np.bool)

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity']
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)


        ego_height = gt_bboxes_3d[0, 2]

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        lidar_to_ego_matrix = info['lidar_to_ego_matrix'].astype(np.float32)
        lidar2ego_translation = lidar_to_ego_matrix[:3, 3]
        lidar2ego_rotation = lidar_to_ego_matrix[:3, :3]

        gt_bboxes_3d.rotate(lidar2ego_rotation.T)
        gt_bboxes_3d.translate(lidar2ego_translation)

        ego_bboxes_3d = None
        if info['vehicle_name'] != 'infrastructure':
            ego_bboxes_3d = np.array([0, 0, ego_height, 3.99, 1.85, 1.62, 0, 0, 0])[np.newaxis, ...]
            ego_bboxes_3d = LiDARInstance3DBoxes(
                ego_bboxes_3d,
                box_dim=ego_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            ego_bboxes_3d.rotate(lidar2ego_rotation.T)
            ego_bboxes_3d.translate(lidar2ego_translation)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_tokens=gt_instance_tokens,
            gt_valid_flag=gt_valid_flag,
            gt_vis_tokens=gt_vis_tokens,
            ego_bboxes_3d=ego_bboxes_3d
        )

        return anns_results

    def check_accident_vehicles_visibility(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        if index == 'invalid':
            return False

        info = self.data_info_all[index]

        last_str = index.split('_scenario')[-1].split('_')[1:-1]
        vehicle_name_ego = '_'.join(last_str)

        lidar_path = info['lidar_path']
        prefix = lidar_path.split('/')[:4]
        prefix.append('meta')
        meta_file_name = lidar_path.split('/')[-2] + '.txt'
        prefix.append(meta_file_name)
        meta_path = '/'.join(prefix)
        with open(meta_path, 'r') as f:
            meta_info = [(line.rstrip()) for line in f]

        id_info = meta_info[2]

        # id in this order: ego, ego_behind, other, other_behind
        id_list = id_info.split('id: ')[-1].split(' ')

        if vehicle_name_ego == 'ego_vehicle':
            ego_id = int(id_list[0])
        elif vehicle_name_ego == 'ego_vehicle_behind':
            ego_id = int(id_list[1])
        elif vehicle_name_ego == 'other_vehicle':
            ego_id = int(id_list[2])
        elif vehicle_name_ego == 'other_vehicle_behind':
            ego_id = int(id_list[3])
        elif vehicle_name_ego == 'infrastructure':
            ego_id = None

        meta_first_line = meta_info[0].split(' ')
        accident_obj1_id, accident_obj1_type, accident_obj2_id, accident_obj2_type = meta_first_line[1:5]

        accident_obj1_id = int(accident_obj1_id)
        accident_obj2_id = int(accident_obj2_id)

        accident_ids = []
        if accident_obj1_id != ego_id:
            accident_ids.append(accident_obj1_id)
        if accident_obj2_id != ego_id:
            accident_ids.append(accident_obj2_id)


        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_instance_tokens = info["vehicle_id"]

        # ego vehicle are marked as -100
        gt_valid_flag = (info["vehicle_id"] != -100).astype(np.int) * (info["vehicle_id"] != -1).astype(np.int)

        close_objects_flag = []

        if info['vehicle_name'] == 'infrastructure':
            close_objects_thre = 15
        else:
            close_objects_thre = 10

        for box in gt_bboxes_3d:
            if np.sqrt(box[0] ** 2 + box[1] ** 2) < close_objects_thre:
                close_objects_flag.append(1)
            else:
                close_objects_flag.append(0)

        for box_index in range(len(gt_bboxes_3d)):
            if gt_names_3d[box_index] == 'pedestrian' or gt_names_3d[box_index] == 'motorcycle' \
                    or gt_names_3d[box_index] == 'cyclist':
                if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(info["num_lidar_pts"][box_index] >= 1)
                elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 2) or int(info["camera_visibility"][box_index])
                else:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(info["num_lidar_pts"][box_index] >= 1)
            else:
                if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(info["num_lidar_pts"][box_index] >= 1)
                elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                    gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 5) or int(info["camera_visibility"][box_index])
                else:
                    gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(info["num_lidar_pts"][box_index] >= 2)


        visible_gts_ego = gt_instance_tokens[np.where(gt_valid_flag != 0)]

        all_accident_vehicles_visible = True
        for accident_id_single in accident_ids:
            if accident_id_single not in visible_gts_ego:
                all_accident_vehicles_visible = False
                break

        return all_accident_vehicles_visible

    def get_detection_ann_info_v2x(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # 'type1_subtype2_accident_Town10HD_scenario00010_other_vehicle_behind_45'
        vehicle_list = ['ego_vehicle', 'ego_vehicle_behind', 'other_vehicle', 'other_vehicle_behind', 'infrastructure']


        ego_info = self.data_info_all[index]
        lidar_path = ego_info['lidar_path']
        prefix = lidar_path.split('/')[:4]
        prefix.append('meta')

        type_num = int(index.split('_')[0].split('type')[-1])
        subtype_num = int(index.split('_')[1].split('subtype')[-1])
        # accident_type = index.split('_')[2]
        town_name = index.split('_')[3]
        scene_name, remaining_str = index.split('_scenario')
        scenario_num = remaining_str.split('_')[0]
        timestep_num = remaining_str.split('_')[-1]

        visible_gts = []
        for vehicle_name in vehicle_list:
            index_agent = scene_name + '_scenario' + scenario_num + '_' + vehicle_name + '_' + timestep_num
            info = self.data_info_all[index_agent]

            gt_bboxes_3d = info['gt_boxes']
            gt_names_3d = info['gt_names']
            gt_instance_tokens = info["vehicle_id"]


            # ego vehicle are marked as -100
            gt_valid_flag = (info["vehicle_id"] != -100).astype(np.int) * (info["vehicle_id"] != -1).astype(np.int)


            close_objects_flag = []

            if info['vehicle_name'] == 'infrastructure':
                close_objects_thre = 15
            else:
                close_objects_thre = 10

            for box in gt_bboxes_3d:
                if np.sqrt(box[0] ** 2 + box[1] ** 2) < close_objects_thre:
                    close_objects_flag.append(1)
                else:
                    close_objects_flag.append(0)

            for box_index in range(len(gt_bboxes_3d)):
                if gt_names_3d[box_index] == 'pedestrian' or gt_names_3d[box_index] == 'motorcycle' \
                        or gt_names_3d[box_index] == 'cyclist':
                    if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                        gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(info["num_lidar_pts"][box_index] >= 1)
                    elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                        gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 2) or int(info["camera_visibility"][box_index])
                    else:
                        gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(info["num_lidar_pts"][box_index] >= 1)
                else:
                    if info['vehicle_name'] == 'infrastructure' and close_objects_flag[box_index]:
                        gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) or int(info["num_lidar_pts"][box_index] >= 1)
                    elif info['vehicle_name'] != 'infrastructure' and close_objects_flag[box_index]:
                        gt_valid_flag[box_index] *= int(info["num_lidar_pts"][box_index] >= 5) or int(info["camera_visibility"][box_index])
                    else:
                        gt_valid_flag[box_index] *= int(info["camera_visibility"][box_index]) and int(info["num_lidar_pts"][box_index] >= 2)

            visible_gts.append(gt_instance_tokens[np.where(gt_valid_flag != 0)])

        visible_gts = np.concatenate(visible_gts)
        visible_gts = np.unique(visible_gts)

        info = self.data_info_all[index]

        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_instance_tokens = info["vehicle_id"]


        gt_valid_flag = np.isin(info["vehicle_id"], visible_gts).astype(np.int8)

        gt_vis_tokens = 3 * (gt_valid_flag.astype(np.int)) + 1

        gt_valid_flag = gt_valid_flag.astype(np.bool)

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity']
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        ego_height = gt_bboxes_3d[0, 2]

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        lidar_to_ego_matrix = info['lidar_to_ego_matrix'].astype(np.float32)
        lidar2ego_translation = lidar_to_ego_matrix[:3, 3]
        lidar2ego_rotation = lidar_to_ego_matrix[:3, :3]

        gt_bboxes_3d.rotate(lidar2ego_rotation.T)
        gt_bboxes_3d.translate(lidar2ego_translation)

        ego_bboxes_3d = None
        ego_annotated_box = gt_bboxes_3d.tensor[gt_instance_tokens==-100]

        if info['vehicle_name'] != 'infrastructure':
            if ego_annotated_box.shape[0] == 1:
                ego_bboxes_3d = ego_annotated_box
            else:
                ego_bboxes_3d = np.array([0, 0, ego_height, 3.99, 1.85, 1.62, 0, 0, 0])[np.newaxis, ...]

            ego_bboxes_3d = LiDARInstance3DBoxes(
                ego_bboxes_3d,
                box_dim=ego_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            ego_bboxes_3d.rotate(lidar2ego_rotation.T)
            ego_bboxes_3d.translate(lidar2ego_translation)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_tokens=gt_instance_tokens,
            gt_valid_flag=gt_valid_flag,
            gt_vis_tokens=gt_vis_tokens,
            ego_bboxes_3d=ego_bboxes_3d
            )
        return anns_results

    def get_map_ann_info(self, info):
        # get the annotations of HD maps
        vectors = self.vector_map.gen_vectorized_samples(
            info['location'], info['ego2global_translation'], info['ego2global_rotation'])

        for vector in vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1)

        return vectors

    def evaluate(self,
                 results,
                 attributes_counter=None,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        import copy
        import time
        from ..metrics import DeepAccident_det_eval
        start_time = time.time()

        if show:
            # pdb.set_trace()
            self.show(results, out_dir, pipeline=pipeline)

            print('saving cost %.5f s' % (time.time() - start_time))
            start_time = time.time()

        do_separate_evaluation = True

        if do_separate_evaluation:
            for attribute_name, attribute_val in attributes_counter.items():
                print('Evaluation divided by %s' % attribute_name)
                for attribute_val_single, selected_index in attribute_val.items():
                    print('\nDetection Evaluation on %s: %s!!!!!!!' % (attribute_name, attribute_val_single))
                    print('\n[Data length for this attribute value {:04d} / {:04d}]'.format(len(selected_index),
                                                                                            len(self.data_infos)))
                    selected_results = [results[selected_index_single] for selected_index_single in selected_index].copy()
                    result_files, tmp_dir = self.format_results(selected_results, pklfile_prefix)
                    gt_annos = []

                    selected_gt_infos = [self.data_infos[selected_index_single] for selected_index_single in selected_index]
                    for info_ori in selected_gt_infos:
                        if isinstance(info_ori, list) and len(info_ori) > 1:
                            info = info_ori[0]
                        else:
                            info = info_ori
                        gt_anno = {}
                        gt_boxes = np.concatenate([info['gt_boxes'], info['gt_velocity']], axis=1)

                        gt_boxes = LiDARInstance3DBoxes(
                            gt_boxes,
                            box_dim=gt_boxes.shape[-1],
                            origin=(0.5, 0.5, 0.5))
                        gt_boxes = gt_boxes.tensor.numpy()

                        gt_anno.update(name=info['gt_names'][:])
                        gt_anno.update(dimensions=gt_boxes[:, 3:6])
                        gt_anno.update(location=gt_boxes[:, :3])
                        gt_anno.update(rotation_y=gt_boxes[:, 6])
                        gt_anno.update(num_lidar_pts=info['num_lidar_pts'][:])
                        gt_anno.update(velocity=gt_boxes[:, 7:9])

                        # might be deleted
                        gt_anno.update(truncated=np.zeros(info['gt_names'].shape[0]))
                        gt_anno.update(occluded=np.zeros(info['gt_names'].shape[0]))
                        gt_anno.update(alpha=np.zeros(info['gt_names'].shape[0]))
                        gt_anno.update(score=np.zeros(info['gt_names'].shape[0]))

                        gt_annos.append(gt_anno)

                    if 'iou_mAP' in metric:
                        eval_types = 'iou_mAP'
                    elif 'distance_mAP' in metric:
                        eval_types = 'distance_mAP'
                    else:
                        eval_types = 'iou_mAP'

                    if isinstance(result_files, dict):
                        ap_dict = dict()
                        for name, result_files_ in result_files.items():
                            ap_result_str, ap_dict_ = DeepAccident_det_eval(
                                gt_annos,
                                result_files_,
                                self.CLASSES,
                                eval_types=eval_types)
                            for ap_type, ap in ap_dict_.items():
                                ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                            print_log(
                                f'Results of {name}:\n' + ap_result_str, logger=logger)
                    else:
                        ap_result_str, ap_dict = DeepAccident_det_eval(
                            gt_annos, result_files, self.CLASSES, eval_types=eval_types)

                        print_log('\n' + ap_result_str, logger=logger)

                    if tmp_dir is not None:
                        tmp_dir.cleanup()

                    print('other part cost %.5f s' % (time.time() - start_time))
                    start_time = time.time()
                print('----------')

        print('\nDetection Evaluation on all data!!!!!!!')
        print('\n[Data length for this attribute value {:04d} / {:04d}]'.format(len(self.data_infos), len(self.data_infos)))
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        gt_annos = []

        for info_ori in self.data_infos:
            if isinstance(info_ori, list) and len(info_ori) > 1:
                info = info_ori[0]
            else:
                info = info_ori
            gt_anno = {}
            gt_boxes = np.concatenate([info['gt_boxes'], info['gt_velocity']], axis=1)
            gt_boxes = LiDARInstance3DBoxes(
                gt_boxes,
                box_dim=gt_boxes.shape[-1],
                origin=(0.5, 0.5, 0.5))
            gt_boxes = gt_boxes.tensor.numpy()
            gt_anno.update(name=info['gt_names'][:])
            gt_anno.update(dimensions=gt_boxes[:, 3:6])
            gt_anno.update(location=gt_boxes[:, :3])
            gt_anno.update(rotation_y=gt_boxes[:, 6])
            gt_anno.update(num_lidar_pts=info['num_lidar_pts'][:])
            gt_anno.update(velocity=gt_boxes[:, 7:9])

            gt_anno.update(truncated=np.zeros(info['gt_names'].shape[0]))
            gt_anno.update(occluded=np.zeros(info['gt_names'].shape[0]))
            gt_anno.update(alpha=np.zeros(info['gt_names'].shape[0]))
            gt_anno.update(score=np.zeros(info['gt_names'].shape[0]))

            gt_annos.append(gt_anno)

        if 'iou_mAP' in metric:
            eval_types = 'iou_mAP'
        elif 'distance_mAP' in metric:
            eval_types = 'distance_mAP'
        else:
            eval_types = 'iou_mAP'

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                ap_result_str, ap_dict_ = DeepAccident_det_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            ap_result_str, ap_dict = DeepAccident_det_eval(
                gt_annos, result_files, self.CLASSES, eval_types=eval_types)

            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        print('other part cost %.5f s' % (time.time() - start_time))

        return ap_dict

    @numba.jit(nopython=True)
    def num_pts_inside_box(pts, vertices):
        # if len(vertices.shape) > 2:
        num_pts_list = np.zeros(vertices.shape[0])
        for i, vertice in enumerate(vertices):
            x_max = np.max(vertice[:, 0])
            x_min = np.min(vertice[:, 0])
            y_max = np.max(vertice[:, 1])
            y_min = np.min(vertice[:, 1])
            z_max = np.max(vertice[:, 2])
            z_min = np.min(vertice[:, 2])
            inside = (pts[:, 0] <= x_max) & (pts[:, 0] >= x_min) & \
                     (pts[:, 1] <= y_max) & (pts[:, 1] >= y_min) & \
                     (pts[:, 2] <= z_max) & (pts[:, 2] >= z_min)
            num_pts_list[i] = np.sum(inside)
        return num_pts_list

    @numba.jit(nopython=True)
    def rotation_3d_in_axis(points, angles, axis=0):
        """Rotate points in specific axis.

        Args:
            points (np.ndarray, shape=[N, point_size, 3]]):
            angles (np.ndarray, shape=[N]]):
            axis (int, optional): Axis to rotate at. Defaults to 0.

        Returns:
            np.ndarray: Rotated points.
        """
        # points: [N, point_size, 3]
        rot_sin = np.sin(angles)
        rot_cos = np.cos(angles)

        ones = np.ones_like(rot_cos)
        zeros = np.zeros_like(rot_cos)

        num_objects = angles.shape[0]
        rot_mat_T = np.zeros((3, 3, num_objects))
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[2, 2] = ones

        for i, point_list in enumerate(points):
            points[i] = np.dot(point_list, rot_mat_T[:, :, i])
        return points

    @numba.jit(nopython=True)
    def get_vertice(boxes):
        locations = boxes[:, :3]
        dimensions = boxes[:, 3:6]
        angles = boxes[:, 6]
        num_boxes = dimensions.shape[0]

        v1 = np.stack((dimensions[:, 0], dimensions[:, 1], np.zeros(num_boxes)), axis=1)
        v2 = np.stack((dimensions[:, 0], -dimensions[:, 1], np.zeros(num_boxes)), axis=1)
        v3 = np.stack((-dimensions[:, 0], -dimensions[:, 1], np.zeros(num_boxes, )), axis=1)
        v4 = np.stack((-dimensions[:, 0], dimensions[:, 1], np.zeros(num_boxes)), axis=1)

        v5 = np.stack((dimensions[:, 0], dimensions[:, 1], 2 * dimensions[:, 2]), axis=1)
        v6 = np.stack((dimensions[:, 0], -dimensions[:, 1], 2 * dimensions[:, 2]), axis=1)
        v7 = np.stack((-dimensions[:, 0], -dimensions[:, 1], 2 * dimensions[:, 2]), axis=1)
        v8 = np.stack((-dimensions[:, 0], dimensions[:, 1], 2 * dimensions[:, 2]), axis=1)

        vertices = np.stack((v1, v2, v3, v4, v5, v6, v7, v8), axis=1)

        vertices = rotation_3d_in_axis(vertices, angles, axis=2)

        # pdb.set_trace()
        for i, vertice in enumerate(vertices):
            vertice += locations[i]
        # vertices = vertices + locations
        return vertices

    def show_results(self, results, out_dir, targets=None):
        # visualize the predictions & ground-truth
        pass

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            info_ori = self.data_infos[i]
            if isinstance(info_ori, list) and len(info_ori) > 1:
                info = info_ori[0]
            else:
                info = info_ori

            pts_path = info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        # assert len(net_outputs) == len(self.data_infos), \
        #     'invalid list length of network outputs'

        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info_ori = self.data_infos[idx]
            if isinstance(info_ori, list) and len(info_ori) > 1:
                info = info_ori[0]
            else:
                info = info_ori
            sample_idx = info['lidar_prefix']
            # image_shape = info['image']['image_shape'][:2]
            # image_shape = [1080, 1920]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
                'velocity': [],
            }

            if len(box_dict['box3d_lidar']) > 0:

                scores = box_dict['scores']

                # box_preds_camera = box_dict['box3d_camera']
                box_preds_camera = box_dict['box3d_lidar']

                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']


                for box, box_lidar, score, label in zip(
                        box_preds_camera, box_preds_lidar, scores, label_preds):

                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    # anno['alpha'].append(
                    #     -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['alpha'].append(np.arctan2(-box_lidar[1], box_lidar[0]))
                    # anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                    anno['velocity'].append(box[7:9])

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                    'velocity': np.zeros([0, 2]),
                }
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']))

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        # pdb.set_trace()
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['lidar_prefix']
        # TODO: remove the hack of yaw
        # box_preds.tensor[:, 6] = box_preds.tensor[:, 6] - np.pi
        # box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM)


        if len(box_preds) == 0:
            return dict(
                # bbox=np.zeros([0, 4]),
                box3d_lidar=np.zeros([0, 7]),
                box3d_camera=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        return dict(
            box3d_lidar=box_preds.tensor.numpy(),
            box3d_camera=box_preds_camera.tensor.numpy(),
            scores=scores.numpy(),
            label_preds=labels.numpy(),
            sample_idx=sample_idx)

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        # pdb.set_trace()
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # pdb.set_trace()
        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti(
                        outputs, self.CLASSES, pklfile_prefix,
                        submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + '_'+ name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if 'img' in name:
                    result_files = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir


@DATASETS.register_module()
class DeepAccidentDataset_V2X(DeepAccidentDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    # CLASSES = ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian')
    CLASSES = [
        'car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian',
        'invalid1', 'invalid2', 'invalid3', 'invalid4'
    ]

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 other_agents_pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 coordinate_system='ego',
                 filter_empty_gt=True,
                 test_mode=False,
                 grid_conf=None,
                 map_grid_conf=None,
                 receptive_field=1,
                 future_frames=0,
                 filter_invalid_sample=False,
                 agent_list=['ego_vehicle', 'infrastructure']):
        self.agent_list = agent_list
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            coordinate_system=coordinate_system,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            grid_conf=grid_conf,
            map_grid_conf=map_grid_conf,
            receptive_field=receptive_field,
            future_frames=future_frames,
            filter_invalid_sample=filter_invalid_sample
        )
        self.other_agents_pipeline = self._get_pipeline(other_agents_pipeline)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        random_py.seed(10)
        data_raw = mmcv.load(ann_file)
        self.data_info_all = {}
        data_infos = []
        if not self.test_mode:
            for data in data_raw['infos']:
                key_name = data['scene_name'] + '_' + data['vehicle_name'] + '_' + str(data['timestamp'])
                self.data_info_all[key_name] = data

            for data in data_raw['infos']:
                if (data['scenario_length'] - data['timestamp']) % 10 == 0:
                    ego_name = data['vehicle_name']
                    name_mapping = {}
                    name_mapping['ego_vehicle'] = ego_name
                    if ego_name == 'ego_vehicle':
                        name_mapping['ego_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['other_vehicle'] = 'other_vehicle'
                        name_mapping['other_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'ego_vehicle_behind':
                        name_mapping['ego_vehicle_behind'] = 'ego_vehicle'
                        name_mapping['other_vehicle'] = 'other_vehicle'
                        name_mapping['other_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'other_vehicle':
                        name_mapping['ego_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['other_vehicle'] = 'ego_vehicle'
                        name_mapping['other_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'other_vehicle_behind':
                        name_mapping['ego_vehicle_behind'] = 'other_vehicle'
                        name_mapping['other_vehicle'] = 'ego_vehicle'
                        name_mapping['other_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'infrastructure':
                        chosen_name = random_py.choice(['ego_vehicle', 'other_vehicle'])
                        if chosen_name == 'ego_vehicle':
                            another_name = 'other_vehicle'
                        else:
                            another_name = 'ego_vehicle'
                        name_mapping['ego_vehicle_behind'] = chosen_name + '_behind'
                        name_mapping['other_vehicle'] = another_name
                        name_mapping['other_vehicle_behind'] = another_name + '_behind'
                        name_mapping['infrastructure'] = chosen_name

                    data_list = []
                    agent_list_to_select = ['ego_vehicle']
                    if 'infrastructure' in self.agent_list:
                        agent_list_to_select += random.sample(self.agent_list[1:-1], len(self.agent_list[1:-1]))
                        agent_list_to_select += ['infrastructure']
                    else:
                        agent_list_to_select += random.sample(self.agent_list[1:], len(self.agent_list[1:]))

                    for agent_name in agent_list_to_select:
                        agent_key = name_mapping[agent_name]
                        key_name = data['scene_name'] + '_' + agent_key + '_' + str(data['timestamp'])
                        data_list.append(self.data_info_all[key_name])
                    data_infos.append(data_list)

        else:
            for data in data_raw['infos']:
                key_name = data['scene_name'] + '_' + data['vehicle_name'] + '_' + str(data['timestamp'])
                self.data_info_all[key_name] = data

            for data in data_raw['infos']:
                if ((data['scenario_length'] - data['timestamp']) <= 4 * 5) and (
                        (data['scenario_length'] - data['timestamp']) > 0) and 'accident' in data['scene_name']:
                    ego_name = data['vehicle_name']
                    name_mapping = {}
                    name_mapping['ego_vehicle'] = ego_name
                    if ego_name == 'ego_vehicle':
                        name_mapping['ego_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['other_vehicle'] = 'other_vehicle'
                        name_mapping['other_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'ego_vehicle_behind':
                        name_mapping['ego_vehicle_behind'] = 'ego_vehicle'
                        name_mapping['other_vehicle'] = 'other_vehicle'
                        name_mapping['other_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'other_vehicle':
                        name_mapping['ego_vehicle_behind'] = 'other_vehicle_behind'
                        name_mapping['other_vehicle'] = 'ego_vehicle'
                        name_mapping['other_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'other_vehicle_behind':
                        name_mapping['ego_vehicle_behind'] = 'other_vehicle'
                        name_mapping['other_vehicle'] = 'ego_vehicle'
                        name_mapping['other_vehicle_behind'] = 'ego_vehicle_behind'
                        name_mapping['infrastructure'] = 'infrastructure'
                    elif ego_name == 'infrastructure':
                        chosen_name = random_py.choice(['ego_vehicle', 'other_vehicle'])
                        if chosen_name == 'ego_vehicle':
                            another_name = 'other_vehicle'
                        else:
                            another_name = 'ego_vehicle'
                        name_mapping['ego_vehicle_behind'] = chosen_name + '_behind'
                        name_mapping['other_vehicle'] = another_name
                        name_mapping['other_vehicle_behind'] = another_name + '_behind'
                        name_mapping['infrastructure'] = chosen_name

                    data_list = []
                    agent_list_to_select = ['ego_vehicle']
                    if 'infrastructure' in self.agent_list:
                        agent_list_to_select += random.sample(self.agent_list[1:-1], len(self.agent_list[1:-1]))
                        agent_list_to_select += ['infrastructure']
                    else:
                        agent_list_to_select += random.sample(self.agent_list[1:], len(self.agent_list[1:]))

                    for agent_name in agent_list_to_select:
                        agent_key = name_mapping[agent_name]
                        key_name = data['scene_name'] + '_' + agent_key + '_' + str(data['timestamp'])
                        data_list.append(self.data_info_all[key_name])

                    data_infos.append(data_list)

            data_infos = list(
                sorted(data_infos, key=lambda x: (x[0]['scene_name'], x[0]['vehicle_name'], x[0]['timestamp']), reverse=False))
            # selected_index = ['type1_subtype2_accident_ego_vehicle_behind_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_infrastructure_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_other_vehicle_behind_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_ego_vehicle_Town04_type001_subtype0002_scenario00021_024',
            #                   'type1_subtype2_accident_other_vehicle_Town04_type001_subtype0002_scenario00021_024']
            # selected_index = ['type1_subtype1_accident_ego_vehicle_behind_Town04_type001_subtype0002_scenario00019',
            #                   ]
            # data_infos = [data_single for data_single in data_infos if data_single[0]['lidar_prefix'] in selected_index]

        data_infos = data_infos[:50]
        self.metadata = data_raw['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict_list = self.get_data_info(index)

        example_list = []
        for idx, input_dict in enumerate(input_dict_list):
            if input_dict is None:
                return None
            # when the labels for future frames are not complete, skip the sample
            if self.filter_invalid_sample and input_dict['has_invalid_frame'] is True:
                return None

            self.pre_pipeline(input_dict)
            if idx == 0:
                example = self.pipeline(input_dict)
            else:
                example = self.other_agents_pipeline(input_dict)
                # example = self.pipeline(input_dict)
            if idx == 0:
                if self.filter_empty_gt and (example is None or
                                             ~(example['gt_labels_3d']._data != -1).any()):
                    return None
            example_list.append(example)

        example_reorganized = {}
        example_len = len(example_list)
        for key in example_list[0].keys():
            example_reorganized[key] = []
            for i in range(example_len):
                if key in example_list[i].keys():
                    example_reorganized[key].append(example_list[i][key])
                else:
                    example_reorganized[key].append([])

        return example_reorganized

    def prepare_test_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict_list = self.get_data_info(index)
        example_list = []
        for idx, input_dict in enumerate(input_dict_list):
            self.pre_pipeline(input_dict)
            if idx == 0:
                example = self.pipeline(input_dict)
            else:
                example = self.other_agents_pipeline(input_dict)
            example_list.append(example)

        example_reorganized = {}
        example_len = len(example_list)
        for key in example_list[0].keys():
            example_reorganized[key] = []
            for i in range(example_len):
                if key in example_list[i].keys():
                    example_reorganized[key].append(example_list[i][key])
                else:
                    example_reorganized[key].append([])
        return example_reorganized

    def get_temporal_indices_by_info(self, info):
        current_scene_name = info['scene_name']
        current_vehicle_name = info['vehicle_name']

        # generate the past
        previous_indices = []
        for t in range(- self.receptive_field + 1, 0):
            # index_t = index + t
            retrive_timestamp = info['timestamp'] + 5 * t
            key_name = info['scene_name'] + '_' + info['vehicle_name'] \
                       + '_' + str(retrive_timestamp)
            if retrive_timestamp > 0 and self.data_info_all[key_name]['scene_name'] == current_scene_name\
                and self.data_info_all[key_name]['vehicle_name'] == current_vehicle_name:
                previous_indices.append(key_name)
            else:
                previous_indices.append('invalid')  # for invalid indices

        # generate the future
        future_indices = []
        for t in range(1, self.n_future + 1):
            retrive_timestamp = info['timestamp'] + 5 * t
            key_name = info['scene_name'] + '_' + info['vehicle_name']\
                        + '_' + str(retrive_timestamp)
            if retrive_timestamp <= info['scenario_length'] and self.data_info_all[key_name]['scene_name'] == current_scene_name\
                and self.data_info_all[key_name]['vehicle_name'] == current_vehicle_name:
                future_indices.append(key_name)
            else:
                future_indices.append('invalid')

        return previous_indices, future_indices

    def get_other_agent_relative_poses(self, key_name_list):
        ego_key = key_name_list[0]
        ego_data = self.data_info_all[ego_key]
        ego_pose = ego_data['ego_to_world_matrix'].astype(dtype=np.float32)

        other_relative_pose_list = []
        for key_name_single in key_name_list:
            # key_name = ego_data['scene_name'] + '_' + agent_key + '_' + str(ego_data['timestamp'])
            other_agent_info = self.data_info_all[key_name_single]
            other_relative_pose = np.eye(4, dtype=np.float32)
            if other_agent_info['scene_name'] == ego_data['scene_name'] and other_agent_info['timestamp'] == ego_data[
                'timestamp']:
                other_pose = other_agent_info['ego_to_world_matrix'].astype(dtype=np.float32)
                # trans from ego to other
                other_relative_pose = invert_matrix_egopose_numpy(ego_pose).dot(other_pose)
                # other_relative_pose[3, :3] = 0.0
                # other_relative_pose[3, 3] = 1.0
            other_relative_pose = torch.Tensor(other_relative_pose).float()
            other_relative_pose = mat2pose_vec(other_relative_pose)
            other_relative_pose_list.append(other_relative_pose)

        return torch.stack(other_relative_pose_list, dim=0)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info_list = self.data_infos[index]
        key_name_list = []
        for info in info_list:
            key_name_list.append(info['scene_name'] + '_' + info['vehicle_name'] + '_' + str(info['timestamp']))
        relative_pose_to_ego = self.get_other_agent_relative_poses(key_name_list)

        input_dict_list = []
        for idx, info in enumerate(info_list):
            # standard protocal modified from SECOND.Pytorch
            input_dict = dict(
                scenario_length=info['scenario_length'],
                sample_idx=info['lidar_prefix'],
                pts_filename=info['lidar_path'],
                timestamp=info['timestamp'],
                data_root=self.data_root,
                bev_path=info['bev_path'],
                lidar_to_ego_matrix=info['lidar_to_ego_matrix'],
                ego_to_world_matrix=info['ego_to_world_matrix'],
            )
            input_dict['relative_pose_to_ego'] = relative_pose_to_ego

            if idx == 0:
                if self.test_mode:
                    lidar_path = info['lidar_path']
                    prefix = lidar_path.split('/')[:4]
                    prefix.append('meta')
                    meta_file_name = lidar_path.split('/')[-2] + '.txt'
                    prefix.append(meta_file_name)
                    meta_path = '/'.join(prefix)
                    with open(meta_path, 'r') as f:
                        meta_info = [(line.rstrip()) for line in f]

                    town_name = info['scene_name'].split('_')[3]
                    weather_and_time = meta_info[0].split(' ')[0]
                    if 'Noon' in weather_and_time:
                        time_of_the_day = 'Noon'
                        weather = weather_and_time.replace('Noon', '')
                    elif 'Night' in weather_and_time:
                        time_of_the_day = 'Night'
                        weather = weather_and_time.replace('Night', '')
                    elif 'Sunset' in weather_and_time:
                        time_of_the_day = 'Sunset'
                        weather = weather_and_time.replace('Sunset', '')

                    if time_of_the_day == 'Night':
                        time_of_the_day = 'Night'
                    else:
                        time_of_the_day = 'Daytime'

                    if 'Rain' in weather:
                        weather = 'Rainy'
                    elif 'Cloud' in weather:
                        weather = 'Cloudy'
                    else:
                        weather = 'Clear'

                    collision_id1, collision_id2 = meta_info[0].split(' ')[2], meta_info[0].split(' ')[4]
                    if collision_id1 == '-1' or collision_id2 == '-1':
                        collision_status = 'no_collision'
                    elif collision_id1 == 'pedestrian' or collision_id2 == 'pedestrian':
                        collision_status = 'collides_with_pedestrian'
                    else:
                        collision_status = 'collides_between_vehicles'
                    junction_type = meta_info[3].split(': ')[1]

                    self_direction = meta_info[-2].split(': ')[-1]
                    other_direction = meta_info[-1].split(': ')[-1]

                    if junction_type == 'four-way junction' and self_direction == 'straight' and other_direction == 'straight':
                        trajectory_type = 'straight_straight'
                    else:
                        trajectory_type = 'one_of_the_agents_turning'

                    if junction_type == 'four-way junction':
                        junction_type = 'four_way_junction'
                    else:
                        junction_type = 'three_way_junction'

                    input_dict['scenario_type'] = info['scenario_type']
                    input_dict['town_name'] = town_name
                    input_dict['weather'] = weather
                    input_dict['time_of_the_day'] = time_of_the_day
                    input_dict['collision_status'] = collision_status
                    input_dict['junction_type'] = junction_type
                    input_dict['trajectory_type'] = trajectory_type
                else:
                    input_dict['scenario_type'] = None
                    input_dict['town_name'] = None
                    input_dict['weather'] = None
                    input_dict['time_of_the_day'] = None
                    input_dict['collision_status'] = None
                    input_dict['junction_type'] = None
                    input_dict['trajectory_type'] = None

            # get temporal indices
            prev_indices, future_indices = self.get_temporal_indices_by_info(info)

            # # ego motions are needed for all frames

            # ego motions are needed for all frames
            retrive_timestamp = info['timestamp']
            key_name = info['scene_name'] + '_' + info['vehicle_name'] \
                       + '_' + str(retrive_timestamp)
            all_frames = prev_indices + [key_name] + future_indices

            future_egomotions = self.get_egomotions(all_frames)

            input_dict['future_egomotions'] = future_egomotions

            # whether invalid frame is present
            has_invalid_frame = 'invalid' in all_frames
            input_dict['has_invalid_frame'] = has_invalid_frame

            input_dict['img_is_valid'] = np.array(all_frames) != 'invalid'

            # for past frames, we need images, camera paramters, depth(optional)
            img_infos = []
            for prev_key_name in prev_indices:
                if prev_key_name != 'invalid':
                    img_infos.append(self.data_info_all[prev_key_name]['cams'])
                else:
                    # get the information of current frame for invalid frames
                    img_infos.append(info['cams'])

            # current frame
            img_infos.append(info['cams'])
            input_dict['img_info'] = img_infos

            lidar_to_ego_matrix = info['lidar_to_ego_matrix'].astype(np.float32)
            lidar2ego_translation = lidar_to_ego_matrix[:3, 3]
            lidar2ego_rotation = lidar_to_ego_matrix[:3, :3]

            input_dict['lidar2ego_rots'] = torch.tensor(lidar2ego_rotation)
            input_dict['lidar2ego_trans'] = torch.tensor(lidar2ego_translation)

            # for future frames, we need detection labels
            if not self.test_submission:
                if idx == 0:
                    # generate detection labels for current + future frames
                    label_frames = [key_name] + future_indices
                    detection_ann_infos = []
                    for label_frame in label_frames:
                        if label_frame != 'invalid':
                            detection_ann_infos.append(
                                self.get_detection_ann_info_v2x(label_frame))
                        else:
                            detection_ann_infos.append(None)

                    if self.test_mode:
                        past_current_frames = prev_indices + [key_name]
                        accident_visibility_past_current = []
                        for checking_visibility_frame in past_current_frames:
                            accident_visibility_past_current.append(
                                self.check_accident_vehicles_visibility(checking_visibility_frame))

                        # vehicles behind steer curtains are not visible for annotations, which is a bug in CARLA
                        exlude_invisible_scenarios = ['Town05_type001_subtype0002_scenario00035',
                                                      'Town05_type001_subtype0002_scenario00004']

                        accident_visibility_list = np.array(accident_visibility_past_current)
                        accident_invisibility = (accident_visibility_list == False).sum() > len(
                            accident_visibility_list) / 2.0 and (accident_visibility_list[-1] == False)
                        scenario_and_timestep = input_dict['pts_filename'].split('/')[-1].split('.')[0]
                        scenario_name_split = scenario_and_timestep.split('_')[:-1]
                        scenario_name = '_'.join(scenario_name_split)

                        accident_invisibility *= (scenario_name not in exlude_invisible_scenarios)
                        accident_invisibility *= (input_dict['collision_status'] != 'no_collision')

                        accident_visibility = ~accident_invisibility
                        input_dict['accident_visibility'] = accident_visibility


                    input_dict['ann_info'] = detection_ann_infos
                    input_dict['vectors'] = []
                    # # generate map labels only for the current frame
                    # input_dict['vectors'] = self.get_map_ann_info(info)
                else:
                    input_dict['ann_info'] = []
                    input_dict['vectors'] = []
                    if self.test_mode:
                        input_dict['accident_visibility'] = []

            input_dict_list.append(input_dict)
        return input_dict_list

