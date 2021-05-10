from functools import partial
import numpy as np

from ...utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        data_processor_list = processor_configs.DATA_PROCESSOR_LIST
        for cur_cfg in data_processor_list:
            pro_method_dict = processor_configs[cur_cfg]
            cur_processor = getattr(self, pro_method_dict.type)(config=pro_method_dict.configs)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def random_select_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_select_points, config=config)

        max_points = config.POINTS_NUM[self.mode]
        far_threshold = config.FAR_THRE
        points = data_dict['points']
        if max_points < len(points):
            # points_depth = np.linalg.norm(points[:, :3], axis=1)
            points_depth = points[:, 0]
            points_near_flag = points_depth < far_threshold
            far_idxs_choice = np.where(points_near_flag == 0)[0]
            near_idxs_choice = np.where(points_near_flag == 1)[0]

            replace = (near_idxs_choice.shape[0] < (max_points - len(far_idxs_choice)))
            near_idxs_choice = np.random.choice(near_idxs_choice, max_points - len(far_idxs_choice), replace=replace)

            if len(far_idxs_choice) > 0:
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
            else:
                choice = near_idxs_choice
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if max_points > len(points):
                replace = (choice.shape[0] < (max_points - len(points)))
                extra_choice = np.random.choice(choice, max_points - len(points), replace=replace)
                choice = np.concatenate((choice, extra_choice), axis=0)

        points = points[choice, :]
        data_dict['points'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
