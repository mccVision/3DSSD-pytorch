import numpy as np
import copy
import pickle
from ...utils import box_utils
from ...ops.iou3d_nms import iou3d_nms_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        easy_dict = {}
        hard_dict = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            easy_dict[class_name] = []
            hard_dict[class_name] = []

        # load database
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        # prepare ground truth database for data augmentation
        for key, dinfos in self.db_infos.items():
            for info in dinfos:
                if info['num_points_in_gt'] > 100:
                    easy_dict[key].append(info)
                else:
                    hard_dict[key].append(info)
        self.gt_database = [easy_dict, hard_dict]

        # prepare method
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:
            sampled_dict:
                db_infos
        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        # same as initialize
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def sample_with_fixed_times(self, class_name, sample_group, existed_boxes):
        """
        sample with fixes try times, for strengthen hard type detection AP
        Args:
            class_name: str
            sample_group:
            existed_boxes: gt boxes
        Returns:
            sampled_dict:
        """
        sample_num = int(sample_group['sample_num'])

        # enlarge boundary to avoid too nearby boxes
        existed_enlarge_boxes = copy.deepcopy(existed_boxes)
        existed_enlarge_boxes[:, 3] += 0.5
        existed_enlarge_boxes[:, 4] += 0.5

        cnt = 0
        try_times = 150
        sampled_dict = []

        while try_times > 0:
            if cnt > sample_num:
                break

            try_times -= 1
            p = np.random.rand()
            if p > self.sampler_cfg.GT_AUG_HARD_RATIO:
                rand_idx = np.random.randint(0, len(self.gt_database[0][class_name]))
                new_sample_dict = self.gt_database[0][class_name][rand_idx]
            else:
                rand_idx = np.random.randint(0, len(self.gt_database[1][class_name]))
                new_sample_dict = self.gt_database[1][class_name][rand_idx]

            sample_box = new_sample_dict['box3d_lidar'].reshape(-1, 7)
            iou_bev = iou3d_nms_utils.boxes_bev_iou_cpu(sample_box[:, :7], existed_enlarge_boxes[:, :7])

            if iou_bev is None:
                continue
            valid_flag = iou_bev.max() < 1e-8
            if not valid_flag:
                continue

            cnt += 1
            sampled_dict.append(new_sample_dict)
            sampled_enlarge_box = copy.deepcopy(sample_box)
            sampled_enlarge_box[:, 3] += 0.5
            sampled_enlarge_box[:, 4] += 0.5
            existed_enlarge_boxes = np.concatenate((existed_enlarge_boxes, sampled_enlarge_box), axis=0)

        if self.logger is not None:
            self.logger.info('Database sample {} class boxes {} count.'.format(class_name, cnt))

        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib: calibration
        Returns:
            gt_boxes: (N, 7 + C)
            mv_height: (N)
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        """
        put the sampled boxes to scene plane
        Args:
            data_dict:
            sampled_gt_boxes:
            total_valid_sampled_dict:
        Returns:
            data_dict:
                gt_boxes:
                gt_names:
                points:
        """
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
            sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
        )
        data_dict.pop('calib')
        data_dict.pop('road_plane')

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)

            if int(sample_group['sample_num']) > 0:
                if self.sampler_cfg.SAMPLE_WITH_FIX_NUM:
                    sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                    # sampled boxes bounding boxes info
                    sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                    # convert to LiDAR coord
                    if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                        sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                    iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])  # (N, M)
                    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])  # (N, N)
                    iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0  # diagonal = 0
                    iou1 = iou1 if iou1.shape[1] > 0 else iou2
                    valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                    valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                    valid_sampled_boxes = sampled_boxes[valid_mask]

                    existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                    total_valid_sampled_dict.extend(valid_sampled_dict)
                else:
                    sampled_dict = self.sample_with_fixed_times(class_name, sample_group, existed_boxes)

                    sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)
                    existed_boxes = np.concatenate((existed_boxes, sampled_boxes), axis=0)
                    total_valid_sampled_dict.extend(sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict.pop('gt_boxes_mask')
        return data_dict
