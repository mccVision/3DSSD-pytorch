import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config
from lib.utils.common_utils import huber_loss
from lib.ops.roiaware_pool3d import roiaware_pool3d_utils
from lib.ops.pointnet2.pointnet2_modules import PointnetSAModuleMSGCG


class CandidateGeneration(nn.Module):
    def __init__(self, cfg):
        super(CandidateGeneration, self).__init__()
        self.backbone_architecture = cfg.backbone.Architecture
        self.vote_architecture = cfg.neck.Vote_Architecture
        self.cg_architecture = cfg.neck.CG_Architecture
        self.loss_config = cfg.neck.Loss_Config
        self.max_translate_range = np.array(cfg.neck.MAX_TRANSLATE_RANGE)
        self.max_translate_range = self.max_translate_range.reshape((1, 1, 3))
        self.min_offset = torch.from_numpy(self.max_translate_range.astype(np.float32)).cuda()

        self.forward_ret_dect = {}

        # -------- Vote Module ----------
        xyz_index = self.vote_architecture[0]
        feature_index = self.vote_architecture[1]
        mlp = self.vote_architecture[2]
        bn = self.vote_architecture[3]

        self.vote_xyz_index = xyz_index
        self.vote_feature_index = feature_index

        vote_input_dim = self.backbone_architecture[feature_index - 1][10]  # just in backbone mlp position
        self.conv1 = nn.Conv1d(vote_input_dim, mlp, 1)
        self.conv2 = nn.Conv1d(mlp, 3, 1)
        self.bn1 = nn.BatchNorm1d(mlp)

        # -------- CG Module -------
        self.CG_module_list = nn.ModuleList()
        cg_output_dim = 0
        for i in range(len(self.cg_architecture)):
            cg_feature_index = self.cg_architecture[i][1]
            cg_input_dim = self.backbone_architecture[cg_feature_index[0] - 1][10]
            self.CG_module_list.append(PointnetSAModuleMSGCG(self.cg_architecture[i], cg_input_dim))
            cg_output_dim += self.cg_architecture[i][8]

    @staticmethod
    def assign_targets(points, gt_boxes):
        """
        assign points to gt boxes
        Args:
            points: ([bs, N, 3], torch.Tensor). candidate points
            gt_boxes: ([bs, M, 8], torch.Tensor). ground truth boxes. [x, y, z, dx, dy, dz, headings, class]
        Returns:
            points_box_mask: ([bs, N], torch.Tensor). mask of points, 1 for positive
            point_box_labels: ([bs, N, 8], torch.Tensor). boxes of all point
        """
        assert len(points.shape) == 3 and points.shape[2] == 3, "points.shape=%s" % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, "gt_boxes.shape=%s" % str(gt_boxes.shape)

        bs, points_num = points.shape[0], points.shape[1]

        points_box_mask = points.new_zeros((bs, points_num))
        points_box_labels = gt_boxes.new_zeros((bs, points_num, 8))

        box_idxs_of_points = roiaware_pool3d_utils.points_in_boxes_gpu(
            points, gt_boxes[:, :, :7]
        ).long()  # (bs, points_num)

        for i in range(bs):
            cur_gt_boxes = gt_boxes[i]  # (M, 8)
            cur_box_idx_of_points = box_idxs_of_points[i, :]  # (points_num,)

            box_fg_flag = (cur_box_idx_of_points >= 0)

            points_box_mask[i, box_fg_flag] = 1.
            points_box_labels[i, box_fg_flag] = cur_gt_boxes[cur_box_idx_of_points[box_fg_flag]]

        return points_box_mask, points_box_labels

    @staticmethod
    def data_clear(data):
        isnan = torch.isnan(data)
        isinf = torch.isinf(data)
        if torch.sum(isnan) > 0:
            data[isnan] = 0.
        if torch.sum(isinf) > 0:
            data[isinf] = 0.

        return data

    def get_loss(self, tb_dict=None):
        """
        calculate voting/shifting loss
        Args:
            tb_dict:
        Returns:
            vote_loss: (scalar)
        """
        gt_boxes = self.forward_ret_dect['gt_boxes']
        base_xyz = self.forward_ret_dect['original_xyz']

        points_box_mask, points_box_labels = self.assign_targets(base_xyz, gt_boxes)
        norm_para = torch.clamp(torch.sum(points_box_mask), min=1.0)

        gt_ctr_offsets = points_box_labels[..., :3] - base_xyz[..., :3]
        pred_ctr_offsets = self.forward_ret_dect['ctr_offset']

        gt_ctr_offsets = self.data_clear(gt_ctr_offsets)
        pred_ctr_offsets = self.data_clear(pred_ctr_offsets)

        vote_loss = huber_loss(gt_ctr_offsets - pred_ctr_offsets, delta=1.)
        vote_loss = torch.mul(torch.sum(vote_loss, dim=-1), points_box_mask)
        vote_loss = torch.sum(vote_loss) / norm_para
        vote_loss = vote_loss * torch.tensor(self.loss_config.Vote_Loss_Weight, dtype=torch.float32)

        tb_dict = {} if tb_dict is None else tb_dict
        tb_dict['vote_loss'] = '%.3f' % vote_loss.item()

        return vote_loss, tb_dict

    def forward(self, output_dict, data_dict):
        xyz_list = output_dict['backbone_xyz_list']
        feature_list = output_dict['backbone_feature_list']
        fps_idx_list = output_dict['backbone_idx_list']

        vote_input_xyz = xyz_list[self.vote_xyz_index]  # (B, npoint, 3)
        vote_input_feature = feature_list[self.vote_feature_index]  # (B, C, npoint)

        # ------- Voting Module -------
        vote_feature = F.relu(self.bn1(self.conv1(vote_input_feature)))
        ctr_offsets_features = self.conv2(vote_feature)  # (B, 3, npoint)
        ctr_offsets = ctr_offsets_features.transpose(2, 1)  # (B, npoint, 3)

        # min_offset = torch.from_numpy(self.max_translate_range.astype(np.float32)).to(ctr_offsets.device)
        limited_ctr_offsets = torch.min(torch.max(ctr_offsets, self.min_offset), -self.min_offset)
        new_xyz = torch.add(vote_input_xyz, limited_ctr_offsets)

        # ------- CG Module --------
        # cg_xyz, cg_feature, cg_fps_idx = self.CG_module(xyz_list, feature_list, fps_idx_list, new_xyz)
        cg_feature_list = []
        for i in range(len(self.CG_module_list)):
            tmp_xyz, tmp_feature, tmp_idx = self.CG_module_list[i](xyz_list, feature_list, fps_idx_list, new_xyz)
            cg_feature_list.append(tmp_feature)
        cg_feature = torch.cat(cg_feature_list, dim=1)

        output_dict['ctr_offset'] = ctr_offsets
        output_dict['original_xyz'] = vote_input_xyz
        # output_dict['original_feature'] = vote_input_feature

        output_dict['candidate_xyz'] = new_xyz
        output_dict['candidate_feature'] = cg_feature
        output_dict['candidate_fps_idx'] = None

        if self.training:
            self.forward_ret_dect['ctr_offset'] = ctr_offsets
            self.forward_ret_dect['original_xyz'] = vote_input_xyz
            self.forward_ret_dect['gt_boxes'] = data_dict['gt_boxes']

        return output_dict


def parse_args():
    parser = argparse.ArgumentParser(description='candidate generation test')
    parser.add_argument('-config', default='../../configs/kitti/car_cfg.py',
                        help='file path of train config')

    args = parser.parse_args()
    return args


def main():
    cg_args = parse_args()

    ssd_cfg = Config.fromfile(cg_args.config)
    print(ssd_cfg)

    CG_Module = CandidateGeneration(ssd_cfg.Model)

    # simulate data list
    xyz_list = []
    feature_list = []
    fps_idx_list = []
    for i in range(4):
        xyz_list.append(None)
        feature_list.append([None])
        fps_idx_list.append([None])

    for i in range(2):
        xyz = torch.rand(2, 256, 3)
        feature = torch.rand(2, 256, 256)
        fps_idx = torch.rand(2, 256)

        xyz_list.append(xyz)
        feature_list.append(feature)
        fps_idx_list.append(fps_idx)

    output = {
        'backbone_xyz_list': xyz_list,
        'backbone_feature_list': feature_list,
        'backbone_idx_list': fps_idx_list,
    }

    output_dict = CG_Module(output)


if __name__ == '__main__':
    main()
