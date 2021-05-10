import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils.box_utils as box_utils
import lib.utils.common_utils as common_utils
import lib.ops.pointnet2.pytorch_utils as pt_utils
import lib.ops.pointnet2.pointnet2_utils as pointnet2_utils
from lib.ops.roiaware_pool3d import roiaware_pool3d_utils


class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        """
        perspective-wise attention
        Args:
            dim_pa: perspective attention channel
            reduction_pa: reduction r of perspective-wise attention
        """
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_pa, dim_pa)
        )

    def forward(self, x):
        """
        perspective-wise attention forward
        Args:
            x: ([bs, points_num, 8, aggregation_channel], torch.Tensor).
        Returns:
            out: ([bs, points_num, 8, 1], torch.Tensor).
        """
        bs, points_num, w, _ = x.size()
        y = torch.max(x, dim=3, keepdim=True)[0].view(bs, points_num, w)
        out = self.fc(y).view(bs, points_num, w, 1)
        return out


class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        """
        channel-wise attention
        Args:
            dim_ca: channel attention dimension
            reduction_ca: reduction r of channel-wise attention
        """
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_ca, dim_ca)
        )

    def forward(self, x):
        """
        channel-wise attention forward
        Args:
            x: ([bs, points_num, 8, channel], torch.Tensor)
        Returns:
            out: ([bs, points_num, 1, channel], torch.Tensor)
        """
        bs, points_num, _, channel = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(bs, points_num, channel)
        out = self.fc(y).view(bs, points_num, 1, channel)

        return out


class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer(dim_pa, dim_pa // reduction_r)
        self.ca = CALayer(dim_ca, dim_ca // reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        PA and CA forward
        Args:
            x: ([bs, points_num, 8, channel], torch.Tensor)
        Returns:
            out: ([bs, points_num, channel], torch.Tensor)
        """
        pa_weight = self.pa(x)  # (bs, points_num, 8, 1)
        ca_weight = self.ca(x)  # (bs, points_num, 1, channel)
        paca_weight = torch.matmul(pa_weight, ca_weight)  # (bs, points_num, 8, channel)
        paca_norm_weight = self.sig(paca_weight)
        out = torch.mul(paca_norm_weight, x)  # (bs, points_num, 8, channel)
        out = torch.sum(out, dim=2, keepdim=False)  # (bs, points_num, channel)

        return out


def grid_rotation(boxes, grid):
    """

    Args:
        boxes: ([bs, npoint, 7/8/9], torch.Tensor).
        grid: ([bs, npoint, K, 3], torch.Tensor).
    Returns:

    """
    heading = boxes[:, :, 6:7]
    rotation = grid.new_zeros(tuple(list(heading.shape) + [3, 3]))
    c = torch.cos(heading)
    s = torch.sin(heading)
    rotation[..., 0, 0] = c
    rotation[..., 0, 1] = s
    rotation[..., 1, 0] = -s
    rotation[..., 1, 1] = c
    rotation[..., 2, 2] = 1

    bs, npoint = boxes.shape[0], boxes.shape[1]
    rotation = rotation.reshape(-1, 3, 3)  # (bs * npoint, 3, 3)
    grid = grid.reshape(bs * npoint, -1, 3)
    grid = torch.matmul(grid, rotation).reshape(bs, npoint, -1, 3)

    return grid


class IoUHead(nn.Module):
    def __init__(self, cfg, num_class, use_xyz=True):
        super(IoUHead, self).__init__()

        iouhead_cfg = cfg.iouhead
        self.use_xyz = use_xyz
        self.num_class = num_class
        self.heading_bin_num = cfg.head.HEADING_BIN_NUM

        # ------- create grid -------
        grid_size = 4
        grid_step = torch.linspace(-1, 1, 4).cuda()

        self.grid_step_x = grid_step.reshape(grid_size, 1, 1).repeat(1, grid_size, grid_size)  # (grid_size, grid_size, gri)
        self.grid_step_y = grid_step.reshape(1, grid_size, 1).repeat(grid_size, 1, grid_size)
        self.grid_step_z = grid_step.reshape(1, 1, grid_size).repeat(grid_size, grid_size, 1)

        # ------- ACA Module -------
        self.cge_cfg = iouhead_cfg.CGE_Config
        if use_xyz:
            self.cge_cfg.MLP[0] += 3
            self.mlp = self.cge_cfg.MLP
        else:
            self.mlp = self.cge_cfg.MLP
        self.cge_mlp = pt_utils.SharedMLP(self.mlp)

        self.cge_conv0 = nn.Conv2d(self.cge_cfg.MLP[-1], self.cge_cfg.MLP[-1], kernel_size=(1, 8))
        self.cge_bn0 = nn.BatchNorm2d(self.cge_cfg.MLP[-1])

        # # ------- MERGE Module -------
        # self.merge_cfg = iouhead_cfg.MERGE_Config
        # head_input_dim = cfg.head.Architecture[0][-1]
        # merge_input_dim = head_input_dim + self.cge_cfg.MLP[-1]
        # self.merge_mlp = self._make_layers(merge_input_dim, self.merge_cfg.MLP)

        # ------- IoU prediction -------
        self.iou_conv0 = nn.Conv1d(self.cge_cfg.MLP[-1], 128, 1)
        self.iou_conv1 = nn.Conv1d(128, 128, 1)
        self.iou_conv2 = nn.Conv1d(128, 1, 1)
        self.iou_bn0 = nn.BatchNorm1d(128)
        self.iou_bn1 = nn.BatchNorm1d(128)

    def _make_layers(self, input_feature_dim, mlps):
        """
        make layers
        Args:
            input_feature_dim: input feature dim
            mlps: list of feature dim
        Returns:
            layers:
        """
        layers = []
        for mlp in mlps:
            layers += [nn.Conv1d(input_feature_dim, mlp, 1),
                       nn.BatchNorm1d(mlp),
                       nn.ReLU(inplace=True)]
            input_feature_dim = mlp
        return nn.Sequential(*layers)

    def decode_prediction(self, output_dict):
        """
        decode all prediction (with cls scores)
        Args:
            output_dict:
        Returns:
            pred_boxes: ([bs, points_num, 9], torch.Tensor). [x, y, z, dx, dy, dz, heading, cls, cls_scores]
        """
        pred_cls = output_dict['pred_cls']
        base_xyz = output_dict['candidate_xyz']
        pred_offset = output_dict['pred_offsets']
        pred_angle_cls = output_dict['pred_angle_cls']  # (bs, points_num, heading_bin_num)
        pred_angle_res = output_dict['pred_angle_res']

        # ------- decode 3D boxes info -------
        angle_cls_one_hot = pred_angle_cls.new_zeros(*list(pred_angle_cls.shape))
        angle_cls_assign = torch.argmax(pred_angle_cls, dim=-1).float()
        angle_cls_one_hot.scatter_(-1, angle_cls_assign.unsqueeze(-1).long(), 1.0)

        angle_res = torch.sum(torch.mul(pred_angle_res, angle_cls_one_hot), dim=-1)  # (bs, points_num)
        pred_angle = common_utils.class2angle_torch(angle_cls_assign, angle_res, self.heading_bin_num).float()

        pred_xyz = torch.add(base_xyz, pred_offset[:, :, :3]).float()
        pred_lwh = torch.clamp(pred_offset[:, :, 3:6] * 2.0, min=0.1)

        # ------- decode prediction class -------
        pred_cls = torch.argmax(pred_cls, dim=-1).unsqueeze(-1).float()  # (bs, points_num, 1)
        pred_cls_scores = torch.gather(pred_cls, 2, pred_cls.long()).float()  # (bs, points_num, 1)

        pred_boxes = torch.cat((pred_xyz, pred_lwh.float(), pred_angle.unsqueeze(-1), pred_cls, pred_cls_scores), dim=-1)
        pred_boxes = pred_boxes.detach()

        return pred_boxes

    def forward(self, output_dict):
        """
        IoU Head froward pass
        Args:
            output_dict:
        Returns:
            pred_iou: ([bs, cls_num, points_num])
        """
        original_xyz = output_dict['original_xyz']  # (bs, npoint, 3)
        original_xyz = original_xyz.detach().contiguous()
        original_feature = output_dict['original_feature']
        original_feature = original_feature.detach().contiguous()

        bs, npoint = original_xyz.shape[0], original_xyz.shape[1]
        in_channel = original_feature.shape[1]
        # ------- decode prediction -------
        pred_boxes = self.decode_prediction(output_dict)  # (bs, npoint, 9)
        # pred_corners = box_utils.batch_boxes_to_corners_3d(pred_boxes[:, :, :7])  # (bs, npoint, 8, 3)
        # pred_corners_reshape = pred_corners.reshape(pred_boxes.shape[0], -1, 3)  # (bs, npoint * 8, 3)

        grid_size = 4
        grid_step = torch.linspace(-1, 1, 4).cuda()

        grid_step_x = grid_step.reshape(grid_size, 1, 1).repeat(1, grid_size, grid_size)  # (grid_size, grid_size, gri)
        grid_step_y = grid_step.reshape(1, grid_size, 1).repeat(grid_size, 1, grid_size)
        grid_step_z = grid_step.reshape(1, 1, grid_size).repeat(grid_size, grid_size, 1)
        grid_step_x = grid_step_x.reshape(1, 1, -1).expand(bs, npoint, -1)  # (bs, npoint, K)
        grid_step_y = grid_step_y.reshape(1, 1, -1).expand(bs, npoint, -1)  # (bs, npoint, K)
        grid_step_z = grid_step_z.reshape(1, 1, -1).expand(bs, npoint, -1)
        x_grid = torch.mul(grid_step_x, pred_boxes[:, :, 3:4])
        y_grid = torch.mul(grid_step_y, pred_boxes[:, :, 4:5])
        z_grid = torch.mul(grid_step_z, pred_boxes[:, :, 5:6])
        whole_grid = torch.cat([x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1)  # (bs, npoint, K, 3)
        relative_corners = grid_rotation(pred_boxes, whole_grid)  # (bs, npoint, K, 3)
        pred_corners = torch.add(relative_corners, pred_boxes[:, :, :3].unsqueeze(2))
        pred_corners_reshape = pred_corners.reshape(bs, -1, 3)

        _, idx = pointnet2_utils.three_nn(pred_corners_reshape, original_xyz)  # (bs, npoint * K, 3)
        interp_points = pointnet2_utils.gather_operation(
            original_xyz.transpose(2, 1).contiguous(),
            idx.reshape(bs, -1).contiguous()
        ).transpose(2, 1).contiguous()  # (bs, npoint * K * 3, 3), point coordinate
        expanded_corners = pred_corners_reshape.unsqueeze(2).expand(-1, -1, 3, -1).contiguous().view(bs, -1, 3)  # (bs, npoint * K * 3, 3)
        dist = interp_points - expanded_corners  # (bs, npoint * K * 3, 3)
        dist = torch.sqrt(torch.sum(torch.mul(dist, dist), dim=2))  # (bs, npoint * K * 3), l2 distance

        dist_weight = torch.div(1., dist + 1e-8)
        dist_weight = dist_weight.view(bs, -1, 3)  # (bs, npoint * K, 3)
        norm = torch.sum(dist_weight, dim=-1, keepdim=True)
        dist_weight = torch.div(dist_weight, norm)
        dist_weight = dist_weight.contiguous()  # (bs, npoint * K, 3)

        interp_feature = pointnet2_utils.gather_operation(
            original_feature,
            idx.reshape(bs, -1).contiguous()
        )  # (bs, C, npoint * K * 3)
        interp_feature = interp_feature.reshape(bs, in_channel, -1, 3)  # (bs, C, npoint * K, 3)
        interp_feature = torch.mul(interp_feature, dist_weight.unsqueeze(1))  # (bs, C, npoint * K, 3)
        interp_feature = torch.sum(interp_feature, dim=-1)  # (bs, C, npoint * K)
        interp_feature = interp_feature.reshape(bs, in_channel, npoint, -1)  # (bs, C, npoint, K)

        if self.use_xyz:
            interp_feature = torch.cat([relative_corners.permute(0, 3, 1, 2), interp_feature], dim=1)

        interp_feature = self.cge_mlp(interp_feature)  # (bs, C', npoint, K)
        interp_feature = F.max_pool2d(interp_feature, kernel_size=[1, interp_feature.shape[3]]).squeeze(-1)

        # IOU regression
        iou_feature = F.relu(self.iou_bn0(self.iou_conv0(interp_feature)))
        iou_feature = F.relu(self.iou_bn1(self.iou_conv1(iou_feature)))
        iou_feature = self.iou_conv2(iou_feature)

        # interp_points = torch.gather(original_xyz, dim=1, index=idx.view(bs, -1, 1).expand(-1, -1, 3).long())  # (bs, npoint * 8 * 3, 3)
        # expanded_corners = pred_corners_reshape.unsqueeze(2).expand(-1, -1, 3, 1).contiguous().view(bs, -1, 3)  # (bs, npoint * 8 * 3, 3)
        # dist = interp_points - expanded_corners
        # dist = torch.sqrt(torch.sum(torch.mul(dist, dist), dim=2))  # (bs, npoint * 8 * 3)
        #
        # dist_weight = torch.div(1., dist + 1e-8)
        # dist_weight = dist_weight.view(bs, -1, 3)  # (bs, npoint * 8, 3)
        # norm = torch.sum(dist_weight, dim=-1, keepdim=True)
        # dist_weight = torch.div(dist_weight, norm)
        # dist_weight = dist_weight.contiguous()

        # group_dist, group_idxs = pointnet2_utils.three_nn(pred_corners_reshape, output_dict['candidate_xyz'])  # (bs, npoint * 8, 3) index and distance of point
        # group_dist_weight = 1. / (group_dist * group_dist)
        # group_dist_sum = torch.sum(group_dist_weight, dim=-1, keepdim=True)  # (bs, npoint * 8, 1)
        # group_dist_weight = torch.div(group_dist_weight, group_dist_sum)  # (bs, npoint * 8, 3)
        # group_feature = pointnet2_utils.three_interpolate(output_dict['candidate_feature'], group_idxs, group_dist_weight)  # (bs, C, npoint * 8)
        # # group_feature = pointnet2_utils.gather_operation(output_dict['candidate_feature'], group_idxs)  # (bs, C, npoint * 8, 3)
        # group_feature = group_feature.reshape(pred_boxes.shape[0], group_feature.shape[1], pred_boxes.shape[1], -1)
        #
        # group_feature = self.cge_mlp(group_feature)
        # group_feature = F.relu(self.cge_bn0(self.cge_conv0(group_feature))).squeeze(-1)  # (bs, C', npoint)

        # ------- CGE module -------
        # cge_feature = self.cge_mlp(pred_corners.permute([0, 3, 1, 2]))  # (bs, cge_mlp[-1], points_num, 8)
        # cge_feature = F.relu(self.cge_bn0(self.cge_conv0(cge_feature))).squeeze(-1)  # (bs, cge_mlp[-1], points_num)

        # # ------- MERGE module -------
        # merge_feature = torch.cat((head_feature, cge_feature), dim=1)  # (bs, C''-sum, points_num)
        # merge_feature = self.merge_mlp(merge_feature)  # (bs, merge_mlp[-1], points_num)

        # ------- IoU prediction -------
        # iou_feature = F.relu(self.iou_bn0(self.iou_conv0(group_feature)))
        # iou_feature = self.iou_conv1(iou_feature)

        return iou_feature


def func_test():
    a = torch.rand(10, 10)

    res = a[torch.arange(10), torch.arange(10)]
    print(res.shape)


if __name__ == '__main__':
    func_test()

