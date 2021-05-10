import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import lib.utils.box_utils as box_utils
from lib.models.iou_head import IoUHead
import lib.utils.loss_utils as loss_utils
import lib.utils.common_utils as common_utils
import lib.ops.pointnet2.pointnet2_utils as pointnet2_utils
from lib.ops.roiaware_pool3d import roiaware_pool3d_utils
from lib.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class RegressionHead(nn.Module):
    def __init__(self, cfg, class_names=None):
        super(RegressionHead, self).__init__()

        head_cfg = cfg.head
        self.loss_config = head_cfg.Loss_Config
        self.head_architecture = head_cfg.Architecture
        self.mlp_list = self.head_architecture[0]
        self.bn = self.head_architecture[1]

        self.num_class = len(class_names)
        self.pred_cls_channel = len(class_names)
        self.pred_reg_channel_num = 6  # offset of (x, y, z); half value of (dx, dy, dz)
        self.heading_bin_num = head_cfg.HEADING_BIN_NUM

        self.global_step = 0
        self.forward_ret_dict = {}

        # input_feature_dim = cfg.neck.CG_Architecture[8]
        input_feature_dim = 0
        for cg_arc in cfg.neck.CG_Architecture:
            input_feature_dim += cg_arc[8]

        # ------- feature learning ------
        self.features_learning = self._make_layers(input_feature_dim)

        # -------- classification -------
        cls_input_dim = self.mlp_list[-1]
        self.cls_conv1 = nn.Conv1d(cls_input_dim, 128, 1)
        self.cls_conv2 = nn.Conv1d(128, self.pred_cls_channel, 1)
        self.cls_bn1 = nn.BatchNorm1d(128)

        # --- bounding box prediction ---
        self.reg_conv1 = nn.Conv1d(cls_input_dim, 128, 1)
        self.reg_conv2 = nn.Conv1d(128,
                                   self.pred_reg_channel_num + self.heading_bin_num * 2,
                                   1)
        self.reg_bn1 = nn.BatchNorm1d(128)

        # ------- IoU prediction -------
        self.bool_iouhead = cfg.head.IoU_HEAD
        if self.bool_iouhead:
            self.iou_head = IoUHead(cfg, self.num_class)

            # self.iou_conv0 = nn.Conv1d(cls_input_dim, 128, 1)
            # self.iou_conv1 = nn.Conv1d(128, 1, 1)
            # self.iou_bn0 = nn.BatchNorm1d(128)

    def update_global_step(self):
        self.global_step = self.global_step + 1

    def _make_layers(self, input_feature_dim):
        layers = []
        for mlp in self.mlp_list:
            layers += [nn.Conv1d(input_feature_dim, mlp, 1),
                       nn.BatchNorm1d(mlp),
                       nn.ReLU(inplace=True)]
            input_feature_dim = mlp
        return nn.Sequential(*layers)

    def decode_prediction(self, points, pred_offsets, angle_cls_one_hot, angle_res):
        """
        decode prediction value to generate prediction boxes
        Args:
            points: ([bs, points_num, 3], torch.Tensor). base points coords
            pred_offsets: ([bs, points_num, 6], torch.Tensor). predict offset
            angle_cls_one_hot: ([bs. points_num, num_heading_bin], torch.Tensor). one-hot encode of angle cls
            angle_res: ([bs, points_num, num_heading_bin], torch.Tensor). prediction value of angle residual
        Returns:
            pred_boxes: ([bs, points_num, 7], torch.Tensor). prediction results of 3D boxes
        """
        angle_cls = torch.argmax(angle_cls_one_hot, dim=-1).float()  # (bs, points_num)
        angle_res = torch.sum(torch.mul(angle_cls_one_hot, angle_res), -1)  # (bs, points_num)
        pred_angle = common_utils.class2angle_torch(angle_cls, angle_res, self.heading_bin_num)

        pred_xyz = torch.add(points, pred_offsets[:, :, :3])
        pred_lwh = torch.clamp(pred_offsets[:, :, 3:6] * 2.0, min=0.1)

        pred_boxes = torch.cat((pred_xyz, pred_lwh, pred_angle.unsqueeze(-1)), dim=-1)
        return pred_boxes

    def assign_targets(self, points, gt_boxes):
        """
        assign points to boxes it belong to
        Args:
            points: ([bs, N, 3], torch.Tensor)
            gt_boxes: ([bs, M, 8], torch.Tensor). gt boxes info. [x, y, z, dx, dy, dz, heading, class]
        Returns:
            points_cls_labels: ([bs, N], torch.Tensor). class of points, 0->background
            points_box_labels: ([bs, N, 8], torch.Tensor). boxes of points. format as gt_boxes
        """
        assert len(points.shape) == 3 and points.shape[2] == 3, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)

        bs, points_num = points.shape[0], points.shape[1]

        points_cls_labels = points.new_zeros((bs, points_num))
        points_box_labels = gt_boxes.new_zeros((bs, points_num, 8))

        box_idxs_of_points = roiaware_pool3d_utils.points_in_boxes_gpu(
            points.contiguous(), gt_boxes[:, :, :7].contiguous()
        ).long()  # (bs, N)

        for i in range(bs):
            cur_box_idxs_of_points = box_idxs_of_points[i]

            box_fg_flag = (cur_box_idxs_of_points >= 0)

            cur_points_box_labels = gt_boxes[i][cur_box_idxs_of_points[box_fg_flag]]
            points_cls_labels[i, box_fg_flag] = 1 if self.num_class == 1 else cur_points_box_labels[:, 7]
            points_box_labels[i, box_fg_flag] = cur_points_box_labels

        return points_cls_labels, points_box_labels

    def iou_assign_targets(self, pred_boxes, gt_boxes):
        """
        assign targets by IoU value
        Args:
            points:
            pred_boxes: ([bs, N, 7], torch.Tensor). prediction boxes info. [x, y, z, dx, dy, dz, heading]
            gt_boxes: ([bs, M, 8], torch.Tensor). gt boxes info. [x, y, z, dx, dy, dz, heading, cls]
        Returns:
            points_cls_labels: ([bs, N], torch.Tensor). class of points, 0->background
            points_box_labels: ([bs, N, 8], torch.Tensor). boxes of points. [x, y, z, dx, dy, dz, heading, gt_iou]
        """
        bs, points_num = pred_boxes.shape[0], pred_boxes.shape[1]

        points_cls_labels = gt_boxes.new_zeros((bs, points_num))
        points_box_labels = gt_boxes.new_zeros((bs, points_num, 8))

        for i in range(bs):
            cur_pred_boxes = pred_boxes[i]  # (N, 7)
            cur_gt_boxes = gt_boxes[i]  # (M, 8)

            cur_iou3d = boxes_iou3d_gpu(cur_pred_boxes, cur_gt_boxes[:, :7])  # (N, M)
            cur_iou3d_max, cur_iou3d_idx = torch.max(cur_iou3d, dim=-1)  # (N)

            cur_keep_flag = (cur_iou3d_max >= 0.55)

            cur_points_box_labels = cur_gt_boxes[cur_iou3d_idx[cur_keep_flag]]
            # print(cur_points_box_labels)
            points_cls_labels[i, cur_keep_flag] = 1 if self.num_class == 1 else cur_points_box_labels[:, 7]
            points_box_labels[i, cur_keep_flag] = cur_points_box_labels
            points_box_labels[i, cur_keep_flag, 7] = cur_iou3d_max[cur_keep_flag]

        return points_cls_labels, points_box_labels

    @staticmethod
    def data_clear(data):
        """
        data clear
        Args:
            data:
        Returns:
            data:
        """
        isnan = torch.isnan(data)
        isinf = torch.isinf(data)
        if torch.sum(isnan).item() > 0:
            data[isnan] = 0.
        if torch.sum(isinf).item() > 0:
            data[isinf] = 0.

        return data

    @staticmethod
    def ctr_data_clear(data):
        data_isnan = torch.isnan(data)
        data_isinf = torch.isinf(data)
        if torch.sum(data_isnan).item() > 0:
            data[data_isnan] = 1.0
        if torch.sum(data_isinf).item() > 0:
            data[data_isinf] = 1.0

        return data

    def _generate_centerness_label(self, base_xyz, point_box_labels, pos_mask_ori, epsilon=1e-9):
        """
        generate centerness value like FCOS
        Args:
            base_xyz: ([bs, points_num, 3], torch.Tensor). candidate xyz
            point_box_labels: ([bs, points_num, 8], torch.Tensor). gt box of points, [x, y, z, dx, dy, dz, heading, cls]
            pos_mask: ([bs, points_num], torch.Tensor). positive point mask
            epsilon:
        Returns:
            ctrness: ([bs, points_num], torch.Tensor). centerness value for classification regression
        """
        assert len(base_xyz.shape) == 3 and base_xyz.shape[2] == 3, 'candidate_xyz.shape=%s' % str(base_xyz.shape)

        pos_mask = pos_mask_ori.clone()
        pos_mask = self.ctr_data_clear(pos_mask)
        bs, points_num = base_xyz.shape[0], base_xyz.shape[1]

        point_box_labels = point_box_labels.clone().detach()

        canonical_xyz = base_xyz - point_box_labels[..., :3]
        canonical_xyz = torch.reshape(canonical_xyz, (bs * points_num, 3)).unsqueeze(1)  # (bs*points_num, 1, 3)
        ry = torch.reshape(point_box_labels[:, :, 6], (bs * points_num,))

        canonical_xyz = common_utils.rotate_points_along_z(canonical_xyz, ry)
        canonical_xyz = torch.reshape(canonical_xyz.squeeze(1), (bs, points_num, 3))  # (bs, points_num, 3)

        x, y, z = canonical_xyz[:, :, 0], canonical_xyz[:, :, 1], canonical_xyz[:, :, 2]
        dx, dy, dz = point_box_labels[:, :, 3], point_box_labels[:, :, 4], point_box_labels[:, :, 5]

        distance_front = torch.abs((dx / 2.) - x)
        distance_front = torch.clamp(self.ctr_data_clear(distance_front), min=epsilon)
        distance_back = torch.abs(x + (dx / 2.))
        distance_back = torch.clamp(self.ctr_data_clear(distance_back), min=epsilon)

        distance_left = torch.abs((dy / 2.) - y)
        distance_left = torch.clamp(self.ctr_data_clear(distance_left), min=epsilon)
        distance_right = torch.abs(y + (dy / 2.))
        distance_right = torch.clamp(self.ctr_data_clear(distance_right), min=epsilon)

        distance_top = torch.abs((dz / 2.) - z)
        distance_top = torch.clamp(self.ctr_data_clear(distance_top), min=epsilon)
        distance_bottom = torch.abs(z + (dz / 2.))
        distance_bottom = torch.clamp(self.ctr_data_clear(distance_bottom), min=epsilon)

        ctrness_l = torch.mul(torch.div(self.ctr_data_clear(torch.min(distance_front, distance_back)),
                                        self.ctr_data_clear(torch.max(distance_front, distance_back))), pos_mask)
        ctrness_w = torch.mul(torch.div(self.ctr_data_clear(torch.min(distance_left, distance_right)),
                                        self.ctr_data_clear(torch.max(distance_left, distance_right))), pos_mask)
        ctrness_h = torch.mul(torch.div(self.ctr_data_clear(torch.min(distance_top, distance_bottom)),
                                        self.ctr_data_clear(torch.max(distance_top, distance_bottom))), pos_mask)

        ctrness_l = self.ctr_data_clear(ctrness_l)
        ctrness_w = self.ctr_data_clear(ctrness_w)
        ctrness_h = self.ctr_data_clear(ctrness_h)

        ctrness = torch.mul(torch.mul(ctrness_l, ctrness_w), ctrness_h)
        ctrness = self.ctr_data_clear(ctrness)

        ctrness = torch.clamp(ctrness, min=epsilon)
        ctrness = torch.pow(ctrness, 1 / 3.)

        # ctrness_mask = torch.ge(ctrness, 0.8)
        # ctrness[ctrness_mask] = 1.0

        return ctrness

    def cls_loss(self, base_xyz, points_cls_labels, points_box_labels, points_pred_cls):
        """
        classification loss
        Args:
            base_xyz: ([bs, points_num, 3], torch.Tensor). candidate coords
            points_cls_labels: ([bs, points_num], torch.Tensor). gt cls of point
            points_box_labels: ([bs, points_num, 8], torch.Tensor). gt boxes of point
            points_pred_cls: ([bs, points_num, num_class], torch.Tensor). pred class value
        Returns:
            cls_loss: classification loss value
        """
        pos_mask = (points_cls_labels > 0) * 1.0
        neg_mask = (points_cls_labels == 0) * 1.0
        cls_mask = (neg_mask + pos_mask).float()
        norm_param = torch.clamp(torch.sum(cls_mask), min=1.0)

        gt_cls_one_hot = torch.zeros((*list(points_cls_labels.shape), self.num_class + 1),
                                     dtype=points_pred_cls.dtype, device=points_pred_cls.device)
        gt_cls_one_hot = gt_cls_one_hot.scatter(-1, (points_cls_labels * cls_mask).unsqueeze(dim=-1).long(), 1.0)
        gt_cls_one_hot = gt_cls_one_hot[:, :, 1:].float()  # (bs, points_num, num_class)

        points_pred_cls = self.data_clear(points_pred_cls)

        ctrness = self._generate_centerness_label(base_xyz, points_box_labels, pos_mask.float())  # (bs, points_num)
        gt_cls_target = torch.mul(gt_cls_one_hot, ctrness.unsqueeze(-1))

        gt_cls_target = torch.clamp(gt_cls_target, min=1e-6, max=1.0 - 1e-6)
        gt_cls_target = self.data_clear(gt_cls_target)

        criterion_cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        cls_loss = criterion_cls_loss(points_pred_cls, gt_cls_target)
        if self.num_class == 1:
            cls_loss = cls_loss.squeeze(-1)
        else:
            cls_loss = torch.sum(cls_loss, dim=-1)
        cls_loss = torch.sum(torch.mul(cls_loss, cls_mask)) / norm_param

        cls_loss = cls_loss * self.loss_config.CLS_LOSS_WEIGHT

        return cls_loss

    def offset_loss(self, base_xyz, points_cls_labels, points_box_labels, pred_offsets):
        """
        calculate offset loss
        Args:
            base_xyz: ([bs, points_num, 3], torch.Tensor).
            points_cls_labels: ([bs, points_num], torch.Tensor)
            points_box_labels: ([bs, points_num, 8], torch.Tensor)
            pred_offsets: ([bs, points_num, 6], torch.Tensor)
        Returns:

        """
        pos_mask = ((points_cls_labels > 0) * 1.0).float()
        norm_param = torch.clamp(torch.sum(pos_mask), min=1.0)

        gt_xyz_offset = points_box_labels[:, :, :3] - base_xyz
        gt_lwh_reg = points_box_labels[:, :, 3:6] / 2.
        gt_offset = torch.cat((gt_xyz_offset, gt_lwh_reg), dim=-1)

        pred_offsets = self.data_clear(pred_offsets)
        gt_offset = self.data_clear(gt_offset)

        offset_loss = common_utils.huber_loss((pred_offsets - gt_offset), delta=1.0)  # (bs, points_num, 6)
        offset_loss = torch.mul(torch.sum(offset_loss, dim=-1), pos_mask)
        offset_loss = torch.sum(offset_loss) / norm_param
        offset_loss = offset_loss * self.loss_config.REGRESSION_LOSS_WEIGHTING

        return offset_loss

    def angle_loss(self, points_cls_labels, points_box_labels, pred_angle_cls, pred_angle_res):
        """
        calculate angle bin classification loss and angle residual loss
        Args:
            points_cls_labels: ([bs, points_num], torch.Tensor). gt cls of points
            points_box_labels: ([bs, points_num, 8], torch.Tensor). gt box of points
            pred_angle_cls: ([bs, points_num, heading_bin_num], torch.Tensor). pred angle cls
            pred_angle_res: ([bs, points_num, heading_bin_num], torch.Tensor). pred angle res
        Returns:
            bin_loss:
            res_loss:
        """
        pos_mask = ((points_cls_labels > 0) * 1.0).float()
        norm_param = torch.clamp(torch.sum(pos_mask), min=1.0)

        gt_angles = points_box_labels[..., 6]  # (bs, points_num)
        gt_angles = common_utils.limit_period(gt_angles, 0, 2 * np.pi)

        gt_angles_cls, gt_angles_res = common_utils.angle2class_torch(gt_angles, num_heading_bin=self.heading_bin_num)

        # bin classification loss
        pred_angle_cls = self.data_clear(pred_angle_cls)
        gt_angles_cls = self.data_clear(gt_angles_cls)

        criterion_angle_loss = nn.CrossEntropyLoss(reduction='none')
        bin_loss = criterion_angle_loss(pred_angle_cls.transpose(2, 1), gt_angles_cls.long())  # (bs, points_num)
        bin_loss = torch.mul(bin_loss, pos_mask)
        bin_loss = torch.sum(bin_loss) / norm_param
        bin_loss = bin_loss * self.loss_config.REGRESSION_LOSS_WEIGHTING

        # residuals regression loss
        gt_angles_cls_one_hot = torch.zeros_like(pred_angle_cls, device=pred_angle_cls.device)
        gt_angles_cls_one_hot = gt_angles_cls_one_hot.scatter(-1, gt_angles_cls.unsqueeze(-1).long(), 1.0)
        pred_angle_res_one_hot = torch.mul(gt_angles_cls_one_hot, pred_angle_res)
        pred_res = torch.sum(pred_angle_res_one_hot, dim=-1)  # (bs, points_num)

        pred_res = self.data_clear(pred_res)
        gt_angles_res = self.data_clear(gt_angles_res)

        res_loss = common_utils.huber_loss((pred_res - gt_angles_res), delta=1.0)
        res_loss = torch.mul(res_loss, pos_mask)
        res_loss = torch.sum(res_loss) / norm_param
        res_loss = res_loss * self.loss_config.REGRESSION_LOSS_WEIGHTING

        return bin_loss, res_loss

    def corner_loss(self, points_cls_labels, points_box_labels, pred_boxes):
        """
        corners loss of 3D boxes
        Args:
            points_cls_labels: ([bs, points_num], torch.Tensor).
            points_box_labels: ([bs, points_num, 8], torch.Tensor).
            pred_boxes: ([bs, points_num, 7], torch.Tensor)
        Returns:
            corners_loss:
        """
        pos_mask = ((points_cls_labels > 0) * 1.0).float()
        norm_param = torch.clamp(torch.sum(pos_mask), min=1.0)

        gt_corners = box_utils.batch_boxes_to_corners_3d(points_box_labels[:, :, :7])
        pred_corners = box_utils.batch_boxes_to_corners_3d(pred_boxes)

        gt_corners = self.data_clear(gt_corners)
        pred_corners = self.data_clear(pred_corners)

        corners_loss = common_utils.huber_loss((gt_corners - pred_corners), delta=1.0)  # (bs, points_num, 8, 3)
        corners_loss = torch.sum(torch.sum(corners_loss, dim=-1), dim=-1)  # (bs, points_num)
        corners_loss = torch.mul(corners_loss, pos_mask)
        corners_loss = torch.sum(corners_loss) / norm_param
        corners_loss = corners_loss * self.loss_config.CORNER_LOSS_WEIGHT

        return corners_loss

    def iou_loss(self, base_xyz, points_cls_labels, points_box_labels, pred_boxes, pred_iou):
        """
        IoU loss
        Args:
            base_xyz: ([bs, points_num, 3], torch.Tensor). points coords
            points_cls_labels: ([bs, points_num], torch.Tensor)
            points_box_labels: ([bs, points_num, 8], torch.Tensor). [x, y, z, dx, dy, dz, heading, cls]
            pred_boxes: ([bs, points_num, 7], torch.Tensor)
            pred_iou:  ([bs, points_num, 1], torch.Tensor)
        Returns:
            iou_loss: scalar
        """
        pos_mask = ((points_cls_labels > 0) * 1.0).float()
        neg_mask = ((points_cls_labels == 0) * 1.0).float()
        norm_param = torch.clamp(torch.sum(pos_mask), min=1.0)

        pos_weight = torch.div(pos_mask, torch.sum(pos_mask, dim=1, keepdim=True))  # (bs, points_num)

        # gt_iou_one_hot = torch.zeros((*list(points_cls_labels.shape), self.num_class + 1),
        #                              dtype=pred_iou.dtype, device=pred_iou.device)
        # gt_iou_one_hot = gt_iou_one_hot.scatter(-1, (points_cls_labels * pos_mask).unsqueeze(dim=-1).long(), 1.0)
        # gt_iou_one_hot = gt_iou_one_hot[:, :, 1:].float()  # (bs, points_num, num_class)

        pred_iou = torch.sigmoid(pred_iou)  # (bs, points_num, pred_cls_channel)
        pred_iou = pred_iou.squeeze(-1)  # (bs, points_num)
        # pred_iou = self.data_clear(pred_iou)

        # gt IoU value
        bs, points_num = pred_iou.shape[0], pred_iou.shape[1]
        gt_iou = torch.zeros((bs, points_num), dtype=pred_iou.dtype, device=pred_iou.device)  # (bs, points_num)

        for i in range(bs):
            cur_box_labels = points_box_labels[i, :, :7]  # (points_num, 7)
            cur_pred_box = pred_boxes[i, :, :7]

            cur_iou3d = boxes_iou3d_gpu(cur_box_labels, cur_pred_box)  # (points_num, points_num)
            cur_gt_iou = cur_iou3d[torch.arange(points_num), torch.arange(points_num)]  # (points_num)

            gt_iou[i] = cur_gt_iou

        # gt_iou = self.data_clear(gt_iou)

        # gt_iou = points_box_labels[:, :, 7]  # (bs, points_num)
        gt_iou = gt_iou.detach()
        # gt_iou = torch.clamp(2. * gt_iou - 1., min=0., max=1.)

        # smooth l1 loss
        # iou_loss = common_utils.huber_loss((pred_iou - gt_iou), delta=1.)  # (bs, points_num)
        # iou_loss = F.binary_cross_entropy(pred_iou, gt_iou, reduction='none')  # (bs, points_num)
        # iou_loss = torch.mul(iou_loss, pos_mask)

        iou_func = loss_utils.WeightedSmoothL1Loss()
        iou_loss = iou_func(pred_iou, gt_iou, pos_weight)

        iou_loss = torch.sum(iou_loss) / norm_param
        # iou_loss = torch.sum(iou_loss) / pos_mask.shape[0]
        iou_loss = iou_loss * self.loss_config.IOU_LOSS_WEIGHT

        return iou_loss

    def get_loss(self, tb_dict=None):
        """
        calculate regression head loss
        Args:

        Returns:
            cls_loss: point classification loss
            reg_loss: loss of regression:
                offset_loss + angle_loss + corner_loss
        """
        pred_cls = self.forward_ret_dict['pred_cls']
        pred_offsets = self.forward_ret_dict['pred_offsets']
        base_xyz = self.forward_ret_dict['candidate_xyz']
        pred_angle_cls = self.forward_ret_dict['pred_angle_cls']
        pred_angle_res = self.forward_ret_dict['pred_angle_res']

        gt_boxes = self.forward_ret_dict['gt_boxes']

        # ------- assign targets -------
        points_cls_labels, points_box_labels = self.assign_targets(base_xyz, gt_boxes)

        # ------- cls loss -------
        cls_loss = self.cls_loss(base_xyz, points_cls_labels, points_box_labels, pred_cls)

        # ------- reg loss -------
        # offset loss
        offset_loss = self.offset_loss(base_xyz, points_cls_labels, points_box_labels, pred_offsets)
        # angle loss
        bin_loss, angle_reg_loss = self.angle_loss(points_cls_labels, points_box_labels, pred_angle_cls, pred_angle_res)
        # corner loss
        gt_angle = points_box_labels[:, :, 6].clone()
        gt_angle_cls, gt_angle_res = common_utils.angle2class_torch(gt_angle, self.heading_bin_num)
        gt_angle_cls_one_hot = torch.zeros_like(pred_angle_cls, device=pred_angle_cls.device)
        gt_angle_cls_one_hot = gt_angle_cls_one_hot.scatter(-1, gt_angle_cls.unsqueeze(-1).long(), 1.0).float()
        fake_pred_boxes = self.decode_prediction(base_xyz, pred_offsets, gt_angle_cls_one_hot, pred_angle_res)

        corner_loss = self.corner_loss(points_cls_labels, points_box_labels, fake_pred_boxes)

        reg_loss = offset_loss + bin_loss + angle_reg_loss + corner_loss

        tb_dict = {} if tb_dict is None else tb_dict
        tb_dict['cls_loss'] = '%.3f' % cls_loss.item()
        tb_dict['reg_loss'] = '%.3f' % reg_loss.item()

        return cls_loss, reg_loss, tb_dict

    def forward(self, output_dict, data_dict):
        """
        Head Regression forward
        Args:
            output_dict:
            data_dict:
        Returns:
            output_dict:
                ...
                pred_cls: ([bs, points_num, cls_channel), torch.Tensor]. classes prediction value
                pred_offsets: ([bs, points_num, 6], torch.Tensor).
                              offset prediction value. [offset(x, y, z), half_(dx, dy, dz)]
                pred_angle_cls: ([bs, points_num, HEADING_BIN_NUM], torch.Tensor). heading bin
                pred_angle_res: ([bs, points_num, HEADING_BIN_NUM], torch.Tensor). heading residuals
        """
        self.update_global_step()

        input_xyz = output_dict['candidate_xyz']
        input_feature = output_dict['candidate_feature']

        features = self.features_learning(input_feature)
        output_dict['head_features'] = features

        # ------- classification --------
        cls_feature = F.relu(self.cls_bn1(self.cls_conv1(features)))
        cls_feature = self.cls_conv2(cls_feature)
        cls_feature = cls_feature.transpose(2, 1).contiguous()  # (B, npoint, pred_cls_channel)

        # --- bounding box prediction ---
        reg_feature = F.relu(self.reg_bn1(self.reg_conv1(features)))
        reg_feature = self.reg_conv2(reg_feature)
        reg_feature = reg_feature.transpose(2, 1).contiguous()  # (B, npoint, C')

        pred_offset = reg_feature[:, :, 0:self.pred_reg_channel_num]

        begin_pos = self.pred_reg_channel_num
        end_pos = self.pred_reg_channel_num + self.heading_bin_num
        pred_angle_cls = reg_feature[:, :, begin_pos:end_pos]

        begin_pos = end_pos
        pred_angle_res = reg_feature[:, :, begin_pos:]

        output_dict['pred_cls'] = cls_feature
        output_dict['pred_offsets'] = pred_offset
        output_dict['pred_angle_cls'] = pred_angle_cls
        output_dict['pred_angle_res'] = pred_angle_res

        if self.training:
            self.forward_ret_dict['pred_cls'] = cls_feature
            self.forward_ret_dict['pred_offsets'] = pred_offset
            self.forward_ret_dict['pred_angle_cls'] = pred_angle_cls
            self.forward_ret_dict['pred_angle_res'] = pred_angle_res

            self.forward_ret_dict['candidate_xyz'] = input_xyz

            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']

        return output_dict
