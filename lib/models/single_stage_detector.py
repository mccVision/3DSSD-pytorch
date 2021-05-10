import torch
import numpy as np
import torch.nn as nn

from lib.models.single_stage_backbone import SingleStageBackbone
from lib.models.candidate_generation import CandidateGeneration
from lib.models.regression_head import RegressionHead
from lib.ops.iou3d_nms import iou3d_nms_utils


class SingleStageDetector(nn.Module):
    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.Model

        self.global_step = 0

        # backbone
        self.backbone_net = SingleStageBackbone(self.model_cfg)

        # candidate generation
        self.cg_net = CandidateGeneration(self.model_cfg)

        # regression head
        self.reg_head = RegressionHead(self.model_cfg, self.cfg.DATASET.class_names)

    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def post_processing(self, pred_boxes, pred_cls, base_xyz, gt_boxes = None):
        """
        post precessing process
        Args:
            pred_boxes: ([bs, points_num, 7], torch.Tensor). 3D boxes of prediction
            pred_cls: ([bs, points_num, cls_num], torch.Tensor).
            gt_boxes: ([bs, npoint, 8], torch.Tensor). 3D boxes of ground truth
            base_xyz: ([bs, points_num, 3], torch.Tensor). candidate xyz
        Returns:
            pred_dicts:
                pred_cls:
                pred_boxes:
                pred_scores:
        """
        pred_dicts = []

        bs, cls_num = pred_cls.shape[0], pred_cls.shape[2]
        for i in range(bs):
            cur_base_xyz = base_xyz[i]  # (points_num, 3)
            cur_pred_cls = pred_cls[i]  # (points_num, cls_num)
            cur_pred_boxes = pred_boxes[i]  # (points_num, 7)

            if cls_num == 1:
                cur_pred_cls = cur_pred_cls.squeeze(-1).contiguous()
                cur_cls_scores = torch.sigmoid(cur_pred_cls.view(-1))  # (points_num)
                cur_cls_type = cur_cls_scores.new_ones(*list(cur_cls_scores.shape))
            else:
                cur_cls_type = torch.argmax(cur_pred_cls, dim=-1).view(-1)
                cur_cls_scores = torch.sigmoid(cur_pred_cls)  # (points_num, cls_num)
                cur_cls_scores = cur_cls_scores[:, cur_cls_type].view(-1)  # (points_num)
                cur_cls_type = cur_cls_type + 1

            keep_idx = cur_cls_scores >= self.model_cfg.post_process.CLS_THRESH
            cur_cls_type = cur_cls_type[keep_idx]
            cur_cls_scores = cur_cls_scores[keep_idx]
            cur_pred_boxes = cur_pred_boxes[keep_idx, :]

            if cur_cls_scores.numel() == 0:
                record_dict = {
                    'pred_labels': cur_cls_type,
                    'pred_boxes': cur_pred_boxes,
                    'pred_scores': cur_cls_scores,
                    'candidate_xyz': cur_base_xyz,
                }
                pred_dicts.append(record_dict)
                continue

            # # replace cls scores with real IoU value
            # cur_gt_boxes = gt_boxes[i]
            # k = cur_gt_boxes.shape[0] - 1
            # while k >= 0 and cur_gt_boxes[k].sum() == 0:
            #     k -= 1
            # # calculate real IoU value
            # if k >= 0:
            #     cur_gt_boxes = cur_gt_boxes[:k+1]
            #
            #     iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_pred_boxes, cur_gt_boxes[:, :7])  # (N, M)
            #     iou_scores = torch.max(iou3d, dim=-1)[0]  # (points_num)
            #     # print(iou_scores.shape)
            #     cur_cls_scores = iou_scores

            # 3D IoU NMS
            keep_idx, selected_scores = iou3d_nms_utils.nms_normal_gpu(cur_pred_boxes,
                                                                       cur_cls_scores,
                                                                       self.model_cfg.post_process.NMS_THRESH)

            cur_cls_type = cur_cls_type[keep_idx]
            cur_cls_scores = cur_cls_scores[keep_idx]
            cur_pred_boxes = cur_pred_boxes[keep_idx, :]

            record_dict = {
                'pred_labels': cur_cls_type,
                'pred_boxes': cur_pred_boxes,
                'pred_scores': cur_cls_scores,
                'candidate_xyz': cur_base_xyz,
            }
            pred_dicts.append(record_dict)

        return pred_dicts

    def get_prediction(self, output_dict, data_dict):
        """
        generate prediction results
        Args:
            output_dict:
            data_dict:
        Returns:
            pred_dicts:
                record_dict:
                    pred_cls:
                    pred_boxes:
                    pred_scores:
        """
        base_xyz = output_dict['candidate_xyz']
        pred_offset = output_dict['pred_offsets']
        pred_angle_cls = output_dict['pred_angle_cls']  # (bs, points_num, heading_bin_num)
        pred_angle_res = output_dict['pred_angle_res']

        # ------- decode 3D boxes info -------
        angle_cls_one_hot = pred_angle_cls.new_zeros(*list(pred_angle_cls.shape))
        angle_cls_assign = torch.argmax(pred_angle_cls, dim=-1).unsqueeze(-1)
        angle_cls_one_hot.scatter_(-1, angle_cls_assign.long(), 1.0)

        pred_boxes = self.reg_head.decode_prediction(base_xyz, pred_offset,
                                                     angle_cls_one_hot, pred_angle_res)  # (bs, points_num, 7)

        # ------- post precessing -------
        pred_cls = output_dict['pred_cls']

        if self.model_cfg.head.IoU_HEAD:
            pred_iou = output_dict['pred_iou']
            pred_dicts = self.post_processing_with_iou(pred_boxes, pred_cls, pred_iou, base_xyz)
        else:
            # gt_boxes = data_dict['gt_boxes']
            pred_dicts = self.post_processing(pred_boxes, pred_cls, base_xyz)

        return pred_dicts

    def get_trianing_loss(self):
        tb_dict = {}

        vote_loss, tb_dict = self.cg_net.get_loss(tb_dict)

        cls_loss, reg_loss, tb_dict = self.reg_head.get_loss(tb_dict)

        total_loss = vote_loss + cls_loss + reg_loss

        tb_dict['loss'] = '%.3f' % total_loss.item()

        return total_loss, tb_dict

    def forward(self, data_dict):
        """
        Single Stage Detector forward pass
        Args:
            data_dict:
        Returns:
            training:
                loss, tb_dict
            testing:
                pred_dicts
        """
        output_dict = {}

        output_dict = self.backbone_net(data_dict, output_dict)

        output_dict = self.cg_net(output_dict, data_dict)

        output_dict = self.reg_head(output_dict, data_dict)

        if self.training:
            loss, tb_dict = self.get_trianing_loss()

            return loss, tb_dict
        else:
            pred_dicts = self.get_prediction(output_dict, data_dict)

            return pred_dicts
