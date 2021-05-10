import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
from mmcv import Config
from datetime import datetime
from tensorboardX import SummaryWriter

from lib.utils import common_utils
from lib.datasets import build_dataloader
from lib.models import build_network, load_data_to_gpu
from lib.ops.roiaware_pool3d import roiaware_pool3d_utils
from tools.train_utils.optimization import build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='single stage detector')
    parser.add_argument('--cfg', default='../configs/kitti/car_cfg.py',
                        help='config file path. (../configs/kitti/car_cfg_FFPS.py)')
    parser.add_argument('--to_cpu', default=False,
                        help='training process whether dist')
    parser.add_argument('--log_dir', default=None,
                        help='the dir to save logs and models [default: None]')
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path [default: None]')

    args = parser.parse_args()
    return args


class Evaluator:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.to_cpu = args.to_cpu

        # ------- logger -------
        datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.cfg.eval.work_dir, datetime_str)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = open(os.path.join(self.log_dir, 'log_test.txt'), 'w')

        self.logger.write(str(args) + '\n')
        self._log_string('--------------------------- EVALUATION ----------------------------')
        self._log_string('------- Saving configure file in {} ----------'.format(self.log_dir))
        os.system('cp \"%s\" \"%s\"' % (args.cfg, self.log_dir))

        # ------- dataset -------
        self.eval_dataset, self.eval_dataloader, self.eval_sample = build_dataloader(
            dataset_cfg=self.cfg.DATASET,
            batch_size=self.cfg.eval.CONFIG.BATCH_SIZE,
            dist=False,
            workers=self.cfg.eval.CONFIG.NUM_THREADS,
            training=False
        )

        # ------- model --------
        self.model = build_network(cfg)
        # self.model = self.model.to(device)

        # ------- checkpoint initialize ----------
        checkpoint_path = cfg.eval.checkpoint_path
        if checkpoint_path.endswith('.pth'):
            self.eval_all = False
            self._initialize_model(checkpoint_path)
            self.model = self.model.to(device)
            # self.model_fp16 = torch.quantization.quantize_dynamic(
            #     self.model,
            #     {nn.Conv1d, nn.BatchNorm1d, nn.Conv2d, nn.Conv1d},
            #     dtype=torch.float16
            # )
        else:
            self.eval_all = True
            checkpoint_list = []
            filelist = os.listdir(checkpoint_path)
            for file in filelist:
                file = os.path.join(checkpoint_path, file)
                if os.path.isfile(file):
                    if file.endswith('.pth'):
                        checkpoint_list.append(file)
            self.checkpoint_list = checkpoint_list
        self.output_dir = cfg.eval.output_dir

    def _initialize_model(self, checkpoint_path):
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            loc_type = torch.device('cpu') if self.to_cpu else None
            checkpoint = torch.load(checkpoint_path, map_location=loc_type)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            # self.last_epoch = self.start_epoch + 1
            # self.it = checkpoint.get('it', 0.0)
            self._log_string(
                "---- Loaded checkpoint {} (epoch: {})".format(checkpoint_path, self.start_epoch))

    def evaluation_statistical(self, output_dir, pred_dicts, data_dict):
        output_mask_dir = os.path.join(output_dir, 'mask')
        output_cg_dir = os.path.join(output_dir, 'candidate_points')
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_cg_dir, exist_ok=True)

        point_fg_ratio = 0.
        gt_boxes = data_dict['gt_boxes']

        bs = gt_boxes.shape[0]
        for i in range(bs):
            tmp_cg_xyz = pred_dicts[i]['candidate_xyz']  # (points_num, 3)
            tmp_gt_boxes = gt_boxes[i][:, :7]  # (gt_num, 8)
            box_of_point_mask = roiaware_pool3d_utils.points_in_boxes_gpu(
                tmp_cg_xyz.unsqueeze(0).contiguous(),
                tmp_gt_boxes.unsqueeze(0).contiguous(),
            )  # (1, points_num)
            point_fg_mask = ((box_of_point_mask.squeeze(0) >= 0) * 1.0).float()  # (points_num)

            tmp_point_mask = point_fg_mask.detach().cpu().numpy()
            frame_id = data_dict['frame_id'][i]

            point_fg_ratio += (torch.sum(point_fg_mask) / torch.numel(point_fg_mask)).item()
            tmp_cg_path = os.path.join(output_cg_dir, '%s.txt' % frame_id)
            tmp_mask_path = os.path.join(output_mask_dir, '%s.txt' % frame_id)

            tmp_cg_xyz = tmp_cg_xyz.detach().cpu().numpy()
            np.savetxt(tmp_cg_path, tmp_cg_xyz, fmt='%.6f')
            np.savetxt(tmp_mask_path, tmp_point_mask, fmt='%f')

        return point_fg_ratio

    def evaluate_mAP(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_res_dir = os.path.join(output_dir, 'results')
        os.makedirs(output_res_dir, exist_ok=True)

        self.model = self.model.cuda()
        self.model = self.model.eval()
        self.model.eval()
        fg_ratio = 0.

        pred_annos = []
        pred_cls_scores = []
        pred_iou_scores = []
        prog_bar = tqdm.tqdm(total=len(self.eval_dataloader), desc='eval')
        for batch_idx, data_dict in enumerate(self.eval_dataloader):
            load_data_to_gpu(data_dict)

            with torch.no_grad():
                pred_dicts = self.model(data_dict)

            det_anno = self.eval_dataset.generate_prediction_dicts(
                data_dict,
                pred_dicts,
                class_names=self.cfg.DATASET.class_names,
                output_path=output_res_dir
            )

            pred_annos += det_anno
            # pred_cls_scores += pred_dicts['pred_scores']
            # pred_iou_scores += pred_dicts['pred_iou']

            tmp_ratio = self.evaluation_statistical(output_dir, pred_dicts, data_dict)
            fg_ratio += tmp_ratio

            prog_bar.update()

        prog_bar.close()

        fg_ratio = fg_ratio / (float(self.eval_dataset.__len__()))
        result_str, result_dict = self.eval_dataset.evaluation(pred_annos, self.cfg.DATASET.class_names)

        self._log_string(result_str)
        self._log_string('positive points ratio: %.6f' % fg_ratio)

    def evaluate(self):
        if self.eval_all:
            for checkpoint_path in self.checkpoint_list:
                self._initialize_model(checkpoint_path)
                self.model = self.model.to(device)
                base_name = os.path.basename(checkpoint_path)
                file_name = base_name.split('.')[0]
                train_dir = os.path.dirname(os.path.dirname(checkpoint_path))
                tmp_output_dir = os.path.join(train_dir, file_name)
                # self.output_dir = train_dir
                # tmp_output_dir = os.path.join(self.output_dir, file_name)
                self.evaluate_mAP(tmp_output_dir)
        else:
            checkpoint_path = self.cfg.eval.checkpoint_path
            base_name = os.path.basename(checkpoint_path)
            file_name = base_name.split('.')[0]
            train_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            tmp_output_dir = os.path.join(train_dir, file_name)
            # tmp_output_dir = os.path.join(self.output_dir, file_name)
            self.evaluate_mAP(tmp_output_dir)

    def _log_string(self, out_str):
        self.logger.write(out_str + '\n')
        self.logger.flush()
        print(out_str)

    def _log_string_no(self, out_str):
        self.logger.write(out_str + '\n')
        self.logger.flush()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)

    if args.log_dir is not None:
        cfg.eval.checkpoint_path = args.log_dir
    if args.checkpoint_path is not None:
        cfg.eval.checkpoint_path = args.checkpoint_path

    evaluator = Evaluator(cfg, args)
    evaluator.evaluate()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
