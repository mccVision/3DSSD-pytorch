import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import tqdm
import mmcv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from mmcv import Config
from datetime import datetime
from mmcv.runner import obj_from_dict
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from lib.utils import common_utils
from lib.datasets import build_dataloader
from lib.datasets.kitti import kitti_dataset
from lib.models import build_network, load_data_to_gpu
from tools.train_utils.optimization import build_optimizer, build_scheduler
from lib.ops.pointnet2.pytorch_utils import BNMomentumScheduler


def parse_args():
    parser = argparse.ArgumentParser(description='single stage detector')
    parser.add_argument('--cfg', default='../configs/kitti/car_cfg.py',
                        help='config file path')
    parser.add_argument('--log_dir', default=None,
                        help='the dir to save logs and models [default: None]')
    parser.add_argument('--batch_size', default=None,
                        help='batch size for training')
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path [default: None]')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument('--tcp_port', type=int, default=18888,
                        help='tcp port for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='whether to use sync bn')

    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.max_iteration = cfg.train.CONFIG.MAX_ITERATION
        self.checkpoint_interval = cfg.train.CONFIG.CHECKPOINT_INTERVAL

        # ------- distributed part -------
        if args.launcher == 'none':
            self.dist_train = False
        else:
            self.cfg.train.CONFIG.BATCH_SIZE, self.args.local_rank = getattr(common_utils, 'init_dist_%s' % args.launcher)(
                self.cfg.train.CONFIG.BATCH_SIZE, self.args.tcp_port, self.args.local_rank, backend='nccl'
            )
            self.dist_train = True

        common_utils.set_random_seed(0)

        # ------- output dir and logger -------
        if self.cfg.train.log_dir is None:
            datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.log_dir = os.path.join(self.cfg.train.work_dir, datetime_str)
            os.makedirs(self.log_dir, exist_ok=True)
            self.logger = open(os.path.join(self.log_dir, 'log_train.txt'), 'w')
            self.ckpt_dir = os.path.join(self.log_dir, 'ckpt')
            os.makedirs(self.ckpt_dir, exist_ok=True)
        else:
            self.log_dir = self.cfg.train.log_dir
            self.logger = open(os.path.join(self.log_dir, 'log_train.txt'), 'a+')
            self.ckpt_dir = os.path.join(self.log_dir, 'ckpt')

        backup_dir = os.path.join(self.log_dir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        os.system("cp -r ../tools %s/" % backup_dir)
        os.system("cp -r ../lib %s/" % backup_dir)

        # ------- logger info -------
        self.logger.write(str(args) + '\n')
        self._log_string('------- Saving models to the path {} -------'.format(self.log_dir))
        self._log_string('------- Saving configure file in {} -------'.format(self.log_dir))
        os.system('cp \"%s\" \"%s\"' % (args.cfg, self.log_dir))

        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        self._log_string('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if self.dist_train:
            total_gpus = dist.get_world_size()
            self._log_string('total_batch_size: %d' % (total_gpus * self.cfg.train.CONFIG.BATCH_SIZE))

        # ------- tensorboard -------
        self.tb_log = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

        # ------- dataset -------
        # self.cfg.DATASET.logger = self.logger
        self.train_dataset, self.train_dataloader, self.train_sampler = build_dataloader(
            dataset_cfg=self.cfg.DATASET,
            batch_size=self.cfg.train.CONFIG.BATCH_SIZE,
            dist=self.dist_train,
            workers=self.cfg.train.CONFIG.NUM_THREADS,
            training=True)

        # ------- model -------
        self.model = build_network(cfg)
        if args.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        # print(self.model)

        self.optimizer = build_optimizer(self.model, self.cfg.train.OPTIMIZATION)

        self.start_epoch = self.it = 0
        self.last_epoch = -1
        self._initialize_model()

        self.lr_scheduler, self.lr_warmup_scheduler = build_scheduler(
            self.optimizer,
            total_iters_each_epoch=len(self.train_dataloader),
            total_epochs=self.max_iteration,
            last_epoch=self.last_epoch,
            optim_cfg=self.cfg.train.OPTIMIZATION
        )

    def _initialize_model(self):
        if self.cfg.train.checkpoint_path is not None and os.path.isfile(self.cfg.train.checkpoint_path):
            checkpoint = torch.load(self.cfg.train.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.last_epoch = self.start_epoch + 1
            self.it = checkpoint.get('it', 0.0)
            self._log_string(
                "---- Loaded checkpoint {} (epoch: {})".format(self.cfg.train.checkpoint_path, self.start_epoch))

    def evaluate_mAP(self, trained_epoch):
        eval_dataset, eval_dataloader, eval_sampler = build_dataloader(
            dataset_cfg=self.cfg.DATASET,
            batch_size=self.cfg.eval.CONFIG.BATCH_SIZE,
            dist=False,
            workers=self.cfg.eval.CONFIG.NUM_THREADS,
            training=False
        )

        output_dir = self.cfg.eval.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()

        pred_annos = []
        prog_bar = mmcv.ProgressBar(len(eval_dataloader))
        for batch_idx, data_dict in enumerate(eval_dataloader):
            load_data_to_gpu(data_dict)

            with torch.no_grad():
                pred_dicts = self.model(data_dict)

            det_anno = eval_dataset.generate_prediction_dicts(
                data_dict, pred_dicts,
                class_names=self.cfg.DATASET.class_names,
                output_path=output_dir
            )

            pred_annos += det_anno

            prog_bar.update()

        result_str, result_dict = eval_dataset.evaluation(pred_annos, self.cfg.DATASET.class_names)

        epoch_info = "epoch %d" % trained_epoch
        self._log_string(epoch_info)
        self._log_string(result_str)

    def train_one_epoch(self, lr_scheduler, tbar):
        total_it_each_epoch = len(self.train_dataloader)

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            # lr_scheduler.step(self.it)

            try:
                cur_lr = float(self.optimizer.lr)
            except:
                cur_lr = self.optimizer.param_groups[0]['lr']

            self.model.train()
            self.optimizer.zero_grad()
            load_data_to_gpu(batch_data)

            # with torch.autograd.set_detect_anomaly(True):
            #     loss, tb_dict = self.model(batch_data)
            #     loss.backward()
            loss, tb_dict = self.model(batch_data)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.cfg.train.OPTIMIZATION.GRAD_NORM_CLIP)
            self.optimizer.step()
            lr_scheduler.step(self.it)

            self.it += 1

            if (batch_idx % 5) == 0 or (batch_idx + 1) == total_it_each_epoch:
                # tensorboard logger
                self.tb_log.add_scalar('learning_rate', cur_lr, self.it)
                for key, val in tb_dict.items():
                    self.tb_log.add_scalar('train_' + key, float(val), self.it)

                tb_dict['batch_idx'] = '{}/{}'.format(batch_idx, total_it_each_epoch)

                tbar.set_postfix(tb_dict)
                tbar.refresh()

    def train_model(self):
        self.model.train()
        if self.dist_train:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank % torch.cuda.device_count()]
            )

        with tqdm.trange(self.start_epoch, self.max_iteration, desc='epochs') as tbar:
            for cur_epoch in tbar:

                self._log_string_no('------------------ Epoch {} -----------------'.format(cur_epoch + 1))
                self._log_string_no('Current time {}'.format(datetime.now()))

                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(cur_epoch)

                if self.lr_warmup_scheduler is not None and cur_epoch < self.cfg.train.OPTIMIZATION.WARMUP_EPOCH:
                    cur_scheduler = self.lr_warmup_scheduler
                else:
                    cur_scheduler = self.lr_scheduler

                self.train_one_epoch(cur_scheduler, tbar)

                # save trained model and evaluation
                trained_epoch = cur_epoch + 1
                if trained_epoch % self.checkpoint_interval == 0 or trained_epoch == self.max_iteration:
                    if trained_epoch >= self.cfg.train.CONFIG.EPOCH_RECORD_POINT:
                        # save model
                        save_dict = {'epoch': trained_epoch,
                                     'optimizer_state_dict': self.optimizer.state_dict(),
                                     'it': self.it,
                                     }

                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            model_state = self.model_state_to_cpu()
                        else:
                            model_state = self.model.state_dict()
                        save_dict['model_state_dict'] = model_state
                        torch.save(save_dict, os.path.join(self.ckpt_dir, 'checkpoint_%d.pth' % trained_epoch))

                    # evaluation
                    if not self.dist_train:
                        self.evaluate_mAP(trained_epoch)

        self._log_string('----------------- End Training ------------------')

    def model_state_to_cpu(self):
        model_state = self.model.module.state_dict()
        model_state_cpu = type(model_state)()
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    def _log_string(self, out_str):
        self.logger.write(out_str)
        self.logger.write('\n')
        self.logger.flush()
        print(out_str)

    def _log_string_no(self, out_str):
        self.logger.write(out_str + '\n')
        self.logger.flush()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)

    if args.log_dir is not None:
        cfg.train.log_dir = args.log_dir
    if args.checkpoint_path is not None:
        cfg.train.checkpoint_path = args.checkpoint_path
    if args.batch_size is not None:
        cfg.train.CONFIG.BATCH_SIZE = args.batch_size

    trainer = Trainer(cfg, args)
    trainer.train_model()


if __name__ == '__main__':
    main()
