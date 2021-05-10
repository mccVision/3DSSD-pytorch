import torch
from mmcv.runner import obj_from_dict
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from . import dataset
from .kitti import kitti_dataset
from lib.utils import common_utils

__all__ = {
    'DatasetTemplate': dataset,
    'KittiDataset': kitti_dataset,
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4, training=True,
                     merge_all_iter_to_one_epoch=False, total_epochs=0):
    if training:
        dset = obj_from_dict(dataset_cfg, __all__[dataset_cfg.type])
    else:
        dataset_cfg.training = False
        dset = obj_from_dict(dataset_cfg, __all__[dataset_cfg.type])

    if merge_all_iter_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist and training:
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
    else:
        sampler = None

    dataloader = DataLoader(
        dset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dset, dataloader, sampler


