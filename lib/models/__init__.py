import torch
import numpy as np
from collections import namedtuple

from .single_stage_detector import SingleStageDetector

__all__ = {
    'SingleStageDetector': SingleStageDetector,
}


def build_network(cfg):
    model = __all__[cfg.Model.type](cfg)

    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metedata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()
