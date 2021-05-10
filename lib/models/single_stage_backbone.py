import torch
import torch.nn as nn

from lib.ops.pointnet2.pointnet2_modules import PointnetSAModuleMSGSSD, PointnetFPModule


class SingleStageBackbone(nn.Module):
    def __init__(self, cfg):
        super(SingleStageBackbone, self).__init__()

        bb_cfg = cfg.backbone
        self.bb_cfg = bb_cfg
        self.architecture_cfg = bb_cfg.Architecture

        # -------- Set Abstraction Module --------
        self.SA_modules = nn.ModuleList()

        input_feature_dim = bb_cfg.INPUT_FEATURE_DIM
        input_channel_list = [input_feature_dim]

        for k in range(len(self.architecture_cfg)):
            # input_channel = input_channel_list[k]

            self.SA_modules.append(
                PointnetSAModuleMSGSSD(
                    layer_idx=k,
                    layer_cfg=self.architecture_cfg,
                    input_channel_list=input_channel_list,
                )
            )

            aggregation_channel = self.architecture_cfg[k][10]
            input_channel_list.append(aggregation_channel)

    @staticmethod
    def _break_up_pc(pc):
        """
        slice the point cloud to coords and features
        Args:
            pc: ([B, N, 3+input_feature_dim], torch.Tensor). Point in the point-cloud, format as (x, y, z, feature)
        Returns:
            xyz: ([B, N, 3], torch.Tensor). point coords
            features: ([B, input_feature_dim, N], torch.Tensor).
        """
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(2, 1).contiguous() if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, data_dict, output_dict):
        """
        forward pass of the Backbone net
        Args:
            data_dict:
                points: ([B, N, 3+input_feature_dim], torch.Tensor).
            output_dict: (dict). store results
        Returns:

        """
        if not output_dict:
            output_dict = {}

        points = data_dict['points']
        xyz, features = self._break_up_pc(points)

        xyz_list, features_list = [xyz], [features]
        idx_list = [None]
        for i in range(len(self.SA_modules)):
            xyz_list, features_list, idx_list = self.SA_modules[i](xyz_list, features_list, idx_list)

        output_dict['backbone_xyz_list'] = xyz_list
        output_dict['backbone_feature_list'] = features_list
        output_dict['backbone_idx_list'] = idx_list

        return output_dict
