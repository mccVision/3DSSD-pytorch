# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Pointnet2 layers.
Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
Extended with the following:
1. Uniform sampling in each local region (sample_uniformly)
2. Return sampled points indices to support votenet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import lib.ops.pointnet2.pointnet2_utils as pointnet2_utils
import lib.ops.pointnet2.pytorch_utils as pt_utils
import lib.utils.model_utils as model_utils
import lib.utils.loss_utils as loss_utils
from typing import List
from lib.ops.roiaware_pool3d import roiaware_pool3d_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), inds


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            mlps: List[List[int]],
            bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz
        )


class PointnetSAModuleVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,  # for RBF pooling
            normalize_xyz: bool = False,  # normalize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                                                         use_xyz=use_xyz, ret_grouped_xyz=True,
                                                         normalize_xyz=normalize_xyz,
                                                         sample_uniformly=sample_uniformly,
                                                         ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert (inds.shape[1] == self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(
            grouped_features
        )  # (B, mlp[-1], npoint, nsample)
        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(
                -1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma ** 2) / 2)  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(
                self.nsample)  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


class PointnetSAModuleMSGSSD(nn.Module):
    def __init__(self, layer_idx, layer_cfg, input_channel_list):
        super().__init__()

        self.layer_idx = layer_idx
        self.layer_architecture = layer_cfg[self.layer_idx]

        self.xyz_index = self.layer_architecture[0]
        self.feature_index = self.layer_architecture[1]
        self.radius_list = list(self.layer_architecture[2])
        self.nsample_list = list(self.layer_architecture[3])
        self.mlp_list = list(self.layer_architecture[4])
        self.bn = self.layer_architecture[5]

        self.fps_method = self.layer_architecture[6]
        self.npoints = self.layer_architecture[7]

        self.scope = self.layer_architecture[8]
        self.cell_type = self.layer_architecture[9]
        self.aggregation_channel = self.layer_architecture[10]
        self.use_xyz = self.layer_architecture[11]

        self.first_layer = (layer_idx == 0)

        input_channel = input_channel_list[self.feature_index]

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.nl_mlps = nn.ModuleList()
        self.rs_mlps = nn.ModuleList()
        output_channel_sum = 0
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]

            mlp_spec = self.mlp_list[i]
            mlp_spec = [input_channel] + mlp_spec
            output_channel_sum += mlp_spec[-1]
            if self.use_xyz:
                mlp_spec[0] += 3

            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=self.use_xyz)
                if self.npoints is not None else pointnet2_utils.GroupAll(self.use_xyz)
            )

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=self.bn))

        self.output_channel_sum = output_channel_sum
        self.conv1 = nn.Conv1d(self.output_channel_sum, self.aggregation_channel, 1)
        self.bn1 = nn.BatchNorm1d(self.aggregation_channel)

    def forward(self, xyz_list, feature_list, fps_idx_list):
        """
        Pointnet set abstraction layer with multiscale grouping for 3DSSD Net
        Args:
            xyz_list: ([B, N', 3], list(torch.Tensor)). the xyz coords of the features
            feature_list: ([B, C, N'], list(torch.Tensor)). the descriptors of the features
            fps_idx_list: ([B, N'],list(torch.Tensor)). the indexs of the features
        Returns:
            xyz_list:
            feature_list:
            fps_idx_list:
        """
        new_features_list = []

        xyz = xyz_list[self.xyz_index]  # (B, N, 3)
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # (B, 3, N)
        feature = feature_list[self.feature_index]  # (B, C, N)

        if self.fps_method == 'F-FPS':
            feature_flipped = feature.transpose(2, 1).contiguous()  # (B, N, C)
            xyz_feature = torch.cat((xyz, feature_flipped), dim=-1)

            xyz_feature_dist = model_utils.calcu_square_dist(xyz_feature, xyz_feature, norm=False)  # (B, N, N)
            inds = pointnet2_utils.furthest_point_sample_with_dist(xyz_feature_dist, self.npoints)
        elif self.fps_method == 'D-FPS':
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoints)
        else:
            raise Exception('No implementation method!')

        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(2, 1).contiguous() if self.npoints is not None else None  # (B, npoint, 3)
        # new_xyz_feature = pointnet2_utils.gather_operation(
        #     feature.contiguous(), inds
        # ).contiguous() if self.npoints is not None else None  # (B, C, npoint)

        for i in range(len(self.groupers)):
            group_feature = self.groupers[i](
                xyz, new_xyz, feature
            )   # (B, C(+3), npoint, nsample)

            new_features = self.mlps[i](
                group_feature
            )   # (B, mlp[-1], npoint, nsample)

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        # Conv1d for feature aggregation
        new_feature = torch.cat(new_features_list, dim=1)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))  # (B, aggregation_channel, npoint)

        xyz_list.append(new_xyz)
        feature_list.append(new_feature)
        fps_idx_list.append(inds)

        return xyz_list, feature_list, fps_idx_list


class PointnetSAModuleMSGCG(nn.Module):
    def __init__(self, layer_cfg, input_channel):
        super().__init__()

        self.xyz_index_list = layer_cfg[0]
        self.feature_index_list = layer_cfg[1]
        self.radius_list = layer_cfg[2]
        self.nsample_list = layer_cfg[3]
        self.mlp_list = layer_cfg[4]
        self.bn = layer_cfg[5]

        self.fps_method = layer_cfg[6]
        self.cell_type = layer_cfg[7]
        self.aggregation_channel = layer_cfg[8]
        self.use_xyz = layer_cfg[9]

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.rs_mlps = nn.ModuleList()
        self.nl_mlps = nn.ModuleList()
        output_channel_sum = 0
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]

            mlp_spec = self.mlp_list[i]
            mlp_spec = [input_channel] + mlp_spec
            output_channel_sum += mlp_spec[-1]
            if self.use_xyz:
                mlp_spec[0] += 3

            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=self.use_xyz)
            )

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=self.bn))

        self.output_channel = output_channel_sum
        self.conv1 = nn.Conv1d(self.output_channel, self.aggregation_channel, 1)
        self.bn1 = nn.BatchNorm1d(self.aggregation_channel)

    def forward(self, xyz_list, feature_list, fps_idx_list, ctr_xyz):
        """
        candidate generation module forward
        Args:
            xyz_list:
            feature_list:
            fps_idx_list:
            ctr_xyz:
        Returns:
            ctr_xyz: ([B, npoints, 3], torch,Tensor)
            ctr_features: ([B, C, npoint], torch.Tensor)
            idx: None
        """
        if len(self.xyz_index_list) > 1:
            temp_xyz_list = []
            for xyz_index in self.xyz_index_list:
                temp_xyz = xyz_list[xyz_index]
                temp_xyz_list.append(temp_xyz)
            xyz = torch.cat(temp_xyz_list, dim=1)
        else:
            xyz = xyz_list[self.xyz_index_list[0]]

        if len(self.feature_index_list) > 1:
            temp_feature_list = []
            for feature_index in self.feature_index_list:

                temp_feature = feature_list[feature_index]
                temp_feature_list.append(temp_feature)
            feature = torch.cat(temp_feature_list, dim=2)
        else:
            feature = feature_list[self.feature_index_list[0]]

        new_features_list = []

        for i in range(len(self.groupers)):
            group_features = self.groupers[i](
                xyz, ctr_xyz, feature
            )   # (B, C(+3), npoint, nsample)

            new_features = self.mlps[i](
                group_features
            )   # (B, mlp[-1], npoint, nsample)

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            if self.cell_type == 'NLC':
                new_features = self.nl_mlps[i](new_features)

            new_features_list.append(new_features)

        # Conv1d for feature aggregation
        new_feature = torch.cat(new_features_list, dim=1)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))  # (B, aggregation_channel, npoint)

        return ctr_xyz, new_feature, None


class PointnetSAModuleMSGVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(
            self,
            *,
            mlps: List[List[int]],
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        super().__init__()

        assert (len(mlps) == len(nsamples) == len(radii))

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None, inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, C) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), inds


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1)
            )

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointnetLFPModuleMSG(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    learnable feature propagation layer.'''

    def __init__(
            self,
            *,
            mlps: List[List[int]],
            radii: List[float],
            nsamples: List[int],
            post_mlp: List[int],
            bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        super().__init__()

        assert (len(mlps) == len(nsamples) == len(radii))

        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz,
                                              sample_uniformly=sample_uniformly)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor,
                features2: torch.Tensor, features1: torch.Tensor) -> torch.Tensor:
        r""" Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \sum_k(mlps[k][-1]), N1) tensor of the new_features descriptors
        """
        new_features_list = []

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz1, xyz2, features1
            )  # (B, C1, N2, nsample)
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], N2, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], N2, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], N2)

            if features2 is not None:
                new_features = torch.cat([new_features, features2],
                                         dim=1)  # (B, mlp[-1] + C2, N2)

            new_features = new_features.unsqueeze(-1)
            new_features = self.post_mlp(new_features)

            new_features_list.append(new_features)

        return torch.cat(new_features_list, dim=1).squeeze(-1)


def main():
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(
            torch.cuda.FloatTensor(*new_features.size()).fill_(1)
        )
        print(new_features)
        print(xyz.grad)


if __name__ == "__main__":
    main()
