import torch
import torch.nn as nn
import numpy as np


def calcu_square_dist(a, b, norm=False):
    a = a.unsqueeze(2)  # (B, N, 1, C)
    b = b.unsqueeze(1)  # (B, 1, M, C)
    a_square = torch.sum(torch.mul(a, a), dim=-1)  # (B, N, 1)
    b_square = torch.sum(torch.mul(b, b), dim=-1)  # (B, 1, M)
    a = a.squeeze(2)  # (B, N, C)
    b = b.squeeze(1)  # (B, M, C)

    xyz_feature_dist = a_square + b_square - 2 * torch.matmul(a, b.transpose(2, 1))
    return xyz_feature_dist


def calcu_weighted_square_dist(a, b, norm=True):
    """
    Calculate square distance between a and b
    Args:
        a: ([B, N, C], torch.Tensor). features
        b: ([B, M, C], torch.Tensor). features
        norm:
    Returns:
        feature_dist: ([B, N, M], torch.Tensor). features dist
    """
    a_xyz = a[:, :, :3]
    b_xyz = b[:, :, :3]
    a_xyz_square = torch.sum(torch.mul(a_xyz.unsqueeze(2), a_xyz.unsqueeze(2)), dim=-1)  # (B, N, 1)
    b_xyz_square = torch.sum(torch.mul(b_xyz.unsqueeze(1), b_xyz.unsqueeze(1)), dim=-1)  # (B, 1, M)

    if norm:
        xyz_dist = torch.sqrt(
            a_xyz_square + b_xyz_square - 2 * torch.matmul(a_xyz, b_xyz.transpose(2, 1))) / 3  # (B, N, M)
        bs_xyz_dist_max = torch.max(torch.max(xyz_dist, dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]
        xyz_dist = torch.div(xyz_dist, bs_xyz_dist_max)
    else:
        xyz_dist = a_xyz_square + b_xyz_square - 2 * torch.matmul(a_xyz, b_xyz.transpose(2, 1))  # (B, N, M)
        bs_xyz_dist_max = torch.max(torch.max(xyz_dist, dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]
        xyz_dist = torch.div(xyz_dist, bs_xyz_dist_max)

    # print("xyz distance max value:")
    # print(bs_xyz_dist_max.shape)
    del a_xyz, b_xyz, a_xyz_square, b_xyz_square, bs_xyz_dist_max

    a_feature = a[:, :, 3:]
    b_feature = b[:, :, 3:]
    a_feature_square = torch.sum(torch.mul(a_feature.unsqueeze(2), a_feature.unsqueeze(2)), dim=-1)  # (B, N, 1)
    b_feature_square = torch.sum(torch.mul(b_feature.unsqueeze(1), b_feature.unsqueeze(1)), dim=-1)  # (B, 1, M)

    if norm:
        C = float(a.shape[-1] - 3)
        feature_dist = torch.sqrt(a_feature_square + b_feature_square - 2 * torch.matmul(a_feature,
                                                                                         b_feature.transpose(2, 1))) / C  # (B, N, M)
        bs_feature_dist_max = torch.max(torch.max(feature_dist, dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]
        feature_dist = torch.div(feature_dist, bs_feature_dist_max)

        # feature_dist = torch.sqrt(a_square + b_square - 2 * torch.matmul(a, b.transpose(2, 1))) / C
    else:
        feature_dist = a_feature_square + b_feature_square - 2 * torch.matmul(a_feature,
                                                                              b_feature.transpose(2, 1))  # (B, N, M)
        bs_feature_dist_max = torch.max(torch.max(feature_dist, dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]
        feature_dist = torch.div(feature_dist, bs_feature_dist_max)

    # print("feature distance max value:")
    # print(bs_feature_dist_max.shape)
    xyz_feature_dist = torch.add(xyz_dist, feature_dist)  # ~
    return xyz_feature_dist


def test_func():
    a = torch.rand(2, 4, 1, 2)
    b = torch.rand(2, 1, 3, 2)
    a_square = torch.sum(torch.mul(a, a), dim=-1)  # (2, 4, 1)
    b_square = torch.sum(torch.mul(b, b), dim=-1)  # (2, 1, 3)
    a = a.squeeze(2)
    b = b.squeeze(1)
    dist = a_square + b_square - 2 * torch.matmul(a, b.transpose(2, 1))
    print(dist)


if __name__ == '__main__':
    test_func()
