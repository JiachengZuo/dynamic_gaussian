# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from gsplat.rendering import rasterization
from tqdm import tqdm
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF


palette_10 = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (70, 130, 180),
}

# 0:Road
# 1:Building
# 2:Vegetation
# 3:Vehicle
# 4:Person
# 5:Cyclist
# 6:Traffic Sign
# 7:truck
# 8:Sidewalk
# 9:Sky

def concat_list(list_1, list_2):
    if list_2[0].shape == torch.Size([0]):
        return list_1
    concated_list = []
    for i in range(len(list_1)):
        item = torch.concat((list_1[i],list_2[i]),dim=0)
        concated_list.append(item)
    return concated_list


def get_masked_gs(point_map, gs_map, mask, idx=None):
    # point_map: B,S,H,W,3 or B,S,H,W,4,3 (multi-gaussian)
    # gs_map: B,S,H,W,C or B,S,H,W,4,C (multi-gaussian [..., 4, 14])
    # mask:  B,S,H,W
    if idx is not None:
        point_map = point_map[:,idx,...]
        gs_map = gs_map[:,idx,...]
        mask = mask[:,idx,...]
    world_points = point_map[mask].reshape(-1, 3)
    # Handle multi-gaussian format [..., 4, C]:
    # gs_map has 2 more dims than mask (the [4, C] dims)
    ndim_diff = gs_map.dim() - mask.dim()
    if ndim_diff == 2 and gs_map.shape[-2] == 4:
        mask_gs = mask.unsqueeze(-1).expand(*mask.shape, gs_map.shape[-2])
        world_points = world_points.repeat_interleave(gs_map.shape[-2], dim=0)
    else:
        mask_gs = mask
    rgbs = gs_map[...,:3][mask_gs].reshape(-1, 3)
    opacity = gs_map[...,3:4][mask_gs].reshape(-1)
    scales = gs_map[...,4:7][mask_gs].reshape(-1, 3)
    rotation = gs_map[...,7:11][mask_gs].reshape(-1, 4)
    return world_points, rgbs, opacity, scales, rotation


def get_split_gs(gs_map, mask):
    """
    Extract Gaussian parameters for active positions.

    Handles two gs_map shapes:
      (A) old:  [..., C]       with mask [B, S, H, W]
      (B) new:  [..., 4, C]    with mask [B, S, H, W]  (multi-gaussian per pixel)
    In case (B), the mask selects pixels; all 4 per-level Gaussians at
    each selected pixel are returned, and inactive ones are pre-zeroed.
    """
    ndim_diff = gs_map.dim() - mask.dim()
    if ndim_diff == 2 and gs_map.shape[-2] == 4:
        # Multi-gaussian [..., 4, C]: broadcast mask across levels
        N_levels = gs_map.shape[-2]
        mask_gs = mask.unsqueeze(-1).expand(*mask.shape, N_levels)
    else:
        mask_gs = mask
    rgbs = gs_map[..., :3][mask_gs].reshape(-1, 3)
    opacity = gs_map[..., 3:4][mask_gs].reshape(-1)
    scales = gs_map[..., 4:7][mask_gs].reshape(-1, 3)
    rotation = gs_map[..., 7:11][mask_gs].reshape(-1, 4)
    conf = gs_map[..., 11:12][mask_gs].reshape(-1)
    return rgbs, opacity, scales, rotation, conf


def gs_dict(points, rgbs, opacity, scales, rotation):
    gs_dict = {}
    gs_dict['means'] = points
    gs_dict['quats'] = rotation
    if opacity.dim() == 1:
        opacity = opacity[...,None]
    gs_dict['opacities'] = opacity
    gs_dict['scales'] = scales
    gs_dict['features_dc'] = rgbs
    return gs_dict

def get_gs_items(gs_dict):
    points = gs_dict['means']
    rotation = gs_dict['quats']
    opacity = gs_dict['opacities'][...,0]
    scales = gs_dict['scales']
    rgbs = gs_dict['features_dc']
    return points, rgbs, opacity, scales, rotation


import numpy as np

import numpy as np

def downsample_3dgs(points, rgbs, opacity, scales, rotation, num_points=200000):
    N = points.shape[0]
    if num_points >= N:
        return points, rgbs, opacity, scales, rotation

    # Compute importance weights: opacity * volume
    volume = scales.prod(dim=1)                    # (N,)
    weights = opacity * volume                     # (N,)
    weights = weights / weights.sum()              # Normalize to sum to 1

    # Sample indices with probability proportional to weights
    indices = torch.multinomial(weights, num_points, replacement=False)  # (num_points,)

    return (
        points[indices],
        rgbs[indices],
        opacity[indices],
        scales[indices],
        rotation[indices]
    )


# ============================================================
# Dynamic Gaussian Budget Allocation Utilities
# ============================================================


def soft_gate_gs(gs_params_list, n_i, temperature=5.0):
    """
    Soft gating for multi-scale Gaussian selection during training.

    For each spatio-temporal position, if n_i ~ k, the first k levels are softly
    activated. Returns per-level gated params WITHOUT averaging.
    n_i=k produces k independent Gaussians at that position.

    Args:
        gs_params_list: list of [B, S, C, H_patch, W_patch] per level (4 levels)
        n_i: [B, S, H_patch, W_patch, 1] continuous gaussian count in [0, 4]
        temperature: softness of gate transition
    Returns:
        gs_stacked: [B, S, num_levels, C, H_patch, W_patch] per-level soft-gated params
        gates: list of [B, S, 1, H_patch, W_patch] per-level gates
    """
    num_levels = len(gs_params_list)
    n_i_c = n_i.permute(0, 1, 4, 2, 3)  # [B, S, 1, H_patch, W_patch]
    B, S = n_i_c.shape[:2]

    gates = []
    for k in range(num_levels):
        gate_k = torch.sigmoid((n_i_c - k) * temperature)
        gates.append(gate_k)  # each [B, S, 1, H_patch, W_patch]

    # Stack per-level gated params WITHOUT averaging
    gs_stacked = torch.stack(
        [gates[k] * gs_params_list[k] for k in range(num_levels)], dim=2
    )  # [B, S, num_levels, C, H_patch, W_patch]
    return gs_stacked, gates


def select_multi_scale_gs(gs_params_list, n_i):
    """
    Hard selection of multi-scale Gaussian params for inference.

    For each spatio-temporal position, if n_i = k, select the first k levels'
    outputs as independent Gaussians. Returns per-level stacked output.

    Args:
        gs_params_list: list of [B, S, C, H_patch, W_patch] per level (4 levels)
        n_i: [B, S, H_patch, W_patch] discrete gaussian count in {0, 1, 2, 3, 4}
    Returns:
        gs_stacked: [B, S, num_levels, C, H_patch, W_patch]
        active_mask: [B, S, num_levels, H_patch, W_patch] bool
    """
    num_levels = len(gs_params_list)
    B, S, C, H_patch, W_patch = gs_params_list[0].shape
    device = gs_params_list[0].device

    # active_mask[k] = True at (b,s,h,w) iff k < n_i[b,s,h,w]
    level_idx = torch.arange(num_levels, device=device).view(1, 1, -1, 1, 1)
    active_mask = level_idx < n_i.unsqueeze(2)  # [B, S, num_levels, H_patch, W_patch]

    # Stack all levels and zero-out inactive ones
    gs_stacked = torch.stack(gs_params_list, dim=2)  # [B, S, num_levels, C, H_patch, W_patch]
    gs_stacked = gs_stacked * active_mask.unsqueeze(3).float()

    return gs_stacked, active_mask


def compute_marginal_gains(S_i, n_i_current):
    """
    Compute marginal utility gain for adding one more Gaussian at each position.

    Args:
        S_i: [H_patch * W_patch] utility scores
        n_i_current: [H_patch * W_patch] current gaussian counts
    Returns:
        delta: [H_patch * W_patch] marginal gains
    """
    delta = S_i * torch.log((n_i_current + 2) / (n_i_current + 1))
    return delta


def greedy_gaussian_allocation(S_i, budget, max_per_position=4):
    """
    Greedy allocation of Gaussians based on utility scores.

    At each step, assigns one Gaussian to the position with highest
    marginal gain, until budget is exhausted.

    Args:
        S_i: [H_patch, W_patch] or [H_patch * W_patch] utility scores
        budget: int, total gaussian budget (e.g., 4 * H_patch * W_patch)
        max_per_position: int, max gaussians per position (default 4)
    Returns:
        n_i: [H_patch * W_patch] discrete gaussian counts {0, ..., 4}
    """
    if S_i.dim() == 2:
        S_i = S_i.flatten()

    num_positions = S_i.shape[0]
    n_i = torch.zeros(num_positions, device=S_i.device, dtype=torch.long)

    # Precompute log ratios for fast lookup
    # log((k+2)/(k+1)) for k = 0, 1, 2, 3
    log_ratios = torch.log(torch.tensor([2.0 / 1.0, 3.0 / 2.0, 4.0 / 3.0, 5.0 / 4.0], device=S_i.device))

    # Compute initial marginal gains (k=0 for all positions)
    marginal_gains = S_i * log_ratios[0]

    for _ in range(budget):
        # Mask positions that already reached max
        valid_mask = (n_i < max_per_position)
        if not valid_mask.any():
            break

        # Find position with highest marginal gain among valid ones
        masked_gains = marginal_gains.clone()
        masked_gains[~valid_mask] = -float('inf')
        best_idx = torch.argmax(masked_gains)

        # Allocate one Gaussian
        n_i[best_idx] += 1
        k = n_i[best_idx]

        # Update marginal gain for this position
        if k < max_per_position:
            marginal_gains[best_idx] = S_i[best_idx] * log_ratios[k]
        else:
            marginal_gains[best_idx] = 0.0

    return n_i


def compute_utility_loss(S_i, n_i):
    """
    Compute utility loss: L_utility = -sum(S_i * log(1 + n_i)).

    Args:
        S_i: [B, S, 1, H_patch, W_patch] utility scores (per spatio-temporal position)
        n_i: [B, S, H_patch, W_patch, 1] continuous gaussian counts
    Returns:
        loss: scalar tensor
    """
    return -torch.mean(S_i * torch.log(1 + n_i + 1e-8))


def compute_budget_loss(n_i, budget, lambda_budget=1.0):
    """
    Compute budget constraint loss: lambda * max(0, sum(n_i) - budget)^2.

    Args:
        n_i: [B, S, H_patch, W_patch, 1] continuous gaussian counts
        budget: int, max allowed total gaussians per sample (across all frames)
        lambda_budget: penalty weight
    Returns:
        loss: scalar tensor (averaged over batch)
    """
    B = n_i.shape[0]
    total_n_per_sample = n_i.sum(dim=[1, 2, 3, 4])  # [B]
    excess = F.relu(total_n_per_sample - budget)  # [B]
    return lambda_budget * torch.mean(excess ** 2)


def collect_active_gaussians(gs_map, active_mask, point_map=None):
    """
    Collect all active Gaussians from pixel-level gs_map.

    gs_map:      [B, S, H, W, N_levels, 14]  — pixel-level, inactive=0
    active_mask: [B, S, H, W, N_levels] bool
    point_map:   [B, S, H, W, 3] or None  — 3D world points per pixel

    Returns flat tensors:
        points:   [N, 3]
        rgbs:     [N, 3]
        opacity:  [N]
        scales:   [N, 3]
        rotation: [N, 4]
        conf:     [N]
    where N = mask.sum(). Each pixel's 3D point is replicated N_levels times
    because one pixel can have up to N_levels Gaussians.
    """
    N_levels = gs_map.shape[-2]
    mask = active_mask  # [B, S, H, W, N_levels]
    gs_flat = gs_map[mask]  # [N, 14]
    rgbs = gs_flat[:, :3]
    opacity = gs_flat[:, 3]
    scales = gs_flat[:, 4:7]
    rotation = gs_flat[:, 7:11]
    conf = gs_flat[:, 11]

    if point_map is not None:
        # point_map: [B, S, H, W, 3] — one 3D point per pixel
        # Tile each pixel's point N_levels times to match gaussians
        points = point_map.unsqueeze(-2).expand(*point_map.shape[:-1], N_levels, 3)
        points = points[mask]  # [N, 3]
    else:
        points = None
    return points, rgbs, opacity, scales, rotation, conf


def compute_vis_loss(vis_sim, covisibility_gt):
    """
    L1 loss between predicted vis_sim and ground-truth covisibility map.

    Args:
        vis_sim: [B, S, 1, H_patch, W_patch] predicted temporal cosine similarity
        covisibility_gt: [B, S, 1, H_patch, W_patch] ground-truth covisibility
    Returns:
        loss: scalar tensor (L1 loss)
    """
    if vis_sim is None or covisibility_gt is None:
        return torch.tensor(0.0, device=vis_sim.device if vis_sim is not None else 'cpu')
    # Ensure matching shapes
    if covisibility_gt.dim() == 3:
        covisibility_gt = covisibility_gt.unsqueeze(0).unsqueeze(-1)
    elif covisibility_gt.dim() == 2:
        covisibility_gt = covisibility_gt.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    return F.l1_loss(vis_sim, covisibility_gt.to(vis_sim.device))

