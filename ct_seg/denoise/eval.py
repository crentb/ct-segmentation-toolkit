"""
eval.py — Laplacian-based edge/sharpness score for monitoring denoising quality.

Provides `laplacian_score_batch`, the metric main.py uses during validation to
track how sharp the denoised output is. For each image it measures edge contrast
as the ratio of mean |Laplacian| in edge pixels to that in flat pixels; for very
flat images (low Laplacian-histogram entropy) it instead rewards smoothness. A
higher score means sharper, better-resolved edges, and main.py keeps the
checkpoint with the highest score (best_edge_model.pth).

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import numpy as np
import torch
import torch.nn.functional as F


def laplacian_batch(x):
    """Apply a 3x3 discrete Laplacian (edge) filter to a batch of images.

    Args:
        x: tensor [B, C, H, W]; the same kernel is applied per channel (grouped conv).
    Returns:
        The Laplacian response, same shape as `x`.
    """
    # x: [B, C, H, W] (assumes grayscale: C=1)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(
        1, 1, 3, 3
    )
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # for grouped conv
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def laplacian_entropy_map(lap, bins=256):
    """Per-image Shannon entropy of the Laplacian-magnitude histogram.

    Higher entropy means a busier (more textured/edge-rich) image. Returns a 1-D
    tensor of length B, one value per image in the batch.
    """
    # Compute entropy for each image in batch
    B = lap.shape[0]
    entropies = []
    for i in range(B):
        hist = torch.histc(lap[i, 0], bins=bins, min=0, max=lap[i, 0].max())
        hist = hist / hist.sum()
        hist = hist + 1e-8  # avoid log(0)
        entropy = -torch.sum(hist * torch.log(hist))
        entropies.append(entropy.item())
    return torch.tensor(entropies, device=lap.device)


def laplacian_score_batch(batch, entropy_thresh=0.2, q=0.9):
    """Mean edge-quality score over a batch (higher = sharper edges).

    Args:
        batch         : tensor [B, 1, H, W] of images to score.
        entropy_thresh: below this Laplacian-histogram entropy an image is treated
                        as "flat" and scored on smoothness instead of edge contrast.
        q             : quantile that separates edge pixels from flat pixels.
    Returns:
        float : the mean per-image score across the batch.
    """
    # batch: [B, 1, H, W]
    lap = torch.abs(laplacian_batch(batch))  # [B, 1, H, W]
    B = lap.shape[0]
    scores = []

    entropies = laplacian_entropy_map(lap)

    for i in range(B):
        lap_i = lap[i, 0]  # [H, W]
        entropy = entropies[i]

        if entropy < entropy_thresh:
            # flat image: reward smoothness
            # score = -torch.mean(lap_i).item()
            smoothness = 1.0 / (lap_i.mean().item() + 1e-6)
            scores.append(smoothness)
        else:
            # compute threshold using quantile
            threshold = torch.quantile(lap_i, q)
            edge_mask = lap_i > threshold
            flat_mask = ~edge_mask

            edge_score = lap_i[edge_mask].mean() if edge_mask.any() else 0.0
            flat_score = lap_i[flat_mask].mean() if flat_mask.any() else 1e-6  # prevent div 0

            contrast = edge_score / (flat_score + 1e-6)
            scores.append(contrast)

    return float(np.mean(scores))
    # return np.array(scores).astype(np.float32)
