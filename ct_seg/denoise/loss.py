"""
loss.py — Laplacian Contrast Loss (LCL) for edge-aware CT denoising.

Defines the auxiliary loss used by main.py after the L1 warm-up. The Laplacian
highlights edges; LCL compares the mean Laplacian magnitude of "edge" pixels (the
top 20% by |Laplacian|) against "flat" pixels and returns flat/edge. Minimizing
this ratio pushes the network to keep edges sharp (high Laplacian) while smoothing
flat regions (low Laplacian), counteracting the over-smoothing that plain L1
denoising tends to produce.

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import torch
import torch.nn as nn
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


class LCL(nn.Module):
    """Laplacian Contrast Loss (see module docstring).

    forward(pred) returns flat_mean / (edge_mean + eps), where edge pixels are the
    top 20% by |Laplacian| and flat pixels are the rest. Lower is better: it favors
    sharp edges relative to flat regions.
    """

    def __init__(
        self,
    ):
        super(LCL, self).__init__()

    def forward(self, pred):
        L = torch.abs(laplacian_batch(pred))
        threshold = torch.quantile(L, 0.80)
        # Split pixels into 'edge' (top 20% by Laplacian magnitude) and 'flat' (the rest).
        edge_mask = L > threshold
        flat_mask = ~edge_mask

        edge_mean = L[edge_mask].mean() if edge_mask.any() else 0.0
        flat_mean = L[flat_mask].mean() if flat_mask.any() else 1e-6

        # Encourage this ratio to grow (i.e., minimize the inverse)
        return flat_mean / (edge_mean + 1e-6)


def laplacian_entropy_map(lap, bins=256):
    """Per-image Shannon entropy of the Laplacian-magnitude histogram.

    Higher entropy means a busier (more textured/edge-rich) image. Returns a 1-D
    tensor of length B, one value per image in the batch.
    """
    # Compute entropy for each image in batch
    B = lap.shape[0]
    entropies = []
    for i in range(B):
        hist = torch.histc(lap[i, 0], bins=bins, min=0, max=lap[i, 0].max().item())
        hist = hist / hist.sum()
        hist = hist + 1e-8  # avoid log(0)
        entropy = -torch.sum(hist * torch.log(hist))
        entropies.append(entropy.item())
    return torch.tensor(entropies, device=lap.device)
