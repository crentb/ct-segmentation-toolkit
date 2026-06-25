"""
ct_seg.denoise — self-supervised CT denoising (2.5D Noise2Inverse).

Cameron Renteria's own implementation of the Noise2Inverse (N2I) self-supervised
tomography-denoising framework: a no-skip U-Net (Group Normalization, LeakyReLU) trained
to map one sub-reconstruction (e.g. even-angle) to another (odd-angle), so it needs no
clean reference image. His additions over the base method include a 2.5D adjacent-slice
input, a Laplacian Contrast Loss (LCL) for edge preservation, edge-aware model selection,
and automatic GPU batch-size optimization.

Upstream method (credited): Noise2Inverse - Hendriksen, Pelt & Batenburg, IEEE Transactions
on Computational Imaging 6 (2020); original code https://github.com/ahendriksen/noise2inverse
(see ACKNOWLEDGMENTS.md). This subpackage is Copyright 2026 Cameron Renteria, Apache-2.0.

Only the lightweight building blocks (model, loss, edge score) are exported here. The
training/inference CLIs and dataset (`ct_seg.denoise.train`, `denoise_volume`,
`denoise_slice`, `data`) require the optional ``[denoise]`` extra (albumentations, PyYAML).
"""

from ct_seg.denoise.eval import laplacian_score_batch
from ct_seg.denoise.loss import LCL
from ct_seg.denoise.model import unet_ns_gn

__all__ = ["unet_ns_gn", "LCL", "laplacian_score_batch"]
