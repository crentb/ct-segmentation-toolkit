"""Smoke tests for the 2.5D Noise2Inverse denoiser building blocks (CPU, synthetic)."""

import math

import torch

from ct_seg.denoise import LCL, laplacian_score_batch, unet_ns_gn


def test_unet_forward_shape():
    # 2.5D: five adjacent input slices -> one denoised slice.
    model = unet_ns_gn(start_filter_size=8, ich=5, och=1)
    out = model(torch.randn(2, 5, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_lcl_loss_is_finite_scalar():
    val = float(LCL()(torch.randn(1, 1, 64, 64)))
    assert math.isfinite(val)


def test_laplacian_score_is_float():
    s = laplacian_score_batch(torch.randn(2, 1, 64, 64))
    assert isinstance(s, float) and math.isfinite(s)
