"""U-Net model smoke tests (synthetic tensors only)."""

import torch

import ct_seg
from ct_seg.model import UNetSegmentation, count_parameters


def test_version():
    assert ct_seg.__version__ == "0.1.0"


def test_unet_forward_shape():
    model = UNetSegmentation(in_channels=1, num_classes=4, base_filters=8)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 4, 64, 64)


def test_unet_predict_classes():
    model = UNetSegmentation(in_channels=1, num_classes=3, base_filters=8)
    pred = model.predict(torch.randn(1, 1, 64, 64))
    assert pred.shape == (1, 64, 64)
    assert int(pred.min()) >= 0 and int(pred.max()) < 3


def test_count_parameters_positive():
    model = UNetSegmentation(in_channels=1, num_classes=2, base_filters=8)
    assert count_parameters(model) > 0
