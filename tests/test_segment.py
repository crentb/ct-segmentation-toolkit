"""Unsupervised segmentation smoke tests on a small synthetic two-phase stack."""

import numpy as np

from ct_seg.segment import (
    normalize_stack,
    segment_gmm,
    segment_kmeans,
    segment_otsu,
)


def _synthetic_stack():
    rng = np.random.default_rng(0)
    dark = rng.normal(0.2, 0.05, size=(3, 32, 32))
    bright = rng.normal(0.8, 0.05, size=(3, 32, 32))
    pick = rng.random((3, 32, 32)) > 0.5
    return normalize_stack(np.where(pick, bright, dark).astype(np.float32))


def test_normalize_range():
    n = normalize_stack(np.array([[[0.0, 2.0], [4.0, 8.0]]], dtype=np.float32))
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_segment_otsu():
    stack = _synthetic_stack()
    labels, info = segment_otsu(stack, num_classes=2)
    assert labels.shape == stack.shape
    assert labels.max() < 2
    assert "thresholds" in info


def test_segment_kmeans():
    stack = _synthetic_stack()
    labels, _ = segment_kmeans(stack, num_classes=2)
    assert labels.shape == stack.shape
    assert set(np.unique(labels)).issubset({0, 1})


def test_segment_gmm():
    stack = _synthetic_stack()
    labels, _ = segment_gmm(stack, num_classes=2)
    assert labels.shape == stack.shape
    assert set(np.unique(labels)).issubset({0, 1})
