"""
ct_seg — a CT image segmentation toolkit.

Supervised (U-Net), unsupervised (multi-Otsu / k-means / GMM), and label-free
spectral-spatial (Self-Organizing Map) segmentation of scientific image stacks,
plus self-supervised denoising (2.5D Noise2Inverse).

Only the lightweight, importable APIs are exported here. The interactive tools
(`labeling`, napari; `viewer`, pyvista) require the optional ``[viz]`` extra and
are imported directly from their modules when needed.
"""

from ct_seg.denoise import LCL, unet_ns_gn
from ct_seg.model import UNetSegmentation, count_parameters
from ct_seg.segment import (
    enhance_contrast,
    normalize_stack,
    segment_gmm,
    segment_kmeans,
    segment_otsu,
    segment_unet,
)
from ct_seg.som_bands import extract_features, som_segment

__version__ = "0.1.0"

__all__ = [
    "UNetSegmentation",
    "count_parameters",
    "normalize_stack",
    "enhance_contrast",
    "segment_otsu",
    "segment_kmeans",
    "segment_gmm",
    "segment_unet",
    "extract_features",
    "som_segment",
    "unet_ns_gn",
    "LCL",
    "__version__",
]
