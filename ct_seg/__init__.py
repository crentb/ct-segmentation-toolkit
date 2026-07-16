"""
ct_seg — a CT image segmentation toolkit.

Supervised (U-Net), unsupervised (multi-Otsu / k-means / GMM), and label-free
spectral-spatial (Self-Organizing Map) segmentation of scientific image stacks,
plus self-supervised denoising (2.5D Noise2Inverse).

Only the lightweight, importable APIs are exported here. The interactive tools
(`labeling`, napari; `viewer`, pyvista) require the optional ``[viz]`` extra and
are imported directly from their modules when needed.

Import policy — torch is loaded LAZILY (PEP 562):
    The torch-backed symbols (``UNetSegmentation``, ``count_parameters``,
    ``unet_ns_gn``, ``LCL``) are resolved on first attribute access instead of
    at package import. Two reasons:
    1. Classical users (k-means / GMM / Otsu / SOM) should not pay torch's
       import cost — or even need torch installed — to segment an image.
    2. Robustness: on some environments (observed on anaconda + macOS ARM,
       2026-07-15) eagerly importing torch alongside scikit-learn is FATAL —
       torch's bundled libomp and MKL's libiomp5 both initialize OpenMP and
       the process aborts with ``OMP: Error #179`` the moment a scikit-learn
       thread pool spins up (e.g., KMeans inside ``som_segment``). Keeping
       torch out of the import path unless a torch API is actually requested
       means the classical paths can never trigger that clash.
"""

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

# torch-backed exports, resolved lazily on first access (PEP 562).
_LAZY_TORCH_EXPORTS = {
    "UNetSegmentation": ("ct_seg.model", "UNetSegmentation"),
    "count_parameters": ("ct_seg.model", "count_parameters"),
    "unet_ns_gn": ("ct_seg.denoise", "unet_ns_gn"),
    "LCL": ("ct_seg.denoise", "LCL"),
}

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


def __getattr__(name):
    """Resolve torch-backed exports on first use (see module docstring)."""
    if name in _LAZY_TORCH_EXPORTS:
        import importlib

        module_name, attr = _LAZY_TORCH_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value  # cache so the import cost is paid once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Keep tab-completion/introspection honest about the lazy exports."""
    return sorted(list(globals().keys()) + list(_LAZY_TORCH_EXPORTS.keys()))
