# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public version of a CT image-stack segmentation toolkit.
- **Supervised**: `ct_seg.model.UNetSegmentation` (U-Net, GroupNorm, skip connections,
  2.5D-capable) with training (`ct_seg.train`) and inference (`ct_seg.segment --method unet`).
- **Unsupervised**: multi-Otsu, k-means, and GMM intensity segmentation
  (`segment_otsu` / `segment_kmeans` / `segment_gmm`), with overlays + per-class statistics.
- **Label-free spectral-spatial**: `ct_seg.som_bands` (`extract_features` + `som_segment`)
  — multi-scale density, structure-tensor orientation/coherence, a Gabor filter bank, and
  local variance, clustered via a Self-Organizing Map; transfers to hyperspectral data.
- **Self-supervised denoising**: `ct_seg.denoise`, the author's own 2.5D Noise2Inverse
  implementation (no-skip U-Net, Laplacian Contrast Loss, edge-aware selection, auto
  batch-sizing), with explicit upstream credit to the N2I paper; optional `[denoise]` extra
  (albumentations, PyYAML) for training.
- Interactive `ct_seg.labeling` (napari) and `ct_seg.viewer` (napari/pyvista) under the
  optional `[viz]` extra.
- pip-installable PEP 621 package (Apache-2.0); pinned dev tools (ruff/black/mypy) for
  reproducible CI; GitHub Actions CI (ruff + black + mypy + pytest on Python 3.10-3.12)
  plus a Docker image build/smoke-test; synthetic-data test suite.
- Optional **MLflow** experiment tracking (`[mlops]` extra) wired into U-Net training
  (params, per-epoch loss/mIoU, best checkpoint), with a graceful no-op fallback.
- Performance-profiling script (`scripts/profile_unet.py`) and a `docs/PERF.md` report.
