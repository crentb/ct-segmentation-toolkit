# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Continuous integration: GitHub Actions updated to their current major versions (Node 24
  runtimes), gitleaks 8.30.1, and the test matrix extended to Python 3.10 through 3.14.
- Container base image `python:3.14-slim`.
- Development tools ruff 0.16.8, black 26.5.1, and mypy 2.3.1, with the pre-commit hooks
  pinned to the same versions.
- README reorganized: status badges (CI, PyPI, Python, DOI, license), installation from
  PyPI and the GitHub Container Registry, a testing and CI section, and a citation entry.
- Overview figure redrawn for legibility, with its LaTeX source in `docs/figures/src/`.

### Added
- `SECURITY.md`: supported versions and private vulnerability reporting through GitHub.
- `doi` in `CITATION.cff` (Zenodo concept DOI).

### Fixed
- `ct_seg.__version__` reports 0.2.0, matching the released package.
- `CITATION.cff` preferred citation updated to the revised manuscript's title and author list.

## [0.2.0] - 2026-07-16

### Added
- **Continuous delivery** (`.github/workflows/release.yml`): pushing a `v*` tag re-runs the
  full CI gate on the tagged commit, then publishes the sdist + wheel to PyPI via Trusted
  Publishing (OIDC — no stored API token) and a signed container image to GHCR carrying a
  SLSA build-provenance attestation. `ci.yml` is now `workflow_call`-able so the release
  re-uses the exact same gates instead of a copy that could drift.
- **DevSecOps stages in CI**, all gating: full-history secret detection (gitleaks), SAST
  (bandit, medium+), dependency-CVE audit (pip-audit), and container vulnerability scanning
  (trivy, CRITICAL/HIGH). The image additionally ships an SPDX SBOM (syft) signed keylessly
  with cosign via GitHub OIDC.

### Changed
- `ct_seg.som_bands`: vectorized the per-pixel best-matching-unit mapping — ~1.7× faster on
  a 512×512 slice.
- Torch exports are lazy-loaded, so the classical (non-deep-learning) API no longer imports
  torch.

### Security
- `torch.load` is now pinned to `weights_only=True` at all three call sites
  (`denoise/denoise_slice.py`, `denoise/denoise_volume.py`, `segment.py`). By default
  `torch.load` unpickles arbitrary Python, so a malicious checkpoint could execute code;
  the weakness (CWE-502) was surfaced by the new bandit gate.
- Patched the 3 fixable HIGH-severity CVEs that trivy found in the container image.

### Fixed
- Corrected the trivy-action pin (`0.24.0`, which does not exist → `v0.36.0`), and scoped
  the trivy gate to *fixable* CRITICAL/HIGH so an unpatchable upstream CVE cannot
  permanently block releases.

## [0.1.0] - 2026-07-03

### Added
- Initial public version of a CT image-stack segmentation toolkit.
- **Supervised**: `ct_seg.model.UNetSegmentation` (U-Net, GroupNorm, skip connections,
  2.5D-capable) with training (`ct_seg.train`) and inference (`ct_seg.segment --method unet`).
- **Unsupervised**: multi-Otsu, k-means, and GMM intensity segmentation
  (`segment_otsu` / `segment_kmeans` / `segment_gmm`), with overlays + per-class statistics.
- **Label-free spectral-spatial**: `ct_seg.som_bands` (`extract_features` + `som_segment`)
  — multi-scale density, structure-tensor orientation/coherence, a Gabor filter bank, and
  local variance, clustered via a Self-Organizing Map; transfers to hyperspectral data.
- **Self-supervised denoising**: `ct_seg.denoise`, a 2.5D Noise2Inverse implementation
  developed together by Austin Yunker (Argonne National Laboratory) and Cameron B.
  Renteria (no-skip U-Net, Laplacian Contrast Loss, edge-aware selection, auto
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
