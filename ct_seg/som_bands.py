"""
Label-free spectral-spatial band segmentation via a Self-Organizing Map (SOM).

Refactored from the author's proof-of-concept script into importable functions
(the algorithm and feature set are unchanged). The idea: instead of hand-labeling
training data, describe every pixel by a multi-scale, spectral-spatial feature
vector (local density, structure-tensor orientation/coherence, a Gabor filter bank,
and local variance), train a SOM on a pixel sample, then k-means-cluster the SOM
nodes into regions. This is unsupervised, modality-agnostic segmentation: the same
spectral-spatial feature engineering applies directly to hyperspectral cubes.

Originally developed to segment enamel "decussation bands" from synchrotron micro-CT
cross-sections, where different bands differ in local rod density, arrangement
anisotropy, and periodic texture.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import gabor_kernel


def extract_features(
    image,
    density_sigmas=(8, 16, 32),
    st_sigma=8,
    gabor_freqs=(0.05, 0.1, 0.15),
    gabor_thetas=None,
    variance_windows=(16, 32),
):
    """Extract a multi-scale, spectral-spatial feature stack from a 2D image.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Input field (e.g., a rod-mask or grayscale slice), any real dtype.
    density_sigmas : sequence of float
        Gaussian smoothing scales for local density features.
    st_sigma : float
        Structure-tensor smoothing scale (orientation + coherence).
    gabor_freqs : sequence of float
        Gabor spatial frequencies.
    gabor_thetas : sequence of float or None
        Gabor orientations (radians). Defaults to 6 orientations over [0, pi).
    variance_windows : sequence of int
        Window sizes for local-variance (texture) features.

    Returns
    -------
    feature_stack : ndarray, shape (H, W, N)
        Per-pixel feature vectors.
    feature_names : list of str
        Human-readable name for each of the N feature channels.
    """
    image = np.asarray(image, dtype=np.float64)
    if gabor_thetas is None:
        gabor_thetas = np.linspace(0, np.pi, 6, endpoint=False)

    features = []
    feature_names = []

    # Multi-scale local density.
    for sigma in density_sigmas:
        features.append(gaussian_filter(image, sigma=sigma))
        feature_names.append(f"density_s{sigma}")

    # Structure tensor -> coherence (anisotropy) + orientation (sin/cos encoded).
    st_elements = structure_tensor(image, sigma=st_sigma)
    lam1, lam2 = structure_tensor_eigenvalues(st_elements)
    coherence = np.where((lam1 + lam2) > 1e-10, (lam1 - lam2) / (lam1 + lam2), 0.0)
    features.append(coherence)
    feature_names.append("coherence")

    axx, axy, ayy = st_elements
    orientation = 0.5 * np.arctan2(2 * axy, ayy - axx)
    features.append(np.sin(2 * orientation))
    feature_names.append("orient_sin2")
    features.append(np.cos(2 * orientation))
    feature_names.append("orient_cos2")

    # Gabor filter bank -> periodic-texture magnitudes at each freq/orientation.
    for freq in gabor_freqs:
        for theta in gabor_thetas:
            kernel = gabor_kernel(freq, theta=theta, sigma_x=3, sigma_y=3)
            resp_real = fftconvolve(image, kernel.real, mode="same")
            resp_imag = fftconvolve(image, kernel.imag, mode="same")
            features.append(np.sqrt(resp_real**2 + resp_imag**2))
            feature_names.append(f"gabor_f{freq:.2f}_t{np.degrees(theta):.0f}")

    # Local variance (texture roughness) at a couple of window sizes.
    for win in variance_windows:
        local_mean = uniform_filter(image, size=win)
        local_sq_mean = uniform_filter(image**2, size=win)
        local_var = np.clip(local_sq_mean - local_mean**2, 0, None)
        features.append(local_var)
        feature_names.append(f"variance_w{win}")

    return np.stack(features, axis=-1), feature_names


def _normalize_per_feature(flat_features):
    """Min-max normalize each feature column to [0, 1]."""
    feat_min = flat_features.min(axis=0)
    feat_max = flat_features.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 1e-10, feat_max - feat_min, 1.0)
    return (flat_features - feat_min) / feat_range


def som_segment(
    image,
    n_clusters=4,
    som_x=12,
    som_y=12,
    iterations=50_000,
    sample_frac=0.05,
    seed=42,
    feature_kwargs=None,
):
    """Label-free segmentation of a 2D image into ``n_clusters`` regions.

    Extracts spectral-spatial features, trains a SOM on a pixel sample, maps every
    pixel to its best-matching unit, then k-means-clusters the SOM nodes into regions.

    Parameters
    ----------
    image : ndarray, shape (H, W)
    n_clusters : int
        Number of output regions.
    som_x, som_y : int
        SOM grid dimensions.
    iterations : int
        SOM training iterations.
    sample_frac : float
        Fraction of pixels used to train the SOM.
    seed : int
        Random seed for reproducibility.
    feature_kwargs : dict or None
        Extra keyword arguments forwarded to :func:`extract_features`.

    Returns
    -------
    cluster_map : ndarray, shape (H, W), dtype int
        Per-pixel region label in ``[0, n_clusters)``.
    """
    # Imported here so the module stays import-light if SOM is unused.
    from minisom import MiniSom
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]

    feature_stack, _ = extract_features(image, **(feature_kwargs or {}))
    n_feat = feature_stack.shape[-1]
    flat_norm = _normalize_per_feature(feature_stack.reshape(-1, n_feat))

    # Train the SOM on a random pixel sample.
    n_sample = max(1, int(flat_norm.shape[0] * sample_frac))
    sample_idx = rng.choice(flat_norm.shape[0], size=n_sample, replace=False)
    som = MiniSom(
        som_x,
        som_y,
        n_feat,
        sigma=2.0,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=seed,
    )
    som.pca_weights_init(flat_norm[sample_idx])
    som.train(flat_norm[sample_idx], iterations, verbose=False)

    # Map each pixel to its best-matching unit (flattened node index).
    #
    # Vectorized on purpose: the naive `[som.winner(x) for x in flat_norm]`
    # makes one Python-level MiniSom call per pixel — 262,144 calls for a
    # 512x512 slice, measured at ~59% of som_segment's total runtime
    # (cProfile, 2026-07-15). Computing all pixel-to-node distances as
    # chunked BLAS matrix products gives the SAME argmin: squared Euclidean
    # distance via (a-b)^2 = a^2 - 2ab + b^2 preserves the ordering (sqrt is
    # monotonic), and row-major argmin over the flattened (som_x*som_y) axis
    # matches MiniSom.winner's unravel_index tie-breaking, so node index
    # x*som_y + y is reproduced exactly.
    weights_flat = som.get_weights().reshape(-1, n_feat)
    w_sq = (weights_flat**2).sum(axis=1)[None, :]
    bmu_flat = np.empty(flat_norm.shape[0], dtype=np.int64)
    chunk = 65536  # ~64k x 144 float64 distance block ~= 75 MB, RAM-friendly
    for i in range(0, flat_norm.shape[0], chunk):
        block = flat_norm[i : i + chunk]
        d2 = (block**2).sum(axis=1)[:, None] - 2.0 * (block @ weights_flat.T) + w_sq
        bmu_flat[i : i + chunk] = np.argmin(d2, axis=1)

    # Cluster the SOM nodes by their weight vectors (weights_flat, computed
    # above for the BMU mapping), then map back to the image.
    node_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        weights_flat
    )
    return node_labels[bmu_flat].reshape(h, w)
