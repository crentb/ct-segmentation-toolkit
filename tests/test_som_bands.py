"""Label-free SOM segmentation smoke tests (small synthetic mask, tiny SOM)."""

import numpy as np

from ct_seg.som_bands import extract_features, som_segment

_FEAT_KW = dict(
    density_sigmas=(4, 8),
    gabor_freqs=(0.1,),
    gabor_thetas=np.linspace(0, np.pi, 3, endpoint=False),
    variance_windows=(8,),
)


def _synthetic_mask():
    rng = np.random.default_rng(0)
    return (rng.random((48, 48)) > 0.5).astype(float)


def test_extract_features_shape():
    feats, names = extract_features(_synthetic_mask(), **_FEAT_KW)
    # 2 density + 1 coherence + 2 orientation + (1 freq * 3 theta) + 1 variance = 9
    assert feats.shape[:2] == (48, 48)
    assert feats.shape[2] == len(names) == 9


def test_som_segment_labels():
    cmap = som_segment(
        _synthetic_mask(),
        n_clusters=3,
        som_x=4,
        som_y=4,
        iterations=100,
        sample_frac=0.3,
        feature_kwargs=_FEAT_KW,
    )
    assert cmap.shape == (48, 48)
    assert set(np.unique(cmap)).issubset(set(range(3)))
