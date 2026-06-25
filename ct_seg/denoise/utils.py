"""
utils.py — Small image-saving and command-line helper utilities.

Helpers shared across the project: write an array to a TIFF or 8-bit PNG preview
(`save2img`), save an RGB preview (`save2img_rgb`), rescale an array to uint8
(`scale2uint8`), and parse boolean command-line strings (`str2bool`).

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import argparse

import skimage.io
from matplotlib import pyplot as plt


def save2img_rgb(img_data, img_fn):
    """Save an array as an RGB PNG preview, sized to the data on a black background."""
    plt.figure(figsize=(img_data.shape[1] / 10.0, img_data.shape[0] / 10.0))
    plt.axes([0, 0, 1, 1])
    plt.imshow(
        img_data,
    )
    plt.axis("off")
    plt.savefig(img_fn, facecolor="black", edgecolor="black", dpi=10)
    plt.close()


def save2img(d_img, fn):
    """Save an array to disk.

    If `fn` ends in 'tiff' the raw values are written unchanged; otherwise the data
    is min-max scaled to 0-255 and saved as an 8-bit image (e.g. a PNG preview).
    """
    if fn[-4:] == "tiff":
        img_norm = d_img.copy()
    else:
        _min, _max = d_img.min(), d_img.max()
        if _max == _min:
            img_norm = d_img - _max
        else:
            img_norm = (d_img - _min) * 255.0 / (_max - _min)
        img_norm = img_norm.astype("uint8")
    skimage.io.imsave(fn, img_norm, check_contrast=False)


def scale2uint8(_img):
    """Min-max scale an array to the 0-255 uint8 range (a constant image maps to 0)."""
    _min, _max = _img.min(), _img.max()
    if _max == _min:
        _img_s = _img - _max
    else:
        _img_s = (_img - _min) * 255.0 / (_max - _min)
    _img_s = _img_s.astype("uint8")
    return _img_s


def str2bool(v):
    """Parse a truthy/falsy command-line string into a bool (for use with argparse)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
