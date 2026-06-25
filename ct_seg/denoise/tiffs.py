"""
tiffs.py — TIFF stack input/output helpers for CT volumes.

Utilities for reading and writing the TIFF image stacks that make up a CT
reconstruction: natural (human) sorting of filenames, loading a directory of TIFFs
into a NumPy volume, globbing a directory for TIFFs, saving a volume back out as
numbered TIFFs, and loading a stack as a sinogram.

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import re
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


def natural_sorted(paths):
    """Sort paths/strings in natural (human) order so that e.g. img2 precedes img10."""

    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]

    return sorted(paths, key=key)


# We use the following function to load a stack of images:
def load_stack(paths, binning=1, use_tqdm=True):
    """Load a stack of tiff files.

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    paths = list(paths)

    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((len(paths), *img0.shape), dtype=dtype)

    for i in tqdm(range(len(paths))):
        imgs[i] = tifffile.imread(str(paths[i]))[::binning, ::binning]
    return imgs


def glob(dir_path):
    """Expand path to list of all tiffs in directory

    :param dir_path: directory
    :returns:
    :rtype:

    """
    dir_path = Path(dir_path).expanduser().resolve()
    return natural_sorted(dir_path.glob("*.tif*"))


def save_stack(path, stack, offset=0, exist_ok=True, parents=False):
    """Write a 3-D volume to `path` as zero-padded numbered TIFFs (NNNNN.tiff).

    `offset` shifts the starting file number so a denoised sub-range keeps the same
    indices as the original slices.
    """
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    for i in tqdm(range(stack.shape[0])):
        opath = path / f"{i+offset:05d}.tiff"
        tifffile.imwrite(str(opath), stack[i])


def load_sino(paths, binning=1, dtype=None, flip_y=False):
    """Load a stack of tiff files into a sinogram

    :param paths: paths to tiff files
    :param binning: whether angles and projection images should be binned.
    :returns: an np.array containing the values in the tiff files
    :rtype: np.array

    """
    # Read first image for shape and dtype information
    paths = list(paths)
    # print(paths[0])
    # sys.exit()
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    if dtype is None:
        dtype = img0.dtype
    # Create empty numpy array to hold result
    imgs = np.empty((img0.shape[0], len(paths), img0.shape[1]), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        # Angles in the middle, "up" in front, "right" at the back.
        if flip_y:
            # Flip in the vertical direction
            imgs[:, i, :] = tifffile.imread(str(p))[::-binning, ::binning]
        else:
            imgs[:, i, :] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs
