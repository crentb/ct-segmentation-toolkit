"""
data.py — Datasets and batch-size tuning for 2.5D Noise2Inverse.

Provides the PyTorch Dataset classes and helpers that feed the training and
inference pipelines:
    * get_5_adjacent_slices       - stack 5 adjacent slices (the 2.5D input), with
                                    edge slices replicated at the volume boundaries.
    * save_normalization_value    - persist the training mean/std back into the
                                    config YAML so inference normalizes identically.
    * TomoDatasetTrain            - load the even/odd sub-reconstructions, normalize
                                    them, and serve augmented 2.5D patch pairs.
    * TomoDatasetInfer            - load the full reconstruction for inference,
                                    normalized with the stored training statistics.
    * InferenceBatchSizeOptimizer - binary-search the largest batch size that fits
                                    in GPU memory to avoid out-of-memory errors.

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import logging

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from ct_seg.denoise import tiffs


def get_5_adjacent_slices(idx, data):
    """
    This function get 5 slices to be used for training.

    params:
        -idx (int) index for the image within the volume
        -data (numpy array) reconstructed volume
    """

    if idx == 0:
        prev_prev_slice = data[idx]
        prev_slice = data[idx]
        inp_slice = data[idx]
        next_slice = data[idx + 1]
        next_next_slice = data[idx + 2]
    elif idx == 1:
        prev_prev_slice = data[idx - 1]
        prev_slice = data[idx - 1]
        inp_slice = data[idx]
        next_slice = data[idx + 1]
        next_next_slice = data[idx + 2]
    elif idx == data.shape[0] - 1:
        prev_prev_slice = data[idx - 2]
        prev_slice = data[idx - 1]
        inp_slice = data[idx]
        next_slice = data[idx]
        next_next_slice = data[idx]
    elif idx == data.shape[0] - 2:
        prev_prev_slice = data[idx - 2]
        prev_slice = data[idx - 1]
        inp_slice = data[idx]
        next_slice = data[idx + 1]
        next_next_slice = data[idx + 1]
    else:
        prev_prev_slice = data[idx - 2]
        prev_slice = data[idx - 1]
        inp_slice = data[idx]
        next_slice = data[idx + 1]
        next_next_slice = data[idx + 2]

    adj_slices = np.concatenate(
        [
            prev_prev_slice[..., np.newaxis],
            prev_slice[..., np.newaxis],
            inp_slice[..., np.newaxis],
            next_slice[..., np.newaxis],
            next_next_slice[..., np.newaxis],
        ],
        axis=-1,
    )
    return adj_slices


def save_normalization_value(config_file, mean, std):
    """
    This functin saves the mean and standard deviation back to the yaml file which is then used during inferencing
    params
        -config_file (str) location of the config file
        -mean (float) mean used for normalization
        -std (float) standard deviation used for normalization
    """
    # safe load
    try:
        with open(config_file, "r") as file:
            data = yaml.safe_load(file)  # Use safe_load for security
    except FileNotFoundError:
        data = {}  # If the file doesn't exist, start with an empty dictionary
    except yaml.YAMLError as exc:
        print(f"Error loading YAML file: {exc}")
        data = {}  # Handle parsing errors

    data["dataset"]["mean4norm"] = float(mean)
    data["dataset"]["std4norm"] = float(std)

    # write the data back to the yaml file
    with open(config_file, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)


class TomoDatasetTrain(Dataset):
    """
    Training class for 2.5D N2I
        -This class loads in two lists corresponding to the two sub reconstructions (saved as .tiffs) and normalizes them
    params
        -params (obj) yaml object, essentially a dictionary
        -config_file (str) location of the configuration file
    """

    def __init__(self, params, config_file):
        super(TomoDatasetTrain, self).__init__()
        dataset_params = params["dataset"]
        train_params = params["train"]

        self.psz = train_params["psz"]

        # specify augmentations for training
        self.augmentations = A.Compose(
            [
                A.SquareSymmetry(p=1.0),
                # A.RandomGridShuffle(grid=[3,3], p=.5),
            ],
            additional_targets={"split1": "image"},
        )

        # load in tiff images for training

        # location to sub reconstructions
        recon_0_path = (
            dataset_params["directory_to_reconstructions"] + "/" + dataset_params["sub_recon_name0"]
        )
        recon_1_path = (
            dataset_params["directory_to_reconstructions"] + "/" + dataset_params["sub_recon_name1"]
        )

        # collect tiff files and optionally slice to a subset (avoids OOM on <128GB machines)
        tiffs_collection_0 = tiffs.glob(recon_0_path)
        tiffs_collection_1 = tiffs.glob(recon_1_path)

        sl_start = dataset_params.get("train_slice_start", None)
        sl_end = dataset_params.get("train_slice_end", None)
        if sl_start is not None and sl_end is not None:
            tiffs_collection_0 = tiffs_collection_0[int(sl_start) : int(sl_end)]
            tiffs_collection_1 = tiffs_collection_1[int(sl_start) : int(sl_end)]
            logging.info(
                f"\nTraining on slice subset [{sl_start}:{sl_end}] ({len(tiffs_collection_0)} slices)"
            )

        self.split0 = tiffs.load_stack(tiffs_collection_0)
        self.split1 = tiffs.load_stack(tiffs_collection_1)

        # convert any nans to zero
        self.split0 = np.nan_to_num(self.split0)
        self.split1 = np.nan_to_num(self.split1)

        # normalize the data
        split0_mean = self.split0.mean()
        split0_std = self.split0.std()
        self.split0 = ((self.split0 - split0_mean) / (split0_std)).astype(np.float32)
        logging.info(f"\nSplit 0 is scaled with calculated mean: {split0_mean}, std: {split0_std}")

        split1_mean = self.split1.mean()
        split1_std = self.split1.std()
        self.split1 = ((self.split1 - split1_mean) / (split1_std)).astype(np.float32)
        logging.info(f"\nSplit 1 is scaled with calculated mean: {split1_mean}, std: {split1_std}")

        # write mean and std to yaml file
        logging.info(
            "Saving training mean and standard deviation to configuration file to be used for inferencing"
        )
        save_normalization_value(config_file=config_file, mean=split0_mean, std=split0_std)

        self.samples = self.__len__()

    def __getitem__(self, idx):

        # get data using patch size
        xst = np.random.randint(0, self.split0[0].shape[-2] - self.psz)
        yst = np.random.randint(0, self.split0[0].shape[-1] - self.psz)

        # get stack of images
        view0 = get_5_adjacent_slices(
            idx, self.split0[:, xst : xst + self.psz, yst : yst + self.psz]
        )
        view1 = get_5_adjacent_slices(
            idx, self.split1[:, xst : xst + self.psz, yst : yst + self.psz]
        )

        # perform augmentations
        augmented = self.augmentations(image=view0, split1=view1)
        view0 = augmented["image"]
        view1 = augmented["split1"]

        return np.transpose(view0, axes=(2, 0, 1)), np.transpose(view1, axes=(2, 0, 1))

    def __len__(self):
        return self.split0.shape[0]


class TomoDatasetInfer(Dataset):
    """
    Inference class for 2.5D N2I
        -This class loads in a lists corresponding to the full reconstructions (saved as .tiffs) and normalizes them based on the training data
    params
        -params (obj) yaml object, essentially a dictionary
        -start_slice (int) start slice for processing a portion of the reconstruction
        -end_slice (int) end slice for processing a portion of the reconstruction
    """

    def __init__(self, params, start_slice, end_slice):
        super(TomoDatasetInfer, self).__init__()
        dataset_params = params["dataset"]

        # location to full reconstruction
        recon_path = (
            dataset_params["directory_to_reconstructions"] + "/" + dataset_params["full_recon_name"]
        )

        # process slice if specified
        if len(start_slice) == 0:
            tiffs_collection = tiffs.glob(recon_path)
        else:
            tiffs_collection = tiffs.glob(recon_path)[int(start_slice) : int(end_slice)]

        self.reconstruction = tiffs.load_stack(tiffs_collection)

        # convert any nans to zero
        self.reconstruction = np.nan_to_num(self.reconstruction)

        mean4norm = dataset_params["mean4norm"]
        std4norm = dataset_params["std4norm"]
        self.reconstruction = ((self.reconstruction - mean4norm) / (std4norm)).astype(np.float32)
        print(f"\nReconstruction is scaled with provided mean: {mean4norm}, std: {std4norm}")

        self.samples = self.__len__()

    def __getitem__(self, idx):

        # get stack of images
        inp = get_5_adjacent_slices(idx, self.reconstruction)

        return np.transpose(inp, axes=(2, 0, 1))

    def __len__(self):
        return self.reconstruction.shape[0]


class InferenceBatchSizeOptimizer:
    """
    Class for determining the optimal batch size to be used for inferencing
        -Differences in GPU memory (32GB V100 vs. 80GB A100), model size, and reconstructed image size can all influence
        how many images can be processed during inference. While we could process 1 image per batch, this is slow and wasteful.
        -This class helps determine the optimal size to be used
    params
        -model (obj) pytorch model to be used for inference
        -input_shape (tuple) size of the images to be denoised
        -device (obj) cuda device
        -max_batch_size (int) maximum batch size to check
        -precision (str) whether to use flaoting point 32 or amp
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 512,
        precision: str = "fp32",
    ):
        self.model = model.eval().to(device)
        self.input_shape = input_shape  # (C, H, W) or (C, D, H, W) for 3D
        self.device = device
        self.max_batch_size = max_batch_size
        self.precision = precision.lower()

        if self.precision not in ["fp32", "amp"]:
            raise ValueError("precision must be either 'fp32' or 'amp'")

        self.cached_optimal_batch_size = None

    def get_available_memory(self):
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.device.index)[0] / 1024**2  # MB

    def estimate_peak_memory(self, batch_size: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        dummy_input = torch.randn((batch_size, 5, *self.input_shape), device=self.device)
        try:
            with torch.no_grad():
                if self.precision == "amp":
                    with torch.autocast(device_type="cuda"):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        except RuntimeError as e:
            raise RuntimeError(f"OOM or other error at batch size {batch_size}: {e}")

        peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        return peak_mem

    def find_optimal_batch_size(self) -> int:
        if self.cached_optimal_batch_size is not None:
            return self.cached_optimal_batch_size

        low, high = 1, self.max_batch_size
        best = 1

        # Binary search for the largest batch size whose forward pass fits in memory:
        # a size that succeeds -> try larger; a size that OOMs (RuntimeError) -> try smaller.
        while low <= high:
            mid = (low + high) // 2
            try:
                _ = self.estimate_peak_memory(mid)
                best = mid
                low = mid + 1
            except RuntimeError:
                high = mid - 1

        self.cached_optimal_batch_size = best
        return best

    def profile(self):
        batch_size = self.find_optimal_batch_size()
        peak_memory = self.estimate_peak_memory(batch_size)
        available_memory = self.get_available_memory()

        # print(f"Optimal batch size: {batch_size}")
        # print(f"Peak memory used: {peak_memory:.2f} MB")
        # print(f"Available GPU memory: {available_memory:.2f} MB")

        return {
            "optimal_batch_size": batch_size,
            "peak_memory_used_MB": peak_memory,
            "available_memory_MB": available_memory,
        }
