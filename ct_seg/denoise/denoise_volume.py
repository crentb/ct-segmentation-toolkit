"""
denoise_volume.py — Denoise a full CT volume (or a slice range) with 2.5D N2I.

Loads the best edge-score checkpoint from training (main.py) and denoises every
slice of the full reconstruction, writing the result as a TIFF stack into a
`denoised_volume/` folder next to the reconstructions. A contiguous slice range
may be given to denoise only part of the volume (useful for quick evaluation); if
omitted, the whole volume is processed.

To use the GPU efficiently the script auto-tunes the inference batch size for the
available GPU memory, model size, and image dimensions (see
InferenceBatchSizeOptimizer in data.py). Normalization uses the mean/std that
training wrote back into the config file.

Inputs  : a YAML config (with mean4norm/std4norm filled in by training) and an
          optional [start_slice, end_slice] range.
Outputs : <reconstruction_dir>/denoised_volume/<index>.tiff stack
          (this folder is deleted and recreated on every run).

Usage:
    python denoise_volume.py -gpus=0 -config=/path/to/config.yaml -start_slice=500 -end_slice=600
    (normally launched via denoise_volume.sh; omit the range to denoise everything)

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import argparse
import logging
import os
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_seg.denoise import tiffs
from ct_seg.denoise.data import InferenceBatchSizeOptimizer, TomoDatasetInfer
from ct_seg.denoise.model import unet_ns_gn

warnings.filterwarnings("ignore")


def main(args):
    """Load the trained model and denoise the configured volume (or slice range)."""

    # Read the YAML file
    with open(args.config, "r") as file:
        params = yaml.safe_load(file)

    # setup output directory
    output_dir = params["dataset"]["directory_to_reconstructions"] + "/" "denoised_volume"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # setup cuda device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in model
    path_to_mdl = (
        params["dataset"]["directory_to_reconstructions"]
        + "/"
        + "TrainOutput"
        + "/"
        + "best_edge_model.pth"
    )
    # weights_only=True: restrict deserialization to tensors/containers.
    # torch.load unpickles arbitrary Python by default, so a malicious
    # checkpoint file would execute code on load (CWE-502 / bandit B614).
    checkpoint = torch.load(path_to_mdl, map_location=torch.device("cpu"), weights_only=True)
    model = unet_ns_gn(ich=5, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev).eval()

    print("\nLoading data into CPU memory, it will take a while ... ...")

    # load in data
    ds_test = TomoDatasetInfer(
        params=params, start_slice=args.start_slice, end_slice=args.end_slice
    )
    print(
        f"\nLoaded in {ds_test.reconstruction.shape[0]} slices of size {ds_test.reconstruction.shape[1]}x{ds_test.reconstruction.shape[2]}.\n"
    )

    # determine optimal batch size given GPU system memory, model size, and image size
    optimal_batch_size = InferenceBatchSizeOptimizer(
        model=model,
        input_shape=ds_test.reconstruction[0].shape,
        device=dev,
        max_batch_size=16,
        precision="fp32",
    )
    stats = optimal_batch_size.profile()
    mbsz = stats["optimal_batch_size"]

    dl_test = DataLoader(
        dataset=ds_test,
        batch_size=mbsz,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        prefetch_factor=6,
        pin_memory=True,
    )

    # initialize empty array for denoised volume
    preds = np.zeros_like(dl_test.dataset.reconstruction)
    insert_cnt = 0
    # denoise volume
    print("Processing data ...")
    with torch.no_grad():
        for X in tqdm(dl_test):
            output = model(X.to(dev)).cpu().squeeze(dim=1).numpy()

            preds[insert_cnt : (insert_cnt + X.shape[0])] = output
            insert_cnt += X.shape[0]

    # rescale volume
    preds = preds * params["dataset"]["std4norm"] + params["dataset"]["mean4norm"]

    # save volume
    print("\nSaving data ...")
    if len(args.start_slice) == 0:
        tiffs.save_stack(output_dir, preds)
    else:
        # Save the processed sub volume with the right tiff number
        tiffs.save_stack(output_dir, preds, offset=int(args.start_slice))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference for 2.5D Noise2Inverse")
    parser.add_argument("-gpus", type=str, default="0", help="list of visiable GPUs")
    parser.add_argument("-start_slice", type=str, default=0, help="minibatch size")
    parser.add_argument("-end_slice", type=str, default=None, help="minibatch size")
    parser.add_argument("-config", type=str, required=True, help="path to config yaml file")
    parser.add_argument(
        "-verbose", type=int, default=1, help="1:print to terminal; 0: redirect to file"
    )

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print("Unrecognized argument(s): \n%s \nProgram exiting ... ... " % "\n".join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib").setLevel(level=logging.CRITICAL)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    start_time = time.time()
    main(args)
    inference_time = time.time() - start_time
    print(f"\nInference Time: {inference_time:.4f} seconds\n")
