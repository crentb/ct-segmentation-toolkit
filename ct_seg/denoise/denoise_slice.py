"""
denoise_slice.py — Denoise a single CT slice with a trained 2.5D N2I model.

Loads the best edge-score checkpoint produced by training (main.py) and denoises
ONE axial slice of the full reconstruction, saving it as a TIFF in a
`denoised_slices/` folder next to the reconstructions. This is a fast way to
spot-check denoising quality without processing the whole volume.

Because the model is 2.5D, the two slices above and below the requested slice are
also loaded and stacked as input channels; this is handled internally and hidden
from the caller. Normalization uses the mean/std that training wrote back into the
config file.

Inputs  : a YAML config (with mean4norm/std4norm filled in by training) and the
          index of the slice to denoise.
Outputs : <reconstruction_dir>/denoised_slices/<slice>.tiff

Usage:
    python denoise_slice.py -gpus=0 -config=/path/to/config.yaml -slice_number=500
    (normally launched via denoise_slice.sh)

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import tifffile
import torch
import yaml

from ct_seg.denoise import tiffs
from ct_seg.denoise.model import unet_ns_gn

warnings.filterwarnings("ignore")


def prepare_stack(tiff_images, slice_number, total_number_of_images):
    """
    This function prepares the stack of images to be used for denoised similar to the data.py approach

    params:
        -tiff_images (list) list of tiff images
        -slice_number (int) slice to process
        -total_number_of_image (int) total number of slices
    """
    images_to_process = []
    # Build the 5-slice input stack [s-2, s-1, s, s+1, s+2]. Near the volume edges
    # there are not two neighbours on one side, so the missing slices are clamped to
    # the nearest valid slice (replicate padding) to keep the channel count at 5.
    if slice_number == 0:
        prev_prev_slice = tiff_images[slice_number]
        prev_slice = tiff_images[slice_number]
        inp_slice = tiff_images[slice_number]
        next_slice = tiff_images[slice_number + 1]
        next_next_slice = tiff_images[slice_number + 2]
    elif slice_number == 1:
        prev_prev_slice = tiff_images[slice_number - 1]
        prev_slice = tiff_images[slice_number - 1]
        inp_slice = tiff_images[slice_number]
        next_slice = tiff_images[slice_number + 1]
        next_next_slice = tiff_images[slice_number + 2]
    elif slice_number == total_number_of_images - 1:
        prev_prev_slice = tiff_images[slice_number - 2]
        prev_slice = tiff_images[slice_number - 1]
        inp_slice = tiff_images[slice_number]
        next_slice = tiff_images[slice_number]
        next_next_slice = tiff_images[slice_number]
    elif slice_number == total_number_of_images - 2:
        prev_prev_slice = tiff_images[slice_number - 2]
        prev_slice = tiff_images[slice_number - 1]
        inp_slice = tiff_images[slice_number]
        next_slice = tiff_images[slice_number + 1]
        next_next_slice = tiff_images[slice_number + 1]
    else:
        prev_prev_slice = tiff_images[slice_number - 2]
        prev_slice = tiff_images[slice_number - 1]
        inp_slice = tiff_images[slice_number]
        next_slice = tiff_images[slice_number + 1]
        next_next_slice = tiff_images[slice_number + 2]

    images_to_process = [prev_prev_slice, prev_slice, inp_slice, next_slice, next_next_slice]
    return images_to_process


def main(args):
    """Load the trained model and denoise the single requested slice."""

    # Read the YAML file
    with open(args.config, "r") as file:
        params = yaml.safe_load(file)

    # create directory for denoised slices
    out_path = params["dataset"]["directory_to_reconstructions"] + "/" "denoised_slices"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

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
    checkpoint = torch.load(path_to_mdl, map_location=torch.device("cpu"))
    model = unet_ns_gn(ich=5, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)

    # get data
    print(f"\nLoading in slice {args.slice_number}.\n")

    # path to data
    full_recon_path = (
        params["dataset"]["directory_to_reconstructions"]
        + "/"
        + params["dataset"]["full_recon_name"]
    )

    # collect tiff files
    tiffs_collection = tiffs.glob(full_recon_path)
    # pull out the requested images to process
    list_of_images_to_process = prepare_stack(
        tiff_images=tiffs_collection,
        slice_number=args.slice_number,
        total_number_of_images=len(tiffs_collection),
    )
    # load in just the images to process
    images = tiffs.load_stack(list_of_images_to_process)[np.newaxis]
    images = torch.from_numpy(images).to(dev)

    # normalize image stack
    mean4norm = params["dataset"]["mean4norm"]
    std4norm = params["dataset"]["std4norm"]
    # mean4norm = images.mean().item()
    # std4norm = images.std().item()
    images = (images - mean4norm) / std4norm

    # denoise image
    with torch.no_grad():
        denoised = model(images).cpu().squeeze().numpy()

    # rescale back to original values
    denoised = denoised * std4norm + mean4norm

    # save denoised slice
    tifffile.imwrite(f"{out_path}/{args.slice_number:05d}.tiff", denoised)

    # save2img(images[0, 2].cpu().numpy(), f'_original_{args.slice_number:05d}.png')
    # save2img(denoised, f'_denoised_{args.slice_number:05d}.png')
    # bash denoise_slice.sh FOAM/config.yaml 500


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Denoise CT slice with 2.5D N2I")
    parser.add_argument("-gpus", type=str, default="0", help="list of visiable GPUs")
    parser.add_argument("-slice_number", type=int, required=True, help="test image")
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

    main(args)
