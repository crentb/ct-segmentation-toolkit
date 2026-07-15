"""
CT Image Stack Segmentation
============================
Supports both unsupervised and supervised segmentation of tiff stacks.

Unsupervised methods:
    python segment.py --input /path/to/tiffs --method otsu --num_classes 4
    python segment.py --input /path/to/tiffs --method kmeans --num_classes 3
    python segment.py --input /path/to/tiffs --method gmm --num_classes 4

Supervised (U-Net):
    python segment.py --input /path/to/tiffs --method unet --model /path/to/model.pth --num_classes 4

Options:
    --slice_range 500 600    Process only slices 500-600
    --enhance                Apply contrast enhancement before segmentation
    --save_overlay           Save overlay images alongside masks
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Tiff I/O (standalone, no dependency on N2I tiffs.py)
# =============================================================================


def natural_sorted(paths):
    import re

    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split(r"([0-9]+)", str(x))]

    return sorted(paths, key=key)


def load_image_stack(dir_path, start=None, end=None):
    """Load a directory of image files (tiff, png, jpg) into a 3D grayscale numpy array."""
    from PIL import Image

    dir_path = Path(dir_path)
    # Support tiff, png, jpg
    paths = []
    for ext in ["*.tif*", "*.png", "*.jpg", "*.jpeg"]:
        paths.extend(dir_path.glob(ext))
    paths = natural_sorted(list(set(paths)))

    if start is not None and end is not None:
        paths = paths[start:end]
    if len(paths) == 0:
        raise FileNotFoundError(f"No image files found in {dir_path}")

    # Read first image to determine shape
    img0 = np.array(Image.open(str(paths[0])).convert("L"))  # convert to grayscale
    stack = np.empty((len(paths), *img0.shape), dtype=img0.dtype)
    for i, p in enumerate(tqdm(paths, desc="Loading images")):
        stack[i] = np.array(Image.open(str(p)).convert("L"))
    return stack, paths


def save_tiff_stack(stack, out_dir, num_classes, offset=0):
    """Save a 3D label array as RGB tiff files with vivid class colors."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vivid colors per class
    color_map = [
        [0, 0, 0],  # class 0: black
        [255, 0, 0],  # class 1: red
        [0, 255, 0],  # class 2: green
        [0, 100, 255],  # class 3: blue
        [255, 255, 0],  # class 4: yellow
        [255, 0, 255],  # class 5: magenta
        [0, 255, 255],  # class 6: cyan
        [255, 165, 0],  # class 7: orange
    ]

    for i in tqdm(range(stack.shape[0]), desc="Saving colored masks"):
        rgb = np.zeros((*stack[i].shape, 3), dtype=np.uint8)
        for c in range(min(num_classes, len(color_map))):
            rgb[stack[i] == c] = color_map[c]
        Image.fromarray(rgb).save(str(out_dir / f"{i + offset:05d}.png"))


# =============================================================================
# Preprocessing
# =============================================================================


def normalize_stack(stack):
    """Normalize stack to [0, 1] range using global min/max."""
    smin, smax = stack.min(), stack.max()
    if smax == smin:
        return np.zeros_like(stack, dtype=np.float32)
    return ((stack - smin) / (smax - smin)).astype(np.float32)


def enhance_contrast(stack, q_low=2, q_high=98):
    """Clip to percentile range and rescale to [0, 1]."""
    p_low = np.percentile(stack, q_low)
    p_high = np.percentile(stack, q_high)
    clipped = np.clip(stack, p_low, p_high)
    return normalize_stack(clipped)


# =============================================================================
# Unsupervised Segmentation Methods
# =============================================================================


def segment_otsu(stack, num_classes):
    """
    Multi-Otsu thresholding.
    Finds (num_classes - 1) thresholds that minimize intra-class variance.
    """
    from skimage.filters import threshold_multiotsu

    print(f"\nRunning Multi-Otsu segmentation with {num_classes} classes...")

    # Compute thresholds on a subsample for speed
    subsample = stack[:: max(1, len(stack) // 20)].ravel()
    subsample = subsample[:: max(1, len(subsample) // 5_000_000)]

    thresholds = threshold_multiotsu(subsample, classes=num_classes)
    print(f"Thresholds: {thresholds}")

    labels = np.digitize(stack, bins=thresholds).astype(np.uint8)
    return labels, {"thresholds": thresholds.tolist()}


def segment_kmeans(stack, num_classes):
    """
    K-Means clustering on pixel intensities.
    Clusters are sorted by centroid value so class 0 = darkest.
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"\nRunning K-Means segmentation with {num_classes} clusters...")

    # Flatten and subsample for fitting
    flat = stack.ravel().astype(np.float32)
    subsample_idx = np.random.choice(len(flat), size=min(2_000_000, len(flat)), replace=False)
    subsample = flat[subsample_idx].reshape(-1, 1)

    kmeans = MiniBatchKMeans(n_clusters=num_classes, batch_size=10000, random_state=42, n_init=3)
    kmeans.fit(subsample)

    # Sort clusters by centroid value (class 0 = darkest)
    sorted_idx = np.argsort(kmeans.cluster_centers_.ravel())
    label_map = np.zeros(num_classes, dtype=np.uint8)
    for new_label, old_label in enumerate(sorted_idx):
        label_map[old_label] = new_label

    # Predict all pixels
    print("Assigning labels to full volume...")
    labels = np.empty(stack.shape, dtype=np.uint8)
    for i in tqdm(range(stack.shape[0]), desc="Segmenting slices"):
        sl = stack[i].ravel().astype(np.float32).reshape(-1, 1)
        pred = kmeans.predict(sl)
        labels[i] = label_map[pred].reshape(stack[i].shape)

    centroids = kmeans.cluster_centers_.ravel()[sorted_idx]
    print(f"Sorted centroids: {centroids}")
    return labels, {"centroids": centroids.tolist()}


def segment_gmm(stack, num_classes):
    """
    Gaussian Mixture Model segmentation.
    Fits a GMM to the intensity histogram, then classifies each pixel
    by posterior probability. Components sorted by mean.
    """
    from sklearn.mixture import GaussianMixture

    print(f"\nRunning GMM segmentation with {num_classes} components...")

    # Subsample for fitting
    flat = stack.ravel().astype(np.float32)
    subsample_idx = np.random.choice(len(flat), size=min(2_000_000, len(flat)), replace=False)
    subsample = flat[subsample_idx].reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=num_classes, covariance_type="full", random_state=42, n_init=3, max_iter=200
    )
    gmm.fit(subsample)

    # Sort components by mean
    sorted_idx = np.argsort(gmm.means_.ravel())
    label_map = np.zeros(num_classes, dtype=np.uint8)
    for new_label, old_label in enumerate(sorted_idx):
        label_map[old_label] = new_label

    # Predict all pixels
    print("Assigning labels to full volume...")
    labels = np.empty(stack.shape, dtype=np.uint8)
    for i in tqdm(range(stack.shape[0]), desc="Segmenting slices"):
        sl = stack[i].ravel().astype(np.float32).reshape(-1, 1)
        pred = gmm.predict(sl)
        labels[i] = label_map[pred].reshape(stack[i].shape)

    means = gmm.means_.ravel()[sorted_idx]
    stds = np.sqrt(gmm.covariances_.ravel()[sorted_idx])
    print(f"Sorted means: {means}")
    print(f"Sorted stds:  {stds}")
    return labels, {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": gmm.weights_[sorted_idx].tolist(),
    }


def segment_unet(stack, model_path, num_classes, device="cuda"):
    """
    Supervised U-Net segmentation using a trained model.
    """
    import torch

    from ct_seg.model import UNetSegmentation

    print(f"\nRunning U-Net segmentation with {num_classes} classes...")
    print(f"Model: {model_path}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # Load model
    model = UNetSegmentation(in_channels=1, num_classes=num_classes, base_filters=32)
    # weights_only=True: restrict deserialization to tensors/containers.
    # torch.load unpickles arbitrary Python by default, so a malicious
    # checkpoint file would execute code on load (CWE-502 / bandit B614).
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(dev).eval()

    # Normalize
    stack_norm = normalize_stack(stack)

    # Segment slice by slice
    labels = np.empty(stack.shape, dtype=np.uint8)
    with torch.no_grad():
        for i in tqdm(range(stack.shape[0]), desc="Segmenting slices"):
            x = torch.from_numpy(stack_norm[i : i + 1][np.newaxis]).to(dev)  # [1, 1, H, W]
            pred = model.predict(x)  # [1, H, W]
            labels[i] = pred[0].cpu().numpy().astype(np.uint8)

    return labels, {}


# =============================================================================
# Overlay Visualization
# =============================================================================


def create_overlay(image, labels, num_classes, alpha=0.4):
    """Create an RGB overlay of segmentation on the original image."""
    # Color map for classes
    colors = [
        [0, 0, 0],  # class 0: black (background)
        [255, 50, 50],  # class 1: red
        [50, 255, 50],  # class 2: green
        [50, 100, 255],  # class 3: blue
        [255, 255, 50],  # class 4: yellow
        [255, 50, 255],  # class 5: magenta
        [50, 255, 255],  # class 6: cyan
        [255, 165, 0],  # class 7: orange
    ]

    # Normalize image to [0, 255]
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)

    # Create colored mask
    mask_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(colors))):
        mask_rgb[labels == c] = colors[c]

    # Blend
    overlay = (img_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay


def save_overlays(stack, labels, out_dir, num_classes, every_n=10, offset=0):
    """Save overlay images for every Nth slice."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, stack.shape[0], every_n):
        overlay = create_overlay(stack[i], labels[i], num_classes)
        Image.fromarray(overlay).save(str(out_dir / f"overlay_{i + offset:05d}.png"))
    print(f"Saved {stack.shape[0] // every_n} overlay images to {out_dir}")


# =============================================================================
# Statistics
# =============================================================================


def compute_statistics(labels, num_classes):
    """Compute volume fraction and voxel count per class."""
    total = labels.size
    print(f"\n{'='*50}")
    print("Segmentation Statistics")
    print(f"{'='*50}")
    print(f"{'Class':<10} {'Voxels':>15} {'Volume %':>12}")
    print(f"{'-'*37}")
    stats = {}
    for c in range(num_classes):
        count = int(np.sum(labels == c))
        frac = count / total * 100
        print(f"{c:<10} {count:>15,} {frac:>11.2f}%")
        stats[f"class_{c}"] = {"voxels": count, "volume_fraction": frac}
    print(f"{'-'*37}")
    print(f"{'Total':<10} {total:>15,} {'100.00%':>12}")
    return stats


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="CT Image Stack Segmentation")
    parser.add_argument("--input", required=True, help="Path to directory of tiff files")
    parser.add_argument(
        "--output", default=None, help="Output directory (default: input_segmented)"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["otsu", "kmeans", "gmm", "unet"],
        help="Segmentation method",
    )
    parser.add_argument(
        "--num_classes", type=int, default=4, help="Number of classes/phases (default: 4)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained U-Net model (for unet method)"
    )
    parser.add_argument(
        "--slice_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Process only a range of slices",
    )
    parser.add_argument(
        "--enhance", action="store_true", help="Apply contrast enhancement before segmentation"
    )
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay images")
    parser.add_argument(
        "--overlay_every", type=int, default=10, help="Save overlay every N slices (default: 10)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for U-Net (default: cuda)"
    )
    args = parser.parse_args()

    # Validate
    if args.method == "unet" and args.model is None:
        parser.error("--model is required for unet method")

    # Setup output directory
    if args.output is None:
        args.output = str(
            Path(args.input).parent / (Path(args.input).name + f"_segmented_{args.method}")
        )

    # Load data
    print(f"\n{'='*60}")
    print("CT Image Stack Segmentation")
    print(f"{'='*60}")
    print(f"Input:    {args.input}")
    print(f"Method:   {args.method}")
    print(f"Classes:  {args.num_classes}")
    print(f"Output:   {args.output}")

    start = args.slice_range[0] if args.slice_range else None
    end = args.slice_range[1] if args.slice_range else None
    stack, paths = load_image_stack(args.input, start=start, end=end)
    print(f"Loaded:   {stack.shape} ({stack.dtype})")

    # Preprocess
    if args.enhance:
        print("Applying contrast enhancement...")
        stack_proc = enhance_contrast(stack)
    else:
        stack_proc = normalize_stack(stack)

    # Segment
    if args.method == "otsu":
        labels, info = segment_otsu(stack_proc, args.num_classes)
    elif args.method == "kmeans":
        labels, info = segment_kmeans(stack_proc, args.num_classes)
    elif args.method == "gmm":
        labels, info = segment_gmm(stack_proc, args.num_classes)
    elif args.method == "unet":
        labels, info = segment_unet(stack, args.model, args.num_classes, args.device)

    # Statistics
    stats = compute_statistics(labels, args.num_classes)

    # Save segmented volume (colored PNGs + raw label tiffs for 3D viewer)
    offset = start if start else 0
    save_tiff_stack(labels, args.output, args.num_classes, offset=offset)

    # Also save raw labels as tiffs for viewer/3D rendering
    raw_label_dir = Path(args.output) / "raw_labels"
    raw_label_dir.mkdir(parents=True, exist_ok=True)
    for i in range(labels.shape[0]):
        tifffile.imwrite(str(raw_label_dir / f"{i + offset:05d}.tiff"), labels[i])

    # Save overlays
    if args.save_overlay:
        overlay_dir = str(Path(args.output) / "overlays")
        save_overlays(
            stack, labels, overlay_dir, args.num_classes, every_n=args.overlay_every, offset=offset
        )

    # Save metadata
    import json

    meta = {
        "method": args.method,
        "num_classes": args.num_classes,
        "input": args.input,
        "shape": list(stack.shape),
        "dtype": str(stack.dtype),
        "enhanced": args.enhance,
        "info": info,
        "statistics": stats,
    }
    meta_path = Path(args.output) / "segmentation_info.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    print(f"\n{'='*60}")
    print("Segmentation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
