"""
Napari-based Labeling Tool for CT Segmentation
===============================================
Opens a tiff stack in Napari with a labels layer for manual annotation.
Supports loading existing masks, painting labels, and saving.

Usage:
    python label_tool.py --images /path/to/tiffs
    python label_tool.py --images /path/to/tiffs --masks /path/to/existing_masks
    python label_tool.py --images /path/to/tiffs --num_classes 4 --slice_range 500 600

Controls in Napari:
    - Select the "Labels" layer in the layer list
    - Use the paint brush (press B) to draw labels
    - Use the eraser (press E) to erase
    - Number keys (1, 2, 3, ...) select the label class
    - Scroll through slices with the slider at the bottom
    - Press Ctrl+S or use File > Save to save labels

When you close Napari, masks are automatically saved.

Requirements:
    pip install napari[all]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile


def natural_sorted(paths):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split(r"([0-9]+)", str(x))]

    return sorted(paths, key=key)


def load_stack(dir_path, start=None, end=None):
    dir_path = Path(dir_path)
    paths = natural_sorted(list(dir_path.glob("*.tif*")))
    if start is not None and end is not None:
        paths = paths[start:end]
    if not paths:
        raise FileNotFoundError(f"No tiff files in {dir_path}")
    from tqdm import tqdm

    imgs = np.stack([tifffile.imread(str(p)) for p in tqdm(paths, desc="Loading")])
    return imgs, paths


def save_masks(masks, out_dir, offset=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    for i in tqdm(range(masks.shape[0]), desc="Saving masks"):
        tifffile.imwrite(str(out_dir / f"{i + offset:05d}.tiff"), masks[i].astype(np.uint8))
    print(f"Saved {masks.shape[0]} masks to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Napari Labeling Tool for CT Segmentation")
    parser.add_argument("--images", required=True, help="Directory of image tiffs")
    parser.add_argument("--masks", default=None, help="Directory of existing mask tiffs (optional)")
    parser.add_argument(
        "--output", default=None, help="Output directory for masks (default: images_masks)"
    )
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes (default: 4)")
    parser.add_argument("--slice_range", nargs=2, type=int, default=None, metavar=("START", "END"))
    args = parser.parse_args()

    try:
        import napari
    except ImportError:
        print("Napari is required for the labeling tool.")
        print("Install with: pip install 'napari[all]'")
        sys.exit(1)

    # Setup output
    if args.output is None:
        args.output = str(Path(args.images).parent / (Path(args.images).name + "_masks"))

    # Load images
    start = args.slice_range[0] if args.slice_range else None
    end = args.slice_range[1] if args.slice_range else None
    images, paths = load_stack(args.images, start=start, end=end)
    print(f"Loaded {images.shape[0]} images of size {images.shape[1]}x{images.shape[2]}")

    # Load or create masks
    if args.masks:
        masks, _ = load_stack(args.masks, start=start, end=end)
        print(f"Loaded existing masks with classes: {np.unique(masks)}")
    else:
        masks = np.zeros(images.shape, dtype=np.uint8)
        print("Created empty mask volume")

    # Normalize images for display
    p2, p98 = np.percentile(images, (2, 98))
    display_imgs = np.clip(images, p2, p98)

    # Color map for labels
    label_colors = {
        0: [0, 0, 0, 0],  # transparent background
        1: [1, 0.2, 0.2, 0.6],  # red
        2: [0.2, 1, 0.2, 0.6],  # green
        3: [0.2, 0.4, 1, 0.6],  # blue
        4: [1, 1, 0.2, 0.6],  # yellow
        5: [1, 0.2, 1, 0.6],  # magenta
        6: [0.2, 1, 1, 0.6],  # cyan
    }

    # Launch Napari
    viewer = napari.Viewer(title="CT Segmentation Labeling Tool")
    viewer.add_image(display_imgs, name="CT Image", colormap="gray")
    labels_layer = viewer.add_labels(masks, name="Labels", color=label_colors)

    # Set brush size
    labels_layer.brush_size = 10

    print(f"\n{'='*50}")
    print("Napari Labeling Tool")
    print(f"{'='*50}")
    print(f"Classes: {args.num_classes}")
    print("Controls:")
    print("  B - Paint brush")
    print("  E - Eraser")
    print("  1,2,3... - Select label class")
    print("  Scroll - Navigate slices")
    print(f"\nClose Napari to save masks to: {args.output}")
    print(f"{'='*50}\n")

    napari.run()

    # Save on close
    final_masks = labels_layer.data
    offset = start if start else 0
    save_masks(final_masks, args.output, offset=offset)


if __name__ == "__main__":
    main()
