"""
3D Volume Viewer for CT Segmentation Results
=============================================
Two viewing modes:
    1. Orthogonal slice viewer (Napari) - interactive XY/XZ/YZ navigation
    2. 3D volume rendering (PyVista) - rotatable 3D view of segmented phases

Usage:
    # Orthogonal viewer (Napari)
    python viewer.py --images /path/to/tiffs --mode ortho
    python viewer.py --images /path/to/tiffs --labels /path/to/masks --mode ortho

    # 3D volume rendering (PyVista)
    python viewer.py --labels /path/to/masks --mode 3d --num_classes 4
    python viewer.py --images /path/to/tiffs --labels /path/to/masks --mode 3d --num_classes 4

    # Both
    python viewer.py --images /path/to/tiffs --labels /path/to/masks --mode both

Options:
    --slice_range 500 600    Load only a subset of slices
    --downsample 2           Downsample volume for faster 3D rendering
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


def load_stack(dir_path, start=None, end=None, downsample=1):
    from PIL import Image

    dir_path = Path(dir_path)
    paths = []
    for ext in ["*.tif*", "*.png", "*.jpg", "*.jpeg"]:
        paths.extend(dir_path.glob(ext))
    paths = natural_sorted(list(set(paths)))
    if start is not None and end is not None:
        paths = paths[start:end]
    if not paths:
        raise FileNotFoundError(f"No image files in {dir_path}")
    from tqdm import tqdm

    # Use tifffile for tiffs (preserves raw values), PIL for others
    def read_img(p):
        if p.suffix.lower() in [".tif", ".tiff"]:
            return tifffile.imread(str(p))
        else:
            return np.array(Image.open(str(p)).convert("L"))

    imgs = np.stack([read_img(p) for p in tqdm(paths, desc=f"Loading {dir_path.name}")])
    if downsample > 1:
        imgs = imgs[::downsample, ::downsample, ::downsample]
    return imgs


# =============================================================================
# Orthogonal Viewer (Napari)
# =============================================================================


def view_ortho(images=None, labels=None):
    """Open Napari with orthogonal slice viewer."""
    try:
        import napari
    except ImportError:
        print("Napari required: pip install 'napari[all]'")
        sys.exit(1)

    viewer = napari.Viewer(title="CT Volume - Orthogonal Viewer", ndisplay=3)

    if images is not None:
        p2, p98 = np.percentile(images, (2, 98))
        viewer.add_image(
            np.clip(images, p2, p98), name="CT Volume", colormap="gray", rendering="mip"
        )

    if labels is not None:
        label_colors = {
            0: [0, 0, 0, 0],
            1: [1, 0.2, 0.2, 0.6],
            2: [0.2, 1, 0.2, 0.6],
            3: [0.2, 0.4, 1, 0.6],
            4: [1, 1, 0.2, 0.6],
            5: [1, 0.2, 1, 0.6],
            6: [0.2, 1, 1, 0.6],
        }
        viewer.add_labels(labels, name="Segmentation", color=label_colors)

    print("\nNapari Controls:")
    print("  - Toggle 2D/3D view: button in bottom-left or press Ctrl+Y")
    print("  - Scroll: navigate slices")
    print("  - Click layer to toggle visibility")
    print("  - Right-click layer for options")

    napari.run()


# =============================================================================
# 3D Volume Rendering (PyVista)
# =============================================================================


def view_3d(labels, num_classes, images=None, downsample=1):
    """Render 3D volume of segmented phases using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista required: pip install pyvista")
        sys.exit(1)

    # Colors for each class
    class_colors = [
        [0.1, 0.1, 0.1],  # class 0: dark (background, often hidden)
        [0.9, 0.2, 0.2],  # class 1: red
        [0.2, 0.9, 0.2],  # class 2: green
        [0.2, 0.4, 0.9],  # class 3: blue
        [0.9, 0.9, 0.2],  # class 4: yellow
        [0.9, 0.2, 0.9],  # class 5: magenta
        [0.2, 0.9, 0.9],  # class 6: cyan
    ]

    class_names = [f"Class {i}" for i in range(num_classes)]

    # Create plotter
    plotter = pv.Plotter(title="3D Segmentation Viewer")

    # Add each class as a separate surface
    for c in range(1, num_classes):  # skip class 0 (usually background)
        binary = (labels == c).astype(np.uint8)

        if binary.sum() == 0:
            print(f"Class {c}: no voxels, skipping")
            continue

        # Create a uniform grid
        grid = pv.ImageData(dimensions=np.array(binary.shape) + 1)
        grid.cell_data["values"] = binary.flatten(order="F")

        # Extract surface via threshold
        surface = grid.threshold(0.5)

        if surface.n_cells > 0:
            color = class_colors[c] if c < len(class_colors) else [0.5, 0.5, 0.5]
            voxel_count = binary.sum()
            label = f"{class_names[c]} ({voxel_count:,} voxels)"
            plotter.add_mesh(surface, color=color, opacity=0.6, label=label, smooth_shading=True)
            print(f"Added {label}")

    plotter.add_legend()
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"

    print("\nPyVista Controls:")
    print("  - Left-click + drag: rotate")
    print("  - Right-click + drag: zoom")
    print("  - Middle-click + drag: pan")
    print("  - Q: quit")

    plotter.show()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="3D CT Volume Viewer")
    parser.add_argument("--images", default=None, help="Directory of image tiffs")
    parser.add_argument("--labels", default=None, help="Directory of segmentation mask tiffs")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ortho", "3d", "both"],
        help="Viewing mode: ortho (Napari), 3d (PyVista), or both",
    )
    parser.add_argument(
        "--num_classes", type=int, default=4, help="Number of classes (for 3D view)"
    )
    parser.add_argument("--slice_range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument(
        "--downsample", type=int, default=1, help="Downsample factor for 3D rendering (default: 1)"
    )
    args = parser.parse_args()

    if args.images is None and args.labels is None:
        parser.error("At least one of --images or --labels is required")

    start = args.slice_range[0] if args.slice_range else None
    end = args.slice_range[1] if args.slice_range else None

    images = None
    labels = None

    if args.images:
        ds = args.downsample if args.mode in ["3d", "both"] else 1
        images = load_stack(args.images, start=start, end=end, downsample=ds)
        print(f"Images: {images.shape} ({images.dtype})")

    if args.labels:
        ds = args.downsample if args.mode in ["3d", "both"] else 1
        labels = load_stack(args.labels, start=start, end=end, downsample=ds)
        print(f"Labels: {labels.shape} ({labels.dtype})")
        print(f"Classes: {np.unique(labels)}")

    if args.mode == "ortho":
        view_ortho(images, labels)
    elif args.mode == "3d":
        if labels is None:
            parser.error("--labels is required for 3D mode")
        view_3d(labels, args.num_classes, images, args.downsample)
    elif args.mode == "both":
        if labels is None:
            parser.error("--labels is required for 3D mode")
        print("\nOpening 3D viewer first (close to continue to ortho viewer)...")
        view_3d(labels, args.num_classes, images, args.downsample)
        print("\nOpening orthogonal viewer...")
        # Reload without downsample for ortho
        if args.images:
            images = load_stack(args.images, start=start, end=end)
        if args.labels:
            labels = load_stack(args.labels, start=start, end=end)
        view_ortho(images, labels)


if __name__ == "__main__":
    main()
