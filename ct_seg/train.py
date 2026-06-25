"""
Train U-Net for Supervised CT Segmentation
===========================================
Trains a U-Net model on labeled CT slices for multi-class segmentation.

Usage:
    python train_seg.py --images /path/to/tiffs --masks /path/to/mask_tiffs --num_classes 4
    python train_seg.py --images /path/to/tiffs --masks /path/to/mask_tiffs --num_classes 4 --epochs 200 --batch_size 8

Mask format:
    - Tiff images with integer pixel values 0, 1, 2, ... (num_classes - 1)
    - Same filenames and dimensions as the corresponding image tiffs
    - Can be created with label_tool.py (Napari-based)
"""

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ct_seg import tracking
from ct_seg.model import UNetSegmentation, count_parameters

# =============================================================================
# Dataset
# =============================================================================


def natural_sorted(paths):
    import re

    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split(r"([0-9]+)", str(x))]

    return sorted(paths, key=key)


class SegmentationDataset(Dataset):
    """
    Dataset for CT segmentation training.
    Loads image-mask pairs from two directories of tiff files.
    """

    def __init__(self, image_dir, mask_dir, patch_size=256, augment=True, slice_range=None):
        self.patch_size = patch_size
        self.augment = augment

        # Load images
        img_paths = natural_sorted(list(Path(image_dir).glob("*.tif*")))
        mask_paths = natural_sorted(list(Path(mask_dir).glob("*.tif*")))

        if slice_range:
            img_paths = img_paths[slice_range[0] : slice_range[1]]
            mask_paths = mask_paths[slice_range[0] : slice_range[1]]

        assert len(img_paths) == len(
            mask_paths
        ), f"Image count ({len(img_paths)}) != mask count ({len(mask_paths)})"

        print(f"Loading {len(img_paths)} image-mask pairs...")
        self.images = np.stack([tifffile.imread(str(p)) for p in tqdm(img_paths, desc="Images")])
        self.masks = np.stack([tifffile.imread(str(p)) for p in tqdm(mask_paths, desc="Masks")])

        # Normalize images
        self.img_mean = self.images.mean()
        self.img_std = self.images.std()
        self.images = ((self.images - self.img_mean) / (self.img_std + 1e-8)).astype(np.float32)

        # Ensure masks are integer class labels
        self.masks = self.masks.astype(np.int64)

        unique_classes = np.unique(self.masks)
        print(f"Image shape: {self.images.shape}, Mask shape: {self.masks.shape}")
        print(f"Classes found in masks: {unique_classes}")
        print(f"Normalization: mean={self.img_mean:.6f}, std={self.img_std:.6f}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        # Random crop
        h, w = img.shape
        if h > self.patch_size and w > self.patch_size:
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            img = img[y : y + self.patch_size, x : x + self.patch_size]
            mask = mask[y : y + self.patch_size, x : x + self.patch_size]

        # Augmentation: random D4 symmetry (rotation + flip)
        if self.augment:
            k = np.random.randint(4)
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

        # Add channel dimension [1, H, W]
        img = img[np.newaxis]
        return torch.from_numpy(img), torch.from_numpy(mask)


# =============================================================================
# Loss Functions
# =============================================================================


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W] (integer class labels)
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        targets_one_hot = torch.nn.functional.one_hot(targets, self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Cross-Entropy + Dice loss."""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


# =============================================================================
# Metrics
# =============================================================================


def compute_iou(pred, target, num_classes):
    """Compute per-class IoU (Intersection over Union)."""
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


# =============================================================================
# Training
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Segmentation")
    parser.add_argument("--images", required=True, help="Directory of image tiffs")
    parser.add_argument(
        "--masks", required=True, help="Directory of mask tiffs (integer class labels)"
    )
    parser.add_argument("--output", default="SegOutput", help="Output directory for checkpoints")
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of segmentation classes"
    )
    parser.add_argument(
        "--base_filters", type=int, default=32, help="Base filter count (default: 32)"
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Training patch size (default: 256)"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument(
        "--val_split", type=float, default=0.15, help="Validation split fraction (default: 0.15)"
    )
    parser.add_argument("--slice_range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--no_mlflow", action="store_true", help="Disable MLflow experiment tracking"
    )
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/results", exist_ok=True)

    logging.basicConfig(filename=f"{args.output}/training.log", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {dev}")

    # Load dataset
    full_dataset = SegmentationDataset(
        args.images,
        args.masks,
        patch_size=args.patch_size,
        augment=True,
        slice_range=args.slice_range,
    )

    # Train/val split
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val])
    val_ds.dataset.augment = False  # no augmentation for validation

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    logging.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # Model
    model = UNetSegmentation(
        in_channels=1, num_classes=args.num_classes, base_filters=args.base_filters
    ).to(dev)
    logging.info(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    criterion = CombinedLoss(args.num_classes)

    # Optional MLflow experiment tracking (no-op if mlflow absent or --no_mlflow).
    track = not args.no_mlflow
    tracking.start(run_name=f"unet_c{args.num_classes}", enabled=track)
    tracking.log_params(
        {
            "num_classes": args.num_classes,
            "base_filters": args.base_filters,
            "patch_size": args.patch_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "val_split": args.val_split,
            "device": str(dev),
            "train_samples": n_train,
            "val_samples": n_val,
        },
        enabled=track,
    )

    # Training loop
    best_val_loss = float("inf")
    best_iou = 0
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tick = time.time()

        # Train
        model.train()
        epoch_loss = []
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(dev), masks.to(dev)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_loss = np.mean(epoch_loss)

        # Validate
        model.eval()
        val_epoch_loss = []
        all_ious = []
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(dev), masks.to(dev)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_epoch_loss.append(loss.item())

                pred = torch.argmax(logits, dim=1)
                ious = compute_iou(pred.cpu().numpy(), masks.cpu().numpy(), args.num_classes)
                all_ious.append(ious)

        val_loss = np.mean(val_epoch_loss)
        mean_iou = np.nanmean(all_ious)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tracking.log_metrics(
            {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "mean_iou": float(mean_iou),
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
            enabled=track,
        )

        ep_time = time.time() - tick
        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"mIoU: {mean_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{ep_time:.1f}s"
        )

        # Save best model (by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "num_classes": args.num_classes,
                    "base_filters": args.base_filters,
                    "img_mean": full_dataset.img_mean,
                    "img_std": full_dataset.img_std,
                },
                f"{args.output}/best_model.pth",
            )
            logging.info(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Save best model (by mIoU)
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "epoch": epoch,
                    "mean_iou": mean_iou,
                    "num_classes": args.num_classes,
                    "base_filters": args.base_filters,
                    "img_mean": full_dataset.img_mean,
                    "img_std": full_dataset.img_std,
                },
                f"{args.output}/best_iou_model.pth",
            )
            logging.info(f"  -> Saved best IoU model (mIoU={mean_iou:.4f})")

        # Plot training curves every 10 epochs
        if epoch % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (CE + Dice)")
            plt.title("Segmentation Training Progress")
            plt.legend()
            plt.savefig(f"{args.output}/results/training_curve.png", dpi=150)
            plt.close()

    total_time = time.time() - start_time
    logging.info(f"\nTraining complete in {total_time:.0f}s")
    logging.info(f"Best val loss: {best_val_loss:.4f}")
    logging.info(f"Best mIoU: {best_iou:.4f}")

    # Save training metadata
    meta = {
        "num_classes": args.num_classes,
        "base_filters": args.base_filters,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_loss": float(best_val_loss),
        "best_iou": float(best_iou),
        "training_time_s": total_time,
        "img_mean": float(full_dataset.img_mean),
        "img_std": float(full_dataset.img_std),
    }
    with open(f"{args.output}/training_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Log final summary + best checkpoint to MLflow, then close the run.
    tracking.log_metrics(
        {
            "best_val_loss": float(best_val_loss),
            "best_iou": float(best_iou),
            "training_time_s": float(total_time),
        },
        enabled=track,
    )
    tracking.log_artifact(f"{args.output}/best_model.pth", enabled=track)
    tracking.log_artifact(f"{args.output}/training_info.json", enabled=track)
    tracking.end(enabled=track)


if __name__ == "__main__":
    main()
