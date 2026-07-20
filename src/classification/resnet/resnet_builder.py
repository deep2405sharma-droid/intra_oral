"""
src/classification/resnet/resnet_builder.py
--------------------------------------------
Model builder + dataset/dataloader builder for the ResNet50
image classification pipeline.

Provides:
    LABEL_CLASS_MAP, NUM_CLASSES, CLASS_WEIGHTS,
    build_lesion_model, build_data_loaders,
    save_checkpoint, load_checkpoint, _resolve_device

Mirrors unet_builder.py exactly:
  - Same patient-wise GroupShuffleSplit train/val split
  - Same checkpoint save/load pattern
  - Same _resolve_device helper
  - Dataset loads image only (no mask/coco_file needed — classification,
    not segmentation)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


# ── Label -> class id mapping ──────────────────────────────────────
# 3-class classification: normal, opmd, variation
# background is NOT a class here — every image has exactly one label
LABEL_CLASS_MAP = {
    "normal":    0,
    "opmd":      1,
    "variation": 2,
}
NUM_CLASSES = len(LABEL_CLASS_MAP)  # 3

# Class weights for CrossEntropyLoss — OPMD highest (rarest, most critical)
CLASS_WEIGHTS = [1.0, 5.0, 3.0]   # normal, opmd, variation


# ══════════════════════════════════════════════════════════════════════════════
# Device helper  (identical to unet_builder.py)
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_device(
    logger: logging.Logger, device: Optional[str] = None
) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    return torch.device(device)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


class ResNetDataset(Dataset):
    """
    Loads (image, label) pairs for ResNet image classification.

    Unlike UNet's CocoSegDataset, no mask or coco_file is needed —
    each image is mapped directly to an integer class label.

    CSV contract:
        image_path : absolute path to RGB image
        label      : 'normal' | 'opmd' | 'variation'
        patient_id : used for patient-wise split (done by build_data_loaders)
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        input_size: Tuple[int, int] = (224, 224),
        label_class_map: dict = LABEL_CLASS_MAP,
        augment: bool = False,
    ):
        self.label_class_map = label_class_map
        self.samples = []
        missing = 0

        for _, row in rows.iterrows():
            img_path = str(row.get("image_path", ""))
            label    = str(row.get("label", "")).lower().strip()

            if label not in label_class_map:
                continue
            if not Path(img_path).exists():
                missing += 1
                continue

            self.samples.append(
                {
                    "image_path": img_path,
                    "label":      label,
                    "class_id":   label_class_map[label],
                }
            )

        # Image transforms
        # Augmentation is applied only during training to improve generalisation
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
                transforms.RandomCrop(input_size),
                # No horizontal/vertical flip — left/right oral anatomy is
                # clinically meaningful, same reasoning as the U-Net augmentation config.
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec     = self.samples[idx]
        img     = PILImage.open(rec["image_path"]).convert("RGB")
        img_t   = self.transform(img)                             # [3, H, W]
        label_t = torch.tensor(rec["class_id"], dtype=torch.long)  # scalar
        return img_t, label_t


def _collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════════════
# build_data_loaders
# ══════════════════════════════════════════════════════════════════════════════


def build_data_loaders(
    logger: logging.Logger,
    csv_path: str,
    label_class_map: dict = LABEL_CLASS_MAP,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    input_size: Tuple[int, int] = (224, 224),
    weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Read the dataset CSV and return (train_loader, val_loader, num_classes).

    Patient-wise split using GroupShuffleSplit — same leak-prevention logic
    as unet_builder.py: all images of one patient go entirely into either
    train or val, never split across both.

    Optionally uses WeightedRandomSampler on the training set to oversample
    minority classes (opmd, variation) so each batch sees a balanced mix
    regardless of the overall dataset imbalance.
    """
    logger.info("Building ResNet data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    # Keep only rows with a valid label and existing image
    df = df[df["label"].isin(label_class_map.keys())].copy()
    df = df[df["image_path"].notna()].copy()
    logger.info("  Rows with valid label and image_path: %d", len(df))

    if len(df) == 0:
        raise RuntimeError(f"No valid rows found in {csv_path}.")

    if "patient_id" not in df.columns:
        raise RuntimeError("Column 'patient_id' is required for patient-wise split.")

    patient_ids = df["patient_id"].dropna().unique().tolist()
    if len(patient_ids) < 2:
        raise RuntimeError(
            f"Need ≥2 unique patient_ids for split, got {len(patient_ids)}."
        )

    # ── Patient-wise split ─────────────────────────────────────────────
    try:
        from sklearn.model_selection import GroupShuffleSplit
        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False
        logger.warning(
            "sklearn not installed — falling back to random.shuffle. "
            "Install with: pip install scikit-learn"
        )

    if _HAS_SKLEARN:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        groups = df["patient_id"].values
        train_idx, val_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
    else:
        import random
        rng = random.Random(seed)
        shuffled = patient_ids.copy()
        rng.shuffle(shuffled)
        n_val      = max(1, min(int(len(shuffled) * val_split), len(shuffled) - 1))
        val_pids   = set(shuffled[:n_val])
        train_pids = set(shuffled[n_val:])
        train_df   = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
        val_df     = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)

    # Sanity check
    train_pids_final = set(train_df["patient_id"].unique())
    val_pids_final   = set(val_df["patient_id"].unique())
    overlap = train_pids_final & val_pids_final
    if overlap:
        raise RuntimeError(
            f"Patient ID overlap detected between train and val: {overlap}"
        )
    else:
        logger.info("No overlap of patient IDs in train and val datasets.")

    logger.info(
        "  Patient split → train_patients=%d (%d rows)  val_patients=%d (%d rows)",
        len(train_pids_final), len(train_df),
        len(val_pids_final),   len(val_df),
    )
    logger.info(
        "  Train label distribution:\n%s",
        train_df["label"].value_counts().to_string(),
    )

    train_ds = ResNetDataset(
        train_df, input_size=input_size,
        label_class_map=label_class_map, augment=True,
    )
    val_ds = ResNetDataset(
        val_df, input_size=input_size,
        label_class_map=label_class_map, augment=False,
    )

    # ── WeightedRandomSampler ─────────────────────────────────────────
    # Oversamples minority classes (opmd, variation) so each training
    # batch sees a balanced class distribution — avoids the model always
    # predicting "normal" due to class imbalance.
    if weighted_sampler:
        class_counts = train_df["label"].value_counts().to_dict()
        sample_weights = [
            1.0 / class_counts.get(rec["label"], 1)
            for rec in train_ds.samples
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,            # sampler and shuffle are mutually exclusive
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("  WeightedRandomSampler enabled for minority class oversampling")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "  DataLoaders ready — num_classes=%d  label_map=%s",
        NUM_CLASSES, label_class_map,
    )
    return train_loader, val_loader, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# build_lesion_model
# ══════════════════════════════════════════════════════════════════════════════


def build_lesion_model(
    logger: logging.Logger,
    num_classes: int,
    device: str,
    pretrained: bool = True,
    backbone: str = "resnet50",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build a ResNet image classifier with a custom FC head.

    Architecture:
        ResNet50 backbone (ImageNet pretrained)
            |
        AdaptiveAvgPool2d (global average pooling — built into ResNet)
            |
        Dropout(p=dropout)     <- reduces overfitting on small medical dataset
            |
        Linear(2048 -> num_classes)   <- fresh classification head

    Two learning rates are used in train_resnet.py:
        backbone_lr  — gentle fine-tuning of pretrained ResNet layers
        head_lr      — faster learning for the fresh FC head
    """
    logger.info(
        "Building ResNet classifier  backbone=%s  pretrained=%s  "
        "num_classes=%d  dropout=%.2f",
        backbone, pretrained, num_classes, dropout,
    )

    # Load pretrained backbone
    backbone_fn = getattr(models, backbone, None)
    if backbone_fn is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: resnet18, resnet34, resnet50, resnet101"
        )

    weights_enum = {
        "resnet18":  getattr(models, "ResNet18_Weights",  None),
        "resnet34":  getattr(models, "ResNet34_Weights",  None),
        "resnet50":  getattr(models, "ResNet50_Weights",  None),
        "resnet101": getattr(models, "ResNet101_Weights", None),
    }.get(backbone)

    weights = weights_enum.IMAGENET1K_V2 if (pretrained and weights_enum is not None) else None
    model   = backbone_fn(weights=weights)

    # Replace the default FC head with Dropout + custom Linear
    in_features = model.fc.in_features   # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("ResNet trainable params: %d", n_params)

    dev = _resolve_device(logger, device)
    model.to(dev)
    logger.info("ResNet classifier built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers  (identical pattern to unet_builder.py)
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    logger: logging.Logger,
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":              metrics,
        },
        path,
    )
    logger.info("Checkpoint saved -> %s  (epoch %d)", path, epoch)


def load_checkpoint(
    logger: logging.Logger,
    model: nn.Module,
    optimizer,
    path,
    device: str = "cpu",
):
    """Load checkpoint into model (and optimizer if present).
    Returns (epoch, metrics)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch   = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    logger.info("Checkpoint loaded from %s  (epoch %d)", path, epoch)
    return epoch, metrics
