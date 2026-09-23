"""
src/classification/efficientnet/efficientnet_builder.py
----------------------------------------------------------
Model builder + dataset/dataloader builder for the EfficientNet
image classification pipeline — a second architecture option for the
same 3-class (normal/opmd/variation) task as resnet_builder.py.

Mirrors resnet_builder.py exactly:
  - Same LABEL_CLASS_MAP / 3-class setup
  - Same patient-wise GroupShuffleSplit train/val split
  - Same checkpoint save/load pattern
  - Same augmentation philosophy (no flips — left/right oral anatomy is
    clinically meaningful)
  - Dataset loads image only (no mask/coco_file needed — classification)

Only real difference from resnet_builder.py: the backbone architecture
(torchvision EfficientNet family instead of ResNet) and how it splits
into backbone/head parameter groups for the two-LR optimiser in
train_efficientnet.py.
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


# ── Label -> class id mapping — SAME 3 classes as resnet_builder.py ──
LABEL_CLASS_MAP = {
    "normal":    0,
    "opmd":      1,
    "variation": 2,
}
NUM_CLASSES = len(LABEL_CLASS_MAP)  # 3

# Same starting point as ResNet's CLASS_WEIGHTS — tune independently per
# architecture, since class-imbalance behaviour can differ between backbones.
CLASS_WEIGHTS = [1.0, 8.0, 5.0]   # normal, opmd, variation


# ══════════════════════════════════════════════════════════════════════════════
# Device helper (identical to resnet_builder.py / unet_builder.py)
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
# Dataset — identical logic to resnet_builder.ResNetDataset
# ══════════════════════════════════════════════════════════════════════════════


class EfficientNetDataset(Dataset):
    """
    Loads (image, label) pairs for EfficientNet image classification.
    Same CSV contract as ResNetDataset: image_path, label, patient_id.
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

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
                transforms.RandomCrop(input_size),
                # No horizontal/vertical flip — left/right oral anatomy is
                # clinically meaningful, same reasoning as resnet_builder.py.
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
        img_t   = self.transform(img)
        label_t = torch.tensor(rec["class_id"], dtype=torch.long)
        return img_t, label_t


def _collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════════════
# build_data_loaders — identical logic to resnet_builder.py
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
    logger.info("Building EfficientNet data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

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
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
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

    train_pids_final = set(train_df["patient_id"].unique())
    val_pids_final   = set(val_df["patient_id"].unique())
    overlap = train_pids_final & val_pids_final
    if overlap:
        raise RuntimeError(f"Patient ID overlap detected between train and val: {overlap}")
    else:
        logger.info("No overlap of patient IDs in train and val datasets.")

    logger.info(
        "  Patient split → train_patients=%d (%d rows)  val_patients=%d (%d rows)",
        len(train_pids_final), len(train_df), len(val_pids_final), len(val_df),
    )
    logger.info(
        "  Train label distribution:\n%s", train_df["label"].value_counts().to_string(),
    )

    train_ds = EfficientNetDataset(
        train_df, input_size=input_size, label_class_map=label_class_map, augment=True,
    )
    val_ds = EfficientNetDataset(
        val_df, input_size=input_size, label_class_map=label_class_map, augment=False,
    )

    if weighted_sampler:
        class_counts = train_df["label"].value_counts().to_dict()
        sample_weights = [
            1.0 / class_counts.get(rec["label"], 1) for rec in train_ds.samples
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            collate_fn=_collate_fn, pin_memory=torch.cuda.is_available(),
        )
        logger.info("  WeightedRandomSampler enabled for minority class oversampling")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=_collate_fn, pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=_collate_fn, pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "  DataLoaders ready — num_classes=%d  label_map=%s", NUM_CLASSES, label_class_map,
    )
    return train_loader, val_loader, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# build_efficientnet_model
# ══════════════════════════════════════════════════════════════════════════════

# Map backbone name -> torchvision weights-enum attribute name
_EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
    "efficientnet_b2": "EfficientNet_B2_Weights",
    "efficientnet_b3": "EfficientNet_B3_Weights",
    "efficientnet_b4": "EfficientNet_B4_Weights",
}


def build_efficientnet_model(
    logger: logging.Logger,
    num_classes: int,
    device: str,
    pretrained: bool = True,
    backbone: str = "efficientnet_b0",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build an EfficientNet image classifier with a custom classifier head.

    Architecture:
        EfficientNet features (ImageNet pretrained)
            |
        AdaptiveAvgPool2d (built into torchvision EfficientNet)
            |
        Dropout(p=dropout)
            |
        Linear(in_features -> num_classes)

    Two learning rates used in train_efficientnet.py:
        backbone_lr — gentle fine-tuning of the pretrained feature
                      extractor (model.features)
        head_lr     — faster learning for the fresh classifier head
                      (model.classifier)
    """
    logger.info(
        "Building EfficientNet classifier  backbone=%s  pretrained=%s  "
        "num_classes=%d  dropout=%.2f",
        backbone, pretrained, num_classes, dropout,
    )

    if backbone not in _EFFICIENTNET_VARIANTS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: "
            f"{', '.join(_EFFICIENTNET_VARIANTS.keys())}"
        )

    backbone_fn = getattr(models, backbone)
    weights_enum = getattr(models, _EFFICIENTNET_VARIANTS[backbone], None)
    weights = weights_enum.IMAGENET1K_V1 if (pretrained and weights_enum is not None) else None
    model = backbone_fn(weights=weights)

    # torchvision EfficientNet already ends in
    # nn.Sequential(nn.Dropout(p=..., inplace=True), nn.Linear(in_features, 1000))
    # — replace it with our own dropout probability and num_classes, same
    # pattern as resnet_builder.py replacing model.fc entirely.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("EfficientNet trainable params: %d", n_params)

    dev = _resolve_device(logger, device)
    model.to(dev)
    logger.info("EfficientNet classifier built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers (identical to resnet_builder.py)
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    logger: logging.Logger, model: nn.Module, optimizer, epoch: int, metrics: dict, path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        },
        path,
    )
    logger.info("Checkpoint saved -> %s  (epoch %d)", path, epoch)


def load_checkpoint(
    logger: logging.Logger, model: nn.Module, optimizer, path, device: str = "cpu",
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    logger.info("Checkpoint loaded from %s  (epoch %d)", path, epoch)
    return epoch, metrics
