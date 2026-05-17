"""
mask_rcnn_builder.py
---------------------------
Builds Mask R-CNN models for two use-cases:

1. build_coco_pretrained()
   → Standard torchvision Mask R-CNN pre-trained on COCO-2017.
     Used for zero-shot inference (no fine-tuning).

2. build_lesion_model(num_classes)
   → Same backbone but with box and mask heads replaced to predict
     num_classes (background + lesion classes).
     Used for fine-tuning on annotated SMART data.

Both return a standard nn.Module (torchvision MaskRCNN).

Changes vs previous version
----------------------------
- _cfg is instantiated once at module level, not referenced before definition.
- build_coco_pretrained() and build_lesion_model() accept logger as first arg;
  the default score_threshold and nms thresholds are set inside the function
  body (not in the signature) to avoid referencing _cfg before it exists.
- Added CocoDataset helper class: reads the per-image COCO JSON files produced
  by the VIA→COCO converter and returns (image_tensor, target) pairs compatible
  with the torchvision Mask R-CNN training loop.
- Added build_data_loaders() which reads smart_merged.csv (with coco_file col)
  and constructs train/val DataLoader objects ready for fine-tuning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.utils.data
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from src.segmentation.maskrcnn.config.maskrcnnconfig import MaskRCNNConfig


# ── COCO class names (torchvision index order, 91 classes) ───────────────────
COCO_CLASS_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# ── Lesion label → integer class id ──────────────────────────────────────────
# background = 0 is always reserved by torchvision Mask R-CNN
# LESION_CLASS_MAP = {
#     "normal": 1,
#     "variation": 2,
#     "opmd": 3,
# }
# NUM_LESION_CLASSES = 1 + len(LESION_CLASS_MAP)

LESION_CLASS_MAP = {
    "left buccal mucosa":  1,
    "right buccal mucosa": 2,
    "dorsal tongue":       3,
    "lower lip":           4,
    "upper lip":           5,
    "upper arch":          6,
    "ventral tongue":      7,
}
NUM_LESION_CLASSES = 1 + len(LESION_CLASS_MAP)
# ══════════════════════════════════════════════════════════════════════════════
# Device helper
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_device(logger, device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    return torch.device(device)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Zero-shot COCO pretrained model
# ══════════════════════════════════════════════════════════════════════════════


def build_coco_pretrained(
    logger: logging.Logger,
    device: torch.device,
    score_threshold: Optional[float] = None,
) -> nn.Module:
    """
    Load COCO-pretrained Mask R-CNN (ResNet50-FPN v1).
    Always returned in eval() mode — do not call .train() for zero-shot use.

    Args:
        logger          : caller's logger instance
        device          : 'cuda' | 'cpu' | None (auto)
        score_threshold : minimum detection confidence
    """
    if score_threshold is None:
        logger.error(f"Invalid value of score_threshold :{score_threshold}")
        return

    logger.info("Loading COCO-pretrained Mask R-CNN (ResNet50-FPN v1) …")
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        box_score_thresh=score_threshold,
    )
    model.eval()
    model.to(device)
    logger.info("COCO pretrained model ready on %s", device)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fine-tuned lesion model
# ══════════════════════════════════════════════════════════════════════════════


def build_lesion_model(
    logger: logging.Logger,
    num_classes: Optional[int] = None,
    device: Optional[str] = None,
    pretrained_backbone: Optional[bool] = None,
) -> nn.Module:
    """
    Mask R-CNN with box & mask heads replaced to predict num_classes classes.
    Backbone weights are COCO-pretrained (includes FPN + RPN).

    Args:
        logger              : caller's logger instance
        num_classes         : total classes including background; defaults to
                              NUM_LESION_CLASSES (4 = bg + normal + variation + opmd)
        device              : 'cuda' | 'cpu' | None (auto)
        pretrained_backbone : whether to start from COCO weights; defaults to
                              _cfg.pretrained_backbone
    """
    if num_classes is None:
        num_classes = NUM_LESION_CLASSES

    logger.info(
        "Building lesion Mask R-CNN  num_classes=%d  pretrained_backbone=%s",
        num_classes,
        pretrained_backbone,
    )

    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params — total: %d  trainable: %d", total_params, trainable_params)
    # Replace box classification head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask head
    # MaskRCNNPredictor has three mandatory inputs
    # in_channels: The number of input channels to the first
    # layer (the transposed convolution). This must match the number of output
    # channels from the preceding mask head.
    # dim_reduced: The number of output channels from the
    # first (transposed convolution) layer and the number of input
    # channels to the final convolution layer. This defines the
    # internal feature dimension.
    # num_classes: The number of output classes, including the background.
    # This determines the number of output channels in the final
    # mask_fcn_logits layer, producing a separate mask prediction for each class.
    #
    # The pre-trained model's MaskRCNNPredictor has its final layer (mask_fcn_logits)
    # configured for the original number of classes (e.g., 81 for COCO).
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced=256, num_classes=num_classes
    )

    dev = _resolve_device(logger, device)
    model.to(dev)
    logger.info("Lesion model built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. CocoDataset — reads per-image COCO JSON files from the metadata CSV
# ══════════════════════════════════════════════════════════════════════════════


class CocoDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs each image with its per-image COCO JSON annotation.

    The metadata CSV (smart_merged.csv + coco_file column) has one row per
    image.  Each row's coco_file points to a COCO-format JSON that contains
    annotations for THAT image only (produced by the VIA→COCO converter).

    COCO JSON format expected (per-image file):
        {
            "images":      [{ "id": int, "file_name": str,
                              "width": int, "height": int }],
            "annotations": [{ "id": int, "image_id": int,
                              "category_id": int,
                              "segmentation": [[x1,y1,x2,y2,...]] or [],
                              "bbox": [x, y, w, h],
                              "area": float,
                              "iscrowd": 0 }],
            "categories":  [{ "id": int, "name": str }]
        }

    Args:
        rows            : DataFrame slice (rows with non-null coco_file and
                          image_path that exist on disk)
        label_class_map : dict mapping label string → integer class id
                          e.g. {"normal": 1, "variation": 2, "opmd": 3}
        transforms      : optional callable applied to (image_tensor, target)
    """

    def __init__(
        self,
        rows,
        label_class_map: dict = LESION_CLASS_MAP,
        transforms: Optional[Callable] = None,
    ):
        self.rows = rows.reset_index(drop=True)
        self.label_class_map = label_class_map
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        img_path = str(row["image_path"])
        coco_path = str(row["coco_file"])

        # ── Load image ────────────────────────────────────────────────────────
        img_pil = PILImage.open(img_path).convert("RGB")
        W, H = img_pil.size
        img_t = TF.to_tensor(img_pil)  # [3, H, W] float32 in [0,1]

        # ── Load COCO annotation ──────────────────────────────────────────────
        with open(coco_path, "r") as f:
            coco = json.load(f)

        annotations = coco.get("annotations", [])

        # Build category_id → class_id mapping from the COCO file's categories
        # (fallback: use label_class_map keyed by category name)
        cat_id_to_class = {}
        for cat in coco.get("categories", []):
            name = cat["name"].lower().strip()
            if name in self.label_class_map:
                cat_id_to_class[cat["id"]] = self.label_class_map[name]
            else:
                # Unknown category — assign background (0), will be filtered
                cat_id_to_class[cat["id"]] = 0

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in annotations:
            cat_id = ann["category_id"]
            class_id = cat_id_to_class.get(cat_id, 0)
            if class_id == 0:
                continue  # skip background-class annotations

            # Bounding box [x, y, w, h] → [x1, y1, x2, y2]
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(W, x + bw), min(H, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            areas.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

            # Segmentation mask from polygon points
            seg = ann.get("segmentation", [])
            mask = self._seg_to_mask(seg, H, W)
            masks.append(mask)

        # ── Assemble torchvision-compatible target dict ───────────────────────
        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.stack(masks),  # [N, H, W] uint8
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
            }
        else:
            # No valid annotations — return empty target (model skips the image
            # during loss computation via iscrowd filter)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
            }

        if self.transforms:
            img_t, target = self.transforms(img_t, target)

        return img_t, target

    @staticmethod
    def _seg_to_mask(segmentation: list, H: int, W: int) -> torch.Tensor:
        """
        Convert a COCO segmentation polygon list to a binary [H, W] uint8 mask.
        segmentation format: [[x1, y1, x2, y2, ...], ...]  (list of rings)
        """
        canvas = np.zeros((H, W), dtype=np.uint8)
        for ring in segmentation:
            if len(ring) < 6:  # need at least 3 points
                continue
            pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(canvas, [pts], color=1)

        return torch.tensor(canvas, dtype=torch.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DataLoader builder
# ══════════════════════════════════════════════════════════════════════════════


def _collate_fn(batch):
    """Collate variable-size images and targets into lists (Mask R-CNN expects lists)."""
    return tuple(zip(*batch))


def build_data_loaders(
    logger: logging.Logger,
    csv_path: str,
    label_class_map: dict = LESION_CLASS_MAP,
    val_split: float = 0.2,
    batch_size: int = 2,
    num_workers: int = 2,
    path_rewrite: Optional[dict] = None,
    augment_labels: Optional[list] = None,
    seed: int = 42,
) -> tuple:
    """
    Read smart_merged.csv (with coco_file column) and return
    (train_loader, val_loader, num_classes).

    Only rows with a non-null coco_file are included (annotated images).
    Rows whose image_path does not exist on disk are silently dropped.
    Rows with label == 'normal' and augment_labels=['opmd','variation'] are
    excluded from training by default (pass augment_labels=None to include all).

    Args:
        csv_path        : path to smart_merged.csv (or subset)
        label_class_map : label → class id mapping
        val_split       : fraction of annotated rows reserved for validation
        batch_size      : images per batch
        num_workers     : DataLoader worker processes
        path_rewrite    : {old_prefix: new_prefix} for path remapping
        augment_labels  : if set, only rows with these labels are kept for
                          training (e.g. ['opmd','variation'])
        seed            : random seed for train/val split

    Returns:
        train_loader, val_loader, num_classes
    """
    import pandas as pd
    from torch.utils.data import random_split

    logger.info("Building data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    # Keep only rows with a COCO annotation file
    df = df[df["coco_file"].notna()].copy()
    logger.info("  Rows with coco_file: %d", len(df))

    # Filter to requested labels only
    if augment_labels:
        df = df[df["label"].str.lower().isin([l.lower() for l in augment_labels])]
        logger.info("  After label filter %s: %d rows", augment_labels, len(df))

    # Drop rows whose image or coco file does not exist on disk
    rewrite = path_rewrite or {}

    def _rw(p: str) -> Path:
        for old, new in rewrite.items():
            if str(p).startswith(old):
                p = new + str(p)[len(old) :]
                break
        return Path(p)

    mask = df.apply(
        lambda r: _rw(str(r["image_path"])).exists()
        and _rw(str(r["coco_file"])).exists(),
        axis=1,
    )
    df = df[mask].reset_index(drop=True)
    logger.info("  Rows with both files on disk: %d", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No annotated images found on disk. Check csv_path and path_rewrite."
        )

    # Build dataset
    dataset = CocoDataset(
        rows=df,
        label_class_map=label_class_map,
    )

    # Train / val split by patient_id
    if "patient_id" not in df.columns:
        raise RuntimeError("Column 'patient_id' is required for patient-wise split.")

    patient_ids = df["patient_id"].dropna().unique().tolist()
    if len(patient_ids) < 2:
        raise RuntimeError(
            f"Need at least 2 unique patient_id values for train/val split, got {len(patient_ids)}."
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(patient_ids), generator=g).tolist()
    patient_ids = [patient_ids[i] for i in perm]

    n_val_patients = max(1, int(len(patient_ids) * val_split))
    if n_val_patients >= len(patient_ids):
        n_val_patients = len(patient_ids) - 1

    val_patient_ids = set(patient_ids[:n_val_patients])
    train_patient_ids = set(patient_ids[n_val_patients:])

    train_df = df[df["patient_id"].isin(train_patient_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patient_ids)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f"Invalid patient-wise split: train={len(train_df)} rows, val={len(val_df)} rows."
        )

    train_ds = CocoDataset(
        rows=train_df,
        label_class_map=label_class_map,
    )
    val_ds = CocoDataset(
        rows=val_df,
        label_class_map=label_class_map,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = 1 + len(label_class_map)  # background + lesion classes
    logger.info(
        "  DataLoaders ready. num_classes=%d  label_map=%s",
        num_classes,
        label_class_map,
    )
    return train_loader, val_loader, num_classes


# ══════════════════════════════════════════════════════════════════════════════
# 5. Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path,
    logger: Optional[logging.Logger] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    if logger:
        logger.info("Checkpoint saved → %s  (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    path,
    device: str = "cpu",
    logger: Optional[logging.Logger] = None,
):
    """Load checkpoint into model (and optionally optimizer). Returns (epoch, metrics)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    if logger:
        logger.info("Checkpoint loaded from %s  (epoch %d)", path, epoch)
    return epoch, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main — smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import configparser
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.common.intraoral_logger import getLogger
    from utils.load_configuration import load_config

    CONFIG_PATH     = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
    MASKRCNN_CONFIG = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\maskrcnn.ini"

    cfg_main     = load_config(MASKRCNN_CONFIG)
    log_file     = cfg_main.get("PATHS", "log_file")
    csv_path     = cfg_main.get("PATHS", "csv.path")
    num_classes  = cfg_main.getint("TRAINING", "num_classes")
    device_str   = cfg_main.get("SYSTEM", "device")
    score_thresh = cfg_main.getfloat("TRAINING", "score_threshold")
    batch_size   = cfg_main.getint("TRAINING", "batch_size")
    num_workers  = cfg_main.getint("TRAINING", "num_workers")
    val_split    = cfg_main.getfloat("TRAINING", "val_split")
    seed         = cfg_main.getint("SYSTEM", "seed")

    logger = getLogger(log_file)

    logger.info("=" * 60)
    logger.info("  mask_rcnn_builder smoke test")
    logger.info("=" * 60)

    # ── 1. Resolve device ─────────────────────────────────────────
    device = _resolve_device(logger, device_str)
    logger.info("Device: %s", device)

    # ── 2. Test COCO pretrained model ─────────────────────────────
    logger.info("\n--- Building COCO pretrained model ---")
    coco_model = build_coco_pretrained(
        logger=logger,
        device=device,
        score_threshold=score_thresh,
    )
    logger.info("COCO model params: %d", sum(p.numel() for p in coco_model.parameters()))

    # ── 3. Test lesion model ──────────────────────────────────────
    logger.info("\n--- Building lesion model ---")
    lesion_model = build_lesion_model(
        logger=logger,
        num_classes=num_classes,
        device=device_str,
        pretrained_backbone=True,
    )
    logger.info("Lesion model params: %d", sum(p.numel() for p in lesion_model.parameters()))

    # ── 4. Test data loaders ──────────────────────────────────────
    logger.info("\n--- Building data loaders ---")
    train_loader, val_loader, n_cls = build_data_loaders(
        logger=logger,
        csv_path=csv_path,
        label_class_map=LESION_CLASS_MAP,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    logger.info("Train batches : %d", len(train_loader))
    logger.info("Val batches   : %d", len(val_loader))
    logger.info("Num classes   : %d", n_cls)

    # ── 5. Test one batch ─────────────────────────────────────────
    logger.info("\n--- Testing one batch ---")
    images, targets = next(iter(train_loader))
    logger.info("Batch images  : %d", len(images))
    logger.info("Image shape   : %s", images[0].shape)
    logger.info("Target keys   : %s", list(targets[0].keys()))
    logger.info("Boxes shape   : %s", targets[0]["boxes"].shape)
    logger.info("Labels        : %s", targets[0]["labels"])

    logger.info("\n" + "=" * 60)
    logger.info("  Smoke test complete — builder is working correctly")
    logger.info("=" * 60)