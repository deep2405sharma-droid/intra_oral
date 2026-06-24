"""
data/unet_dataset.py
---------------------
Clean & production-ready Dataset for U-Net semantic segmentation.

- No on-the-fly augmentation (augmentations are pre-generated)
- Supports original + augmented data
- Proper path rewriting
- Compatible with segmentation_models_pytorch U-Net

Mirrors dataset.py (Mask R-CNN version)
----------------------------------------
- LesionInferenceDataset      -> UNetInferenceDataset
  Same role: inference only, no mask needed.
- LesionAnnotatedDataset      -> UNetSegmentationDataset
  Same role: main training dataset, but loads a single rasterised
  mask PNG per image instead of per-instance boxes/masks parsed from
  VIA JSON. No region-by-region loop is needed because the mask is
  already a flat pixel-value image (0=bg, 1=normal, 2=opmd, 3=variation).
- collate_fn                  -> collate_fn
  Mask R-CNN needs tuple-of-lists (variable-size targets per image).
  U-Net needs stacked tensors (fixed-size after resize) — see note below.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image as PILImage
from torch.utils.data import Dataset

from src.common import intraoral_logger as iolog
logger = iolog.getLogger("unet_dataset")


# ── Path Rewriting  (identical to dataset.py) ──────────────────────────────────
def rewrite_path(raw_path: str, base_rewrite: Optional[Dict[str, str]] = None) -> Path:
    """Remap paths (Windows -> Linux/Docker)."""
    p = str(raw_path)
    if base_rewrite:
        for old, new in base_rewrite.items():
            if p.startswith(old):
                p = p.replace(old, new, 1)
                break
    return Path(p)


# ── Default label -> mask pixel value mapping ─────────────────────────────────
# 0 is always reserved for background, same convention as Mask R-CNN's
# class 0. Kept as a module-level default so callers can override via
# the label_map argument exactly like LesionAnnotatedDataset.label_map.
DEFAULT_LABEL_MAP = {
    "normal":    1,
    "opmd":      2,
    "variation": 3,
}


def _build_image_transform(input_size: tuple) -> Callable:
    """ImageNet-normalised resize pipeline — required because the encoder
    (resnet34 etc.) is ImageNet-pretrained, same reasoning as encoder_lr
    being lower than decoder_lr in train_unet.py."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _build_mask_resize(input_size: tuple) -> Callable:
    """NEAREST interpolation is mandatory for masks — any other mode would
    blend pixel class values (e.g. 1 and 2 averaging to 1.5), corrupting
    the class labels."""
    return transforms.Resize(
        input_size,
        interpolation=transforms.InterpolationMode.NEAREST,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Inference Dataset (No Mask)
# ══════════════════════════════════════════════════════════════════════════════
class UNetInferenceDataset(Dataset):
    """For inference only - no mask_path needed."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: tuple = (512, 512),
        transform: Optional[Callable] = None,
        base_rewrite: Optional[Dict[str, str]] = None,
        skip_no_image: bool = True,
    ):
        self.transform = transform or _build_image_transform(input_size)
        self.base_rewrite = base_rewrite

        subset = df.copy()
        if skip_no_image:
            subset = subset[subset["image_path"].notna()]

        self.records = []
        missing = 0

        for _, row in subset.iterrows():
            img_p = rewrite_path(row["image_path"], base_rewrite)
            if not img_p.exists():
                missing += 1
                continue
            self.records.append(
                {
                    "image_path": img_p,
                    "patient_id": row.get("patient_id", ""),
                    "label": row.get("label", ""),
                }
            )

        logger.info(
            f"UNetInferenceDataset: {len(self.records)} valid images, {missing} missing"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = PILImage.open(rec["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, rec


# ══════════════════════════════════════════════════════════════════════════════
# Annotated/Mask Dataset (Main Training Dataset)
# ══════════════════════════════════════════════════════════════════════════════
class UNetSegmentationDataset(Dataset):
    """
    Main dataset for training U-Net.
    Loads pre-augmented + original images with their pre-rasterised
    mask PNGs (one pixel value per class, no on-the-fly augmentation).

    Unlike LesionAnnotatedDataset (Mask R-CNN), there is no region-by-
    region polygon loop here — the prepare stage has already burned all
    polygons for an image into a single mask PNG, so __getitem__ only
    needs to load two files: image_path and mask_path.

    Mask PNG convention (must match the prepare stage / via_to_mask.py):
        0 = background, 1 = normal, 2 = opmd, 3 = variation
    """

    _LABEL_ALIASES = {
        "homogenous leukoplakia": "opmd",
        "non-homogenous leukoplakia": "opmd",
        "erythroplakia": "opmd",
        "erythroleukoplakia": "opmd",
        "lichen planus": "opmd",
        "no opmd": "normal",
        "normal": "normal",
        "variation from normal": "variation",
        "variation": "variation",
        "opmd": "opmd",
        "lesion": "variation",
    }

    def __init__(
        self,
        rows: pd.DataFrame,
        input_size: tuple = (512, 512),
        label_map: Optional[dict] = None,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        base_rewrite: Optional[Dict[str, str]] = None,
    ):
        self.input_size = input_size
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.transform = transform or _build_image_transform(input_size)
        self.mask_transform = mask_transform or _build_mask_resize(input_size)
        self.base_rewrite = base_rewrite

        self.samples: List[Tuple[Path, Path, str]] = []
        missing = 0

        for _, row in rows.iterrows():
            img_path = rewrite_path(str(row["image_path"]), base_rewrite)
            mask_path = rewrite_path(str(row["mask_path"]), base_rewrite)

            if img_path.exists() and mask_path.exists():
                self.samples.append((img_path, mask_path, row.get("label", "")))
            else:
                missing += 1

        logger.info(
            f"UNetSegmentationDataset: {len(self.samples)} samples | {missing} missing files"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, _label = self.samples[idx]

        # Load image
        img = PILImage.open(img_path).convert("RGB")
        img_t = self.transform(img)  # [3, H, W] float32, ImageNet-normalised

        # Load mask — single channel, pixel value = class id
        mask = PILImage.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)  # [H, W] int64

        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
# Collate Function
# ══════════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    """
    Required for U-Net batching.

    Unlike Mask R-CNN's collate_fn (tuple(zip(*batch)), needed because
    each image can have a different number of instances/boxes), U-Net
    inputs are a fixed (H, W) after the resize transform, so images and
    masks can be stacked directly into batched tensors.
    """
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)