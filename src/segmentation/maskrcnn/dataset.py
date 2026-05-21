"""
data/dataset.py
---------------------
Clean & Production-ready Dataset for Mask R-CNN.

- No on-the-fly augmentation (augmentations are pre-generated)
- Supports original + augmented data
- Proper path rewriting
- Compatible with torchvision Mask R-CNN
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from torch.utils.data import Dataset

from utils.annotation_parser import Annotation, polygon_to_mask


# ── Path Rewriting ─────────────────────────────────────────────────────────────
def rewrite_path(raw_path: str, base_rewrite: Optional[Dict[str, str]] = None) -> Path:
    """Remap paths (Windows → Linux/Docker)."""
    p = str(raw_path)
    if base_rewrite:
        for old, new in base_rewrite.items():
            if p.startswith(old):
                p = p.replace(old, new, 1)
                break
    return Path(p)


# ══════════════════════════════════════════════════════════════════════════════
# Inference Dataset (No Annotations)
# ══════════════════════════════════════════════════════════════════════════════
class LesionInferenceDataset(Dataset):
    """For inference only - no JSON needed."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        base_rewrite: Optional[Dict[str, str]] = None,
        skip_json: bool = True,
    ):
        self.transform = transform or (lambda x: TF.to_tensor(x))
        self.base_rewrite = base_rewrite

        subset = df.copy()
        if skip_json:
            subset = subset[subset["lesion_location"] != "json_file"]

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
                    "lesion_location": row.get("lesion_location", ""),
                }
            )

        logger.info(
            f"InferenceDataset: {len(self.records)} valid images, {missing} missing"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = PILImage.open(rec["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, rec


# ══════════════════════════════════════════════════════════════════════════════
# Annotated Dataset (Main Training Dataset)
# ══════════════════════════════════════════════════════════════════════════════
class LesionAnnotatedDataset(Dataset):
    """
    Main dataset for training Mask R-CNN.
    Loads pre-augmented + original images with their VIA JSON annotations.
    No on-the-fly augmentation.
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
        annotations: List[Annotation],
        transform: Optional[Callable] = None,
        base_rewrite: Optional[Dict[str, str]] = None,
        min_area: int = 100,
    ):
        self.transform = transform or (lambda x: TF.to_tensor(x))
        self.base_rewrite = base_rewrite
        self.min_area = min_area
        self.label_map = _cfg.LABEL_MAP

        self.samples: List[Tuple[Annotation, Path]] = []
        missing = 0

        for ann in annotations:
            json_path = rewrite_path(str(ann.json_path), base_rewrite)
            if not json_path.exists():
                missing += 1
                continue

            # Use the image_path already resolved in annotation_parser
            img_path = rewrite_path(str(ann.image_path), base_rewrite)

            if img_path.exists():
                self.samples.append((ann, img_path))
            else:
                missing += 1

        logger.info(
            f"LesionAnnotatedDataset: {len(self.samples)} samples | {missing} missing files"
        )

    def _resolve_label(self, csv_label: str, region_label: str = "") -> int:
        """Priority: CSV label > Region label > Alias mapping"""

        # 1. Highest priority: CSV label
        if csv_label:
            key = str(csv_label).strip().lower()
            if key in ["normal", "variation", "opmd"]:
                return self.label_map.get(key, 1)
            if key in ["variation from normal", "pigmentation"]:
                return self.label_map.get("variation", 2)
            if "opmd" in key:
                return self.label_map.get("opmd", 3)

        # 2. Fallback: Region label from JSON
        if region_label:
            key = str(region_label).strip().lower()
            canonical = self._LABEL_ALIASES.get(key, key)
            return self.label_map.get(canonical, self.label_map.get("normal", 1))

        # 3. Default
        return self.label_map.get("normal", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ann, img_path = self.samples[idx]

        # Load image
        img = PILImage.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels, masks = [], [], []

        for reg in ann.regions:
            if len(reg.polygon_x) < 3:
                continue

            mask = polygon_to_mask(reg.polygon_x, reg.polygon_y, H, W)
            if mask.sum() < self.min_area:
                continue

            x1, y1, x2, y2 = reg.bbox
            # Clamp coordinates
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))

            boxes.append([x1, y1, x2, y2])
            labels.append(
                self._resolve_label(
                    csv_label=ann._csv_label if hasattr(ann, "_csv_label") else "",
                    region_label=reg.label,
                )
            )
            masks.append(mask)

        # Convert to tensors
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area_t = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_t,
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }

        img = self.transform(img)

        return img, target


# ══════════════════════════════════════════════════════════════════════════════
# Collate Function
# ══════════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    """Required for Mask R-CNN batching."""
    return tuple(zip(*batch))
