"""
unet_builder.py
---------------
Builds UNet models using segmentation_models_pytorch (SMP).

Dataset source
--------------
Masks are derived directly from the COCO JSON annotation files that already
exist for Mask R-CNN — there is NO separate mask_path column required.

The per-image COCO JSON (produced by the VIA→COCO converter) contains polygon
segmentation coordinates under annotations[*].segmentation.  CocoSegDataset
rasterises those polygons into a single semantic mask [H, W] where each pixel
holds the integer class id (0 = background, 1..N = lesion classes).

This is the same polygon-to-mask logic already in mask_rcnn_builder.CocoDataset
._seg_to_mask(), reused here without modification.  The only difference is the
output format:

    Mask R-CNN needs  [N, H, W]  — one binary mask per instance (N detections).
    UNet needs        [H, W]     — one semantic map, pixel = class id.

CocoSegDataset collapses the per-instance masks into a single semantic canvas
by painting each instance's polygon with its class id.  Overlapping polygons
are handled by painting in annotation order (last annotation wins).

CSV contract  (same as mask_rcnn_builder — no new columns needed)
--------------------------------------------------------------
    image_path  : absolute path to the RGB image
    coco_file   : absolute path to the per-image COCO JSON annotation
    patient_id  : used for patient-wise train / val split
    label       : lesion label string
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image as PILImage

try:
    from torchinfo import summary
    _HAS_TORCHINFO = True
except ImportError:
    _HAS_TORCHINFO = False

import segmentation_models_pytorch as smp

# ══════════════════════════════════════════════════════════════════════════════
# Class map  (mirrors mask_rcnn_builder.LESION_CLASS_MAP)
# ══════════════════════════════════════════════════════════════════════════════

LESION_CLASS_MAP = {
    "left buccal mucosa": 1,
    "right buccal mucosa": 1,
    "dorsal tongue": 1,
    "lower lip": 1,
    "upper lip": 1,
    "upper arch": 1,
    "ventral tongue": 1,
}
NUM_LESION_CLASSES = 1  # background(0) + lesion(1)


# ══════════════════════════════════════════════════════════════════════════════
# Device helper
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_device(logger, device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"
    return torch.device(device)


# ══════════════════════════════════════════════════════════════════════════════
# Polygon → semantic mask  (taken verbatim from mask_rcnn_builder._seg_to_mask)
# ══════════════════════════════════════════════════════════════════════════════


def _seg_to_binary_mask(segmentation: list, H: int, W: int) -> np.ndarray:
    """
    Rasterise one COCO annotation's polygon list into a binary [H, W] uint8 array.
    segmentation format: [[x1, y1, x2, y2, ...], ...]  (list of rings)
    Identical to CocoDataset._seg_to_mask() in mask_rcnn_builder.
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    for ring in segmentation:
        if len(ring) < 6:  # need at least 3 (x, y) pairs
            continue
        pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        cv2.fillPoly(canvas, [pts], color=1)
    return canvas


def _coco_to_semantic_mask(
    coco: dict,
    H: int,
    W: int,
    label_class_map: dict,
    min_area: int = 0,
) -> np.ndarray:
    """
    Convert a per-image COCO annotation dict into a single semantic mask.

    Output: [H, W] int32 array where pixel value = class id.
            0 → background, 1..N → lesion classes from label_class_map.

    Steps
    -----
    1. Build category_id → class_id from the JSON's "categories" list
       (same lookup used in CocoDataset.__getitem__).
    2. For each annotation, rasterise the polygon(s) with _seg_to_binary_mask.
    3. Paint pixels belonging to the annotation with the annotation's class_id.
       Annotations are painted in order — later annotations overwrite earlier
       ones at overlapping pixels (consistent with Mask R-CNN training data).
    4. Annotations whose rasterised area < min_area are skipped (artefact filter,
       mirrors the DATASET.min_area setting used in mask_rcnn training).

    Why a single [H, W] tensor for UNet rather than [N, H, W]?
    -----------------------------------------------------------
    Mask R-CNN predicts one binary mask per detected instance.  Its loss is
    computed per-instance so it needs the N-mask format.
    UNet predicts a class score at every pixel in a single forward pass.  Its
    loss (BCE / Dice / cross-entropy) is computed on the full [H, W] map, so
    it needs one integer label per pixel — i.e. a semantic segmentation mask.
    """
    # category_id → class_id  (same logic as CocoDataset.__getitem__)
    cat_id_to_class = {}
    for cat in coco.get("categories", []):
        name = cat["name"].lower().strip()
        cat_id_to_class[cat["id"]] = label_class_map.get(name, 0)

    semantic = np.zeros((H, W), dtype=np.int32)  # start: all background

    for ann in coco.get("annotations", []):
        class_id = cat_id_to_class.get(ann["category_id"], 0)
        if class_id == 0:
            continue  # skip unknown / background categories

        seg = ann.get("segmentation", [])
        if not seg:
            continue

        binary = _seg_to_binary_mask(seg, H, W)  # [H, W] uint8

        if int(binary.sum()) < min_area:
            continue  # too small — annotation artefact

        # Paint class_id onto semantic canvas wherever binary mask is 1
        semantic[binary == 1] = class_id

    return semantic  # [H, W] int32


# ══════════════════════════════════════════════════════════════════════════════
# CocoSegDataset  — image + COCO JSON → (image_tensor, semantic_mask_tensor)
# ══════════════════════════════════════════════════════════════════════════════


class CocoSegDataset(torch.utils.data.Dataset):
    """
    Segmentation dataset that derives masks from COCO JSON polygon annotations.

    Reads the same CSV + coco_file column that CocoDataset (Mask R-CNN) uses.
    No mask_path column is required — masks are rasterised on the fly from
    the polygon coordinates already in the JSON.

    Returns
    -------
    image_tensor : [3, H, W] float32, normalised with SMP per-encoder mean/std
    mask_tensor  : [H, W] int64, pixel value = class id (0=bg, 1=lesion, ...)

    Args
    ----
    rows              : DataFrame slice (rows with non-null coco_file and
                        image_path that exist on disk) — same as CocoDataset
    label_class_map   : category name → class id  (lowercase keys)
    preprocessing_fn  : SMP per-encoder normalisation from get_preprocessing_fn()
    transforms        : optional albumentations Compose applied before normalisation
                        Expected signature: fn(image=np_array, mask=np_array)
    min_area          : skip annotations whose rasterised pixel count < min_area
    target_size       : (H, W) to resize image and mask; must be divisible by 32
                        for SMP UNet.  None keeps original size.
    """

    def __init__(
        self,
        rows,
        label_class_map: dict = LESION_CLASS_MAP,
        preprocessing_fn: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        min_area: int = 500,
        target_size: Optional[Tuple[int, int]] = (512, 512),
    ):
        self.rows = rows.reset_index(drop=True)
        self.label_class_map = label_class_map
        self.preprocessing_fn = preprocessing_fn
        self.transforms = transforms
        self.min_area = min_area
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        img_path = str(row["image_path"])
        coco_path = str(row["coco_file"])

        # ── Load image ────────────────────────────────────────────────
        img_pil = PILImage.open(img_path).convert("RGB")
        W_orig, H_orig = img_pil.size  # PIL gives (W, H)

        # ── Load COCO JSON and rasterise polygons → semantic mask ─────
        with open(coco_path, "r") as f:
            coco = json.load(f)

        semantic_np = _coco_to_semantic_mask(
            coco,
            H=H_orig,
            W=W_orig,
            label_class_map=self.label_class_map,
            min_area=self.min_area,
        )  # [H_orig, W_orig] int32

        # ── Resize (image: bilinear, mask: nearest to preserve class ids) ──
        if self.target_size:
            H_t, W_t = self.target_size
            img_pil = img_pil.resize((W_t, H_t), PILImage.BILINEAR)
            semantic_pil = PILImage.fromarray(semantic_np.astype(np.int32))
            # PIL does not support int32 resize directly — use cv2
            semantic_np = cv2.resize(
                semantic_np.astype(np.int32),
                (W_t, H_t),
                interpolation=cv2.INTER_NEAREST,
            )

        img_np = np.array(img_pil, dtype=np.float32) / 255.0  # [H, W, 3]

        # ── Optional albumentations transforms ────────────────────────
        # Albumentations handles paired image+mask augmentation and keeps
        # spatial transforms (flip, rotate, elastic) consistent between the
        # image and the class-id mask.
        if self.transforms:
            augmented = self.transforms(image=img_np, mask=semantic_np)
            img_np = augmented["image"]
            semantic_np = augmented["mask"]

        # ── SMP per-encoder normalisation ─────────────────────────────
        # Must be applied AFTER augmentation (augmentation works on [0,1] float
        # images; normalisation shifts them to encoder-specific mean/std).
        if self.preprocessing_fn is not None:
            img_np = self.preprocessing_fn(img_np)

        # ── To tensors ────────────────────────────────────────────────
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]
        mask_t = torch.from_numpy(semantic_np.astype(np.int64)).long()  # [H, W]

        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
# Factory functions
# ══════════════════════════════════════════════════════════════════════════════


def get_preprocessing_fn(encoder_name: str = "resnet50") -> Callable:
    """
    Return the SMP preprocessing function for the given encoder.
    Exposes the exact ImageNet mean/std the encoder was pretrained with,
    so you never hard-code normalization values per backbone.
    """
    return smp.encoders.get_preprocessing_fn(encoder_name, pretrained="imagenet")


def build_pretrained_encoder(
    logger: logging.Logger,
    device: torch.device,
    decoder_channels: Tuple[int, ...],
    encoder_name: str = "resnet50",
    num_classes: int = NUM_LESION_CLASSES,
) -> nn.Module:
    """
    SMP UNet with ImageNet-pretrained encoder. Returned in eval() mode.
    """
    logger.info(
        "Building pretrained SMP UNet  encoder=%s  num_classes=%d",
        encoder_name,
        num_classes,
    )
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        decoder_channels=decoder_channels,
        decoder_use_batchnorm=True,
        activation=None,  # raw logits
    )
    model.eval()
    model.to(device)
    logger.info("Pretrained SMP UNet ready on %s", device)
    return model


def build_lesion_model(
    logger: logging.Logger,
    num_classes: Optional[int] = None,
    device: Optional[str] = None,
    pretrained_backbone: Optional[bool] = None,
    encoder_name: str = "resnet50",
    decoder_channels: Tuple[int] = [256, 128, 64, 32, 16],
    bilinear: bool = False,
) -> nn.Module:
    """
    SMP UNet for lesion segmentation fine-tuning.

    Encoder: ImageNet-pretrained (trained with lower LR).
    Decoder: randomly initialised (trained with higher LR).
    """
    if num_classes is None:
        num_classes = NUM_LESION_CLASSES

    encoder_weights = "imagenet" if pretrained_backbone else None

    logger.info(
        "Building lesion SMP UNet  encoder=%s  num_classes=%d  pretrained=%s",
        encoder_name,
        num_classes,
        pretrained_backbone,
    )
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        decoder_channels=decoder_channels,
        decoder_use_batchnorm=True,
        activation=None,
    )
    dev = _resolve_device(logger, device)
    model.to(dev)

    # Model summary is purely informational -- never let a forward-pass
    # failure here (e.g. incompatible GPU kernel) block training itself.
    if _HAS_TORCHINFO:
        try:
            logger.info(
                "Model Summary:\n%s",
                summary(model, input_size=(1, 3, 512, 512), device=dev, verbose=0),
            )
        except Exception as e:
            logger.warning("Skipping model summary (forward pass failed): %s", e)
    else:
        logger.info(
            "Model Summary: (torchinfo not installed -- skipping summary; "
            "run `pip install torchinfo` to enable)"
        )
    logger.info("Lesion SMP UNet built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader builder
# ══════════════════════════════════════════════════════════════════════════════


def _collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)


def build_data_loaders(
    logger: logging.Logger,
    csv_path: str,
    label_class_map: dict = LESION_CLASS_MAP,
    val_split: float = 0.2,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    min_area: int = 500,
    target_size: Optional[Tuple[int, int]] = (512, 512),
    path_rewrite: Optional[dict] = None,
    encoder_name: str = "resnet50",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Read the metadata CSV and return (train_loader, val_loader, num_classes).

    Uses the same CSV + coco_file column as Mask R-CNN — no mask_path needed.
    Masks are rasterised from COCO JSON polygons inside CocoSegDataset.__getitem__.

    Patient-wise split: no patient appears in both train and val.
    Only rows with non-null coco_file where both image_path and coco_file
    exist on disk are included — same filter as build_data_loaders in
    mask_rcnn_builder.
    """
    import pandas as pd

    logger.info("Building UNet data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    # Keep only annotated rows  (same filter as mask_rcnn_builder)
    df = df[df["coco_file"].notna()].copy()
    logger.info("  Rows with coco_file: %d", len(df))

    # Drop rows where files do not exist on disk
    rewrite = path_rewrite or {}

    def _rw(p: str) -> Path:
        for old, new in rewrite.items():
            if str(p).startswith(old):
                return Path(new + str(p)[len(old) :])
        return Path(p)

    exists_mask = df.apply(
        lambda r: _rw(str(r["image_path"])).exists()
        and _rw(str(r["coco_file"])).exists(),
        axis=1,
    )
    df = df[exists_mask].reset_index(drop=True)
    logger.info("  Rows with both files on disk: %d", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No annotated images found on disk. Check csv_path and path_rewrite."
        )

    # ── Patient-wise split ────────────────────────────────────────────
    # Each patient_id goes ENTIRELY into train or val — never split
    # across sets.  This prevents data leakage where the model could
    # memorise patient-specific tissue appearance.
    #
    # Uses sklearn GroupShuffleSplit instead of a simple torch.randperm
    # shuffler so the split is guaranteed non-overlapping at the patient
    # level, not just at the row level.
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
        # groups= maps every row to its patient_id so GroupShuffleSplit
        # knows which rows belong to the same patient and keeps them together.
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        groups = df["patient_id"].values
        train_idx, val_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
    else:
        # Fallback: manual patient-wise shuffle with fixed seed
        import random
        rng = random.Random(seed)
        shuffled = patient_ids.copy()
        rng.shuffle(shuffled)
        n_val      = max(1, min(int(len(shuffled) * val_split), len(shuffled) - 1))
        val_pids   = set(shuffled[:n_val])
        train_pids = set(shuffled[n_val:])
        train_df   = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
        val_df     = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)

    # Sanity check — confirm zero patient overlap between train and val
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
        len(train_pids_final),
        len(train_df),
        len(val_pids_final),
        len(val_df),
    )

    preprocessing_fn = get_preprocessing_fn(encoder_name)
    logger.info("  SMP preprocessing for encoder '%s'", encoder_name)

    train_ds = CocoSegDataset(
        rows=train_df,
        label_class_map=label_class_map,
        preprocessing_fn=preprocessing_fn,
        min_area=min_area,
        target_size=target_size,
    )
    val_ds = CocoSegDataset(
        rows=val_df,
        label_class_map=label_class_map,
        preprocessing_fn=preprocessing_fn,
        min_area=min_area,
        target_size=target_size,
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

    num_classes = NUM_LESION_CLASSES
    logger.info(
        "  DataLoaders ready — num_classes=%d  encoder=%s",
        num_classes,
        encoder_name,
    )
    return train_loader, val_loader, num_classes


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers  (identical API to mask_rcnn_builder)
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    logger: Optional[logging.Logger],
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
    logger: Optional[logging.Logger],
    model: nn.Module,
    optimizer,
    path,
    device: str = "cpu",
) -> Tuple[int, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if logger:
        logger.info("Checkpoint loaded from %s  (epoch %d)", path, ckpt.get("epoch", 0))
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})