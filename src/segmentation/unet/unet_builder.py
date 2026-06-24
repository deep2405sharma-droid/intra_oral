"""
unet_builder.py
----------------
Builds U-Net models for semantic segmentation of intraoral lesions.

1. build_imagenet_pretrained()
   -> Standard U-Net (segmentation_models_pytorch) with an ImageNet
      pretrained encoder. Used as the starting point before fine-tuning.

2. build_lesion_unet(num_classes)
   -> Same encoder, decoder head configured to output num_classes
      channels (background + lesion classes). Used for fine-tuning
      on annotated SMART data.

Both return a standard nn.Module (smp.Unet).

Mirrors mask_rcnn_builder.py
-----------------------------
- _resolve_device() reused as-is.
- build_lesion_unet() accepts logger as first arg, same as
  build_lesion_model() in mask_rcnn_builder.py.
- UNetMaskDataset is the semantic-segmentation equivalent of CocoDataset:
  instead of parsing COCO annotations into boxes/masks per instance,
  it loads a single pre-rasterised mask PNG per image (one pixel value
  per class, no per-instance separation needed).
- build_data_loaders() mirrors the Mask R-CNN version: reads the merged
  CSV (with mask_path column), filters to rows that exist on disk, splits
  by patient_id to avoid leakage, returns (train_loader, val_loader,
  num_classes).
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image as PILImage
from torchvision import transforms

# from src.segmentation.unet.config.unetconfig import UNetConfig


# ── Lesion label -> integer class id ──────────────────────────────────────────
# background = 0 is always reserved, same convention as Mask R-CNN
LESION_CLASS_MAP = {
    "normal":    1,
    "opmd":      2,
    "variation": 3,
}
NUM_LESION_CLASSES = 1 + len(LESION_CLASS_MAP)  # 4


# ══════════════════════════════════════════════════════════════════════════════
# Device helper  (identical to mask_rcnn_builder.py)
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_device(logger, device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    return torch.device(device)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ImageNet-pretrained U-Net  (zero-shot encoder, fresh decoder)
# ══════════════════════════════════════════════════════════════════════════════


def build_imagenet_pretrained(
    logger: logging.Logger,
    device: torch.device,
    encoder: str = "resnet34",
) -> nn.Module:
    """
    Load a U-Net with an ImageNet-pretrained encoder and a randomly
    initialised decoder (1 output channel, i.e. binary lesion/no-lesion).
    Useful as a quick sanity check before fine-tuning on num_classes.

    Args:
        logger  : caller's logger instance
        device  : torch.device
        encoder : timm/smp encoder name, default resnet34
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch not installed.\n"
            "Install with: pip install segmentation-models-pytorch"
        )

    logger.info("Loading ImageNet-pretrained U-Net (encoder=%s) ...", encoder)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.eval()
    model.to(device)
    logger.info("ImageNet pretrained U-Net ready on %s", device)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fine-tuned lesion U-Net
# ══════════════════════════════════════════════════════════════════════════════


def build_lesion_unet(
    logger: logging.Logger,
    num_classes: Optional[int] = None,
    device: Optional[str] = None,
    encoder: str = "resnet34",
    pretrained_encoder: Optional[bool] = None,
) -> nn.Module:
    """
    U-Net with decoder configured to output num_classes channels
    (background + lesion classes). Encoder weights are ImageNet-pretrained
    by default for transfer learning, same role as the COCO-pretrained
    backbone in build_lesion_model().

    Args:
        logger              : caller's logger instance
        num_classes         : total classes including background; defaults
                              to NUM_LESION_CLASSES (4 = bg + normal +
                              opmd + variation)
        device              : 'cuda' | 'cpu' | None (auto -> cpu)
        encoder              : smp encoder name, e.g. resnet34, resnet50,
                              efficientnet-b0
        pretrained_encoder  : whether to start encoder from ImageNet
                              weights; defaults to True
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch not installed.\n"
            "Install with: pip install segmentation-models-pytorch"
        )

    if num_classes is None:
        num_classes = NUM_LESION_CLASSES

    if pretrained_encoder is None:
        pretrained_encoder = True

    logger.info(
        "Building lesion U-Net  num_classes=%d  encoder=%s  pretrained_encoder=%s",
        num_classes,
        encoder,
        pretrained_encoder,
    )

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=3,
        classes=num_classes,
        activation=None,  # raw logits — softmax applied in loss function
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("U-Net trainable params: %d", n_params)

    dev = _resolve_device(logger, device)
    model.to(dev)
    logger.info("Lesion U-Net built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNetMaskDataset — pairs each image with its rasterised mask PNG
# ══════════════════════════════════════════════════════════════════════════════


class UNetMaskDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs each image with its pre-rasterised mask PNG.

    The metadata CSV (smart_merged.csv + mask_path column) has one row
    per image. Each row's mask_path points to a single-channel PNG where
    pixel value = class id (produced by the VIA -> mask rasteriser, the
    semantic-segmentation equivalent of the VIA -> COCO converter used
    for Mask R-CNN).

    Mask PNG convention:
        0 = background, 1 = normal, 2 = opmd, 3 = variation

    Args:
        rows            : DataFrame slice (rows with non-null mask_path
                          and image_path that exist on disk)
        input_size      : (H, W) to resize image and mask to
        label_class_map : dict mapping label string -> integer class id
                          e.g. {"normal": 1, "opmd": 2, "variation": 3}
                          (kept for parity with CocoDataset signature;
                          mask pixel values are already baked in)
        transforms      : optional callable applied to (image_t, mask_t)
    """

    def __init__(
        self,
        rows,
        input_size: tuple = (512, 512),
        label_class_map: dict = LESION_CLASS_MAP,
        transforms: Optional[Callable] = None,
    ):
        self.rows = rows.reset_index(drop=True)
        self.label_class_map = label_class_map
        self.transforms = transforms

        self.img_transform = transforms_module_resize(input_size)
        self.mask_resize = transforms.Resize(
            input_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        ) if hasattr(transforms, "Resize") else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]

        img_path  = str(row["image_path"])
        mask_path = str(row["mask_path"])

        # ── Load image ──────────────────────────────────────────────────
        img_pil = PILImage.open(img_path).convert("RGB")
        img_t   = self.img_transform(img_pil)  # [3, H, W] float32, normalised

        # ── Load mask ───────────────────────────────────────────────────
        mask_pil = PILImage.open(mask_path).convert("L")  # single channel
        if self.mask_resize is not None:
            mask_pil = self.mask_resize(mask_pil)
        mask_t = torch.as_tensor(np.array(mask_pil), dtype=torch.long)  # [H, W]

        if self.transforms:
            img_t, mask_t = self.transforms(img_t, mask_t)

        return img_t, mask_t


def transforms_module_resize(input_size: tuple):
    """Build the standard ImageNet-normalised image transform pipeline."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 4. DataLoader builder  (mirrors build_data_loaders in mask_rcnn_builder.py)
# ══════════════════════════════════════════════════════════════════════════════


def _collate_fn(batch):
    """Stack images and masks into batched tensors (unlike Mask R-CNN, U-Net
    inputs are fixed-size after resize, so stacking works directly)."""
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)


def build_data_loaders(
    logger: logging.Logger,
    csv_path: str,
    label_class_map: dict = LESION_CLASS_MAP,
    input_size: tuple = (512, 512),
    val_split: float = 0.2,
    batch_size: int = 4,
    num_workers: int = 2,
    path_rewrite: Optional[dict] = None,
    seed: int = 42,
) -> tuple:
    """
    Read smart_merged.csv (with mask_path column) and return
    (train_loader, val_loader, num_classes).

    Only rows with a non-null mask_path are included (annotated or
    normal images that have a rasterised mask). Rows whose image_path
    or mask_path do not exist on disk are silently dropped.

    Args:
        csv_path        : path to smart_merged.csv (or subset)
        label_class_map : label -> class id mapping
        input_size      : (H, W) resize target for image and mask
        val_split       : fraction of rows reserved for validation
        batch_size      : images per batch
        num_workers     : DataLoader worker processes
        path_rewrite    : {old_prefix: new_prefix} for path remapping
        seed            : random seed for train/val split

    Returns:
        train_loader, val_loader, num_classes
    """
    import pandas as pd

    logger.info("Building data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    # Keep only rows with a mask file
    df = df[df["mask_path"].notna()].copy()
    logger.info("  Rows with mask_path: %d", len(df))

    # Drop rows whose image or mask file does not exist on disk
    rewrite = path_rewrite or {}

    def _rw(p: str) -> Path:
        for old, new in rewrite.items():
            if str(p).startswith(old):
                p = new + str(p)[len(old):]
                break
        return Path(p)

    mask_exists = df.apply(
        lambda r: _rw(str(r["image_path"])).exists()
        and _rw(str(r["mask_path"])).exists(),
        axis=1,
    )
    df = df[mask_exists].reset_index(drop=True)
    logger.info("  Rows with both files on disk: %d", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No image+mask pairs found on disk. Check csv_path and path_rewrite."
        )

    # Train / val split by patient_id  (identical leak-prevention logic
    # to build_data_loaders in mask_rcnn_builder.py)
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

    val_patient_ids   = set(patient_ids[:n_val_patients])
    train_patient_ids = set(patient_ids[n_val_patients:])

    train_df = df[df["patient_id"].isin(train_patient_ids)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_patient_ids)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f"Invalid patient-wise split: train={len(train_df)} rows, val={len(val_df)} rows."
        )

    # Verify no overlap, same sanity check style as Mask R-CNN pipeline
    if set(train_df["patient_id"]) & set(val_df["patient_id"]):
        raise RuntimeError("Patient ID overlap detected between train and val sets.")
    else:
        logger.info("No overlap of patient IDs in train and val datasets.")

    train_ds = UNetMaskDataset(
        rows=train_df,
        input_size=input_size,
        label_class_map=label_class_map,
    )
    val_ds = UNetMaskDataset(
        rows=val_df,
        input_size=input_size,
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
# 5. Checkpoint helpers  (identical pattern to mask_rcnn_builder.py)
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
        logger.info("Checkpoint saved -> %s  (epoch %d)", path, epoch)


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