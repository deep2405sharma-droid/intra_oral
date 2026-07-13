"""
train_unet.py
-------------
Fine-tunes a ResNet-encoder UNet on the annotated SMART intraoral dataset
for pixel-wise lesion segmentation.

Pipeline
--------
1. build_data_loaders()    →  patient-wise train / val split
2. build_lesion_model()    →  ImageNet encoder + randomly-initialised decoder
3. Training loop           →  Adam/SGD with separate encoder / decoder LRs
4. Validation loop         →  Dice coefficient and pixel-wise IoU
5. Checkpointing           →  best val Dice model saved; resume supported

Dataset contract
----------------
The CSV must have at minimum:
    image_path  : absolute path to the RGB image
    mask_path   : absolute path to the PNG mask
                  (pixel value = class id; 0 = background, 1 = lesion)
    patient_id  : used for patient-wise train/val split
    label       : lesion label string

Loss functions
--------------
bce        → Binary Cross-Entropy with logits (binary segmentation only)
dice       → Soft Dice loss           (handles class imbalance well)
bce_dice   → bce_weight * BCE + (1 - bce_weight) * Dice  [default]
focal_dice → Focal loss + Dice        (for severe class imbalance)

Usage
-----
    python train_unet.py                              # uses config defaults
    python train_unet.py --epochs 30 \\
        --csv ./data/smart_merged.csv \\
        --checkpoint_dir ./checkpoints/unet \\
        --resume ./checkpoints/unet/last.pth
"""

import argparse
import json
import logging
import sys
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Dict
from torchvision.utils import make_grid

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.segmentation.unet2.unet_config import UNetConfig
from src.segmentation.unet2.unet_builder import (
    LESION_CLASS_MAP,
    NUM_LESION_CLASSES,
    build_lesion_model,
    build_data_loaders,
    save_checkpoint,
    load_checkpoint,
    _resolve_device,
)

# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════


def _soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.

    Dice = (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    Loss = 1 - Dice

    Using sigmoid probabilities (not hard binarisation) keeps the loss
    differentiable end-to-end.

    Why Dice?  BCE treats every pixel equally, so on intraoral images where
    the lesion occupies a small fraction of the frame the loss is dominated
    by background pixels and the model converges to predicting all-background.
    Dice directly maximises the overlap fraction, giving equal weight to the
    foreground region regardless of its size.
    """
    probs = torch.sigmoid(logits)
    # Flatten spatial dimensions while keeping batch dimension
    p_flat = probs.view(probs.size(0), -1)
    g_flat = targets.float().view(targets.size(0), -1)

    intersection = (p_flat * g_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        p_flat.sum(dim=1) + g_flat.sum(dim=1) + smooth
    )
    return (1.0 - dice).mean()


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Focal loss (Lin et al., 2017) for handling extreme class imbalance.
    FL(p_t) = -α_t * (1 − p_t)^γ * log(p_t)
    γ = 2 down-weights easy examples (well-classified background pixels).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal = alpha * (1 - p_t) ** gamma * bce
    return focal.mean()


def _tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Tversky Loss for binary segmentation.

    Args:
        logits:  [B,H,W] raw logits from model
        targets: [B,H,W] float tensor with values {0,1}
        alpha: penalty for False Positives
        beta: penalty for False Negatives
        smooth: numerical stability

    Returns:
        Scalar loss tensor
    """
    probs = torch.sigmoid(logits)

    # flatten per batch
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    TP = (probs * targets).sum(dim=1)
    FP = (probs * (1 - targets)).sum(dim=1)
    FN = ((1 - probs) * targets).sum(dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - tversky
    return loss.mean()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_function: str = "bce_tversky",
    bce_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    focal_gamma=2.0,
    focal_alpha=0.25,
) -> Dict[str, torch.Tensor]:
    """
    Dispatch to the configured loss function.

    Args:
        logits        : raw model output [B, num_classes, H, W]
        targets       : ground-truth class map [B, H, W] int64
        loss_function : loss function name, default bce_tversky
        bce_weight    : weight of BCE term in combined losses

    Binary segmentation loss only.
    logits shape: [B, 1, H, W]
    """
    # Force binary format
    if logits.size(1) != 1:
        raise ValueError(
            f"Expected binary logits [B,1,H,W], got {logits.shape}. "
            f"Use NUM_LESION_CLASSES=1 for this pipeline."
        )

    logits = logits.squeeze(1)  # [B, H, W]
    targets = targets.float()  # [B, H, W]

    # Optional class weighting (helps with background dominance)
    weights = torch.ones_like(targets)
    weights[targets == 0] = 2.0  # background
    weights[targets == 1] = 1.0  # lesion / mouth

    if loss_function == "bce":
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        return {"loss": bce, "bce": bce}

    elif loss_function == "dice":
        dice = _soft_dice_loss(logits, targets)
        return {"loss": dice, "dice": dice}

    elif loss_function == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        dice = _soft_dice_loss(logits, targets)
        total = bce_weight * bce + (1.0 - bce_weight) * dice
        return {"loss": total, "bce": bce, "dice": dice}

    elif loss_function == "tversky":
        tversky = _tversky_loss(logits, targets, alpha, beta)
        return {"loss": tversky, "tversky": tversky}

    elif loss_function == "bce_tversky":
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        tversky = _tversky_loss(logits, targets, alpha, beta)
        total = bce_weight * bce + (1 - bce_weight) * tversky
        return {"loss": total, "bce": bce, "tversky": tversky}

    elif loss_function == "focal_dice":
        focal = _focal_loss(logits, targets, focal_gamma, focal_alpha)
        dice = _soft_dice_loss(logits, targets)
        total = bce_weight * focal + (1.0 - bce_weight) * dice
        return {"loss": total, "focal": focal, "dice": dice}

    else:
        raise ValueError(f"Unknown loss_function '{loss_function}'")


# def compute_loss(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     loss_function: List = ["bce_dice"],
#     bce_weight: float = 0.5,
# ) -> torch.Tensor:
#     """
#     Dispatch to the configured loss function.

#     Args:
#         logits        : raw model output [B, num_classes, H, W]
#         targets       : ground-truth class map [B, H, W] int64
#         loss_function : 'bce' | 'dice' | 'bce_dice' | 'focal_dice'
#         bce_weight    : weight of BCE term in combined losses
#     """
#     # Binary segmentation: squeeze class dim and cast targets to float
#     if logits.size(1) == 1:
#         logits = logits.squeeze(1)
#         targets = targets.float()
#     else:
#         # Multi-class: use cross-entropy directly
#         return F.cross_entropy(logits, targets)

#     weights = torch.ones_like(targets)

#     weights[targets == 0] = 2.0  # background pixels
#     weights[targets == 1] = 1.0  # foreground pixels

#     if loss_function == "bce":
#         return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)

#     elif loss_function == "dice":
#         return _soft_dice_loss(logits, targets)

#     elif loss_function == "bce_dice":
#         bce = F.binary_cross_entropy_with_logits(logits, targets)
#         dice = _soft_dice_loss(logits, targets)
#         return bce_weight * bce + (1.0 - bce_weight) * dice

#     elif loss_function == "focal_dice":
#         focal = _focal_loss(logits, targets)
#         dice = _soft_dice_loss(logits, targets)
#         return bce_weight * focal + (1.0 - bce_weight) * dice

#     else:
#         raise ValueError(f"Unknown loss_function '{loss_function}'")


# ══════════════════════════════════════════════════════════════════════════════
# Validation metrics
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_metrics(
    preds_bin: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> dict:
    """
    Compute Dice coefficient and pixel-wise IoU for a batch.

    Dice = (2 * TP + ε) / (2 * TP + FP + FN + ε)
    IoU  = TP / (TP + FP + FN + ε)

    These are the standard metrics for segmentation quality:
    - Dice is more sensitive to small structures.
    - IoU (Jaccard index) is the stricter metric used in most benchmarks.

    Both range from 0 (no overlap) to 1 (perfect overlap).
    """
    p = preds_bin.view(preds_bin.size(0), -1).float()
    g = targets.view(targets.size(0), -1).float()

    tp = (p * g).sum(dim=1)
    fp = (p * (1 - g)).sum(dim=1)
    fn = ((1 - p) * g).sum(dim=1)

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# One training epoch
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    loss_function: str = "bce_dice",
    bce_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    gradient_clip: float = 1.0,
    log_every: int = 10,
) -> dict:
    """
    Run one full pass over the training set.

    UNet returns [B, num_classes, H, W] logits.  The loss function converts
    these to a scalar; backward() computes gradients; optimizer.step() updates
    parameters.  Gradient clipping (max_norm=gradient_clip) prevents
    exploding gradients, which can occur in the decoder when fine-tuning a
    deep ResNet encoder.

    Returns a dict with averaged loss over the epoch.
    """
    model.train()

    component_totals: dict[str, float] = {}
    n_batches = 0
    t_start = time.time()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss_dict = compute_loss(
            logits=logits,
            targets=masks,
            loss_function=loss_function,
            bce_weight=bce_weight,
            alpha=alpha,
            beta=beta,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        loss = loss_dict["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # Accumulate components
        for key, val in loss_dict.items():
            component_totals[key] = component_totals.get(key, 0.0) + val.item()

        n_batches += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            avg_components = {k: v / n_batches for k, v in component_totals.items()}
            component_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_components.items())
            logger.info(
                "Epoch %d [%d/%d]  %s",
                epoch,
                batch_idx + 1,
                len(loader),
                component_str,
            )

    elapsed = time.time() - t_start
    avg_components = {k: v / max(n_batches, 1) for k, v in component_totals.items()}
    component_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_components.items())

    logger.info(
        "Epoch %d  train  %s  time=%.1fs",
        epoch,
        component_str,
        elapsed,
    )
    return {f"train_{k}": round(v, 4) for k, v in avg_components.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    mask_threshold: float = 0.5,
    loss_function: str = "bce_dice",
    bce_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> dict:
    """
    Run inference on the val set and compute:
        val_loss  : same loss function as training (plus all components)
        val_dice  : mean Dice coefficient across the val set
        val_iou   : mean IoU across the val set

    Loss components are accumulated dynamically from whatever keys
    compute_loss returns, so switching loss_function never breaks logging.

    mask_threshold converts the raw sigmoid probability to a binary mask:
        pixel > mask_threshold → foreground (1)
        pixel ≤ mask_threshold → background (0)
    """
    model.eval()

    component_totals: dict[str, float] = {}  # accumulates all loss components
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss_dict = compute_loss(
                logits=logits,
                targets=masks,
                loss_function=loss_function,
                bce_weight=bce_weight,
                alpha=alpha,
                beta=beta,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )

            # Accumulate every component compute_loss returns
            for key, val in loss_dict.items():
                component_totals[key] = component_totals.get(key, 0.0) + val.item()

            # Binarise predictions
            if logits.size(1) == 1:
                probs = torch.sigmoid(logits.squeeze(1))
                preds_bin = (probs > mask_threshold).long()
            else:
                preds_bin = logits.argmax(dim=1)

            m = compute_metrics(preds_bin, masks)
            total_dice += m["dice"]
            total_iou += m["iou"]
            n_batches += 1

    if n_batches == 0:
        logger.warning("Validation loader was empty — no metrics computed.")
        return {}

    avg_components = {k: v / n_batches for k, v in component_totals.items()}
    component_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_components.items())

    metrics = {
        **{f"val_{k}": round(v, 4) for k, v in avg_components.items()},
        "val_dice": round(total_dice / n_batches, 4),
        "val_iou": round(total_iou / n_batches, 4),
    }

    logger.info(
        "Epoch %d  val  %s  dice=%.4f  iou=%.4f",
        epoch,
        component_str,
        metrics["val_dice"],
        metrics["val_iou"],
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════


def train(logger: logging.Logger, cfg: UNetConfig, csv_path: str) -> None:
    """
    Full UNet training pipeline:
        1. Build data loaders
        2. Build model
        3. Build optimiser + LR scheduler
        4. (Optional) resume from checkpoint
        5. Epoch loop: train → validate → checkpoint
    """
    device = _resolve_device(logger, cfg.device)
    logger.info("=" * 60)
    logger.info("  UNet Fine-Tuning  |  device=%s", device)
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, num_classes = build_data_loaders(
        logger=logger,
        csv_path=csv_path,
        label_class_map=LESION_CLASS_MAP,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        min_area=cfg.min_area,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_lesion_model(
        logger=logger,
        num_classes=num_classes,
        device=str(device),
        pretrained_backbone=cfg.pretrained,
        encoder_name=cfg.backbone,
        decoder_channels=cfg.decoder_channels,  # explicit from .ini — no silent defaults
        bilinear=cfg.bilinear_upsample,
    )

    # ── Optimiser — separate encoder (lower LR) / decoder (higher LR) ─
    # The pretrained encoder contains rich ImageNet features.  Updating it
    # too aggressively would destroy those representations.  The decoder is
    # randomly initialised and needs to learn faster — same rationale as
    # Mask R-CNN's backbone_lr / head_lr split.
    enc_params = []
    dec_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.startswith("layer")
            or name.startswith("layer0")
            or name.startswith("pool0")
        ):
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = optim.SGD(
        [
            {"params": enc_params, "lr": cfg.encoder_lr},
            {"params": dec_params, "lr": cfg.decoder_lr},
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # ── LR Scheduler ──────────────────────────────────────────────────
    if cfg.lr_scheduler == "cosine":
        # Cosine annealing smoothly decays LR from initial value to min_lr
        # over cfg.epochs steps.  This prevents aggressive drops that can
        # destabilise the pretrained encoder weights.
        # Formula: η_t = η_min + ½(η_max − η_min)(1 + cos(π · t / T_max))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr,
        )
    elif cfg.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_patience, gamma=cfg.lr_factor
        )
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
        )

    # ── Checkpoint setup ──────────────────────────────────────────────
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"
    last_ckpt_path = checkpoint_dir / "last.pth"

    # Optional: resume from a previously-trained checkpoint shipped as a
    # Kaggle input dataset (read-only). If present, training resumes from
    # it; either way, new checkpoints are always written to checkpoint_dir
    # (writable, under /kaggle/working/...), never back into /kaggle/input.
    pretrained_ckpt_path = getattr(cfg, "pretrained_checkpoint", None)

    start_epoch = 1
    best_dice = 0.0
    history = []

    if best_ckpt_path.exists():
        resume_path = best_ckpt_path
    elif last_ckpt_path.exists():
        resume_path = last_ckpt_path
    elif pretrained_ckpt_path and Path(pretrained_ckpt_path).exists():
        # First run on a fresh Kaggle session: no local checkpoint yet, but
        # a previously-trained model was attached as a read-only input
        # dataset. Resume from it; subsequent epochs still save to the
        # writable checkpoint_dir, not back into /kaggle/input.
        resume_path = Path(pretrained_ckpt_path)
        logger.info("Using pretrained checkpoint from input dataset: %s", resume_path)
    else:
        resume_path = None
    if resume_path and Path(resume_path).exists():
        start_epoch, prev_metrics = load_checkpoint(
            logger=logger,
            model=model,
            optimizer=optimizer,
            path=resume_path,
            device=str(device),
        )
        best_dice = prev_metrics.get("val_dice", 0.0)
        start_epoch += 1
        logger.info(
            "Resuming from epoch %d  (best val_dice=%.4f)",
            start_epoch,
            best_dice,
        )
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        logger.info("\n--- Epoch %d / %d ---", epoch, cfg.epochs)

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            logger=logger,
            loss_function=cfg.loss_function,
            bce_weight=cfg.bce_weight,
            alpha=cfg.alpha,
            beta=cfg.beta,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
            gradient_clip=cfg.gradient_clip,
            log_every=cfg.log_every,
        )
        print(f"train_metrics: {train_metrics}")
        # Step cosine / step schedulers per epoch
        if cfg.lr_scheduler in ("cosine", "step"):
            scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(
            "  LR after step: encoder=%.2e  decoder=%.2e",
            current_lrs[0],
            current_lrs[1],
        )

        # Validate every val_every epochs (always on last epoch)
        val_metrics = {}
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                logger=logger,
                mask_threshold=cfg.mask_threshold,
                loss_function=cfg.loss_function,
                bce_weight=cfg.bce_weight,
                alpha=cfg.alpha,
                beta=cfg.beta,
                focal_gamma=cfg.focal_gamma,
                focal_alpha=cfg.focal_alpha,
            )

            # ReduceLROnPlateau monitors val_dice
            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_metrics.get("val_dice", 0.0))

        # Always save last checkpoint for resume
        all_metrics = {**train_metrics, **val_metrics}
        # save_checkpoint(logger, model, optimizer, epoch, all_metrics, last_ckpt_path)

        # Save best checkpoint based on val Dice
        current_dice = val_metrics.get("val_dice", best_dice)
        if val_metrics and current_dice > best_dice:
            best_dice = current_dice
            save_checkpoint(
                logger, model, optimizer, epoch, all_metrics, best_ckpt_path
            )
            logger.info(
                "   New best val_dice=%.4f — saved to %s",
                best_dice,
                best_ckpt_path,
            )

        history.append({"epoch": epoch, **all_metrics})

    # ── Save final best model to model_dir ─────────────────────────────
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "best_unet_model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        },
        final_model_path,
    )
    logger.info(f" Best model saved to: {final_model_path}")

    # ── Save test/visualization results ───────────────────────────────
    if val_loader is not None and len(val_loader) > 0:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating validation visualizations...")
        model.eval()
        results = []

        # Load val CSV to look up patient_id and image_path per sample index.
        # val_loader iterates in the same order as the underlying val dataset
        # rows, so index i in the loop matches row i in the val subset of the CSV.
        import pandas as _pd
        _full_df   = _pd.read_csv(csv_path, dtype=str)
        _full_df   = _full_df[_full_df["coco_file"].notna()].reset_index(drop=True)
        # Reconstruct the same val split using the same seed so indices align
        try:
            from sklearn.model_selection import GroupShuffleSplit as _GSS
            _gss = _GSS(n_splits=1, test_size=cfg.val_split, random_state=cfg.seed)
            _, _val_idx = next(_gss.split(_full_df, groups=_full_df["patient_id"].values))
            _val_df = _full_df.iloc[_val_idx].reset_index(drop=True)
        except Exception:
            _val_df = None

        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > cfg.mask_threshold).long()

                # Save visualization
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)

                pred_np = preds[0].cpu().numpy()
                gt_np = masks[0].cpu().numpy()

                # Create overlay
                overlay = img_np.copy()
                overlay[pred_np == 1] = [0, 255, 0]  # Green prediction
                overlay[gt_np == 1] = [0, 0, 255]  # Blue ground truth (overwrites)

                # Build patient ID + image stem for the title
                if _val_df is not None and i < len(_val_df):
                    _row       = _val_df.iloc[i]
                    _patient   = _row.get("patient_id", "unknown")
                    _img_stem  = Path(str(_row.get("image_path", ""))).stem
                    _sup_title = f"Patient: {_patient}  |  Image: {_img_stem}"
                else:
                    _sup_title = f"Sample {i:04d}"

                fig = plt.figure(figsize=(15, 5))
                fig.suptitle(_sup_title, fontsize=11, fontweight="bold")
                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.title("Original")
                plt.subplot(1, 3, 2)
                plt.imshow(pred_np, cmap="gray")
                plt.title("Prediction")
                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title("Overlay (Green=Pred, Blue=GT)")
                plt.savefig(
                    viz_dir / f"val_sample_{i:04d}.png", bbox_inches="tight", dpi=200
                )
                plt.close()

                results.append(
                    {
                        "image_id": i,
                        "dice": compute_metrics(preds, masks)["dice"],
                        "iou": compute_metrics(preds, masks)["iou"],
                    }
                )

        # Save summary
        pd.DataFrame(results).to_csv(output_dir / "val_results.csv", index=False)
        logger.info(f" Validation results and visualizations saved to: {output_dir}")

    # ── Save training history ─────────────────────────────────────────
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)
    logger.info("Training complete. Best val_dice=%.4f", best_dice)


def update_merged_df_paths(smart_base_path, smartom_base_path, df):
    mask1 = df["source"] == "smart_II"
    mask2 = df["source"] == "smart_om"
    df.loc[mask1, "image_path"] = (
        smart_base_path + "/" + df.loc[mask1, "image_path"].astype(str)
    )
    df.loc[mask1, "json_file"] = (
        smart_base_path + "/" + df.loc[mask1, "json_file"].astype(str)
    )
    df.loc[mask1, "coco_file"] = (
        smart_base_path + "/" + df.loc[mask1, "coco_file"].astype(str)
    )

    df.loc[mask2, "image_path"] = (
        smartom_base_path + "/" + df.loc[mask2, "image_path"].astype(str)
    )
    df.loc[mask2, "json_file"] = (
        smartom_base_path + "/" + df.loc[mask2, "json_file"].astype(str)
    )
    df.loc[mask2, "coco_file"] = (
        smartom_base_path + "/" + df.loc[mask2, "coco_file"].astype(str)
    )
    return df


def update_paths(base_path, df):
    df["image_path"] = (
        base_path + "/" + df["image_path"].astype(str).where(df["image_path"].notna())
    )
    df["json_file"] = (
        base_path + "/" + df["json_file"].astype(str).where(df["json_file"].notna())
    )
    df["coco_file"] = (
        base_path + "/" + df["coco_file"].astype(str).where(df["coco_file"].notna())
    )
    return df


def get_dataset_path(logger, config, cfg: UNetConfig) -> pd.DataFrame:
    """
    Two datasets are available:
    Based on the config we will take rows from different datasets
    if normal_dataset == SMART_II
        take normal rows only from SMART_II
    elif normal_dataset == SMART_OM
        take normal rows only from SMART_OM
    elif normal_dataset == BOTH
        take normal rows from SMART_II and SMART_OM
    """
    train_dataset_path = None
    # Base folders
    smart_merged_basepath = config.get("TRAIN", "smart.merged.basepath")
    smart_base = config.get("TRAIN", "smart.basepath")
    smartom_base = config.get("TRAIN", "smartom.basepath")
    augment_smart_base = config.get("TRAIN", "augment.smart.baseroot")
    augment_smartom_base = config.get("TRAIN", "augment.smartom.baseroot")

    merged_data_path = f"{smart_merged_basepath}/{config.get('SMART_MERGED', 'merged.coco.output.filename')}"
    augment_smart_filename = config.get(
        "AUGMENT_SMART", "augment.patient.coco.metadata.filename"
    )
    augment_smart_data = f"{augment_smart_base}/{augment_smart_filename}"
    augment_smartom_filename = config.get(
        "AUGMENT_SMARTOM", "augment.patient.coco.metadata.filename"
    )
    augment_smartom_data = f"{augment_smartom_base}/{augment_smartom_filename}"
    dataset = pd.DataFrame()
    # Load merged_data
    if Path(merged_data_path).exists():
        merged_df = pd.read_csv(merged_data_path)
        merged_df = update_merged_df_paths(
            smart_base_path=smart_base, smartom_base_path=smartom_base, df=merged_df
        )
        logger.info(f"Loaded Merged data, shape: {merged_df.shape}")
    else:
        logger.error(f"Merged Dataset Not Found at path: {merged_data_path}")
        raise FileNotFoundError(f"Merged Dataset Not Found at Path: {merged_data_path}")

    # Check for Smart_II Augmented Dataset
    if Path(augment_smart_data).exists():
        aug_smart_df = pd.read_csv(augment_smart_data)
        aug_smart_df = update_paths(base_path=augment_smart_base, df=aug_smart_df)
        logger.info(f"Shape of Augmented Smart Dataset: {aug_smart_df.shape}")
        # First update in dataset
        dataset = pd.concat([dataset, aug_smart_df])
        logger.info(f"Added Smart_II augmented dataset: {dataset.shape}")
    else:
        logger.info("Smart_II augmented dataset not found at path {augment_smart_data}")

    # Check for Smart OM Augmented Dataset
    if Path(augment_smartom_data).exists():
        aug_smartom_df = pd.read_csv(augment_smartom_data)
        aug_smartom_df = update_paths(base_path=augment_smartom_base, df=aug_smartom_df)
        logger.info(f"Shape of Augmented Smart_OM Dataset: {aug_smartom_df.shape}")
        # Second update in dataset
        dataset = pd.concat([dataset, aug_smartom_df])
        logger.info(
            f"Merged Smart_OM augmented and Smart_II augmented datasets. Total augmented dataset size: {dataset.shape}"
        )
    else:
        logger.info(
            "Smart OM augmented dataset not found at path {augment_smartom_data}"
        )
    if cfg.normal_dataset == "SMART_II":
        logger.info("Processing for smart II dataset")
        temp = merged_df[
            (merged_df.source == "smart_II")
            | ((merged_df.source == "smart_om") & ~(merged_df.label == "normal"))
        ]
        dataset = pd.concat([dataset, temp])
        logger.info(
            f"After merging Smart_II, Smart_OM only OPMD and Variation and complete augmented dataset shape: {dataset.shape}"
        )
    elif cfg.normal_dataset == "SMART_OM":
        logger.info("Processing for smart OM dataset")
        temp = merged_df[
            (merged_df.source == "smart_om")
            | ((merged_df.source == "smart_II") & ~(merged_df.label == "normal"))
        ]
        dataset = pd.concat([dataset, temp])
        logger.info(
            f"After merging Smart OM, Smart_II only OPMD and Variation and complete augmented dataset shape: {dataset.shape}"
        )
    elif cfg.normal_dataset == "BOTH":
        dataset = pd.concat([dataset, merged_df])
        logger.info(f"After merging merged dataset shape: {dataset.shape}")

    if dataset is not None and not dataset.empty:
        train_dataset_path = config.get("SEGMENT-UNET", "train.dataset")
        if not Path(train_dataset_path).parent.exists():
            os.makedirs(str(Path(train_dataset_path).parent), exist_ok=True)
        dataset = dataset.replace([np.nan, "nan", "NaN"], None)
        df = dataset[dataset[["image_path", "coco_file"]].notnull().all(axis=1)].copy()
        logger.info(
            f"Rows with image path and coco path not null in train dataset: {len(df)}"
        )
        logger.info(f"Rows with null coco file columns: {df.coco_file.isna().sum()}")
        df.to_csv(train_dataset_path)

    return train_dataset_path


def get_configpath():
    parser = argparse.ArgumentParser(
        description="A comprehensive argparse example",
        epilog="Thank you for using this tool!",
    )
    parser.add_argument("-p", "--profile")
    args = parser.parse_args()
    config_path = "config/config.ini"
    if args.profile and args.profile.lower() == "kaggle":
        config_path = "config/kaggle_config.ini"
    return config_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config_path = get_configpath()
    print(f"config_path: {config_path}")
    config = load_config(config_path)
    logger = initialize_logger(config=config)
    unet_ini = load_config(config.get("SEGMENT-UNET", "unet.config"))
    cfg = UNetConfig(unet_ini)
    csv_path = get_dataset_path(logger=logger, config=config, cfg=cfg)
    train(logger=logger, cfg=cfg, csv_path=csv_path)