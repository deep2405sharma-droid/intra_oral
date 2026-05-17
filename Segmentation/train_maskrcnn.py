"""
train_maskrcnn.py
-----------------
Fine-tunes Mask R-CNN (ResNet50-FPN, COCO-pretrained backbone) on the
annotated SMART intraoral dataset.

Pipeline
--------
1. build_data_loaders()  →  patient-wise train / val split
2. build_lesion_model()  →  COCO backbone + new box/mask heads
3. Training loop         →  SGD with momentum + cosine LR decay
4. Validation loop       →  mAP@50 and mAP@50:95 via torchmetrics
5. Checkpointing         →  best val mAP model saved; resume supported

Dataset contract
----------------
The CSV must have at minimum:
    image_path   : absolute path to the unannotated image
    coco_file    : absolute path to the per-image COCO JSON annotation
    patient_id   : used for patient-wise train/val split (no patient
                   appears in both train and val)
    label        : lesion label

COCO JSON format (per image, produced by VIA→COCO converter):
    { "images": [...], "annotations": [...], "categories": [...] }

Usage
-----
    python train_maskrcnn.py                    # uses config defaults
    python train_maskrcnn.py --epochs 20 \\
        --csv ./data/temp_coco_dataset.csv \\
        --checkpoint_dir ./checkpoints/maskrcnn \\
        --resume ./checkpoints/maskrcnn/best.pth
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

import torch
import torch.optim as optim
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.common.intraoral_logger import getLogger
from utils.load_configuration import load_config
from Segmentation.maskrcnnconfig import MaskRCNNConfig
from Segmentation.mask_rcnn_builder import (
    LESION_CLASS_MAP,
    NUM_LESION_CLASSES,
    CocoDataset,
    build_lesion_model,
    save_checkpoint,
    load_checkpoint,
    _resolve_device,
    _collate_fn,
)

# ══════════════════════════════════════════════════════════════════════════════
# Data loaders  (patient-wise split — fixed here, not in builder)
# ══════════════════════════════════════════════════════════════════════════════


def build_data_loaders(
    logger,
    csv_path: str,
    label_class_map: dict,
    val_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    debug_n_images: int = 20,
) -> tuple:
    """
    Read the dataset CSV and return (train_loader, val_loader, num_classes).

    Split is done on unique patient_ids so no patient appears in both
    train and val sets.  Only rows with a non-null coco_file and whose
    both image_path and coco_file exist on disk are kept.

    Args:
        debug_n_images: if set, limits total dataset to this many rows
                        (for quick smoke-test runs e.g. 20 images)
    """
    logger.info("Loading dataset: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    # Keep only annotated rows
    df = df[df["coco_file"].notna()].copy()
    logger.info("  Rows with coco_file: %d", len(df))

    # Drop rows where files do not exist on disk
    exists_mask = df.apply(
        lambda r: Path(str(r["image_path"])).exists()
        and Path(str(r["coco_file"])).exists(),
        axis=1,
    )
    df = df[exists_mask].reset_index(drop=True)
    logger.info("  Rows with both files on disk: %d", len(df))

    # ── DEBUG: limit dataset size ─────────────────────────────────────────────
    if debug_n_images is not None:
        df = df.iloc[:debug_n_images].reset_index(drop=True)
        logger.info("  DEBUG mode: limited to %d images", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No annotated images found on disk. "
            "Check csv_path and path_rewrite settings."
        )

    # ── Patient-wise train / val split ───────────────────────────────────────
    if "patient_id" not in df.columns:
        raise RuntimeError("Column 'patient_id' is required for patient-wise split.")

    patient_ids = df["patient_id"].dropna().unique().tolist()
    if len(patient_ids) < 2:
        raise RuntimeError(
            f"Need ≥2 unique patient_ids for split, got {len(patient_ids)}."
        )

    # Shuffle patient list with fixed seed for reproducibility
    g = torch.Generator().manual_seed(seed)
    shuffled_ids = [
        patient_ids[i] for i in torch.randperm(len(patient_ids), generator=g).tolist()
    ]

    n_val = max(1, int(len(shuffled_ids) * val_split))
    n_val = min(n_val, len(shuffled_ids) - 1)  # always keep ≥1 train patient

    val_pids = set(shuffled_ids[:n_val])
    train_pids = set(shuffled_ids[n_val:])

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)

    logger.info(
        "  Patient-wise split → train_patients=%d (%d rows)  "
        "val_patients=%d (%d rows)",
        len(train_pids),
        len(train_df),
        len(val_pids),
        len(val_df),
    )

    # ── Wrap in CocoDataset (not raw DataFrame) ───────────────────────────────
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

    num_classes = 1 + len(label_class_map)
    logger.info(
        "  DataLoaders ready — num_classes=%d  label_map=%s",
        num_classes,
        label_class_map,
    )
    return train_loader, val_loader, num_classes


# ══════════════════════════════════════════════════════════════════════════════
# One training epoch
# ══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    epoch: int,
    logger,
    log_every: int = 10,
) -> dict:
    """
    Run one full pass over the training set.
    Mask R-CNN returns a dict of losses when called in train mode with targets.
    Returns averaged loss dict for the epoch.

    zero_grad() resets the gradients of all model parameters to zero (or sets them to None)
    before computing the gradients for the next batch. This is necessary because
    PyTorch accumulates gradients by default—each call to loss.backward() adds new
    gradients to the existing ones.  Without zeroing, gradients from previous batches
    would accumulate, leading to incorrect parameter updates.

    backward()- calculate gradients
    parameter = parameter - learning_rate * gradient
    parameters are weight and biases

    step() update parameters
    """
    model.train()

    total_losses = {}
    n_batches = 0
    t_epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass — returns loss dict in train mode
        loss_dict = model(images, targets)

        # Total loss = sum of all component losses
        total_loss = sum(loss_dict.values())

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate for logging
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        total_losses["total"] = total_losses.get("total", 0.0) + total_loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            avg_total = total_losses["total"] / n_batches
            logger.info(
                "  Epoch %d [%d/%d]  total_loss=%.4f",
                epoch,
                batch_idx + 1,
                len(loader),
                avg_total,
            )

    # Average over batches
    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    elapsed = time.time() - t_epoch_start
    logger.info(
        "Epoch %d  train  total=%.4f  "
        "cls=%.4f  box_reg=%.4f  mask=%.4f  obj=%.4f  rpn_box=%.4f  "
        "time=%.1fs",
        epoch,
        avg.get("total", 0.0),
        avg.get("loss_classifier", 0.0),
        avg.get("loss_box_reg", 0.0),
        avg.get("loss_mask", 0.0),
        avg.get("loss_objectness", 0.0),
        avg.get("loss_rpn_box_reg", 0.0),
        elapsed,
    )
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch: int,
    logger,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
) -> dict:
    """
    Run inference on the val set and compute mAP@50 / mAP@50:95
    using torchmetrics.detection.MeanAveragePrecision.

    Returns a metrics dict with keys:
        box_mAP50, box_mAP50_95, mask_mAP50, mask_mAP50_95
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        logger.warning(
            "torchmetrics not installed — skipping mAP validation. "
            "pip install torchmetrics"
        )
        return {}

    model.eval()
    metric_box = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    metric_mask = MeanAveragePrecision(iou_type="segm", class_metrics=False)

    has_masks = False
    n_images = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]

        # In eval mode, Mask R-CNN returns predictions (no targets needed)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            n_det = len(output["boxes"])

            pred = {
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            }
            if "masks" in output and n_det > 0:
                # Binarise soft masks: [N,1,H,W] → [N,H,W] uint8
                pred["masks"] = (
                    (output["masks"].cpu() > mask_threshold).squeeze(1).to(torch.uint8)
                )
                has_masks = True
            elif "masks" in output:
                pred["masks"] = torch.zeros(
                    (0, *images[0].shape[-2:]), dtype=torch.uint8
                )

            gt = {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            }
            if "masks" in target:
                gt["masks"] = target["masks"].cpu()

            metric_box.update([pred], [gt])
            if has_masks:
                metric_mask.update([pred], [gt])

        n_images += len(images)

    if n_images == 0:
        logger.warning("Validation loader was empty — no metrics computed.")
        return {}

    box_result = metric_box.compute()
    mask_result = metric_mask.compute() if has_masks else {}

    metrics = {
        "box_mAP50": round(float(box_result.get("map_50", 0.0)), 4),
        "box_mAP50_95": round(float(box_result.get("map", 0.0)), 4),
        "mask_mAP50": round(float(mask_result.get("map_50", 0.0)), 4),
        "mask_mAP50_95": round(float(mask_result.get("map", 0.0)), 4),
    }
    logger.info(
        "Epoch %d  val  box_mAP50=%.4f  box_mAP50:95=%.4f  "
        "mask_mAP50=%.4f  mask_mAP50:95=%.4f",
        epoch,
        metrics["box_mAP50"],
        metrics["box_mAP50_95"],
        metrics["mask_mAP50"],
        metrics["mask_mAP50_95"],
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════


def train(logger: logging.Logger, cfg: MaskRCNNConfig, csv_path: str, debug_n_images: int = None) -> None:
    """
    Full training pipeline:
        1. Build data loaders
        2. Build model
        3. Build optimiser + LR scheduler
        4. (Optional) resume from checkpoint
        5. Epoch loop: train → validate → checkpoint

    Args:
        debug_n_images: if set, limits total dataset to this many images
                        (useful for quick smoke tests, e.g. 20 images)
    """
    device = _resolve_device(logger, cfg.device)
    logger.info("=" * 60)
    logger.info("  Mask R-CNN Fine-Tuning  |  device=%s", device)
    if debug_n_images is not None:
        logger.info("  DEBUG MODE: training on %d images only", debug_n_images)
    logger.info("=" * 60)

    # Data
    train_loader, val_loader, num_classes = build_data_loaders(
        logger=logger,
        csv_path=csv_path,
        label_class_map=LESION_CLASS_MAP,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        debug_n_images=debug_n_images,
    )

    # Model
    model = build_lesion_model(
        logger=logger,
        num_classes=num_classes,
        device=str(device),
        pretrained_backbone=cfg.pretrained,
    )

    # Optimiser
    # Separate backbone params (lower LR) from head params (higher LR)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    """
    The backbone is pre-trained and learns general features, its weights are 
    only slightly adjusted (fine-tuned) during training. 
    The heads (RPN, box, and mask), which are often randomly initialized or 
    adapted to a new number of classes, need to learn more specific tasks from 
    the custom dataset and therefore use a higher learning rate to learn faster.
    This approach stabilizes training and prevents the pre-trained features in the 
    backbone from being destroyed by large gradient updates.
    
    Weight decay adds a penalty to the loss function based on the magnitude 
    of the model's weights.
    Its primary function is to prevent overfitting by encouraging the model to 
    learn simpler patterns.  It does this by:

    1. Regularizing the model, discouraging it from relying too heavily on any single 
    feature by keeping the weight values small. 
    2. Improving generalization, helping the model perform better on unseen data by 
    reducing its tendency to memorize noise in the training data. 
    3. Stabilizing training, preventing weights from growing too large, which can 
    make the learning process more robust. 

    CosineAnnealingLR is a PyTorch learning rate scheduler that decreases the learning 
    rate following a cosine curve. 

    It starts at the initial learning rate and smoothly reduces it to a specified 
    minimum (eta_min) over a set number of iterations (T_max). The formula used is:
    η_t = η_min + (1/2)(η_max - η_min)(1 + cos(π * T_cur / T_max)) where:
    T_cur is the current number of iterations (or epochs) since the last restart. 
    T_max is the maximum number of iterations (or epochs) in one cosine cycle.
    eta_min is the minimum learning rate (often referred to as n_min in your question)
    η_min is the minimum learning rate.  This is the lowest value the learning rate 
    will decay to, defined by the eta_min parameter.
    η_max is the initial (maximum) learning rate.  This is the starting learning rate,
    typically set to the initial learning rate of the optimizer.
    Suppose the learning rate started from .01 which is and min lr is 0.00025.
    Current iteration is 3 and max iteration is 5.
    The calculation is: η_t = 0.00025 + (1/2)(0.01 - 0.00025)(1 + cos(π * 3 / 5))
    The cosine term cos(π * 3 / 5) is cos(108°), which is approximately -0.309.
    This results in a learning rate of approximately 0.0027 at iteration 3, 
    smoothly decreasing from 0.01 towards 0.00025. 

    This gradual decay helps the model converge more stably 
    into a good minimum. After T_max iterations, the learning rate remains 
    constant at eta_min.

    """
    optimizer = optim.SGD(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # Cosine annealing — smoothly decays LR to near zero over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )

    # Resume from checkpoint
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"
    last_ckpt_path = checkpoint_dir / "last.pth"

    start_epoch = 1
    best_map50 = 0.0
    history = []

    resume_path = last_ckpt_path if last_ckpt_path.exists() else None
    if resume_path and Path(resume_path).exists():
        start_epoch, prev_metrics = load_checkpoint(
            model, optimizer, resume_path, device=str(device), logger=logger
        )
        best_map50 = prev_metrics.get("box_mAP50", 0.0)
        start_epoch += 1
        logger.info(
            "Resuming from epoch %d  (best box_mAP50=%.4f)",
            start_epoch,
            best_map50,
        )
    else:
        logger.warning("Resume path not found: %s — starting fresh.")

    # Epoch loop
    n_epochs = cfg.epochs

    for epoch in range(start_epoch, n_epochs + 1):
        logger.info("\n--- Epoch %d / %d ---", epoch, n_epochs)

        # Train
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            logger,
            log_every=cfg.log_every,
        )

        # LR step
        scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(
            "  LR after step: backbone=%.2e  heads=%.2e", current_lrs[0], current_lrs[1]
        )

        # Validate every val_every epochs (always on last epoch)
        """
        Two key thresholds mask and score govern detection and segmentation quality 
        The score threshold (typically set to 0.5) filters bounding box 
        proposals based on their classification confidence, ignoring any 
        detections with a score below this limit. This helps eliminate low-confidence 
        detections during inference.
        Additionally, Mask R-CNN uses Non-Maximum Suppression (NMS) after applying the 
        score threshold to remove overlapping bounding boxes, ensuring each 
        object is detected only once.

        The mask threshold applies to the raw, continuous-valued output mask 
        (typically in [0, 1]) generated by the mask head.  This threshold binarizes 
        the mask: pixels above the value (e.g., 0.5) are considered part of the 
        object; others are background. 
        A lower mask threshold (e.g., 0.3) results in larger, more inclusive masks, 
        which may capture faint object boundaries but risk including background noise. 
        A higher mask threshold (e.g., 0.7) produces tighter, more precise masks, 
        reducing false positives but potentially missing weak edges. 
        The mask threshold works because the mask head in Mask R-CNN performs 
        pixel-wise binary classification.
        If the pixel value > threshold → classified as foreground
        If the pixel value ≤ threshold → classified as background  
        """
        val_metrics = {}
        if epoch % cfg.val_every == 0 or epoch == n_epochs:
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                device,
                epoch,
                logger,
                score_threshold=cfg.score_threshold,
                mask_threshold=cfg.mask_threshold,
            )

        # Always save last checkpoint so training can be resumed
        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(model, optimizer, epoch, all_metrics, last_ckpt_path, logger)

        # Save best checkpoint based on box mAP@50
        current_map50 = val_metrics.get("box_mAP50", 0.0)
        if val_metrics and current_map50 > best_map50:
            best_map50 = current_map50
            save_checkpoint(
                model, optimizer, epoch, all_metrics, best_ckpt_path, logger
            )
            logger.info(
                "  ★ New best box_mAP50=%.4f — saved to %s",
                best_map50,
                best_ckpt_path,
            )

        # Epoch-level log entry
        history.append({"epoch": epoch, **all_metrics})

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)
    logger.info("Training complete. Best box_mAP50=%.4f", best_map50)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CONFIG_PATH     = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
    MASKRCNN_CONFIG = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\maskrcnn.ini"

    config       = load_config(CONFIG_PATH)
    maskrcnn_cfg = load_config(MASKRCNN_CONFIG)
    cfg          = MaskRCNNConfig(maskrcnn_cfg)
    logger       = getLogger(cfg.log_file)

    # Use csv_path from maskrcnn.ini
    csv_path = cfg.csv_path

    # ── DEBUG: limit to 20 images ─────────────────────────────────
    # Set to None to train on full dataset
    DEBUG_N_IMAGES = 20

    train(logger=logger, cfg=cfg, csv_path=csv_path, debug_n_images=DEBUG_N_IMAGES)
"""
mAP (mean Average Precision) is a key metric for evaluating object detection 
and instance segmentation models like Mask R-CNN.  It measures 
overall accuracy by combining precision (fraction of correct detections) 
and recall (fraction of detected objects). 

AP (Average Precision): Calculated per class as the area under the 
precision-recall curve at different confidence thresholds. 
mAP: The mean of AP across all classes.

Common variants:
mAP@50: Uses an IoU (Intersection over Union) threshold of 0.5—counts a 
detection as correct if predicted and ground truth boxes overlap by at 
least 50%. Easier to achieve. 
mAP@50:95: Averages mAP over IoU thresholds from 0.5 to 0.95 (step 0.05). 
Stricter and more comprehensive, commonly used in benchmarks like COCO. 
A higher mAP means better detection accuracy, balancing low false positives 
and false negatives.

Interpretation of Result

total_loss=1.2876: Overall loss combining all components.
cls=1.2735: High classification loss indicates the model struggles 
to assign correct labels to detected objects.

box_reg=0.0000: No bounding box regression loss—suggests no updates to 
predicted box coordinates.

mask=0.0000: No mask loss—indicates the mask branch is not being 
trained, a major issue for instance segmentation.

obj=0.0140: RPN (Region Proposal Network) objectness loss—low, 
meaning RPN is decent at distinguishing foreground vs. background.

rpn_box=0.0000: No RPN bounding box loss—RPN isn't learning to 
refine region proposals.

time=21.3s: Duration of the epoch. 

backbone=9.88e-04: Lower LR for backbone (feature extractor).
heads=9.85e-03: Higher LR for task-specific heads (e.g., classification, mask). 

box_mAP50=-1.0000, mask_mAP50=-1.0000, etc.: mAP = -1.0 indicates 
evaluation failed or was skipped—common when no valid detections 
are made or due to a bug (e.g., empty predictions, incorrect format, 
or missing ground truth matches).

Since mask=0.0000 and box_reg=0.0000 during training, the model likely 
isn't learning to predict masks or refine boxes, explaining the 
failed evaluation
"""