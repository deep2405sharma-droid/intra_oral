"""
train_resnet.py
----------------
Fine-tunes a ResNet50 classifier on the annotated SMART intraoral
dataset for 3-class image classification:
    0 = normal, 1 = opmd, 2 = variation

Differences from train_unet.py (segmentation):
  - No mask/coco_file needed — images are fed directly to ResNet
  - Output: single class label per image (not pixel-wise mask)
  - Loss: weighted CrossEntropyLoss (not Dice + Tversky)
  - Metrics: Accuracy, F1, Precision, Recall per class (not Dice/IoU)
  - Two separate LRs: backbone_lr (fine-tune) + head_lr (FC head)

Pipeline
--------
1. get_dataset_path()    ->  build/load training CSV
2. build_data_loaders()  ->  patient-wise train / val split
3. build_lesion_model()  ->  ResNet50 + custom FC head
4. Training loop         ->  weighted CE loss + cosine LR decay
5. Validation loop       ->  accuracy, F1 (via sklearn classification_report)
6. Checkpointing         ->  best val_macro_f1 model saved

Dataset contract
----------------
The CSV must have at minimum:
    image_path   : absolute path to the image
    label        : 'normal' | 'opmd' | 'variation'
    patient_id   : used for patient-wise train/val split

Usage
-----
    python -m src.classification.resnet.train_resnet
    python -m src.classification.resnet.train_resnet -p kaggle
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.classification.resnet.resnet_builder import (
    LABEL_CLASS_MAP,
    NUM_CLASSES,
    CLASS_WEIGHTS,
    build_lesion_model,
    build_data_loaders,
    save_checkpoint,
    load_checkpoint,
    _resolve_device,
)
from src.classification.resnet.resnet_config import ResNetConfig

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

# Suppress noisy library logs
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Class names in index order (0=normal, 1=opmd, 2=variation), derived from
# LABEL_CLASS_MAP so it can never drift out of sync with the model's output
# indices.
CLASS_NAMES = [k for k, v in sorted(LABEL_CLASS_MAP.items(), key=lambda x: x[1])]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Batch accuracy from raw logits.

    logits  : [B, num_classes] — raw model output (pre-softmax)
    targets : [B]              — ground truth class index per image
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Hand-rolled classification metrics from batched predictions/targets.
    Still used for train_one_epoch(); validate_one_epoch() now uses
    sklearn.metrics.classification_report instead (see below) since it
    gives macro/weighted averages for free and is less error-prone.

    preds   : [N] — predicted class index per image (argmax of logits)
    targets : [N] — ground truth class index per image
    """
    results = {}
    smooth  = 1e-6

    preds_np   = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Overall accuracy
    results["accuracy"] = float((preds_np == targets_np).mean())

    class_names = {v: k for k, v in LABEL_CLASS_MAP.items()}
    f1_list     = []

    for c in range(num_classes):
        tp = int(((preds_np == c) & (targets_np == c)).sum())
        fp = int(((preds_np == c) & (targets_np != c)).sum())
        fn = int(((preds_np != c) & (targets_np == c)).sum())

        precision = (tp + smooth) / (tp + fp + smooth)
        recall    = (tp + smooth) / (tp + fn + smooth)
        f1        = 2 * precision * recall / (precision + recall + smooth)
        f1_list.append(f1)

        name = class_names.get(c, str(c))
        results[f"precision_{name}"] = round(precision, 4)
        results[f"recall_{name}"]    = round(recall,    4)
        results[f"f1_{name}"]        = round(f1,        4)

    results["f1_macro"] = round(float(np.mean(f1_list)), 4)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# One training epoch
# ══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model, optimizer, criterion, loader, device,
    epoch: int, logger, log_every: int = 10,
) -> dict:
    """
    Run one full pass over the training set.

    ResNet returns raw logits [B, num_classes] — CrossEntropyLoss applies
    softmax internally, so no activation is needed from the model.
    """
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0
    t_start    = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)   # [B, 3, H, W]
        labels = labels.to(device)   # [B]

        # Forward pass
        logits = model(images)       # [B, num_classes]
        loss   = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)  # [B] predicted class
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            avg = total_loss / n_batches
            logger.info(
                "Epoch %d | batch %d/%d | loss=%.4f | avg_loss=%.4f",
                epoch, batch_idx + 1, len(loader), loss.item(), avg,
            )

    avg_loss = total_loss / max(n_batches, 1)
    elapsed  = time.time() - t_start

    # Training metrics
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_metrics = compute_metrics(all_preds, all_labels)

    logger.info(
        "Epoch %d  train  avg_loss=%.4f  accuracy=%.4f  f1_macro=%.4f  time=%.1fs",
        epoch, avg_loss, train_metrics["accuracy"], train_metrics["f1_macro"], elapsed,
    )
    return {"train_loss": avg_loss, **{f"train_{k}": v for k, v in train_metrics.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def validate_one_epoch(
    model, criterion, loader, device,
    epoch: int, logger, num_classes: int = NUM_CLASSES,
    output_dir: Path = None, val_df: pd.DataFrame = None,
) -> dict:
    """
    Run inference on the val set and compute classification metrics via
    sklearn.metrics.classification_report (macro/weighted precision,
    recall, F1 — plus per-class breakdown).

    Also rebuilds val_results.csv: one row per validation sample with
    patient_id/image_path/true_label/pred_label/correct, so predictions
    can be traced back to specific patients — same pattern as U-Net's
    val_results.csv.

    NOTE: the row-alignment against val_df assumes `loader` (the val
    DataLoader) does NOT shuffle and does NOT drop the last batch — i.e.
    sample order out of the loader matches val_df's row order exactly.
    If build_data_loaders() ever turns on shuffling for the val split,
    this alignment will silently break — double check that before relying
    on val_results.csv.

    @torch.no_grad() disables gradient computation — saves memory and
    speeds up inference since we're not calling .backward() here.
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    all_preds  = []
    all_targets = []
    results    = []
    n_batches  = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, labels)
        n_batches  += 1

        batch_preds = logits.argmax(dim=1)
        all_preds.append(batch_preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

        # Per-sample result rows — iterate every sample in the batch
        # (previous version only ever recorded sample 0 per batch, which
        # produced a val_results.csv that was both wrong-sized and
        # misaligned with val_df for batch_size > 1).
        batch_size = labels.size(0)
        for j in range(batch_size):
            global_idx = i * loader.batch_size + j
            patient_id = ""
            image_path = ""
            true_label = ""
            if val_df is not None and global_idx < len(val_df):
                row        = val_df.iloc[global_idx]
                patient_id = row.get("patient_id", "")
                image_path = str(row.get("image_path", ""))
                true_label = str(row.get("label", ""))

            results.append({
                "image_id":   global_idx,
                "patient_id": patient_id,
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": CLASS_NAMES[batch_preds[j].item()],
                "correct":    int(batch_preds[j].item() == labels[j].item()),
            })

    if n_batches == 0:
        logger.warning("Validation loader was empty — no metrics computed.")
        return {}

    avg_loss = total_loss / n_batches
    avg_acc  = total_acc / n_batches

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1          = report["macro avg"]["f1-score"]
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall    = report["weighted avg"]["recall"]
    weighted_f1        = report["weighted avg"]["f1-score"]

    logger.info(
        "Epoch %d  val  loss=%.4f  acc=%.4f  macro_f1=%.4f weighted_f1=%.4f "
        "weighted_precision=%.4f weighted_recall=%.4f",
        epoch, avg_loss, avg_acc, macro_f1, weighted_f1,
        weighted_precision, weighted_recall,
    )
    logger.info(
        "  Per-class F1: %s",
        {name: round(report[name]["f1-score"], 4) for name in CLASS_NAMES},
    )
    # Recall specifically called out — this is the metric that has been
    # driving the class-weight tuning (OPMD getting misclassified as
    # Variation shows up here first).
    logger.info(
        "  Per-class Recall: %s",
        {name: round(report[name]["recall"], 4) for name in CLASS_NAMES},
    )

    # Save per-sample results CSV
    if output_dir is not None:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "val_results.csv", index=False)

    return {
        "val_loss":      round(avg_loss, 4),
        "val_acc":       round(avg_acc, 4),
        "val_macro_f1":  round(macro_f1, 4),
        "val_weighted_f1": round(weighted_f1, 4),
        # Kept for training_history.json / potential downstream use, but
        # stripped out before json.dump() — see history_entry below.
        "_y_true": y_true,
        "_y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════


def train(logger, cfg: ResNetConfig, csv_path: str) -> None:
    """
    Full ResNet classification training pipeline:
        1. Build data loaders (patient-wise split)
        2. Build model (ResNet50 + custom FC head)
        3. Build weighted CrossEntropy loss
        4. Build optimizer (separate LR for backbone and head)
        5. Build LR scheduler (cosine / step / plateau)
        6. Resume from checkpoint if available
        7. Epoch loop: train -> validate -> checkpoint
    """
    device = _resolve_device(logger, cfg.device)
    logger.info("=" * 60)
    logger.info("  ResNet50 Classification  |  device=%s", device)
    logger.info("=" * 60)

    # Output dirs
    output_dir     = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, val_loader, num_classes = build_data_loaders(
        logger=logger,
        csv_path=csv_path,
        label_class_map=LABEL_CLASS_MAP,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        input_size=cfg.input_size,
        weighted_sampler=cfg.weighted_sampler,
    )

    # Reconstruct val_df (same seed/split) for per-sample result logging
    try:
        from sklearn.model_selection import GroupShuffleSplit
        _full_df = pd.read_csv(csv_path, dtype=str)
        _full_df = _full_df[_full_df["label"].isin(LABEL_CLASS_MAP.keys())].reset_index(drop=True)
        _gss     = GroupShuffleSplit(n_splits=1, test_size=cfg.val_split, random_state=cfg.seed)
        _, _val_idx = next(_gss.split(_full_df, groups=_full_df["patient_id"].values))
        val_df   = _full_df.iloc[_val_idx].reset_index(drop=True)
    except Exception:
        val_df = None

    # Model
    model = build_lesion_model(
        logger=logger,
        num_classes=num_classes,
        device=str(device),
        pretrained=cfg.pretrained,
        backbone=cfg.backbone,
        dropout=cfg.dropout,
    )

    # Loss — weighted CrossEntropy so rare OPMD class is penalised more
    weights   = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer — two param groups, same pattern as unet backbone_lr / head_lr
    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    head_params     = list(model.fc.parameters())

    optimizer = optim.SGD(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params,     "lr": cfg.head_lr},
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    if cfg.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif cfg.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_patience, gamma=cfg.lr_factor
        )
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=cfg.lr_patience,
            factor=cfg.lr_factor, min_lr=cfg.min_lr,
        )

    # Checkpoint paths
    best_ckpt_path = checkpoint_dir / "best.pth"
    last_ckpt_path = checkpoint_dir / "last.pth"

    start_epoch = 1
    best_f1     = 0.0
    history     = []

    # Resume from checkpoint if available
    if best_ckpt_path.exists():
        resume_path = best_ckpt_path
    elif last_ckpt_path.exists():
        resume_path = last_ckpt_path
    else:
        resume_path = None

    if resume_path:
        start_epoch, prev_metrics = load_checkpoint(
            logger=logger, model=model, optimizer=optimizer,
            path=resume_path, device=str(device),
        )
        best_f1      = prev_metrics.get("val_macro_f1", 0.0)
        start_epoch += 1
        logger.info("Resuming from epoch %d  (best val_macro_f1=%.4f)", start_epoch, best_f1)
    else:
        logger.info("No checkpoint found — starting fresh training.")

    # Epoch loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        logger.info("\n--- Epoch %d / %d ---", epoch, cfg.epochs)

        train_metrics = train_one_epoch(
            model, optimizer, criterion, train_loader,
            device, epoch, logger, log_every=cfg.log_every,
        )

        # LR step
        if cfg.lr_scheduler in ("cosine", "step"):
            scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(
            "  LR after step: backbone=%.2e  head=%.2e",
            current_lrs[0], current_lrs[1],
        )

        # Validation
        val_metrics = {}
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = validate_one_epoch(
                model, criterion, val_loader, device,
                epoch, logger, num_classes=num_classes,
                output_dir=output_dir, val_df=val_df,
            )
            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_metrics.get("val_macro_f1", 0.0))

        # Save last checkpoint (enables resume)
        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(logger, model, optimizer, epoch, all_metrics, last_ckpt_path)

        # Save best checkpoint based on val_macro_f1
        # Macro F1 is preferred over accuracy due to class imbalance —
        # accuracy is inflated by the dominant normal class
        current_f1 = val_metrics.get("val_macro_f1", 0.0)
        if val_metrics and current_f1 > best_f1:
            best_f1 = current_f1
            save_checkpoint(logger, model, optimizer, epoch, all_metrics, best_ckpt_path)
            logger.info("  New best val_macro_f1=%.4f — saved to %s", best_f1, best_ckpt_path)

        # _y_true/_y_pred are numpy arrays — not JSON-serializable, so strip
        # any underscore-prefixed keys before they hit training_history.json.
        history_entry = {
            "epoch": epoch,
            **{k: v for k, v in all_metrics.items() if not k.startswith("_")},
        }
        history.append(history_entry)

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved: %s", history_path)
    logger.info("Training complete. Best val_macro_f1=%.4f", best_f1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset path builder  (mirrors get_dataset_path in train_unet.py)
# ══════════════════════════════════════════════════════════════════════════════
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


def get_dataset_path(logger, config, cfg: ResNetConfig) -> str:
    """
    Build the training CSV from merged + augmented datasets.
    ResNet only needs image_path, label, patient_id —
    no coco_file or mask_path columns required.
    """
    smart_merged_basepath = config.get("TRAIN", "smart.merged.basepath")
    smart_base            = config.get("TRAIN", "smart.basepath")
    smartom_base          = config.get("TRAIN", "smartom.basepath")
    augment_smart_base    = config.get("TRAIN", "augment.smart.baseroot")
    augment_smartom_base  = config.get("TRAIN", "augment.smartom.baseroot")

    merged_data_path = (
        f"{smart_merged_basepath}/"
        f"{config.get('SMART_MERGED', 'merged.output.filename')}"
    )
    aug_smart_csv = (
        f"{augment_smart_base}/"
        f"{config.get('AUGMENT_SMART', 'augment.patient.metadata.filename')}"
    )
    aug_smartom_csv = (
        f"{augment_smartom_base}/"
        f"{config.get('AUGMENT_SMARTOM', 'augment.patient.metadata.filename')}"
    )

    dataset = pd.DataFrame()

    # Load merged dataset
    if Path(merged_data_path).exists():
        merged_df = pd.read_csv(merged_data_path)
        # Prepend base paths depending on source
        m1 = merged_df["source"] == "smart_II"
        m2 = merged_df["source"] == "smart_om"
        if "image_path" in merged_df.columns:
            merged_df.loc[m1, "image_path"] = smart_base   + "/" + merged_df.loc[m1, "image_path"].astype(str)
            merged_df.loc[m2, "image_path"] = smartom_base + "/" + merged_df.loc[m2, "image_path"].astype(str)
        logger.info("Loaded merged data: %s", merged_df.shape)
        dataset = pd.concat([dataset, merged_df])
    else:
        raise FileNotFoundError(f"Merged dataset not found: {merged_data_path}")

    # Augmented SMART-II
    if Path(aug_smart_csv).exists():
        aug_s = pd.read_csv(aug_smart_csv)
        if "image_path" in aug_s.columns:
            aug_s["image_path"] = augment_smart_base + "/" + aug_s["image_path"].astype(str)
        dataset = pd.concat([dataset, aug_s])
        logger.info("Added SMART-II augmented: %s", aug_s.shape)

    # Augmented SMART-OM
    if Path(aug_smartom_csv).exists():
        aug_om = pd.read_csv(aug_smartom_csv)
        if "image_path" in aug_om.columns:
            aug_om["image_path"] = augment_smartom_base + "/" + aug_om["image_path"].astype(str)
        dataset = pd.concat([dataset, aug_om])
        logger.info("Added SMART-OM augmented: %s", aug_om.shape)

    logger.info("Final dataset shape: %s", dataset.shape)
    logger.info("Label distribution:\n%s", dataset["label"].value_counts().to_string())

    # Save training CSV
    train_csv_path = config.get("CLASSIFICATION-RESNET", "train.dataset")
    Path(train_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df = dataset[dataset["label"].isin(LABEL_CLASS_MAP.keys())].copy()
    df = df[df["image_path"].notna()].copy()
    df.to_csv(train_csv_path, index=False)
    logger.info("Training CSV saved: %s  (%d rows)", train_csv_path, len(df))
    return train_csv_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

# same as unet
def get_configpath():
    parser = argparse.ArgumentParser(description="ResNet classification training")
    parser.add_argument("-p", "--profile")
    args        = parser.parse_args()
    config_path = "config/config.ini"
    if args.profile and args.profile.lower() == "kaggle":
        config_path = "config/kaggle_config.ini"
    return config_path


if __name__ == "__main__":
    config_path = get_configpath()
    config      = load_config(config_path)
    resnet_ini  = config.get("CLASSIFICATION-RESNET", "resnet.config")
    logger      = initialize_logger(resnet_ini)
    cfg         = ResNetConfig(resnet_ini)
    csv_path    = get_dataset_path(logger=logger, config=config, cfg=cfg)
    train(logger=logger, cfg=cfg, csv_path=csv_path)