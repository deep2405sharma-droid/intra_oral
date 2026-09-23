"""
train_efficientnet.py
-----------------------
Fine-tunes an EfficientNet classifier on the same annotated SMART
intraoral dataset as train_resnet.py, for the same 3-class
(normal/opmd/variation) image classification task.

This is a third pipeline, not a replacement — same classes, same
dataset, same training loop as train_resnet.py; only the backbone
architecture differs. Reuses get_dataset_path() directly from
train_resnet.py rather than duplicating it, since building the training
CSV doesn't depend on which classifier architecture will read it.

Usage
-----
    python -m src.classification.efficientnet.train_efficientnet -p kaggle
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
from src.classification.efficientnet.efficientnet_builder import (
    LABEL_CLASS_MAP,
    NUM_CLASSES,
    CLASS_WEIGHTS,
    build_efficientnet_model,
    build_data_loaders,
    save_checkpoint,
    load_checkpoint,
    _resolve_device,
)
from src.classification.efficientnet.efficientnet_config import EfficientNetConfig
from src.classification.resnet.train_resnet import get_dataset_path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

CLASS_NAMES = [k for k, v in sorted(LABEL_CLASS_MAP.items(), key=lambda x: x[1])]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics — identical to train_resnet.py
# ══════════════════════════════════════════════════════════════════════════════


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int = NUM_CLASSES,
) -> dict:
    """Hand-rolled metrics used for train_one_epoch(); validate_one_epoch()
    uses sklearn.metrics.classification_report instead — same split as
    train_resnet.py."""
    results = {}
    smooth  = 1e-6

    preds_np   = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

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
# One training epoch — identical to train_resnet.py
# ══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model, optimizer, criterion, loader, device,
    epoch: int, logger, log_every: int = 10,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0
    t_start    = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
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

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_metrics = compute_metrics(all_preds, all_labels)

    logger.info(
        "Epoch %d  train  avg_loss=%.4f  accuracy=%.4f  f1_macro=%.4f  time=%.1fs",
        epoch, avg_loss, train_metrics["accuracy"], train_metrics["f1_macro"], elapsed,
    )
    return {"train_loss": avg_loss, **{f"train_{k}": v for k, v in train_metrics.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch — identical to train_resnet.py, including the
# per-sample val_results.csv fix and the classification_report metrics.
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def validate_one_epoch(
    model, criterion, loader, device,
    epoch: int, logger, num_classes: int = NUM_CLASSES,
    output_dir: Path = None, val_df: pd.DataFrame = None,
) -> dict:
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
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    macro_f1           = report["macro avg"]["f1-score"]
    weighted_precision  = report["weighted avg"]["precision"]
    weighted_recall     = report["weighted avg"]["recall"]
    weighted_f1         = report["weighted avg"]["f1-score"]

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
    logger.info(
        "  Per-class Recall: %s",
        {name: round(report[name]["recall"], 4) for name in CLASS_NAMES},
    )

    if output_dir is not None:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "val_results.csv", index=False)

    return {
        "val_loss": round(avg_loss, 4),
        "val_acc": round(avg_acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_weighted_f1": round(weighted_f1, 4),
        "_y_true": y_true,
        "_y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop — same structure as train_resnet.py
# ══════════════════════════════════════════════════════════════════════════════


def train(logger, cfg: EfficientNetConfig, csv_path: str) -> None:
    device = _resolve_device(logger, cfg.device)
    logger.info("=" * 60)
    logger.info("  EfficientNet Classification  |  device=%s", device)
    logger.info("=" * 60)

    output_dir     = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    model = build_efficientnet_model(
        logger=logger,
        num_classes=num_classes,
        device=str(device),
        pretrained=cfg.pretrained,
        backbone=cfg.backbone,
        dropout=cfg.dropout,
    )

    weights   = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Two param groups: model.features (pretrained backbone) vs
    # model.classifier (fresh head) — EfficientNet's equivalent split to
    # ResNet's "everything except fc" vs "fc".
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("classifier")]
    head_params     = list(model.classifier.parameters())

    optimizer = optim.SGD(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params,     "lr": cfg.head_lr},
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    if cfg.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif cfg.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_patience, gamma=cfg.lr_factor
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=cfg.lr_patience,
            factor=cfg.lr_factor, min_lr=cfg.min_lr,
        )

    best_ckpt_path = checkpoint_dir / "best.pth"
    last_ckpt_path = checkpoint_dir / "last.pth"

    start_epoch = 1
    best_f1     = 0.0
    history     = []

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

    for epoch in range(start_epoch, cfg.epochs + 1):
        logger.info("\n--- Epoch %d / %d ---", epoch, cfg.epochs)

        train_metrics = train_one_epoch(
            model, optimizer, criterion, train_loader,
            device, epoch, logger, log_every=cfg.log_every,
        )

        if cfg.lr_scheduler in ("cosine", "step"):
            scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(
            "  LR after step: backbone=%.2e  head=%.2e", current_lrs[0], current_lrs[1],
        )

        val_metrics = {}
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = validate_one_epoch(
                model, criterion, val_loader, device,
                epoch, logger, num_classes=num_classes,
                output_dir=output_dir, val_df=val_df,
            )
            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_metrics.get("val_macro_f1", 0.0))

        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(logger, model, optimizer, epoch, all_metrics, last_ckpt_path)

        current_f1 = val_metrics.get("val_macro_f1", 0.0)
        if val_metrics and current_f1 > best_f1:
            best_f1 = current_f1
            save_checkpoint(logger, model, optimizer, epoch, all_metrics, best_ckpt_path)
            logger.info("  New best val_macro_f1=%.4f — saved to %s", best_f1, best_ckpt_path)

        history_entry = {
            "epoch": epoch,
            **{k: v for k, v in all_metrics.items() if not k.startswith("_")},
        }
        history.append(history_entry)

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved: %s", history_path)
    logger.info("Training complete. Best val_macro_f1=%.4f", best_f1)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def get_configpath():
    parser = argparse.ArgumentParser(description="EfficientNet classification training")
    parser.add_argument("-p", "--profile")
    args        = parser.parse_args()
    config_path = "config/config.ini"
    if args.profile and args.profile.lower() == "kaggle":
        config_path = "config/kaggle_config.ini"
    return config_path


if __name__ == "__main__":
    config_path = get_configpath()
    config      = load_config(config_path)

    efficientnet_ini = config.get("CLASSIFICATION-EFFICIENTNET", "efficientnet.config")
    logger            = initialize_logger(config)
    cfg               = EfficientNetConfig(load_config(efficientnet_ini))

    # csv_path reuses the SAME dataset-building function and the SAME
    # underlying CSV as train_resnet.py — the raw images/labels don't
    # depend on which classifier architecture will train on them.
    # get_dataset_path() only needs cfg.normal_dataset, which exists on
    # both ResNetConfig and EfficientNetConfig with identical meaning, so
    # a ResNetConfig instance is built here purely to satisfy that
    # function's signature — no ResNet training happens in this script.
    from src.classification.resnet.resnet_config import ResNetConfig
    resnet_ini_path = config.get("CLASSIFICATION-RESNET", "resnet.config")
    resnet_cfg      = ResNetConfig(load_config(resnet_ini_path))
    csv_path        = get_dataset_path(logger=logger, config=config, cfg=resnet_cfg)

    train(logger=logger, cfg=cfg, csv_path=csv_path)
