"""
utils/evaluation.py
-------------------
Evaluation utilities for Mask R-CNN lesion detection.

Computes:
  - Box mAP50  (IoU threshold 0.50)
  - Box mAP50-95  (COCO-style, average over IoU 0.50:0.05:0.95)
  - Mask mAP50
  - Mask mAP50-95
  - Per-class AP
  - Confusion-matrix style counts (TP/FP/FN)

Two modes:
  A. evaluate_model()  — runs the model on a DataLoader and collects results
  B. compute_map()     — pure metric computation from collected results

Works with:
  - torchvision Mask R-CNN output dicts
  - Our LesionAnnotatedDataset target dicts

Requires: torchvision >= 0.13 (which ships torchmetrics-compatible helpers)
Uses a lightweight custom implementation so no extra dep on pycocotools
(though pycocotools is noted as optional for strict COCO-style eval).
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# IoU helpers
# ══════════════════════════════════════════════════════════════════════════════


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes [N,4] and [M,4] (xyxy).
    Returns [N,M] tensor.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Binary mask IoU.  Masks are bool or uint8 arrays of same shape.
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return float(intersection) / (float(union) + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# AP computation (per class, single IoU threshold)
# ══════════════════════════════════════════════════════════════════════════════


def _compute_ap(
    all_tp: List[int],
    all_fp: List[int],
    n_gt: int,
) -> float:
    """
    Compute Average Precision from sorted TP/FP lists and total GT count.
    11-point interpolation (VOC-style).
    """
    if n_gt == 0:
        return 0.0

    tp = np.cumsum(all_tp)
    fp = np.cumsum(all_fp)

    recall = tp / (n_gt + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_recall = precision[recall >= t]
        ap += prec_at_recall.max() if len(prec_at_recall) else 0.0
    return ap / 11.0


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = None,
    use_masks: bool = False,
) -> Dict:
    """
    Compute mAP over all images.

    Parameters
    ----------
    predictions   : list of dicts {boxes, labels, scores, masks}
    ground_truths : list of dicts {boxes, labels, masks}
    num_classes   : total classes INCLUDING background (0)
    iou_thresholds: list of IoU thresholds  [default: 0.50:0.05:0.95]
    use_masks     : if True, compute mask-based IoU instead of box IoU

    Returns dict with:
      mAP50, mAP50_95, per_class_ap50, per_class_ap50_95
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

    # class indices excluding background (0)
    classes = list(range(1, num_classes))

    ap_per_class_per_threshold: Dict[float, Dict[int, float]] = defaultdict(dict)

    for iou_thresh in iou_thresholds:
        for cls in classes:
            # Gather all detections for this class across all images,
            # sorted by descending score
            det_scores: List[float] = []
            det_tp: List[int] = []
            det_fp: List[int] = []
            n_gt_total = 0

            for pred, gt in zip(predictions, ground_truths):
                # Ground-truth for this class
                gt_cls_mask = gt["labels"] == cls
                gt_boxes_cls = gt["boxes"][gt_cls_mask]
                n_gt = int(gt_cls_mask.sum())
                n_gt_total += n_gt

                # Predictions for this class
                pred_cls_mask = pred["labels"] == cls
                pred_boxes = pred["boxes"][pred_cls_mask]
                pred_scores = pred["scores"][pred_cls_mask]
                pred_masks = pred.get("masks", None)

                if pred_cls_mask.any():
                    pred_masks_cls = (
                        pred_masks[pred_cls_mask] if pred_masks is not None else None
                    )
                else:
                    pred_masks_cls = None

                if len(pred_boxes) == 0:
                    continue

                # Sort predictions by score
                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

                gt_matched = torch.zeros(n_gt, dtype=torch.bool)

                for pi in range(len(pred_boxes)):
                    score = float(pred_scores[pi])
                    det_scores.append(score)

                    if n_gt == 0:
                        det_fp.append(1)
                        det_tp.append(0)
                        continue

                    if use_masks and pred_masks_cls is not None:
                        # Mask IoU
                        pred_m = pred_masks_cls[order[pi], 0].numpy() > 0.5
                        ious_m = torch.tensor(
                            [
                                mask_iou(pred_m, gt["masks"][j].numpy().astype(bool))
                                for j in range(n_gt)
                            ]
                        )
                        ious = ious_m
                    else:
                        ious = box_iou(
                            pred_boxes[pi : pi + 1],
                            gt_boxes_cls,
                        )[0]

                    best_iou, best_j = (
                        ious.max(0)
                        if len(ious)
                        else (torch.tensor(0.0), torch.tensor(0))
                    )
                    best_iou = float(best_iou)
                    best_j = int(best_j)

                    if best_iou >= iou_thresh and not gt_matched[best_j]:
                        gt_matched[best_j] = True
                        det_tp.append(1)
                        det_fp.append(0)
                    else:
                        det_tp.append(0)
                        det_fp.append(1)

            # Sort all detections by score
            if det_scores:
                order_all = np.argsort(-np.array(det_scores))
                det_tp = [det_tp[i] for i in order_all]
                det_fp = [det_fp[i] for i in order_all]

            ap = _compute_ap(det_tp, det_fp, n_gt_total)
            ap_per_class_per_threshold[iou_thresh][cls] = ap

    # Aggregate
    def _mean_ap(thresh: float) -> float:
        aps = [v for v in ap_per_class_per_threshold[thresh].values()]
        return float(np.mean(aps)) if aps else 0.0

    map50 = _mean_ap(0.50)
    map50_95 = float(np.mean([_mean_ap(t) for t in iou_thresholds]))

    per_class_ap50 = {
        cls: ap_per_class_per_threshold[0.50].get(cls, 0.0) for cls in classes
    }

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_class_ap50": {int(k): round(v, 4) for k, v in per_class_ap50.items()},
        "iou_thresholds": iou_thresholds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model evaluation loop
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_model(
    logger,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    score_threshold: float,
    mask_threshold: float,
    class_names: Optional[List[str]] = None,
    use_masks: bool = True,
) -> Dict:
    """
    Run model on dataloader and compute box + mask mAP.

    Returns dict:
      {
        "box_mAP50":    float,
        "box_mAP50_95": float,
        "mask_mAP50":   float,
        "mask_mAP50_95":float,
        "per_class_box_ap50": {class_idx: ap},
        "per_class_mask_ap50":{class_idx: ap},
        "class_names":  {class_idx: name},
        "inference_time_per_image_sec": float,
        "num_images":   int,
      }
    """
    model.eval()
    all_preds: List[Dict] = []
    all_gts: List[Dict] = []

    total_time = 0.0
    num_images = 0

    logger.info("Starting evaluation on %d batches …", len(dataloader))

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            t0 = time.time()
            outputs = model(images)
            total_time += time.time() - t0
            num_images += len(images)

            for out, tgt in zip(outputs, targets):
                # Move to CPU for metric computation
                pred = {
                    "boxes": out["boxes"].cpu(),
                    "labels": out["labels"].cpu(),
                    "scores": out["scores"].cpu(),
                    "masks": out["masks"].cpu() if "masks" in out else None,
                }
                # Apply score threshold
                keep = pred["scores"] >= score_threshold
                pred = {
                    k: (
                        v[keep]
                        if v is not None and k != "scores"
                        else (v[keep] if k == "scores" else v)
                    )
                    for k, v in pred.items()
                }

                gt = {
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                    "masks": tgt["masks"].cpu() if "masks" in tgt else None,
                }
                all_preds.append(pred)
                all_gts.append(gt)

            if batch_idx % 10 == 0:
                logger.info("  Eval batch %d / %d", batch_idx + 1, len(dataloader))

    logger.info("Running box mAP computation …")
    box_metrics = compute_map(all_preds, all_gts, num_classes, use_masks=False)

    mask_metrics: Dict = {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_ap50": {}}
    if use_masks and all(p.get("masks") is not None for p in all_preds):
        logger.info("Running mask mAP computation …")
        mask_metrics = compute_map(all_preds, all_gts, num_classes, use_masks=True)

    cnames = class_names or {i: f"class_{i}" for i in range(1, num_classes)}
    if isinstance(cnames, list):
        cnames = {i: cnames[i] for i in range(len(cnames))}

    result = {
        "box_mAP50": box_metrics["mAP50"],
        "box_mAP50_95": box_metrics["mAP50_95"],
        "mask_mAP50": mask_metrics["mAP50"],
        "mask_mAP50_95": mask_metrics["mAP50_95"],
        "per_class_box_ap50": box_metrics["per_class_ap50"],
        "per_class_mask_ap50": mask_metrics.get("per_class_ap50", {}),
        "class_names": cnames,
        "inference_time_per_image_sec": round(total_time / max(num_images, 1), 4),
        "num_images": num_images,
    }

    _log_eval_summary(result, cnames)
    return result


def _log_eval_summary(logger, result: Dict, class_names: Dict) -> None:
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("  Images evaluated       : %d", result["num_images"])
    logger.info(
        "  Avg inference / image  : %.4f s", result["inference_time_per_image_sec"]
    )
    logger.info("  Box  mAP@50            : %.4f", result["box_mAP50"])
    logger.info("  Box  mAP@50:95         : %.4f", result["box_mAP50_95"])
    logger.info("  Mask mAP@50            : %.4f", result["mask_mAP50"])
    logger.info("  Mask mAP@50:95         : %.4f", result["mask_mAP50_95"])
    logger.info("  Per-class Box AP@50:")
    for cls_idx, ap in result["per_class_box_ap50"].items():
        cname = class_names.get(cls_idx, f"cls_{cls_idx}")
        logger.info("    %-20s : %.4f", cname, ap)
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation report saver
# ══════════════════════════════════════════════════════════════════════════════


def save_eval_report(logger, result: Dict, path, class_names=None) -> None:
    """Save evaluation metrics to a JSON file."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {**result}
    if class_names:
        report["class_names"] = (
            class_names
            if isinstance(class_names, dict)
            else {i: n for i, n in enumerate(class_names)}
        )

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved → %s", path)
