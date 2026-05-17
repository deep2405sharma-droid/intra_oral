"""
inference_zeroshot.py
---------------------------------
Zero-shot lesion segmentation using COCO-pretrained Mask R-CNN.

What it does
------------
1.  Reads smart_merged.csv (path from --csv or config default).
2.  Optionally filters by lesion_location (--location flag).
3.  Runs COCO-pretrained Mask R-CNN on each image (no fine-tuning).
4.  Since COCO has no oral-anatomy classes, ALL detections above
    score_threshold are treated as candidate lesion masks.
5.  Saves per-image results:
      - annotated PNG with bounding boxes + masks overlaid
      - inference_results.csv with per-image scores
6.  Evaluates against COCO ground-truth annotations (rows with a
    non-null coco_file column) using torchmetrics mAP.

Changes vs previous version
----------------------------
- Evaluation now uses the coco_file column instead of the removed
  json_file / lesion_location=='json_file' approach.
  Each image row already has its own per-image COCO JSON path; no
  separate filtering step is needed.
- Removed dependency on utils.annotation_parser / utils.evaluation
  (replaced with torchmetrics.detection.MeanAveragePrecision).
- CocoDataset from mask_rcnn_builder is used for the eval loop so
  ground-truth loading is consistent between zero-shot and fine-tuning.
- rewrite_path helper added inline (was commented-out import).
- build_coco_pretrained() now receives logger as first argument.

Usage
-----
    python inference_zeroshot.py \
        --csv ./data/smart_merged/patient_metadata/smart_merged.csv \
        --output_dir ./outputs/zeroshot \
        --score_threshold 0.5 \
        --location ALL \
        --base_rewrite "C:/Users/User/Documents/ManthanShala/v18hub/Projects/intraoral_leison=/data/intraoral_leison"
"""

import argparse
import json
import sys
import time
import cv2
from pathlib import Path
import numpy as np

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage, ImageDraw
from torchmetrics.detection import MeanAveragePrecision

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.intraoral_logger import getLogger
from utils.load_configuration import load_config
from Segmentation.maskrcnnconfig import MaskRCNNConfig
from Segmentation.mask_rcnn_builder import (
    COCO_CLASS_NAMES,
    LESION_CLASS_MAP,
    CocoDataset,
    build_coco_pretrained,
    _resolve_device,
)

# ── Colours for overlaying multiple detections ────────────────────────────────
_COLOURS = [
    (255, 70, 70),
    (70, 255, 145),
    (70, 138, 255),
    (255, 208, 70),
    (255, 70, 240),
    (70, 255, 255),
]


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════


def overlay_predictions(
    image: PILImage.Image,
    boxes,
    labels,
    scores,
    masks,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> PILImage.Image:
    """Draw boxes + semi-transparent masks on a PIL image copy."""
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out)
    W, H = image.size

    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        if float(score) < score_threshold:
            continue
        colour = _COLOURS[i % len(_COLOURS)]

        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(*colour, 255), width=2)

        cls_name = (
            COCO_CLASS_NAMES[int(label)]
            if int(label) < len(COCO_CLASS_NAMES)
            else str(label)
        )
        draw.text((x1 + 2, y1 + 2), f"{cls_name} {score:.2f}", fill=(*colour, 255))

        if mask is not None:
            import numpy as np
            from PIL import Image as _PIL

            binary = (mask[0].numpy() > mask_threshold).astype("uint8") * 120
            overlay = _PIL.fromarray(binary, mode="L")
            colour_img = _PIL.new("RGBA", (W, H), (*colour, 0))
            colour_img.putalpha(overlay)
            out = PILImage.alpha_composite(out, colour_img)
            draw = ImageDraw.Draw(out)

    return out.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth loader from per-image COCO JSON
# ══════════════════════════════════════════════════════════════════════════════


def load_gt_from_coco(coco_path: Path, img_path: Path) -> dict | None:
    """
    Load ground-truth boxes, labels and masks from a per-image COCO JSON.

    The COCO JSON produced by the VIA→COCO converter covers a single image.
    Format:
        {
            "images":      [{ "id", "file_name", "width", "height" }],
            "annotations": [{ "id", "image_id", "category_id",
                              "bbox": [x,y,w,h], "segmentation": [[x1,y1,...]],
                              "area", "iscrowd" }],
            "categories":  [{ "id", "name" }]
        }

    Returns a dict compatible with torchmetrics MeanAveragePrecision:
        { "boxes": Tensor[N,4], "labels": Tensor[N], "masks": Tensor[N,H,W] }
    or None if there are no valid annotations.
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Image dimensions
    images = coco.get("images", [])
    if not images:
        return None
    img_info = images[0]
    H, W = img_info["height"], img_info["width"]

    # category_id → class label (all → 1 for binary zero-shot evaluation)
    # (multi-class fine-tuning evaluation would map per LESION_CLASS_MAP)
    cat_id_to_label = {cat["id"]: 1 for cat in coco.get("categories", [])}

    boxes, labels, masks = [], [], []

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue

        class_label = cat_id_to_label.get(ann["category_id"], 1)

        # bbox [x, y, w, h] → [x1, y1, x2, y2]
        x, y, bw, bh = ann["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(W, x + bw), min(H, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue

        # Rasterise segmentation polygon → binary mask
        canvas = np.zeros((H, W), dtype=np.uint8)
        for ring in ann.get("segmentation", []):
            if len(ring) < 6:
                continue
            pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(canvas, [pts], color=1)

        boxes.append([x1, y1, x2, y2])
        labels.append(class_label)
        masks.append(canvas)

    if not boxes:
        return None
    boxes = np.array(boxes)
    labels = np.array(labels)
    masks = np.array(masks)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.tensor(masks, dtype=torch.uint8),  # [N, H, W]
    }


def run_inference(
    logger, device, df, output_dir, score_threshold, location="ALL", mask_threshold=0.5
) -> None:
    device = _resolve_device(logger, device)
    logger.info("Device: %s", device)
    logger.info("Total rows of dataset: %d", len(df))

    # Filter by lesion_location
    if location and location.upper() != "ALL":
        df = df[df["lesion_location"] == location]
        logger.info("Filtered to location='%s': %d rows", location, len(df))

    # Build COCO pretrained model
    model = build_coco_pretrained(
        logger=logger,
        device=device,
        score_threshold=score_threshold,
    )
    model.eval()

    # Output directories
    out_dir = Path(output_dir)
    vis_dir = out_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Run inference ONCE on every image row.
    # Store predictions in a dict keyed by resolved image_path string
    # ══════════════════════════════════════════════════════════════════════════

    image_rows = df[df["image_path"].notna()].copy()
    logger.info("Images for inference: %d", len(image_rows))

    predictions: dict[str, dict] = {}
    results = []
    skipped = 0

    for idx, (_, row) in enumerate(image_rows.iterrows()):
        img_path = Path(str(row["image_path"]))

        if not img_path.exists():
            skipped += 1
            logger.debug("Image not found (skipped): %s", img_path)
            continue

        img_pil = PILImage.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img_pil).to(device)

        t0 = time.time()
        with torch.no_grad():
            output = model([img_t])[0]
        elapsed = time.time() - t0

        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        masks = output["masks"].cpu() if "masks" in output else [None] * len(boxes)

        n_det = len(boxes)
        max_score = float(scores.max()) if n_det > 0 else 0.0

        logger.info(
            "[%d/%d] %s | n_det=%d max_score=%.3f t=%.2fs",
            idx + 1,
            len(image_rows),
            img_path.name,
            n_det,
            max_score,
            elapsed,
        )

        # Store prediction keyed by resolved path so eval can look it up
        predictions[str(img_path)] = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
        }

        # Save visualisation
        vis = overlay_predictions(
            img_pil,
            boxes,
            labels,
            scores,
            masks,
            score_threshold=score_threshold,
        )
        vis_path = vis_dir / f"{img_path.stem}_pred.jpg"
        vis.save(vis_path)

        results.append(
            {
                "patient_id": row.get("patient_id", ""),
                "label": row.get("label", ""),
                "source": row.get("source", ""),
                "lesion_location": row.get("lesion_location", ""),
                "image_path": str(img_path),
                "coco_file": row.get("coco_file", ""),
                "n_detections": n_det,
                "max_score": round(max_score, 4),
                "inference_time_s": round(elapsed, 4),
                "vis_path": str(vis_path),
            }
        )

    logger.info("Inference done. Processed=%d  Skipped=%d", len(results), skipped)

    results_csv = out_dir / "inference_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    logger.info("Results saved → %s", results_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Evaluate STORED predictions against COCO ground truth.
    # Only rows that have a non-null coco_file can be evaluated.
    # For each such row we:
    # a) look up the prediction from Step 1 by image_path
    # b) load the GT from the coco_file JSON
    # c) pass both to torchmetrics MeanAveragePrecision
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- Evaluation against COCO ground-truth annotations ---")

    # Only rows that have a coco_file can be evaluated
    eval_rows = df[df["coco_file"].notna()].copy()
    logger.info("Rows with coco_file (evaluable): %d", len(eval_rows))

    if eval_rows.empty:
        logger.warning("No annotated rows available. Skipping evaluation.")
        return

    try:
        metric_box = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
        metric_mask = MeanAveragePrecision(iou_type="segm", class_metrics=False)
        has_torchmetrics = True
    except ImportError:
        logger.warning(
            "torchmetrics not installed — skipping mAP. "
            "Run: pip install torchmetrics"
        )
        has_torchmetrics = False

    eval_matched = 0  # rows where we have both pred and GT
    eval_no_pred = 0  # coco_file row whose image was not inferred (missing on disk)
    eval_no_gt = 0  # coco JSON had no valid annotations
    has_mask_pred = False

    for _, row in eval_rows.iterrows():
        img_path = Path(str(row["image_path"]))
        coco_path = Path(str(row["coco_file"]))

        # Look up stored prediction
        pred_raw = predictions.get(str(img_path))
        if pred_raw is None:
            eval_no_pred += 1
            logger.debug(
                "No stored prediction for %s "
                "(image was skipped or not in inference rows)",
                img_path.name,
            )
            continue

        if not coco_path.exists():
            eval_no_gt += 1
            logger.debug("COCO file not found: %s", coco_path)
            continue

        # Load ground truth from COCO JSON
        gt = load_gt_from_coco(coco_path, img_path)
        if gt is None:
            eval_no_gt += 1
            logger.debug("No valid GT annotations in %s", coco_path.name)
            continue
        if pred_raw is None or len(pred_raw["boxes"]) == 0:
            pred = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.tensor([], dtype=torch.float32),
                "labels": torch.tensor([], dtype=torch.int64),
            }
        else:
            # Build valid prediction dict
            n_det = len(pred_raw["boxes"])
            pred = {
                "boxes": pred_raw["boxes"],
                "scores": pred_raw["scores"],
                "labels": torch.ones(n_det, dtype=torch.int64),
            }
        if isinstance(pred_raw["masks"], torch.Tensor):
            pred["masks"] = (
                (pred_raw["masks"] > mask_threshold).squeeze(1).to(torch.uint8)
            )
            has_mask_pred = True
        # Below metric_box and metric_mask update should be removed if we want to evaluate only if any object is found.
        metric_box.update([pred], [gt])
        if "masks" in pred:
            metric_mask.update([pred], [gt])
        if has_torchmetrics:
            metric_box.update([pred], [gt])
            if "masks" in pred:
                metric_mask.update([pred], [gt])

        eval_matched += 1

    logger.info(
        "Evaluation summary: matched=%d  no_pred=%d  no_gt=%d",
        eval_matched,
        eval_no_pred,
        eval_no_gt,
    )

    # Below if block comment should be removed if we want to evaluate only if any object is found.
    # if not has_torchmetrics or eval_matched == 0:
    #     logger.warning(
    #         "Nothing to report (torchmetrics=%s, matched=%d).",
    #         has_torchmetrics,
    #         eval_matched,
    #     )
    #     return

    # Compute metrics
    box_result = metric_box.compute()
    mask_result = metric_mask.compute() if has_mask_pred else {}

    eval_result = {
        "mode": "zero_shot_coco_pretrained",
        "num_inferred": len(predictions),
        "num_evaluated": eval_matched,
        "eval_no_pred": eval_no_pred,
        "eval_no_gt": eval_no_gt,
        "box_mAP50": round(float(box_result.get("map_50", 0.0)), 4),
        "box_mAP50_95": round(float(box_result.get("map", 0.0)), 4),
        "mask_mAP50": round(float(mask_result.get("map_50", 0.0)), 4),
        "mask_mAP50_95": round(float(mask_result.get("map", 0.0)), 4),
    }

    logger.info("Box  mAP@50    : %.4f", eval_result["box_mAP50"])
    logger.info("Box  mAP@50:95 : %.4f", eval_result["box_mAP50_95"])
    logger.info("Mask mAP@50    : %.4f", eval_result["mask_mAP50"])
    logger.info("Mask mAP@50:95 : %.4f", eval_result["mask_mAP50_95"])

    report_path = out_dir / "eval_zeroshot_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    logger.info("Eval report saved → %s", report_path)


if __name__ == "__main__":
    CONFIG_PATH     = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
    MASKRCNN_CONFIG = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\maskrcnn.ini"

    config          = load_config(CONFIG_PATH)
    maskrcnn_config = load_config(MASKRCNN_CONFIG)
    cfg             = MaskRCNNConfig(maskrcnn_config)
    logger          = getLogger(cfg.log_file)

    dataset         = pd.read_csv(cfg.csv_path)
    output_dir      = cfg.zeroshot_dir
    score_threshold = cfg.score_threshold
    device          = cfg.device

    run_inference(
        logger=logger,
        device=device,
        df=dataset,
        output_dir=output_dir,
        score_threshold=score_threshold,
    )