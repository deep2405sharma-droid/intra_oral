"""
evaluate_mask_rcnn.py
─────────────────────
Evaluates trained Mask R-CNN on test set.
Computes COCO metrics: AP, AP50, AP75, mAP.

Usage:
    python -m src.evaluate_mask_rcnn

Requirements:
    pip install pycocotools
"""

import os
import json
import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config
from Segmentation.train_mask_rcnn import COCOLesionDataset, build_model, collate_fn


# ── Config ────────────────────────────────────────────────────────
COCO_DIR   = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\data\coco"
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\models\mask_rcnn_best.pth"
RESULTS_DIR = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\reports\evaluation"
NUM_CLASSES = 8
SCORE_THRESH = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_NAMES = {
    1: "DT",
    2: "LB",
    3: "RB",
    4: "UL",
    5: "UA",
    6: "VT",
    7: "LL",
}


# ── Inference ─────────────────────────────────────────────────────

def run_inference(model, data_loader, device, score_thresh=SCORE_THRESH):
    """Run model on all test images, return predictions in COCO format."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds  = model(images)

            for pred, target in zip(preds, targets):
                img_id = target["image_id"].item()
                scores = pred["scores"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()
                boxes  = pred["boxes"].cpu().numpy()
                masks  = pred["masks"].cpu().numpy()  # shape: [N, 1, H, W]

                for score, label, box, mask in zip(scores, labels, boxes, masks):
                    if score < score_thresh:
                        continue

                    # Convert mask to polygon (RLE or polygon)
                    mask_bin = (mask[0] > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue

                    # Take largest contour
                    contour = max(contours, key=cv2.contourArea)
                    segmentation = contour.flatten().tolist()

                    x1, y1, x2, y2 = box
                    results.append({
                        "image_id":    img_id,
                        "category_id": int(label),
                        "bbox":        [float(x1), float(y1),
                                        float(x2 - x1), float(y2 - y1)],
                        "score":       float(score),
                        "segmentation": [segmentation],
                    })

    return results


# ── Per-image visualization ───────────────────────────────────────

def save_prediction_images(model, dataset, device, output_dir, n=10):
    """Save first n images with predicted masks overlaid."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for i in range(min(n, len(dataset))):
        img_tensor, target = dataset[i]
        img_id   = target["image_id"].item()
        img_path = dataset.images[i]["file_name"]

        with torch.no_grad():
            pred = model([img_tensor.to(device)])[0]

        # Load original image
        img = cv2.imread(img_path)
        overlay = img.copy()

        # Draw predicted masks
        masks  = pred["masks"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        colors = {
            1: (0,   255, 255),   # DT  — yellow
            2: (255,   0,   0),   # LB  — blue
            3: (0,     0, 255),   # RB  — red
            4: (0,   255,   0),   # UL  — green
            5: (255,   0, 255),   # UA  — magenta
            6: (255, 165,   0),   # VT  — orange
            7: (128,   0, 128),   # LL  — purple
        }

        for mask, label, score in zip(masks, labels, scores):
            if score < SCORE_THRESH:
                continue
            mask_bin = (mask[0] > 0.5).astype(np.uint8)
            color    = colors.get(label, (255, 0, 0))
            overlay[mask_bin == 1] = color

        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # Draw boxes
        boxes = pred["boxes"].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if score < SCORE_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(label, (255, 0, 0))
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
            cat_name = CATEGORY_NAMES.get(label, str(label))
            cv2.putText(blended, f"{cat_name} {score:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1)

        out_path = os.path.join(output_dir, f"pred_{i:04d}_{Path(img_path).stem}.jpg")
        cv2.imwrite(out_path, blended)


# ── Main ──────────────────────────────────────────────────────────

def initialize_logger(config):
    return iolog.getLogger(config.get("LOGGER", "logger.filename"))


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config)

    logger.info("=" * 65)
    logger.info(f"  Mask R-CNN Evaluation  |  Device: {DEVICE}")
    logger.info("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test dataset
    test_json    = os.path.join(COCO_DIR, "test.json")
    test_dataset = COCOLesionDataset(test_json)
    test_loader  = DataLoader(test_dataset, batch_size=1,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    logger.info(f"Test images: {len(test_dataset)}")

    # Load model
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    logger.info(f"Model loaded from {MODEL_PATH}")

    # Run inference
    logger.info("Running inference...")
    results = run_inference(model, test_loader, DEVICE)
    logger.info(f"Predictions: {len(results)}")

    # Save predictions
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)

    # COCO evaluation
    logger.info("Running COCO evaluation...")
    coco_gt  = COCO(test_json)
    coco_dt  = coco_gt.loadRes(pred_path)

    # BBox evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    bbox_metrics = {
        "AP":    coco_eval.stats[0],
        "AP50":  coco_eval.stats[1],
        "AP75":  coco_eval.stats[2],
        "AP_S":  coco_eval.stats[3],
        "AP_M":  coco_eval.stats[4],
        "AP_L":  coco_eval.stats[5],
    }

    # Segmentation evaluation
    coco_eval_seg = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval_seg.evaluate()
    coco_eval_seg.accumulate()
    coco_eval_seg.summarize()

    segm_metrics = {
        "AP":    coco_eval_seg.stats[0],
        "AP50":  coco_eval_seg.stats[1],
        "AP75":  coco_eval_seg.stats[2],
    }

    logger.info(f"\nBBox metrics:  {bbox_metrics}")
    logger.info(f"Segm metrics:  {segm_metrics}")

    # Save metrics
    metrics = {"bbox": bbox_metrics, "segmentation": segm_metrics}
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # Save prediction images
    logger.info("Saving prediction visualizations...")
    save_prediction_images(
        model, test_dataset, DEVICE,
        output_dir=os.path.join(RESULTS_DIR, "visualizations")
    )

    logger.info("=" * 65)
    logger.info("  Evaluation complete")
    logger.info("=" * 65)