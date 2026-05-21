import os
import datetime
import pandas as pd
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from pathlib import Path
from typing import List

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.segmentation.maskrcnn.config.maskrcnnconfig import MaskRCNNConfig
from src.segmentation.maskrcnn.prepare_coco_json import build_coco_from_via


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(logger, config, cfg: MaskRCNNConfig, df: pd.DataFrame):
    logger.info("=" * 55)
    logger.info(f"  Mask R-CNN ROI Pipeline  — {datetime.now():%Y-%m-%d %H:%M}")
    logger.info("=" * 55)
    logger.info("\n" + cfg.summary())

    # Todo add stages
    logger.info(f"Starting Training Pipeline for stages: {cfg.stages}")

    for stage in cfg.stages:
        if stage == "prepare":
            build_coco_from_via(logger, cfg.min_area, df)
        # elif stage == "zero_shot":
        #     run_inference()

        # elif stage == "train":
        #     train(cfg, logger)

        # elif stage == "evaluate":
        #     predictor = load_predictor(cfg)
        #     img_path = cfg.image or next(Path(cfg.images_dir).glob("*.jpg"), None)
        #     if img_path:
        #         infer_image(str(img_path), cfg, predictor, logger)

        # elif stage == "extract":
        #     run_extraction(cfg, logger, use_ground_truth=cfg.ground_truth)

    logger.info("\n  Pipeline complete ✓")


def visualize_coco(coco_json_path):
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    # Create lookup for images
    images = {img["id"]: img for img in data["images"]}

    for ann in data["annotations"]:
        img_meta = images.get(ann["image_id"])
        if not img_meta:
            continue

        img_path = img_meta["path"]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # COCO segmentation is a list of polygons (list of lists)
        for poly in ann["segmentation"]:
            # Convert flat [x1, y1, x2, y2...] to list of tuples [(x1,y1), (x2,y2)...]
            points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline="red", width=3)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Annotation: {ann['id']}")
        plt.axis("off")
        plt.show()


def get_dataset(logger, config) -> List:
    datasets = []

    # Smart Augmented Dataset
    smart_aug_df_path = (
        f"{config.get('AUGMENT_SMART', 'augment.baseroot')}/"
        f"{config.get('AUGMENT_SMART', 'augment.patient.metadata.filename')}"
    )
    if Path(smart_aug_df_path).exists():
        smart_aug_dataset = pd.read_csv(smart_aug_df_path)
        smart_aug_coco_df_path = (
            f"{config.get('AUGMENT_SMART', 'augment.baseroot')}/"
            f"{config.get('AUGMENT_SMART', 'augment.patient.coco.metadata.filename')}"
        )
        datasets.append(["SMART_II_AUG", smart_aug_dataset, smart_aug_coco_df_path])
    else:
        logger.warning(f"Smart Augmented Dataset Not Found: [{smart_aug_df_path}]")

    # Smart OM Augmented Dataset
    smartom_aug_df_path = (
        f"{config.get('AUGMENT_SMARTOM', 'augment.baseroot')}/"
        f"{config.get('AUGMENT_SMARTOM', 'augment.patient.metadata.filename')}"
    )
    if Path(smartom_aug_df_path).exists():
        smartom_aug_dataset = pd.read_csv(smartom_aug_df_path)
        smartom_aug_coco_df_path = (
            f"{config.get('AUGMENT_SMARTOM', 'augment.baseroot')}/"
            f"{config.get('AUGMENT_SMARTOM', 'augment.patient.coco.metadata.filename')}"
        )
        datasets.append(["SMART_OM_AUG", smartom_aug_dataset, smartom_aug_coco_df_path])
    else:
        logger.warning(f"Smart_OM Augmeneted Dataset Not Found: {smartom_aug_df_path}")

    # Merged Dataset
    merged_df_path = f"{config.get('SMART_MERGED', 'baseroot')}{config.get('SMART_MERGED', 'merged.output.filename')}"
    if Path(merged_df_path).exists():
        merged_dataset = pd.read_csv(merged_df_path)
        merged_coco_dataset_filename = (
            f"{config.get('SMART_MERGED', 'baseroot')}"
            f"{config.get('SMART_MERGED', 'merged.coco.output.filename')}"
        )
        datasets.append(["MERGED", merged_dataset, merged_coco_dataset_filename])
    else:
        logger.warning(f"Merged Dataset Not Found: [{merged_df_path}]")

    return datasets


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


if __name__ == "__main__":
    config_path = get_configpath()
    config = load_config(config_path)
    logger = initialize_logger(config=config)
    maskrcnn_config = load_config(config.get("SEGMENT-MASKRCNN", "maskrcnn.config"))
    cfg = MaskRCNNConfig(maskrcnn_config)
    datasets = get_dataset(logger, config)
    for img_type, df, output_df_path in datasets:
        if df is not None and not df.empty:
            coco_dataset = build_coco_from_via(
                logger, config, cfg.min_area, df, img_type
            )
            coco_dataset.to_csv(output_df_path)
    if len(datasets) == 0:
        logger.error("Got final dataset empty. Exiting the process.")
        raise FileNotFoundError("Final dataset is empty. Exiting the process.")
        # later can be replaced by custom exception
