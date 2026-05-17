"""
augment_pipeline.py
───────────────────
Full augmentation pipeline for intraoral / oral lesion images.

Steps:
  1. Read config.ini  -> paths, labels to augment, output dirs
  2. Read augmentation_config.py -> transform definitions, target counts
  3. Load clean_patient_data.csv -> get image + JSON paths per patient
  4. For each (patient, image, label) matching configured labels:
       a. Load image
       b. Load VIA annotation JSON (polygons + rects)
       c. Run N augmentations (N = AUGMENTATION_TARGETS[label])
       d. For each augmentation:
            - Apply spatial + pixel transforms
            - Remap all polygon and rect coordinates
            - Build new VIA-format JSON with updated coordinates
            - Save augmented image  -> output/label/patient_id/Unannotated/
            - Save remapped JSON    -> output/label/patient_id/
       e. Append one metadata row per augmentation to augmented CSV
  5. Save / append augmented metadata CSV

Usage:
    python augment_pipeline.py --config config.ini

Requirements:
    pip install albumentations opencv-python numpy pandas
"""

from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config

import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import argparse
import configparser
import copy
import importlib.util
import json
import logging
import os
import random
import sys
import cv2
import re
import numpy as np
import pandas as pd
import matplotlib
import albumentations as A

matplotlib.use("Agg")


def load_aug_config(py_path: str):
    """Dynamically import augmentation_config.py and return the module."""
    spec = importlib.util.spec_from_file_location("augmentation_config", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ══════════════════════════════════════════════════════════════════════════════
# ALBUMENTATIONS PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════


def _build_transform(name: str, params: dict, p: float, img_h: int, img_w: int):
    """
    Instantiate a single Albumentations transform by name.
    RandomResizedCrop needs image dimensions filled in at runtime.
    """

    # Fill runtime image size for RandomResizedCrop
    if name == "RandomResizedCrop":
        params = dict(params)
        params["height"] = img_h
        params["width"] = img_w

    cls = getattr(A, name)
    return cls(**params, p=p)


def build_pipeline(aug_cfg, img_h: int, img_w: int, enable_flags: dict = None):
    """
    Read transform definitions from augmentation_config.py and build an
    Albumentations A.Compose pipeline.

    augmentation_config.py owns WHAT transforms to apply and their parameters.
    This function owns HOW to instantiate them using the Albumentations API.

    KeypointParams wires all polygon + rect vertices into every spatial
    transform so Albumentations remaps coordinates automatically.

    Args:
        aug_cfg : the loaded augmentation_config module
        img_h   : actual image height (needed by RandomResizedCrop)
        img_w   : actual image width

    Returns:
        A.Compose pipeline — call as:
            result = pipeline(image=img_array, keypoints=[(x,y), ...])
            aug_image     = result["image"]
            aug_keypoints = result["keypoints"]   # automatically remapped
    """
    import albumentations as A

    groups = []
    if aug_cfg.ENABLE_GEOMETRIC:
        groups += aug_cfg.GEOMETRIC
    if aug_cfg.ENABLE_COLOR:
        groups += aug_cfg.COLOR
    if aug_cfg.ENABLE_NOISE:
        groups += aug_cfg.NOISE
    if aug_cfg.ENABLE_BLUR:
        groups += aug_cfg.BLUR
    if aug_cfg.ENABLE_COMPRESSION:
        groups += aug_cfg.COMPRESSION

    transforms = []
    for t in groups:
        name = t["name"]
        params = dict(t["params"])  # copy so we don't mutate the config
        p = t["p"]

        # RandomResizedCrop: newer Albumentations uses size=(h,w) not height+width
        if name == "RandomResizedCrop":
            params["size"] = (img_h, img_w)

        # Affine with scale=None means no scaling — remove the key entirely
        # so Albumentations uses its default (no scale change)
        if name == "Affine" and params.get("scale") is None:
            params.pop("scale", None)

        try:
            cls = getattr(A, name)
            transforms.append(cls(**params, p=p))
        except AttributeError:
            logger.warning(f"Transform '{name}' not found in albumentations — skipping")
        except TypeError as e:
            logger.warning(f"Transform '{name}' bad params: {e} — skipping")

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,  # keep OOB points — pipeline clips after
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIA JSON PARSING & COORDINATE HANDLING
# ══════════════════════════════════════════════════════════════════════════════


def load_via_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_keypoints_from_regions(regions: list) -> tuple:
    """
    Extract all polygon + rect corner points as a flat list of (x, y) keypoints.
    Returns:
        keypoints : list of (x, y)
        index_map : list of (region_idx, shape, point_role)
                    so we can reconstruct after augmentation
    """
    keypoints = []
    index_map = []

    for r_idx, region in enumerate(regions):
        sa = region["shape_attributes"]
        name = sa["name"]

        if name == "polygon":
            xs = sa["all_points_x"]
            ys = sa["all_points_y"]
            for i, (x, y) in enumerate(zip(xs, ys)):
                keypoints.append((float(x), float(y)))
                index_map.append((r_idx, "polygon", i))

        elif name == "rect":
            x, y, w, h = sa["x"], sa["y"], sa["width"], sa["height"]
            # Store all 4 corners so rotation/perspective is handled correctly
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for i, (cx, cy) in enumerate(corners):
                keypoints.append((float(cx), float(cy)))
                index_map.append((r_idx, "rect", i))

    return keypoints, index_map


def reconstruct_regions(
    original_regions: list, aug_keypoints: list, index_map: list, img_w: int, img_h: int
) -> list:
    """
    Rebuild region shape_attributes from augmented keypoints.
    Clips all coordinates to image bounds.
    For rects: re-derives x,y,w,h from the 4 augmented corners (handles rotation).
    """
    # Group augmented keypoints back to their regions
    region_points: dict = {}  # r_idx -> list of (point_role, aug_kp)
    for kp, (r_idx, shape, point_role) in zip(aug_keypoints, index_map):
        region_points.setdefault(r_idx, []).append((point_role, shape, kp))

    new_regions = copy.deepcopy(original_regions)

    for r_idx, points in region_points.items():
        sa = new_regions[r_idx]["shape_attributes"]
        name = sa["name"]

        if name == "polygon":
            xs = []
            ys = []
            for _, _, kp in sorted(points, key=lambda x: x[0]):
                x = int(round(np.clip(kp[0], 0, img_w - 1)))
                y = int(round(np.clip(kp[1], 0, img_h - 1)))
                xs.append(x)
                ys.append(y)
            sa["all_points_x"] = xs
            sa["all_points_y"] = ys

        elif name == "rect":
            # 4 corners -> derive axis-aligned bounding rect
            corner_xs = [np.clip(kp[0], 0, img_w - 1) for (_, _, kp) in points]
            corner_ys = [np.clip(kp[1], 0, img_h - 1) for (_, _, kp) in points]
            x_min = int(round(min(corner_xs)))
            y_min = int(round(min(corner_ys)))
            x_max = int(round(max(corner_xs)))
            y_max = int(round(max(corner_ys)))
            sa["x"] = x_min
            sa["y"] = y_min
            sa["width"] = max(1, x_max - x_min)
            sa["height"] = max(1, y_max - y_min)

    return new_regions


def remap_via_json(
    original_via: dict,
    matched_key: str,
    aug_image: np.ndarray,
    aug_keypoints: list,
    index_map: list,
    new_filename: str,
    new_filesize: int,
) -> dict:
    """
    Build a new VIA JSON for the augmented image with remapped coordinates.

    The JSON format is a flat dict:
        { "<filename><size>": { filename, size, file_attributes, regions } }

    matched_key is the original key (from find_via_entry_for_image) so we
    update exactly the right entry and rename it for the augmented image.
    """
    img_h, img_w = aug_image.shape[:2]

    # Detect format
    is_via2_format = "_via_img_metadata" in original_via

    if is_via2_format:
        metadata = original_via["_via_img_metadata"]
        original_entry = metadata[matched_key]
    else:
        original_entry = original_via[matched_key]

    # Rebuild regions with augmented coordinates
    new_regions = reconstruct_regions(
        original_entry["regions"], aug_keypoints, index_map, img_w, img_h
    )

    # Create updated entry
    new_entry = copy.deepcopy(original_entry)
    new_entry["filename"] = new_filename
    new_entry["size"] = new_filesize
    new_entry["regions"] = new_regions

    new_key = f"{new_filename}{new_filesize}"

    # Return in the SAME format as the original JSON
    if is_via2_format:
        new_via = copy.deepcopy(original_via)  # preserve all settings, attributes, etc.
        new_via["_via_img_metadata"][new_key] = new_entry
        # Optional: keep image_id_list in sync
        if "_via_image_id_list" in new_via:
            if new_key not in new_via["_via_image_id_list"]:
                new_via["_via_image_id_list"].append(new_key)
        return new_via
    else:
        # Old flat format → just return the single entry
        return {new_key: new_entry}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════


def build_aug_metadata_row(
    original_row: pd.Series, new_img_path: str, new_json_path: str, aug_index: int
) -> dict:
    """
    Copy all patient metadata from original row.
    Replace image paths with augmented paths.
    Add image_type = 'augmented'.
    """
    row = original_row.to_dict()

    # Update image path columns
    row["image_path"] = new_img_path
    row["json_file"] = new_json_path
    row["image_type"] = "augmented"
    row["aug_index"] = aug_index  # which augmentation run (1..N)
    return row


def save_metadata(rows: list, csv_path: str):
    """Append to existing CSV if it exists, otherwise create new."""
    new_df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Avoid duplicate rows (same patient_id + aug_index)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        logger.info(
            f"Appended {len(new_df)} rows to {csv_path}  (total: {len(combined)})"
        )
    else:
        new_df.to_csv(csv_path, index=False)
        logger.info(f"Created metadata CSV: {csv_path}  ({len(new_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# CORE AUGMENTATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════


def find_via_entry_for_image(via_json: dict, img_path: str):
    """
    The VIA JSON is a flat dict keyed by '<filename><size>' (no separator).
    One JSON file may cover multiple images for the same patient — one entry
    per image, each storing only the regions annotated for that image.

    Match by comparing the 'filename' field inside each entry against the
    basename of img_path.

    Returns:
        (key, entry_dict)  if a matching entry is found
        (None, None)       if no matching entry exists (image has no annotation)
    Handles both standard VIA format and possible malformed cases.
    """
    if not isinstance(via_json, dict):
        return None, None

    img_basename = Path(img_path).name  # e.g. SMITA00402_R_LB.jpg
    img_stem = Path(img_path).stem  # without extension

    # Case 1: New VIA 2.0 format (most common now)
    metadata_dict = via_json.get("_via_img_metadata")
    if isinstance(metadata_dict, dict):
        for key, entry in metadata_dict.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("filename") == img_basename:
                return key, entry
            # Fallback: key contains filename
            if img_basename in key or img_stem in key:
                return key, entry

    # Case 2: Old flat format (what you had in SMITA00368_R_OPMD.json)
    for key, entry in via_json.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("filename") == img_basename:
            return key, entry
        if img_basename in key or img_stem in key:
            return key, entry

    logger.warning(f"No matching annotation entry found for image: {img_basename}")
    return None, None


def augment_single_image(
    image: np.ndarray,
    via_json: dict,
    img_path: str,
    pipeline,
    aug_index: int,
    base_seed: int,
) -> tuple | None:
    """
    Run one augmentation pass on an image + its VIA annotations.

    The JSON file contains one entry per image keyed by '<filename><size>'.
    We locate the entry matching img_path so only the regions for THIS image
    are extracted and remapped — not all regions in the file.

    Returns:
        aug_image      : augmented np.ndarray
        aug_keypoints  : remapped flat keypoint list (empty if no annotation)
        index_map      : parallel index map for reconstruction (empty if no annotation)
        matched_key    : the original JSON key (None if no annotation)
    """
    matched_key, entry = find_via_entry_for_image(via_json, img_path)

    if entry is None:
        return None  # ← Signal to skip

    regions = entry.get("regions", [])
    if not isinstance(regions, list):
        regions = []

    keypoints, index_map = extract_keypoints_from_regions(regions)

    random.seed(base_seed + aug_index)
    np.random.seed(base_seed + aug_index)

    try:
        result = pipeline(image=image, keypoints=keypoints)
        return result["image"], result["keypoints"], index_map, matched_key
    except Exception as e:
        logger.error(f"Pipeline failed for {img_path}: {e}")
        return None


def run_augmentations_for_patient(
    logger,
    patient_row: pd.Series,
    aug_cfg,
    augment_baseroot: str,
    output_metadata_csv: str,
    n_augmentations: int,
    base_seed: int = 42,
) -> list:
    """
    Run all augmentations for a single patient row.
    Returns list of metadata dicts (one per augmented image).
    """
    patient_id = patient_row["patient_id"]
    label = patient_row["label"]
    json_path = patient_row.get("json_file")
    img_path = patient_row.get("image_path")
    metadata_rows = []
    logger.info(f"Augmentation for Patient id: {patient_id}")
    # Load VIA JSON (may be NaN if not present)
    via_json = None
    if pd.notna(json_path) and Path(json_path).exists():
        try:
            via_json = load_via_json(json_path)
            if not isinstance(via_json, dict):
                logger.warning(f"JSON loaded but not a dict for {patient_id}")
                via_json = None
        except Exception as e:
            logger.warning(f"Failed to load JSON {json_path}: {e}")
            via_json = None
    else:
        logger.debug(f"No JSON for patient {patient_id}")

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        logger.warning(f"  Cannot load image: {img_path} — skipping")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # Build pipeline at actual image size (needed for RandomResizedCrop)
    pipeline = build_pipeline(aug_cfg, img_h, img_w)

    for aug_idx in range(1, n_augmentations + 1):
        try:
            # === NEW: Skip if no annotation ===
            if via_json:
                result = augment_single_image(
                    image, via_json, img_path, pipeline, aug_idx, base_seed
                )
                if result is None:
                    logger.warning(
                        f"Skipping augmentation {aug_idx} for {patient_id} "
                        f"({Path(img_path).name}) — no annotation found"
                    )
                    continue
                aug_image, aug_keypoints, index_map, matched_key = result
            else:
                # No JSON at all → skip
                logger.warning(f"Skipping {patient_id} — no JSON file")
                return []

            # Rest of your code (build paths, save image, save JSON, metadata row) remains the same
            # ── Build output paths ────────────────────────────────────
            orig_stem = Path(img_path).stem
            aug_fname = f"{orig_stem}_aug{aug_idx:04d}.jpg"

            out_img_dir = Path(augment_baseroot) / label / patient_id / "Unannotated"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = out_img_dir / aug_fname

            # Save image
            aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            file_size = out_img_path.stat().st_size

            # Save remapped JSON (only if we had annotation)
            out_json_path = ""
            if via_json and matched_key is not None and index_map:
                aug_via = remap_via_json(
                    via_json,
                    matched_key,
                    aug_image,
                    aug_keypoints,
                    index_map,
                    aug_fname,
                    file_size,
                )
                out_json_dir = Path(augment_baseroot) / label / patient_id / "Json file"
                out_json_dir.mkdir(parents=True, exist_ok=True)
                json_fname = f"{orig_stem}_aug{aug_idx:04d}.json"
                out_json_path = str(out_json_dir / json_fname)
                with open(out_json_path, "w") as f:
                    json.dump(aug_via, f, indent=2)

            # Build metadata row
            new_img_path = str(out_img_path)
            meta_row = build_aug_metadata_row(
                patient_row, new_img_path, out_json_path, aug_idx
            )
            metadata_rows.append(meta_row)

            if aug_idx % 25 == 0 or aug_idx == n_augmentations:
                logger.info(f"  {patient_id}: {aug_idx}/{n_augmentations} done")

        except Exception as e:
            logger.error(
                f"  Augmentation {aug_idx} failed for {patient_id}: {e}", exc_info=True
            )

    return metadata_rows


def _augment_image_only(image, pipeline, aug_idx, base_seed):
    """Augment image when no annotation JSON is available."""
    random.seed(base_seed + aug_idx)
    np.random.seed(base_seed + aug_idx)
    result = pipeline(image=image, keypoints=[])
    return result["image"]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def main(logger, config, aug_config_path: str):
    logger.info("=" * 65)
    logger.info("  Intraoral Image Augmentation Pipeline")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # ── 1. Load configs ────────────────────────────────────────────────────
    aug_cfg = load_aug_config(aug_config_path)

    # Paths from config.ini
    metadata_csv_path = config.get("SMART_MERGED", "merged.output.filename")
    augment_smart_baseroot = config.get("AUGMENT_SMART", "augment.baseroot")
    augment_smartom_baseroot = config.get("AUGMENT_SMARTOM", "augment.baseroot")
    output_smart_metadata_csv = os.path.join(
        augment_smart_baseroot,
        config.get("AUGMENT_SMART", "augment.patient.metadata.filename"),
    )
    output_smartom_metadata_csv = os.path.join(
        augment_smartom_baseroot,
        config.get("AUGMENT_SMARTOM", "augment.patient.metadata.filename"),
    )

    # Labels to augment — read from config.ini [AUGMENT] section
    # Format in config.ini:  augment.labels = OPMD,Variation
    raw_smart_labels = config.get("AUGMENT_SMART", "augment.labels")
    raw_smartom_labels = config.get("AUGMENT_SMARTOM", "augment.labels")
    smart_labels_to_augment = [lbl.strip() for lbl in raw_smart_labels.split(",")]
    smartom_labels_to_augment = [lbl.strip() for lbl in raw_smartom_labels.split(",")]
    logger.info(f"Smart labels to augment: {smart_labels_to_augment}")
    logger.info(f"Smartom labels to augment: {smartom_labels_to_augment}")

    # Load metadata doing for smart dataset
    df = pd.read_csv(metadata_csv_path)
    df = df[df.source == "smart_II"]
    logger.info(
        f"Loaded metadata: {len(df)} patients  |  "
        f"Labels: {df['label'].value_counts().to_dict()}"
    )

        # ── SMART-II augmentation (already exists) ────────────────────────
    df = pd.read_csv(metadata_csv_path)
    df_smart = df[df["source"] == "smart_II"]
    df_smart_target = df_smart[df_smart["label"].isin(smart_labels_to_augment)].copy()
    logger.info(f"SMART-II patients to augment: {len(df_smart_target)}")

    all_smart_rows = []
    for _, patient_row in df_smart_target.iterrows():
        label = patient_row["label"]
        cfg_key = f"augment.count.{label.lower()}"
        n_aug = config.getint("AUGMENT_SMART", cfg_key)
        if n_aug == 0:
            continue
        rows = run_augmentations_for_patient(
            logger=logger,
            patient_row=patient_row,
            aug_cfg=aug_cfg,
            augment_baseroot=augment_smart_baseroot,
            output_metadata_csv=output_smart_metadata_csv,
            n_augmentations=n_aug,
            base_seed=aug_cfg.RANDOM_SEED,
        )
        all_smart_rows.extend(rows)

    if all_smart_rows:
        save_metadata(all_smart_rows, output_smart_metadata_csv)
        logger.info(f"SMART-II done. Total: {len(all_smart_rows)}")

    # ── SMART-OM augmentation (new block) ────────────────────────────
    df_smartom = df[df["source"] == "smart_om"]
    df_smartom_target = df_smartom[df_smartom["label"].isin(
        [l.lower() for l in smartom_labels_to_augment]
    )].copy()
    logger.info(f"SMART-OM patients to augment: {len(df_smartom_target)}")

    all_smartom_rows = []
    for _, patient_row in df_smartom_target.iterrows():
        label = patient_row["label"]
        cfg_key = f"augment.count.{label.lower()}"
        n_aug = config.getint("AUGMENT_SMARTOM", cfg_key)
        if n_aug == 0:
            continue
        rows = run_augmentations_for_patient(
            logger=logger,
            patient_row=patient_row,
            aug_cfg=aug_cfg,
            augment_baseroot=augment_smartom_baseroot,
            output_metadata_csv=output_smartom_metadata_csv,
            n_augmentations=n_aug,
            base_seed=aug_cfg.RANDOM_SEED,
        )
        all_smartom_rows.extend(rows)

    if all_smartom_rows:
        save_metadata(all_smartom_rows, output_smartom_metadata_csv)
        logger.info(f"SMART-OM done. Total: {len(all_smartom_rows)}")

        logger.info("=" * 65)
        logger.info(f"  Done. SMART-II: {len(all_smart_rows)} | SMART-OM: {len(all_smartom_rows)}")
        logger.info(f"  Metadata saved to: {output_smart_metadata_csv}")
        logger.info("=" * 65)


def initialize_logger(config):
    log_filename = config.get("LOGGER", "logger.filename")
    logger = iolog.getLogger(log_filename)
    return logger


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config=config)
    aug_config = config.get("AUGMENT_SMART", "augment.config")
    main(logger, config, aug_config)
