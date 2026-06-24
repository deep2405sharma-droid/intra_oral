"""
via_to_mask.py
---------------
Converts VIA polygon JSON annotations + metadata CSV
into rasterised mask PNGs for U-Net semantic segmentation.

Unlike via_to_coco.py (which stores polygon coordinates as bbox/segmentation
lists for instance segmentation), this script paints the polygon directly
onto a blank canvas and saves it as a single-channel PNG.

Mask PNG pixel value convention:
    0 = background
    1 = normal
    2 = opmd
    3 = variation

Normal images get an all-zero (background) mask since they have no
lesion annotation -- there is nothing to paint.

Usage:
    python -m Segmentation.via_to_mask

Output:
    data/masks/
        smart_ii/<label>/<patient_id>/<image_stem>_mask.png
        smart_om/<label>/<patient_id>/<image_stem>_mask.png
    Updated CSVs with new `mask_path` column:
        data/masks/smart_ii_with_masks.csv
        data/masks/smart_om_with_masks.csv
        data/masks/combined_with_masks.csv
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import configparser
from pathlib import Path

from src.common.intraoral_logger import initialize_logger

def update_merged_df_paths(smart_base_path, smartom_base_path, df):
    """Prepend base paths to relative image/mask paths in merged dataset."""
    mask1 = df["source"] == "smart_II"
    mask2 = df["source"] == "smart_om"
    for col in ["image_path", "json_file"]:
        if col in df.columns:
            df.loc[mask1, col] = (
                smart_base_path + "/" + df.loc[mask1, col].astype(str)
            )
            df.loc[mask2, col] = (
                smartom_base_path + "/" + df.loc[mask2, col].astype(str)
            )
    return df


def update_paths(base_path, df):
    """Prepend base path to relative paths in augmented dataset."""
    for col in ["image_path", "json_file"]:
        if col in df.columns:
            df[col] = (
                base_path + "/" + df[col].astype(str).where(df[col].notna())
            )
    return df


# ── Load config ───────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "unet.ini")

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


# ── Class -> pixel value mapping (must match unet.ini [MASK] section) ──
LABEL_PIXEL_MAP = {
    "normal":    1,
    "opmd":      2,
    "variation": 3,
}


# ── VIA JSON helpers (identical to via_to_coco.py) ────────────────

def load_via_json(json_path: str, logger) -> dict:
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {json_path}: {e}")
        return None


def get_entry_for_image_smartii(via_json: dict, img_filename: str) -> dict:
    img_basename = Path(img_filename).name
    img_stem     = Path(img_filename).stem
    base_stem    = img_stem.split("_aug")[0] if "_aug" in img_stem else img_stem

    for key, entry in via_json.items():
        if not isinstance(entry, dict):
            continue
        entry_fname = entry.get("filename", "")
        entry_stem  = Path(entry_fname).stem
        if (entry_fname == img_basename
                or img_basename in key
                or img_stem in key
                or entry_stem == base_stem
                or base_stem in key):
            return entry
    return None


def get_entry_for_image_smartom(via_json: dict, img_filename: str) -> dict:
    img_basename = Path(img_filename).name
    img_stem     = Path(img_filename).stem
    base_stem    = img_stem.split("_aug")[0] if "_aug" in img_stem else img_stem

    alt_basenames = set([
        img_basename,
        img_basename.lower(),
        img_basename.upper(),
        img_stem + ".jpg",
        img_stem + ".jpeg",
        img_stem + ".JPG",
        img_stem.lower() + ".jpg",
        img_stem.lower() + ".jpeg",
    ])

    # VIA 2.0 format
    metadata = via_json.get("_via_img_metadata")
    if isinstance(metadata, dict):
        for key, entry in metadata.items():
            if not isinstance(entry, dict):
                continue
            entry_fname = entry.get("filename", "")
            if (entry_fname in alt_basenames
                    or any(b in key for b in alt_basenames)
                    or base_stem in key
                    or Path(entry_fname).stem == base_stem):
                return entry

    # Old flat format
    for key, entry in via_json.items():
        if not isinstance(entry, dict):
            continue
        entry_fname = entry.get("filename", "")
        if (entry_fname in alt_basenames
                or any(b in key for b in alt_basenames)
                or base_stem in key
                or Path(entry_fname).stem == base_stem):
            return entry

    return None


def extract_polygons_from_entry(entry: dict) -> list:
    """Returns list of (xs, ys) tuples for valid polygons."""
    polygons = []
    for region in entry.get("regions", []):
        if region is None:
            continue
        if not isinstance(region, dict):
            continue
        sa = region.get("shape_attributes", {})
        if not sa:
            continue
        if sa.get("name") != "polygon":
            continue
        xs = sa.get("all_points_x", [])
        ys = sa.get("all_points_y", [])
        if not xs or not ys:
            continue
        if max(xs) < 10:
            continue
        polygons.append((list(xs), list(ys)))
    return polygons


# ── Mask rasteriser ────────────────────────────────────────────────

def rasterise_mask(h: int, w: int, polygons: list, pixel_value: int) -> np.ndarray:
    """
    Paint all polygons onto a blank (h, w) canvas with the given pixel value.
    Returns a single-channel uint8 mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for xs, ys in polygons:
        xs_clipped = [max(0, min(int(x), w - 1)) for x in xs]
        ys_clipped = [max(0, min(int(y), h - 1)) for y in ys]
        pts = np.array(list(zip(xs_clipped, ys_clipped)), dtype=np.int32)
        cv2.fillPoly(mask, [pts], pixel_value)
    return mask


# ── Main per-source processor ──────────────────────────────────────

def build_masks_for_source(rows: list, source: str, output_root: str, logger) -> list:
    """
    For each row:
      - normal     -> save an all-zero mask, no JSON needed
      - opmd/variation -> load JSON, rasterise polygons with class pixel value

    Returns list of dicts (original row + mask_path) for rows that succeeded.
    """
    stats = {
        "total":          len(rows),
        "no_label":       0,
        "img_load_fail":  0,
        "no_json":        0,
        "json_load_fail": 0,
        "no_entry":       0,
        "no_polygon":     0,
        "normal_added":   0,
        "success":        0,
    }

    output_rows = []

    for row in rows:
        img_path  = str(row.get("image_path", ""))
        json_path = row.get("json_file")
        label     = str(row.get("label", "")).lower().strip()
        patient_id = str(row.get("patient_id", "unknown"))

        if label not in LABEL_PIXEL_MAP and label != "normal":
            stats["no_label"] += 1
            continue

        if not os.path.exists(img_path):
            stats["img_load_fail"] += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            stats["img_load_fail"] += 1
            continue
        h, w = img.shape[:2]

        # ── Normal images: all-zero mask, no JSON needed ──────────
        if label == "normal":
            mask = np.zeros((h, w), dtype=np.uint8)
            stats["normal_added"] += 1
        else:
            # ── OPMD / Variation: requires JSON annotation ────────
            if pd.isna(json_path) or not json_path or not os.path.exists(str(json_path)):
                stats["no_json"] += 1
                continue

            via_json = load_via_json(str(json_path), logger)
            if via_json is None:
                stats["json_load_fail"] += 1
                continue

            if source == "smart_ii":
                entry = get_entry_for_image_smartii(via_json, img_path)
            else:
                entry = get_entry_for_image_smartom(via_json, img_path)

            if entry is None:
                stats["no_entry"] += 1
                logger.warning(f"No annotation entry for: {Path(img_path).name}")
                continue

            polygons = extract_polygons_from_entry(entry)
            if not polygons:
                stats["no_polygon"] += 1
                continue

            pixel_value = LABEL_PIXEL_MAP[label]
            mask = rasterise_mask(h, w, polygons, pixel_value)

        # ── Save mask PNG ──────────────────────────────────────────
        img_stem  = Path(img_path).stem
        out_dir   = Path(output_root) / source / label / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"{img_stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        out_row = dict(row)
        out_row["mask_path"] = str(mask_path)
        output_rows.append(out_row)
        stats["success"] += 1

    logger.info(f"\n{'='*55}")
    logger.info(f"  Mask Generation -- {source}")
    logger.info(f"{'='*55}")
    logger.info(f"  Total input rows   : {stats['total']}")
    logger.info(f"  Successfully built : {stats['success']}")
    logger.info(f"    of which normal  : {stats['normal_added']} (all-zero mask)")
    logger.info(f"  Skipped -- no label: {stats['no_label']}")
    logger.info(f"  Skipped -- img fail: {stats['img_load_fail']}")
    logger.info(f"  Skipped -- no JSON : {stats['no_json']}")
    logger.info(f"  Skipped -- no entry: {stats['no_entry']}")
    logger.info(f"  Skipped -- no poly : {stats['no_polygon']}")
    logger.info(f"{'='*55}\n")

    return output_rows


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    logger = initialize_logger(config)

    logger.info("=" * 65)
    logger.info("  VIA to Mask PNG Conversion (for U-Net)")
    logger.info("=" * 65)

    merged_csv      = config.get("DATA", "merged.csv")
    aug_smart_csv   = config.get("DATA", "aug.smart.csv")
    aug_smartom_csv = config.get("DATA", "aug.smartom.csv")
    mask_output_root = config.get("PATHS", "mask_output_dir") \
        if config.has_option("PATHS", "mask_output_dir") \
        else r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\data\masks"
    os.makedirs(mask_output_root, exist_ok=True)

    # Base paths for prepending relative paths in CSVs
    smart_base       = config.get("TRAIN", "smart.basepath")
    smartom_base     = config.get("TRAIN", "smartom.basepath")
    augment_smart_base   = config.get("TRAIN", "augment.smart.baseroot")
    augment_smartom_base = config.get("TRAIN", "augment.smartom.baseroot")

    # Load merged data -- all 3 labels
    df_merged     = pd.read_csv(merged_csv)
    df_merged     = update_merged_df_paths(smart_base, smartom_base, df_merged)
    df_all_labels = df_merged[df_merged["label"].isin(["normal", "opmd", "variation"])].copy()
    logger.info(f"Merged rows (normal+opmd+variation) : {len(df_all_labels)}")
    logger.info(f"  normal    : {len(df_all_labels[df_all_labels['label'] == 'normal'])}")
    logger.info(f"  opmd      : {len(df_all_labels[df_all_labels['label'] == 'opmd'])}")
    logger.info(f"  variation : {len(df_all_labels[df_all_labels['label'] == 'variation'])}")

    # Load augmented data
    df_aug_s  = pd.read_csv(aug_smart_csv)   if os.path.exists(aug_smart_csv)   else pd.DataFrame()
    df_aug_om = pd.read_csv(aug_smartom_csv) if os.path.exists(aug_smartom_csv) else pd.DataFrame()
    if not df_aug_s.empty:
        df_aug_s = update_paths(augment_smart_base, df_aug_s)
    if not df_aug_om.empty:
        df_aug_om = update_paths(augment_smartom_base, df_aug_om)
    logger.info(f"Aug SMART-II rows  : {len(df_aug_s)}")
    logger.info(f"Aug SMART-OM rows  : {len(df_aug_om)}")

    # ... rest stays the same (split by source, build masks, save CSVs)