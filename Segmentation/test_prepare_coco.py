"""
test_prepare_coco.py
────────────────────
Test script for prepare_coco_json.build_coco_from_via.

Steps:
    1. Take one row from smart_II and one from smart_om
    2. Run build_coco_from_via on these two rows
    3. Save the returned dataset to a CSV

Usage:
    python -m Segmentation.test_prepare_coco
"""

import os
import logging
import configparser
import pandas as pd
from Segmentation.prepare_coco_json import build_coco_from_via

# ── Config ────────────────────────────────────────────────────────
CONFIG_PATH = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# ── Logger ────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load merged CSV ───────────────────────────────────────────────
merged_csv = config.get("DATA", "merged.csv")
df = pd.read_csv(merged_csv)

VALID_LOCATIONS = {"DT", "LB", "RB", "UL", "UA", "VT", "LL"}
LOCATION_REMAP  = {"RB1": "RB", "RB2": "RB", "RB3": "RB"}

def clean_location(loc):
    k = str(loc).strip()
    return LOCATION_REMAP.get(k, k)

df["lesion_location"] = df["lesion_location"].apply(clean_location)
df = df[df["lesion_location"].isin(VALID_LOCATIONS)].copy()
df = df.dropna(subset=["json_file", "image_path"]).copy()

# ── Pick one row from each source ─────────────────────────────────
row_smart_ii = (
    df[df["source"] == "smart_II"]
    .iloc[0:1]
    .copy()
)

row_smart_om = (
    df[df["source"] == "smart_om"]
    .iloc[0:1]
    .copy()
)

test_dataset = pd.concat([row_smart_ii, row_smart_om], ignore_index=True)

logger.info("=" * 60)
logger.info("  Test: build_coco_from_via on 2 rows")
logger.info("=" * 60)
logger.info(f"\nTest dataset ({len(test_dataset)} rows):")
logger.info(
    test_dataset[["patient_id", "source", "lesion_location", "image_path"]]
    .to_string(index=False)
)

# ── Run build_coco_from_via ───────────────────────────────────────
MIN_AREA = 100

result = build_coco_from_via(
    logger   = logger,
    min_area = MIN_AREA,
    dataset  = test_dataset,
)

# ── Show result ───────────────────────────────────────────────────
logger.info("\nResult dataset:")
logger.info(
    result[["patient_id", "source", "lesion_location", "coco_file"]]
    .to_string(index=False)
)

# ── Save result ───────────────────────────────────────────────────
coco_output_dir = config.get("COCO", "coco.output.dir")
os.makedirs(coco_output_dir, exist_ok=True)

out_csv = os.path.join(coco_output_dir, "test_coco_result.csv")
result.to_csv(out_csv, index=False)
logger.info(f"\nResult saved → {out_csv}")

n_saved = result["coco_file"].notna().sum()
logger.info(f"COCO JSONs saved : {n_saved} / {len(result)}")
logger.info("=" * 60)
logger.info("  Done")
logger.info("=" * 60)