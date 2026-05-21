"""
utils/annotation_parser.py
---------------------------
Final version - Optimized for your use case:
- Prioritizes CSV label
- Matches exact image inside multi-image JSONs (_full.json)
- Skips artifacts
- Ready for Mask R-CNN + Augmentation pipeline
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class RegionAnnotation:
    shape: str
    label: str  # "normal", "variation", "opmd"
    polygon_x: List[float] = field(default_factory=list)
    polygon_y: List[float] = field(default_factory=list)
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Annotation:
    json_path: Path
    image_path: Path
    regions: List[RegionAnnotation] = field(default_factory=list)
    via_key: Optional[str] = None  # Original key in VIA JSON
    csv_label: str = ""


# ── Shape to Polygon ────────────────────────────────────────────────────────
def _shape_to_polygon(shape_attrs: Dict) -> Tuple[List[float], List[float], str]:
    name = shape_attrs.get("name", "polygon")

    if name == "polygon":
        xs = shape_attrs.get("all_points_x", [])
        ys = shape_attrs.get("all_points_y", [])
    elif name == "rect":
        x = shape_attrs.get("x", 0)
        y = shape_attrs.get("y", 0)
        w = shape_attrs.get("width", 0)
        h = shape_attrs.get("height", 0)
        xs = [x, x + w, x + w, x]
        ys = [y, y, y + h, y + h]
    else:
        xs = ys = []

    return [float(v) for v in xs], [float(v) for v in ys], name


# ── Label Resolution (CSV First) ────────────────────────────────────────────
def _get_target_label(
    region_attrs: Dict, json_stem: str, csv_label: str = ""
) -> Optional[str]:
    """Priority: CSV > JSON Description > JSON filename"""

    # 1. Highest priority: CSV label (from smart_merged.csv)
    if csv_label:
        cl = str(csv_label).strip().lower()
        if cl in ["normal", "variation", "opmd"]:
            return cl
        if cl == "variation from normal":
            return "variation"

    # 2. JSON content fallback
    if not isinstance(region_attrs, dict):
        region_attrs = {}

    desc = str(region_attrs.get("Description", "")).lower()
    label_key = str(region_attrs.get("label", "")).lower()

    # Skip non-lesion artifacts
    artifacts = {
        "retractor",
        "tongue",
        "caries",
        "wood",
        "instrument",
        "shadow",
        "glare",
        "cheek",
    }
    if any(art in desc or art in label_key for art in artifacts):
        return None

    # Clinical description
    if "opmd: yes" in desc or ("opmd" in desc and "no opmd" not in desc):
        return "opmd"
    if any(x in desc for x in ["variation from normal", "pigmentation", "variation"]):
        return "variation"
    if "normal" in desc:
        return "normal"

    # Region label
    if "opmd" in label_key:
        return "opmd"
    if "variation" in label_key:
        return "variation"
    if "normal" in label_key:
        return "normal"

    # JSON filename fallback
    js = json_stem.lower()
    if "opmd" in js:
        return "opmd"
    if "variation" in js:
        return "variation"
    if "normal" in js:
        return "normal"

    return None


# ── Main Parser ──────────────────────────────────────────────────────────────
def parse_via_json(
    logger, json_path: Path, image_path: Path, csv_label: str = ""
) -> List[Annotation]:
    """
    Parse JSON and return annotations ONLY for the given image_path.
    """
    json_path = Path(json_path)
    image_path = Path(image_path)

    if not json_path.exists():
        logger.error(f"JSON not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON {json_path}: {e}")
        return []

    # Support both flat and VIA v2 (_via_img_metadata)
    image_dict = raw.get("_via_img_metadata", raw)

    annotations = []

    for via_key, entry in image_dict.items():
        if not isinstance(entry, dict):
            continue

        filename_in_json = entry.get("filename")
        if not filename_in_json:
            continue

        # Critical: Match exact image name (handles _full.json perfectly)
        if Path(filename_in_json).name != image_path.name:
            continue

        regions_raw = entry.get("regions", [])
        if not regions_raw:
            continue

        ann = Annotation(
            json_path=json_path,
            image_path=image_path,
            via_key=via_key,
            csv_label=csv_label,
        )

        json_stem = json_path.stem

        for reg in regions_raw:
            shape_attrs = reg.get("shape_attributes", {})
            region_attrs = reg.get("region_attributes", {}) or {}

            xs, ys, shape_name = _shape_to_polygon(shape_attrs)
            if len(xs) < 3:
                continue

            label = _get_target_label(region_attrs, json_stem, csv_label)
            if label is None:
                continue  # Skip artifact or unknown

            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            region = RegionAnnotation(
                shape=shape_name,
                label=label,
                polygon_x=xs,
                polygon_y=ys,
                bbox=(x1, y1, x2, y2),
            )
            ann.regions.append(region)

        if ann.regions:
            annotations.append(ann)

    return annotations


# ── Load from DataFrame (Recommended) ───────────────────────────────────────
def load_annotations_from_df(
    logger: logging.Logger,
    df: pd.DataFrame,
    image_path_col: str = "image_path",
    json_col: str = "json_file",
    label_col: str = "label",
) -> List[Annotation]:
    """Load using direct paths from CSV"""
    annotations: List[Annotation] = []

    for _, row in df.iterrows():
        img_path = Path(str(row[image_path_col]))
        json_path = Path(str(row[json_col])) if pd.notna(row.get(json_col)) else None
        csv_label = str(row.get(label_col, ""))

        if json_path and json_path.exists():
            parsed_list = parse_via_json(logger, json_path, img_path, csv_label)
            annotations.extend(parsed_list)
        else:
            logger.warning(f"JSON not found for image: {img_path.name}")

    logger.info(
        f"Successfully loaded {len(annotations)} annotated images from DataFrame"
    )
    return annotations
