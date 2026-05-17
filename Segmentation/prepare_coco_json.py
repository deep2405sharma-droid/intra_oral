import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image

import numpy as np
import pandas as pd


# LABEL_TO_CLASS = {
#     "left buccal mucosa": "left buccal mucosa",
#     "left bucccal mucosa": "left buccal mucosa",
#     "location: left buccal mucosa colour: white margin: not well demarcated surface texture: smooth description of the lesion: single opmd: no opmd others: hyperkeratosis": "left buccal mucosa",
#     "lefrt buccal mucosa": "left buccal mucosa",
#     "location: left buccal mucosa color: brown to black margin: not well demarcated surface feature: smooth description of the lesion: non-scrappable no opmd: pigmentation": "left buccal mucosa",
#     "right buccal mucosa": "right buccal mucosa",
#     "location: right buccal mucosa colour: brown to black margin: not well demarcated surface texture: smooth description of the lesion:multiple opmd: no opmd others: pigmentation": "right buccal mucosa",
#     "location: right buccal mucosa colour: white margin: not well demarcated surface texture: smooth description of the lesion: multiple opmd: no opmd others: tobacco pouch keratosis": "right buccal mucosa",
#     "location: right buccal mucosa color: brown margin: not well demarcated surface texture: smooth description of the lesion: patch opmd: no opmd others: pigmentation": "right buccal mucosa",
#     "location: right buccal mucosa color: brown to black margin: not well demarcated surface feature: smooth description of the lesion: non-scrappable no opmd: pigmentation": "right buccal mucosa",
#     "dorsal tongue": "dorsal tongue",
#     "location: dorsal tongue colour: red margin: not well demarcated surface texture:smooth description of the lesion: multiple opmd: no opmd others: extrinsic stains": "dorsal tongue",
#     "location: left lateral tongue color: brown to black surface texture: smooth description of the lesion: prominent tongue papilla opmd: no opmd others: hyperpigmentation": "dorsal tongue",
#     "location: dorsal tongue color: brown margin: not well demarcated surface texture: smooth description of the lesion: opmd: no others: pigmentation": "dorsal tongue",
#     "dorsa tongue": "dorsal tongue",
#     "ventral tongue": "ventral tongue",
#     "upper lip": "upper lip",
#     "hard palate": "upper arch",
#     "upper labial mucosa": "upper arch",
#     "location: hard palate color: brown to black margin: not well demarcated surface texture: smooth description of the lesion: non-scrappable no opmd: smoker's palate": "upper arch",
#     "lower labial mucosa": "lower lip",
#     "lower lip mucosa": "lower lip",
#     "lower lip": "lower lip",
# }

NOISE_LABELS = {
    "retractor wood",
    "out of focus",
    "upper left first molar",
    "upper left first premolar",
    "reflection of light",
    "shadow",
    "upper vermillion border",
}

CLASS_NAMES = [
    "LEFT BUCCAL MUCOSA",
    "RIGHT BUCCAL MUCOSA",
    "DORSAL TONGUE",
    "LOWER LIP",
    "UPPER LIP",
    "UPPER ARCH",
    "VENTRAL TONGUE",
]
CLASS_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}

SITE_ROI_CLASS = {
    "LB": "LEFT BUCCAL MUCOSA",
    "RB": "RIGHT BUCCAL MUCOSA",
    "DT": "DORSAL TONGUE",
    "VT": "VENTRAL TONGUE",
    "UL": "UPPER LIP",
    "LL": "LOWER LIP",
    "UA": "UPPER ARCH",
}


def load_via_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    if "_via_img_metadata" in data:
        return data["_via_img_metadata"]
    return data


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip().lower()
    text = " ".join(text.split())
    return text


def polygon_area(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def rect_to_polygon(x: float, y: float, w: float, h: float) -> List[float]:
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def bbox_from_polygon(segmentation: List[float]) -> List[float]:
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]


def shape_to_coco(region: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    shape = region.get("shape_attributes", {}) or {}
    name = str(shape.get("name", "")).lower()

    if name == "polygon":
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if len(xs) < 3 or len(xs) != len(ys):
            return None

        segmentation = []
        for x, y in zip(xs, ys):
            segmentation.extend([float(x), float(y)])

        area = polygon_area(xs, ys)
        bbox = bbox_from_polygon(segmentation)

        return {
            "segmentation": [segmentation],  # <--- Flat list here
            "area": float(area),
            "bbox": bbox,
            "iscrowd": 0,
        }

    if name == "rect":
        x = float(shape.get("x", 0))
        y = float(shape.get("y", 0))
        w = float(shape.get("width", 0))
        h = float(shape.get("height", 0))
        if w <= 0 or h <= 0:
            return None
        segmentation = rect_to_polygon(x, y, w, h)
        return {
            "segmentation": [segmentation],
            "area": float(w * h),
            "bbox": [x, y, w, h],
            "iscrowd": 0,
        }

    return None


def canonical_to_class_id(canonical_label: str) -> Optional[int]:
    mapping = {
        "LB": CLASS_ID["LEFT BUCCAL MUCOSA"],
        "RB": CLASS_ID["RIGHT BUCCAL MUCOSA"],
        "DT": CLASS_ID["DORSAL TONGUE"],
        "LL": CLASS_ID["LOWER LIP"],
        "UL": CLASS_ID["UPPER LIP"],
        "UA": CLASS_ID["UPPER ARCH"],
        "VT": CLASS_ID["VENTRAL TONGUE"],
    }
    return mapping.get(canonical_label)


def build_categories() -> List[Dict[str, Any]]:
    return [
        {"id": CLASS_ID[name], "name": name.lower(), "supercategory": "oral_site"}
        for name in CLASS_NAMES
        if name != "BACKGROUND"
    ]


def extract_image_id(image_path: str) -> str:
    return Path(image_path).stem


def extract_image_name(image_path: str) -> str:
    return Path(image_path).name


def derive_coco_output_path(json_path: str, source: str, image_id: str) -> str:
    p = Path(json_path)
    parent = p.parent
    parent_name = parent.name.lower()
    print(f"Source: {source}")
    if source == "smart_II":
        if parent_name in {"json", "json file"}:
            coco_dir = parent.parent / "coco_json"
        else:
            coco_dir = parent / "coco_json"
        print(f"coco dir smart II: {coco_dir}")
    elif source == "smart_om":
        if parent_name in {"full json", "09. json files"}:
            coco_dir = parent.parent / "coco_json"
        else:
            coco_dir = parent / "coco_json"
        print(f"coco dir smart OM: {coco_dir}")
    else:
        coco_dir = parent / "coco_json"

    coco_dir.mkdir(parents=True, exist_ok=True)
    return str(f"{coco_dir}/{image_id}.json")


def get_region_with_max_area(regions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the region with the largest area."""
    max_area = -1.0
    best_region = None

    for region in regions:
        if not isinstance(region, dict):
            continue

        # Calculate area of this region
        coco_shape = shape_to_coco(region)
        if coco_shape is None:
            continue

        area = coco_shape["area"]
        if area > max_area:
            max_area = area
            best_region = region

    return best_region


def build_coco_from_via(
    logger: Optional[logging.Logger], min_area: int, dataset: pd.DataFrame
) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["coco_file"] = pd.Series(
        [None] * len(dataset), index=dataset.index, dtype="object"
    )

    for idx, row in dataset.iterrows():
        img_name = row.get("image_path", np.nan)
        json_path = row.get("json_file", np.nan)

        if (
            type(img_name) is float
            or img_name == "nan"
            or type(json_path) is float
            or json_path == "nan"
        ):
            logger.info(f"Skipping row {idx} due to missing image/json path")
            continue

        image_path = str(img_name)
        json_path = str(json_path)

        image_name = extract_image_name(image_path)

        raw_label = row.get("lesion_location", "")
        if not raw_label or raw_label == "LA":
            continue

        image_id = extract_image_id(image_path)
        # patient_id = image_id

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.exception(f"Failed to read json {json_path}: {e}")
            continue

        metadata = load_via_metadata(data)
        logger.info(f"Metadata for file {image_name} is: {metadata.keys()}")

        key_name = [key for key in list(metadata.keys()) if image_name in key]

        # Fallback: normal patients share one JSON — try matching by patient stem
        # e.g. SMITA00363_R_DT.jpg not found → try any key containing SMITA00363
        if len(key_name) <= 0:
            image_stem = Path(image_path).stem          # SMITA00363_R_DT
            patient_id = image_stem.split("_")[0]       # SMITA00363
            key_name   = [key for key in list(metadata.keys()) if patient_id in key]

        if len(key_name) <= 0:
            logger.info(f"Metadata for file {image_name} is missing: {list(metadata.keys())[:3]}")
            continue

        key_name = key_name[0]
        logger.info(f"Metadata for {key_name}: {metadata[key_name].keys()}")

        image_info = metadata[key_name]
        filename = image_info.get("filename", "")
        regions = image_info.get("regions", []) or []
        with Image.open(image_path) as img:
            w, h = img.size
        coco = {
            "info": {
                "description": "COCO converted from VIA",
                "version": "1.0",
            },
            "images": [
                {
                    "id": image_id,
                    "file_name": filename if filename else image_name,
                    "path": image_path,
                    "width": w,
                    "height": h,
                }
            ],
            "annotations": [],
            "categories": build_categories(),
        }

        best_region = get_region_with_max_area(regions)
        if best_region:
            if raw_label and raw_label not in NOISE_LABELS:
                canonical_label = SITE_ROI_CLASS.get(raw_label, None)
                class_id = canonical_to_class_id(raw_label)

            if canonical_label and class_id:
                coco_shape = shape_to_coco(best_region)

                # You can still keep the min_area check if you want
                if coco_shape and coco_shape["area"] >= min_area:
                    ann_id = f"{image_id}_1"
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "label": canonical_label,
                            **coco_shape,
                        }
                    )

        source = row.get("source", "")
        coco_path = derive_coco_output_path(json_path, source, image_id)
        try:
            with open(coco_path, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=2)
            dataset.at[idx, "coco_file"] = coco_path
            logger.info(f"Saved COCO json for {image_name} at {coco_path}")
        except Exception as e:
            logger.exception(f"Failed to write coco json {coco_path}: {e}")

    return dataset


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import configparser
    from src.common import intraoral_logger as iolog

    # ── Config ────────────────────────────────────────────────────
    CONFIG_PATH = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    # ── Logger ────────────────────────────────────────────────────
    log_file = config.get("LOGGER", "logger.filename")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = iolog.getLogger(log_file)

    logger.info("=" * 65)
    logger.info("  prepare_coco_json  —  per-image COCO from VIA")
    logger.info("=" * 65)

    # ── Load CSVs ─────────────────────────────────────────────────
    merged_csv      = config.get("DATA", "merged.csv")
    aug_smart_csv   = config.get("DATA", "aug.smart.csv")
    aug_smartom_csv = config.get("DATA", "aug.smartom.csv")

    VALID_LOCATIONS = {"DT", "LB", "RB", "UL", "UA", "VT", "LL"}
    LOCATION_REMAP  = {"RB1": "RB", "RB2": "RB", "RB3": "RB"}

    def clean_location(loc):
        k = str(loc).strip()
        return LOCATION_REMAP.get(k, k)

    # Merged
    df_merged = pd.read_csv(merged_csv)
    df_merged["lesion_location"] = df_merged["lesion_location"].apply(clean_location)
    df_merged = df_merged[df_merged["lesion_location"].isin(VALID_LOCATIONS)].copy()
    logger.info(f"Merged rows      : {len(df_merged)}")

    # Augmented smart_II
    if os.path.exists(aug_smart_csv):
        df_aug_s = pd.read_csv(aug_smart_csv)
        df_aug_s["lesion_location"] = df_aug_s["lesion_location"].apply(clean_location)
        df_aug_s = df_aug_s[df_aug_s["lesion_location"].isin(VALID_LOCATIONS)].copy()
    else:
        df_aug_s = pd.DataFrame()
    logger.info(f"Aug SMART-II rows: {len(df_aug_s)}")

    # Augmented smart_om
    if os.path.exists(aug_smartom_csv):
        df_aug_om = pd.read_csv(aug_smartom_csv)
        df_aug_om["lesion_location"] = df_aug_om["lesion_location"].apply(clean_location)
        df_aug_om = df_aug_om[df_aug_om["lesion_location"].isin(VALID_LOCATIONS)].copy()
    else:
        df_aug_om = pd.DataFrame()
    logger.info(f"Aug SMART-OM rows: {len(df_aug_om)}")

    # Combine all
    df_all = pd.concat([df_merged, df_aug_s, df_aug_om], ignore_index=True)
    logger.info(f"Total rows       : {len(df_all)}")

    # ── Run ───────────────────────────────────────────────────────
    MIN_AREA = 100   # minimum polygon area in pixels — adjust as needed

    result = build_coco_from_via(
        logger   = logger,
        min_area = MIN_AREA,
        dataset  = df_all,
    )

    # ── Summary ───────────────────────────────────────────────────
    n_saved  = result["coco_file"].notna().sum()
    n_failed = len(result) - n_saved
    logger.info(f"\nCOCO JSONs saved  : {n_saved}")
    logger.info(f"Rows skipped      : {n_failed}")

    # ── Save result CSV with coco_file column ─────────────────────
    coco_output_dir = config.get("COCO", "coco.output.dir")
    os.makedirs(coco_output_dir, exist_ok=True)
    result_csv = os.path.join(coco_output_dir, "smart_merged_with_coco.csv")
    result.to_csv(result_csv, index=False)
    logger.info(f"Result CSV saved  → {result_csv}")

    logger.info("=" * 65)
    logger.info("  Done")
    logger.info("=" * 65)