import pandas as pd
import json

from pathlib import Path

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config


def polygon_area(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def shape_area(shape_attrs):
    shape_name = shape_attrs.get("name", "").lower()

    if shape_name == "polygon":
        xs = shape_attrs.get("all_points_x", [])
        ys = shape_attrs.get("all_points_y", [])
        return polygon_area(xs, ys)

    elif shape_name == "rect":
        w = shape_attrs.get("width", 0)
        h = shape_attrs.get("height", 0)
        return float(w * h)

    return 0.0


def normalize_label(label):
    if not label:
        return None

    label = label.strip().lower()
    label = " ".join(label.split())

    alias_map = {
        "dt": "dorsal tongue",
        "ll": "lower lip",
        "vt": "ventral tongue",
        "ll": "lower lip",
        "ua": "upper arch",
        "rb": "right buccal mucosa",
        "lb": "left buccal mucosa",
    }
    return alias_map.get(label, label)


def load_via_metadata(data):
    if "_via_img_metadata" in data:
        return data["_via_img_metadata"]
    return data


def is_lower_arch_image(image_name):
    stem = Path(image_name).stem.upper()
    return stem.endswith("_LA") or stem.endswith("RLA") or stem.endswith("LA")


def collect_biggest_unique_labels(json_paths, image_paths):
    """
    Unique labels is to collect all unique description used for same
    label like dorsal tongue and dorsa tongue
    Just run once to get the labels
    """
    # image_names = {Path(p).name for p in image_paths}
    unique_labels = []
    # imagewise_results = []

    for img_name, json_path in zip(image_paths, json_paths):
        logger.info(
            f"Starting conversion of VIA json to coco json format for"
            f"image name: {img_name} and json_path : {type(json_path)}"
        )
        if (
            type(img_name) is float
            or img_name == "nan"
            or type(json_path) is float
            or json_path == "nan"
        ):
            logger.warning(
                f"Image name : {img_name} or Json file : {json_path} is missing"
            )
            continue
        json_path = Path(json_path)
        image_name = img_name.split("/")[-1]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = load_via_metadata(data)
        logger.info(f"Metadata for file {image_name} is: {metadata.keys()}")
        key_name = [key for key in list(metadata.keys()) if image_name in key]
        if len(key_name) <= 0:
            logger.info(f"Metadata for file {image_name} is missing: {metadata.keys()}")
            continue
        key_name = key_name[0]
        logger.info(f"Metadata for {key_name}: {metadata[key_name].keys()}")
        # for _, image_info in metadata[key_name].items():
        image_info = metadata[key_name]
        filename = image_info.get("filename", "")

        if not filename:
            continue

        # if filename not in image_names:
        #     continue

        if is_lower_arch_image(filename):
            continue

        regions = image_info.get("regions", [])
        max_area = -1
        max_label = None
        for region in regions:
            if region:
                shape_attrs = region.get("shape_attributes", {})
                region_attrs = region.get("region_attributes", {})
                raw_label = region_attrs.get("Description", "")
                norm_label = normalize_label(raw_label)
                area = shape_area(shape_attrs)
                logger.info(
                    f"Image {key_name} values: {shape_attrs}, {region_attrs}, "
                    f"{raw_label}, {norm_label}, {area}"
                )
                if area > max_area and norm_label:
                    max_area = area
                    max_label = norm_label

        if max_label is not None:
            # row = {
            #     "json_file": str(json_path),
            #     "filename": filename,
            #     "largest_label": max_label,
            #     "largest_area": max_area,
            # }
            # imagewise_results.append(row)

            if max_label not in unique_labels:
                unique_labels.append(max_label)  # [max_label] = row

    return unique_labels  # , imagewise_results


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config=config)
    merged_csv_path = config.get("SMART_MERGED", "merged.output.filename")
    merged_csv = pd.read_csv(merged_csv_path)
    json_paths = list(merged_csv.json_file.values)
    img_paths = list(merged_csv.image_path.values)
    # Unique labels is to collect all unique description used for same
    # label like dorsal tongue and dorsa tongue
    # Just run once to get the labels
    unique_labels = collect_biggest_unique_labels(json_paths, img_paths)
    logger.info(f"Unique Labels Length: {len(unique_labels)}")
    logger.info(f"Unique Labels: {unique_labels}")
