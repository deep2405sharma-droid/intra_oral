"""
src/classification/resnet/precompute_masked_dataset.py
--------------------------------------------------------
Builds Pipeline B's masked-image dataset using EXISTING ground-truth
lesion annotations (the same coco_file polygons U-Net itself was trained
on) — no U-Net inference is run here.

For each row in the ResNet training CSV:
  - If a coco_file annotation exists and its file is on disk: rasterise
    the lesion polygon(s) into a binary mask — reusing
    unet_builder._coco_to_semantic_mask, the exact same rasterisation
    code U-Net was trained against — and zero out the background.
  - If no coco_file exists (expected for the 'normal' class — healthy
    images have no lesion to annotate): keep the ORIGINAL, UNMASKED image
    unchanged. Rasterising a missing/empty annotation would produce an
    all-zero mask -> a solid black image, wiping out the entire
    normal-class training signal, so normal images are deliberately left
    untouched rather than blacked out.
  - If an annotation exists but rasterises to nothing usable (every
    polygon below min_area) it's treated the same as "no annotation".

Output: masked images saved to disk + a new CSV with the same schema as
the original ResNet training CSV (image_path updated, everything else
unchanged) — feeds straight into resnet_builder.build_data_loaders()
unchanged, via train_resnet_masked.py.

Config keys needed (add to config.ini / kaggle_config.ini)
------------------------------------------------------------
[CLASSIFICATION-RESNET-MASKED]
masked.output.dir  = directory to save masked images into
masked.dataset.csv = output CSV path (same schema as the raw ResNet train.dataset)
min_area           = minimum rasterised lesion pixel count to count as a
                      real annotation (artefact filter). Optional, default 500 —
                      mirrors unet.ini's [DATASET] min_area.

Usage
-----
    python -m src.classification.resnet.precompute_masked_dataset -p kaggle
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.segmentation.unet2.unet_builder import (
    _coco_to_semantic_mask,
    LESION_CLASS_MAP,
)
from src.classification.resnet.resnet_config import ResNetConfig
from src.classification.resnet.train_resnet import get_dataset_path


def apply_mask_to_image(
    image_pil: PILImage.Image, binary_mask: np.ndarray
) -> PILImage.Image:
    """
    Zero out background pixels (mask == 0); keep lesion-region pixels as-is.
    binary_mask must be the same [H, W] size as image_pil.

    Inlined from unet_inference.py (removed — this script no longer runs
    U-Net inference, only rasterises ground-truth coco_file annotations,
    so apply_mask_to_image was the only piece of that file still in use).
    """
    img_np = np.array(image_pil.convert("RGB"), dtype=np.uint8)  # [H, W, 3]
    masked_np = img_np.copy()
    masked_np[binary_mask == 0] = 0
    return PILImage.fromarray(masked_np)


def get_configpath():
    parser = argparse.ArgumentParser(description="Precompute masked ResNet dataset")
    parser.add_argument("-p", "--profile")
    args = parser.parse_args()
    config_path = "config/config.ini"
    if args.profile and args.profile.lower() == "kaggle":
        config_path = "config/kaggle_config.ini"
    return config_path


def main():
    config_path = get_configpath()
    config = load_config(config_path)
    logger = initialize_logger(config)

    section = "CLASSIFICATION-RESNET-MASKED"
    output_dir = Path(config.get(section, "masked.output.dir"))
    out_csv = config.get(section, "masked.dataset.csv")
    min_area = config.getint(section, "min_area", fallback=500)

    # Source: the SAME CSV Pipeline A trains on — must already carry a
    # coco_file column (inherited from the merged dataset shared with
    # Mask R-CNN / U-Net), even though Pipeline A itself never reads it.
    src_csv = config.get("CLASSIFICATION-RESNET", "train.dataset")

    # Kaggle sessions start fresh each time, so this CSV won't exist yet
    # unless train_resnet.py has already run once in this session. Rather
    # than requiring that as a manual prerequisite, build it here the
    # first time — same get_dataset_path() function train_resnet.py uses,
    # so there's no duplicated dataset-building logic to drift out of sync.
    if not Path(src_csv).exists():
        logger.info(
            "%s not found — building it now via get_dataset_path() "
            "(same step train_resnet.py runs).",
            src_csv,
        )
        resnet_ini_path = config.get("CLASSIFICATION-RESNET", "resnet.config")
        resnet_cfg = ResNetConfig(load_config(resnet_ini_path))
        src_csv = get_dataset_path(logger=logger, config=config, cfg=resnet_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv, dtype=str)
    logger.info("Source dataset rows: %d", len(df))
    has_coco_column = "coco_file" in df.columns
    if not has_coco_column:
        logger.warning(
            "No 'coco_file' column found in %s — every row will be treated "
            "as unannotated and copied through unmasked.",
            src_csv,
        )

    out_rows = []
    n_masked = 0
    n_unmasked = 0
    n_failed = 0

    for i, row in df.iterrows():
        img_path = str(row.get("image_path", ""))
        if not Path(img_path).exists():
            n_failed += 1
            continue

        coco_path = str(row.get("coco_file", "")) if has_coco_column else ""
        has_annotation = bool(
            coco_path and coco_path.lower() != "nan" and Path(coco_path).exists()
        )

        try:
            image = PILImage.open(img_path).convert("RGB")
            out_image = image  # default: unmasked (normal / no annotation)

            if has_annotation:
                W_orig, H_orig = image.size
                with open(coco_path, "r") as f:
                    coco = json.load(f)

                semantic = _coco_to_semantic_mask(
                    coco,
                    H=H_orig,
                    W=W_orig,
                    label_class_map=LESION_CLASS_MAP,
                    min_area=min_area,
                )
                binary_mask = (semantic > 0).astype(np.uint8)

                if binary_mask.sum() > 0:
                    out_image = apply_mask_to_image(image, binary_mask)
                    n_masked += 1
                else:
                    # Annotation existed but rasterised to nothing usable
                    # (e.g. every polygon fell below min_area) — fall back
                    # to unmasked rather than saving a blank image.
                    n_unmasked += 1
            else:
                n_unmasked += 1

            out_name = f"{i:06d}_{Path(img_path).name}"
            out_path = output_dir / out_name
            out_image.save(out_path)

            new_row = row.to_dict()
            new_row["image_path"] = str(out_path)
            out_rows.append(new_row)

        except Exception as e:
            logger.warning("Failed on row %d (%s): %s", i, img_path, e)
            n_failed += 1

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d images", i + 1, len(df))

    logger.info(
        "Done. %d masked (lesion annotation applied), %d unmasked "
        "(normal / no usable annotation), %d failed.",
        n_masked,
        n_unmasked,
        n_failed,
    )

    out_df = pd.DataFrame(out_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("Masked dataset CSV saved: %s  (%d rows)", out_csv, len(out_df))


if __name__ == "__main__":
    main()