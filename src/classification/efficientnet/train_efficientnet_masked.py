"""
src/classification/efficientnet/train_efficientnet_masked.py
--------------------------------------------------------------
Masked-image EfficientNet: trains a SEPARATE EfficientNet classifier on the
lesion-masked images (background zeroed, lesion region kept) instead of raw
images. This is the EfficientNet counterpart of train_resnet_masked.py.

Reuses train() from train_efficientnet.py UNCHANGED — the only differences
from the raw-image EfficientNet run are which CSV is loaded and which
config / checkpoint / output directories are used, so the two runs can never
drift apart in their training logic.

The masked dataset itself is architecture-agnostic (it is just images on
disk plus a CSV), so there is NO second precompute step: this script reuses
the exact masked images and CSV that precompute_masked_dataset.py already
produced for the ResNet masked run. That also guarantees ResNet-masked and
EfficientNet-masked are compared on identical inputs.

IMPORTANT: run precompute_masked_dataset.py FIRST in each fresh Kaggle
session (it writes the masked images + CSV this script reads):

    python -m src.classification.resnet.precompute_masked_dataset -p kaggle

Config keys needed (add to config.ini / kaggle_config.ini)
------------------------------------------------------------
[CLASSIFICATION-EFFICIENTNET-MASKED]
efficientnet.config = path to efficientnet_masked.ini (separate model_dir /
                      output_dir / checkpoint_dir so this run never overwrites
                      the raw-image EfficientNet results)
masked.dataset.csv  = OPTIONAL. Leave it out to reuse the CSV from
                      [CLASSIFICATION-RESNET-MASKED] masked.dataset.csv
                      (single source of truth for the masked dataset path).

Usage
-----
    python -m src.classification.efficientnet.train_efficientnet_masked -p kaggle
"""

from pathlib import Path

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.classification.efficientnet.efficientnet_config import EfficientNetConfig
from src.classification.efficientnet.train_efficientnet import train, get_configpath

SECTION = "CLASSIFICATION-EFFICIENTNET-MASKED"
SHARED_SECTION = "CLASSIFICATION-RESNET-MASKED"


def get_masked_csv(config) -> str:
    """
    Path of the masked-image CSV. Uses this pipeline's own
    masked.dataset.csv if set, otherwise falls back to the one the ResNet
    masked pipeline already uses, so the path only has to be configured in
    one place.
    """
    own = config.get(SECTION, "masked.dataset.csv", fallback=None)
    if own:
        return own
    return config.get(SHARED_SECTION, "masked.dataset.csv")


if __name__ == "__main__":
    config_path = get_configpath()
    config = load_config(config_path)

    # Separate ini so this run gets its own checkpoint/output directories
    # and can use different hyperparameters without touching efficientnet.ini.
    efficientnet_ini = config.get(SECTION, "efficientnet.config")
    efficientnet_config = load_config(efficientnet_ini)

    logger = initialize_logger(config)
    cfg = EfficientNetConfig(efficientnet_config)

    # csv_path is the OUTPUT of precompute_masked_dataset.py — not rebuilt
    # here. Fail with an instruction instead of a bare pandas error if the
    # masking step has not been run in this session yet.
    csv_path = get_masked_csv(config)
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Masked dataset CSV not found: {csv_path}\n"
            "Run the masking step first:\n"
            "    python -m src.classification.resnet.precompute_masked_dataset -p kaggle"
        )

    train(logger=logger, cfg=cfg, csv_path=csv_path)
