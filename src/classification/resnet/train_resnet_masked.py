"""
src/classification/resnet/train_resnet_masked.py
---------------------------------------------------
Pipeline B: trains a SEPARATE ResNet50 classifier on U-Net-masked images
(background zeroed, lesion region kept) instead of raw images.

Reuses every function from train_resnet.py (train_one_epoch,
validate_one_epoch, compute_accuracy, compute_metrics, train()) UNCHANGED —
the only difference between Pipeline A (raw images) and Pipeline B (masked
images) is which CSV is loaded and which config section / checkpoint
directory is used. This keeps the two pipelines' training logic from
drifting apart over time.

IMPORTANT: run precompute_masked_dataset.py FIRST — this script reads the
masked-image CSV that script produces; it does not run U-Net itself.

Config keys needed (add to config.ini / kaggle_config.ini)
------------------------------------------------------------
[CLASSIFICATION-RESNET-MASKED]
resnet.config       = path to a resnet.ini for Pipeline B (can reuse the
                       same resnet.ini as Pipeline A, or point to a
                       separate one if you want different epochs/LR/
                       checkpoint_dir/output_dir for the masked model)
masked.dataset.csv  = output CSV from precompute_masked_dataset.py

Usage
-----
    python -m src.classification.resnet.train_resnet_masked -p kaggle
"""

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.classification.resnet.resnet_config import ResNetConfig
from src.classification.resnet.train_resnet import train, get_configpath


if __name__ == "__main__":
    config_path = get_configpath()
    config = load_config(config_path)

    # Separate ini section so Pipeline B can use a different resnet.ini
    # (different LR/epochs/checkpoint_dir/output_dir) than Pipeline A
    # without touching Pipeline A's config at all.
    resnet_ini = config.get("CLASSIFICATION-RESNET-MASKED", "resnet.config")
    resnet_config = load_config(resnet_ini)

    logger = initialize_logger(config)
    cfg = ResNetConfig(resnet_config)

    # csv_path is the OUTPUT of precompute_masked_dataset.py — not rebuilt
    # here, since the masking step only needs to run once, not every time
    # you retrain.
    csv_path = config.get("CLASSIFICATION-RESNET-MASKED", "masked.dataset.csv")

    train(logger=logger, cfg=cfg, csv_path=csv_path)
