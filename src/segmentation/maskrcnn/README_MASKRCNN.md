# Mask R-CNN Intraoral Lesion Segmentation

End-to-end pipeline for segmenting intraoral lesions using Mask R-CNN on the SMART-II / SMART-OM dataset.

---

## Project Structure

```
mask_rcnn_lesion/
├── config/
│   └── maskrcnn.ini          # Reads config.ini → typed config objects
├── data/
│   ├── dataset.py           # LesionInferenceDataset + LesionAnnotatedDataset
├── models/
│   └── mask_rcnn_builder.py # build_coco_pretrained() + build_lesion_model()
├── utils/
│   └── evaluation.py        # mAP50 / mAP50-95 (box + mask), per-class AP\
├── config/
│   └── maskrcnnconfig.py    # Warpper over maskrcnn.ini
├── scripts/
│   ├── inference_zeroshot.py   # Zero-shot COCO-pretrained inference
│   └── train_mask_rcnn.py      # Full training + validation + test eval
└── outputs/
    ├── zeroshot/            # Zero-shot results
    ├── training/            # Checkpoints, history, test report
    └── logs/                # Rotating log files
```

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow pandas numpy
```

Place `config.ini` in the project root (same directory as this README).

---

## Script 1 — Zero-shot Inference (`inference_zeroshot.py`)

Uses **COCO-pretrained** Mask R-CNN without any fine-tuning.

Since COCO has no oral-anatomy classes, ALL detections above the score threshold
are treated as candidate lesion regions. Useful as a baseline and for visual
exploration.

### What it does
1. Loads `smart_merged.csv`
2. Runs Mask R-CNN on every image
3. Saves annotated visualisations (boxes + masks overlaid)
4. From json_file, parses the polygon annotations
   and evaluates box mAP50 + mask mAP50 against the COCO detections

### Run
```bash
python scripts/inference_zeroshot.py \
    --csv /path/to/smart_merged.csv \
    --output_dir ./outputs/zeroshot \
    --score_threshold 0.5 \
    --base_rewrite "/mnt/c/Users/User/Documents/ManthanShala/v18hub/Projects/intraoral_leison=/your/local/data"
```

### Key arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | from config | Path to smart_merged.csv |
| `--output_dir` | `./outputs/zeroshot` | Results directory |
| `--score_threshold` | 0.5 | Minimum detection confidence |
| `--mask_threshold` | 0.5 | Sigmoid threshold for binary mask |
| `--location` | ALL | Filter by lesion_location code (DT / VT / LB …) |
| `--base_rewrite` | None | `OLD=NEW` path prefix rewrite |
| `--device` | cuda | cuda or cpu |

### Outputs
```
outputs/zeroshot/
├── inference_results.csv        # Per-image: n_detections, max_score, timing
├── eval_zeroshot_report.json    # mAP50, mAP50-95 vs JSON ground truths
└── visualisations/
    └── <image_stem>_pred.jpg    # Annotated image with boxes + masks
```

---

## Script 2 — Training Pipeline (`train_mask_rcnn.py`)

Fine-tunes Mask R-CNN on the **annotated subset** (`lesion_location == 'json_file'`).

### What it does
1. Loads `smart_merged.csv` and parses VIA JSON polygons from the `image_path` column
2. Stratified patient-aware split → train / val / test (no patient leakage)
3. Trains Mask R-CNN (ResNet50-FPN, COCO-pretrained backbone)
4. Validates each epoch (box mAP50 + mask mAP50)
5. Saves best model checkpoint + full training history
6. Runs final evaluation on test set

### Label mapping
| CSV `label` | Training class index |
|-------------|---------------------|
| background  | 0 (implicit)        |
| normal      | 1                   |
| variation   | 2                   |
| opmd        | 3                   |

### Run
```bash
python scripts/02_train_mask_rcnn.py \
    --csv /path/to/smart_merged.csv \
    --output_dir ./outputs/training \
    --epochs 30 \
    --batch_size 2 \
    --lr 0.001 \
    --num_classes 3 \
    --base_rewrite "/mnt/c/Users/User/Documents/ManthanShala/v18hub/Projects/intraoral_leison=/your/local/data"
```

### Resume training
```bash
python scripts/02_train_mask_rcnn.py \
    --resume ./outputs/training/checkpoints/epoch_010.pth \
    --epochs 50 \
    ...
```

### Key arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | from config | Path to smart_merged.csv |
| `--output_dir` | `./outputs/training` | Results directory |
| `--base_rewrite` | None | `OLD=NEW` path prefix rewrite |
| `--num_classes` | 3 | Lesion classes (excl. background): normal / variation / opmd |
| `--epochs` | 30 | Training epochs |
| `--batch_size` | 2 | Mask R-CNN is memory heavy — 2 is typical on 8GB GPU |
| `--lr` | 0.001 | Initial learning rate (CosineAnnealing scheduler) |
| `--patience` | 10 | Early stopping patience (0 = disabled) |
| `--val_split` | 0.15 | Validation fraction |
| `--test_split` | 0.10 | Test fraction |
| `--resume` | None | Path to `.pth` checkpoint to resume from |
| `--device` | cuda | cuda or cpu |

### Outputs
```
outputs/training/
├── best_mask_rcnn.pth           # Best checkpoint (by val box mAP50)
├── training_history.json        # Per-epoch train losses + val mAP
├── test_eval_report.json        # Final test set metrics
└── checkpoints/
    ├── epoch_001.pth
    ├── epoch_002.pth
    └── …
```

---

## Path Rewriting

The `smart_merged.csv` was created on a Windows machine. Paths look like:
```
/mnt/c/Users/User/Documents/ManthanShala/v18hub/Projects/intraoral_leison/...
```

If running on a different machine, use `--base_rewrite`:
```
--base_rewrite "/mnt/c/Users/User/Documents/ManthanShala/v18hub/Projects/intraoral_leison=/data/intraoral_leison"
```

---

## Annotation Format

The JSON files follow the **VGG Image Annotator (VIA)** format with polygon regions:
```json
{
  "image.jpg": {
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [x0, x1, ...],
          "all_points_y": [y0, y1, ...]
        },
        "region_attributes": { "label": "opmd" }
      }
    ]
  }
}
```

The parser also handles `rect`, `circle`, and `ellipse` shapes (all converted to polygons).

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `box_mAP50` | Box detection mAP at IoU=0.50 |
| `box_mAP50_95` | Box detection mAP averaged over IoU 0.50:0.05:0.95 |
| `mask_mAP50` | Instance mask mAP at IoU=0.50 |
| `mask_mAP50_95` | Instance mask mAP averaged over IoU 0.50:0.05:0.95 |
| `per_class_ap50` | Per-class AP at IoU=0.50 |

---

## GPU Memory Tips

- Mask R-CNN is memory-heavy. On a 6–8 GB GPU use `--batch_size 2`
- On 4 GB GPU: `--batch_size 1` and reduce `image_min_side` in `config/settings.py` to 600
- On CPU: inference works but training will be very slow (use `--batch_size 1`)

---

## Extending

- **Add a new class**: update `LABEL_MAP` and `CLASS_NAMES` in `config/settings.py`,
  increase `--num_classes`
- **Different backbone**: change `maskrcnn_resnet50_fpn` → `maskrcnn_resnet50_fpn_v2`
  in `models/mask_rcnn_builder.py`
- **More augmentation**: edit `data/transforms.py` → `TrainingTransform`
- **COCO-style strict eval**: install `pycocotools` and replace `compute_map` with
  `COCOeval` — the prediction / GT format is already compatible
