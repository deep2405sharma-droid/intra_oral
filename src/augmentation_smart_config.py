"""
augmentation_config.py
──────────────────────
Defines all augmentation transforms for intraoral images.

Design principles:
  - NO horizontal/vertical flips   (anatomical left/right/top/bottom are clinically meaningful)
  - All spatial transforms use REFLECT border mode to avoid black padding artifacts
  - Color transforms are subtle — lesion color (red/white/brown) is a diagnostic feature
  - Elastic deformation is very gentle — soft tissue deformation only

Each transform group can be toggled via ENABLE_* flags.
Probabilities and limits are tuned for intraoral / oral lesion images.

Augmentations needed to balance dataset:
  - OPMD      :  5 source images  -> ~175 augmentations per image  (target ~880)
  - Variation : 30 source images  ->  ~28 augmentations per image  (target ~870)
  - Normal    : controlled by config.ini augment.labels — skip if not requested
"""

import cv2

# ── Toggle entire groups ───────────────────────────────────────────────────
ENABLE_GEOMETRIC = True
ENABLE_COLOR = False
ENABLE_NOISE = False
ENABLE_BLUR = False
ENABLE_COMPRESSION = True

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TRANSFORMS
# Applied to BOTH image AND coordinates (polygons / rects)
# ══════════════════════════════════════════════════════════════════════════════
GEOMETRIC = [
    {
        # Small-angle rotation — simulates camera tilt during capture
        "name": "Rotate",
        "params": {"limit": 12, "border_mode": cv2.BORDER_REFLECT_101},
        "p": 0.70,
    },
    {
        # Pure translation — patient not perfectly centered in frame.
        "name": "Affine",
        "params": {
            "translate_percent": {"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            "scale": None,  # no scale change
            "rotate": 0,  # no rotation here
            "mode": cv2.BORDER_REFLECT_101,
        },
        "p": 0.60,
    },
    {
        # Mild zoom in/out — distance from camera varies
        "name": "RandomScale",
        "params": {"scale_limit": 0.10},  # ±10%
        "p": 0.40,
    },
    {
        # Random crop + resize — forces zoom-invariant features
        "name": "RandomResizedCrop",
        "params": {
            "size": None,
            "scale": (0.85, 1.00),
            "ratio": (0.95, 1.05),
        },
        "p": 0.40,
    },
    {
        # Very subtle elastic deformation — soft tissue natural deformation
        # Keep alpha LOW to avoid unrealistic tissue distortion
        "name": "ElasticTransform",
        "params": {
            "alpha": 25,  # displacement magnitude — keep ≤30
            "sigma": 5,  # smoothness of displacement field
            "border_mode": cv2.BORDER_REFLECT_101,
        },
        "p": 0.25,
    },
    {
        # Mild perspective shift — camera rarely perfectly orthogonal
        "name": "Perspective",
        "params": {"scale": (0.02, 0.04)},
        "p": 0.25,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# COLOR / ILLUMINATION TRANSFORMS
# Pixel-only — no coordinate mapping needed
# ══════════════════════════════════════════════════════════════════════════════
COLOR = [
    {
        # Brightness + contrast — intraoral light intensity variation
        "name": "RandomBrightnessContrast",
        "params": {"brightness_limit": 0.10, "contrast_limit": 0.10},
        "p": 0.60,
    },
    {
        # HSV jitter — different intraoral cameras have different color profiles
        # Keep hue_shift SMALL — lesion color (red/white/brown) is diagnostic
        "name": "HueSaturationValue",
        "params": {
            "hue_shift_limit": 6,  # very small — color category must be preserved
            "sat_shift_limit": 12,
            "val_shift_limit": 10,
        },
        "p": 0.50,
    },
    {
        # CLAHE — simulates variation in intraoral illumination uniformity
        "name": "CLAHE",
        "params": {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
        "p": 0.35,
    },
    {
        # Gamma — over/underexposed shots from intraoral cameras
        "name": "RandomGamma",
        "params": {"gamma_limit": (88, 112)},  # very mild ±12%
        "p": 0.35,
    },
    {
        # Tone curve — non-linear brightness response of different cameras
        "name": "RandomToneCurve",
        "params": {"scale": 0.08},
        "p": 0.20,
    },
    {
        # Shadow simulation — partial shadow from retractor or cheek
        "name": "RandomShadow",
        "params": {
            "shadow_roi": (0.0, 0.0, 1.0, 0.5),
            "num_shadows_lower": 1,
            "num_shadows_upper": 1,
            "shadow_dimension": 4,
        },
        "p": 0.15,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# NOISE — sensor and sensor noise from intraoral cameras
# ══════════════════════════════════════════════════════════════════════════════
NOISE = [
    {
        "name": "GaussNoise",
        "params": {"var_limit": (5.0, 20.0), "per_channel": True},
        "p": 0.30,
    },
    {
        # ISO noise — simulates low-light intraoral captures
        "name": "ISONoise",
        "params": {"color_shift": (0.01, 0.03), "intensity": (0.05, 0.15)},
        "p": 0.15,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# BLUR — slight focus variation, motion blur from patient movement
# ══════════════════════════════════════════════════════════════════════════════
BLUR = [
    {
        "name": "GaussianBlur",
        "params": {"blur_limit": (3, 5), "sigma_limit": (0.1, 0.5)},
        "p": 0.30,
    },
    {
        # Motion blur — slight patient/camera movement
        "name": "MotionBlur",
        "params": {"blur_limit": 5},
        "p": 0.10,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSION ARTIFACTS — JPEG images at varying quality levels
# ══════════════════════════════════════════════════════════════════════════════
COMPRESSION = [
    {
        "name": "ImageCompression",
        "params": {"quality_lower": 80, "quality_upper": 100},
        "p": 0.20,
    },
]

# Random seed for reproducibility
RANDOM_SEED = 42
