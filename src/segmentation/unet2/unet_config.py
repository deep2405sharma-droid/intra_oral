"""
unetconfig.py
-------------
Singleton configuration wrapper for UNet / Attention UNet training.

Mirrors MaskRCNNConfig exactly:
  - Singleton via __new__ so the config is parsed only once per process.
  - _load() maps every .ini key to a typed Python attribute.
  - Callers import UNetConfig and pass the ConfigParser object on first use;
    subsequent calls return the cached instance.

Usage
-----
    from utils.load_configuration import load_config
    from src.segmentation.unet.config.unetconfig import UNetConfig

    base_cfg    = load_config()                                  # config.ini
    unet_ini    = load_config(base_cfg.get("SEGMENT-UNET", "unet.config"))
    cfg         = UNetConfig(unet_ini)
    print(cfg.encoder_lr, cfg.decoder_channels)

    # Subsequent import anywhere in the process — same instance, no re-parse:
    cfg2 = UNetConfig(None)
    assert cfg is cfg2
"""


class UNetConfig:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            if config is None:
                raise ValueError("config must be provided on first initialization")
            instance = super().__new__(cls)
            instance._load(config)
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Internal loader — called exactly once
    # ------------------------------------------------------------------

    def _load(self, config):
        # ── TRAINING ─────────────────────────────────────────────────
        train = config["TRAINING"]
        self.num_classes = train.getint("num_classes")
        self.epochs = train.getint("epochs")
        self.batch_size = train.getint("batch_size")
        self.encoder_lr = train.getfloat("encoder_lr")
        # Pretrained encoder uses a lower LR (fine-tuning), randomly-initialised
        # decoder blocks use a higher LR — same rationale as Mask R-CNN's
        # backbone_lr / head_lr split.
        self.decoder_lr = train.getfloat("decoder_lr")
        self.momentum = train.getfloat("momentum")
        self.weight_decay = train.getfloat("weight_decay")
        self.gradient_clip = train.getfloat("gradient_clip")
        self.val_split = train.getfloat("val_split")
        self.test_split = train.getfloat("test_split")
        self.loss_function = train.get("loss_function")
        # Supported values: "bce" | "dice" | "bce_dice" | "focal_dice"
        self.bce_weight = train.getfloat("bce_weight")
        self.lr_scheduler = train.get("lr_scheduler")
        self.min_lr = train.getfloat("min_lr")
        self.lr_patience = train.getint("lr_patience")
        self.lr_factor = train.getfloat("lr_factor")
        self.num_workers = train.getint("num_workers")
        self.mask_threshold = train.getfloat("mask_threshold")
        self.score_threshold = train.getfloat("score_threshold")
        self.log_every = train.getint("log_every")
        self.val_every = train.getint("val_every")
        self.alpha = train.getfloat("alpha")
        self.beta = train.getfloat("beta")
        self.focal_gamma = train.getfloat("focal_gamma")
        self.focal_alpha = train.getfloat("focal_alpha")

        # ── SAMPLER ───────────────────────────────────────────────────
        sampler = config["SAMPLER"]
        self.weighted_sampler = sampler.getboolean("weighted_sampler")
        self.oversample_minority = sampler.getboolean("oversample_minority")

        # ── MODEL ─────────────────────────────────────────────────────
        model = config["MODEL"]
        self.backbone = model.get("backbone")
        self.pretrained = model.getboolean("pretrained")
        self.decoder_channels = tuple(
            int(c.strip()) for c in model.get("decoder_channels").split(",")
        )
        self.bilinear_upsample = model.getboolean("bilinear_upsample")

        # Validate decoder_channels length matches the encoder's skip-connection
        # count.  SMP determines depth from the encoder name; mismatches produce
        # a cryptic shape error mid-training rather than a clear message at startup.
        _ENCODER_DEPTHS = {
            "resnet34": 5,
            "resnet50": 5,
            "resnet101": 5,
            "mobilenet_v2": 5,
            "efficientnet-b0": 6,
            "efficientnet-b1": 6,
            "efficientnet-b2": 6,
            "efficientnet-b3": 6,
            "efficientnet-b4": 6,
            "efficientnet-b5": 6,
        }
        expected_depth = _ENCODER_DEPTHS.get(self.backbone)
        if expected_depth is not None and len(self.decoder_channels) != expected_depth:
            raise ValueError(
                f"decoder_channels has {len(self.decoder_channels)} values but "
                f"encoder '{self.backbone}' has {expected_depth} skip-connection "
                f"levels. Update decoder_channels in unet.ini to have exactly "
                f"{expected_depth} comma-separated values."
            )

        # ── SYSTEM ────────────────────────────────────────────────────
        system = config["SYSTEM"]
        self.device = system.get("device")
        self.seed = system.getint("seed")

        # ── LOGGING ───────────────────────────────────────────────────
        logging_ = config["LOGGING"]
        self.save_best_model = logging_.getboolean("save_best_model")
        self.save_every_epoch = logging_.getboolean("save_every_epoch")

        # ── PATHS ─────────────────────────────────────────────────────
        paths = config["PATHS"]
        self.images_dir = paths.get("images_dir")
        self.masks_dir = paths.get("masks_dir")
        self.json_dir = paths.get("json_dir")
        self.coco_dir = paths.get("coco_dir")
        self.csv_path = paths.get("csv_path")
        self.model_dir = paths.get("model_dir")
        self.output_dir = paths.get("output_dir")
        self.infer_output_dir = paths.get("infer_output_dir")
        self.checkpoint_dir = paths.get("checkpoint_dir")
        # Optional: path to a previously-trained checkpoint shipped as a
        # read-only Kaggle input dataset, used to resume training on a
        # fresh session when no local checkpoint exists yet. Safe to omit
        # from unet.ini -- defaults to None.
        self.pretrained_checkpoint = paths.get("pretrained_checkpoint", fallback=None)
        self.log_file = paths.get("log_file")
        self.zeroshot_dir = paths.get("zeroshot_output_dir")

        # ── DATASET ───────────────────────────────────────────────────
        dataset = config["DATASET"]
        self.dataset_val_split = dataset.getfloat("val_split")
        self.random_seed = dataset.getint("random_seed")
        self.min_area = dataset.getint("min_area")
        self.normal_dataset = dataset.get("normal_dataset")

        # ── VISUAL ────────────────────────────────────────────────────
        self.visualize_predictions = config.getboolean(
            "VISUAL", "visualize_predictions"
        )

        # ── STAGES ────────────────────────────────────────────────────
        _stages = config.get("STAGES", "stages")
        self.stages = [s.strip() for s in _stages.split(",")]