"""
resnet_config.py
-----------------
Singleton configuration wrapper for ResNet50 classification training.

Mirrors UNetConfig exactly:
  - Singleton via __new__ so the config is parsed only once per process.
  - _load() maps every .ini key to a typed Python attribute.
  - Callers import ResNetConfig and pass the ConfigParser object on first use;
    subsequent calls return the cached instance.

Usage
-----
    from utils.load_configuration import load_config
    from src.classification.resnet.resnet_config import ResNetConfig

    base_cfg   = load_config()                                        # config.ini
    resnet_ini = load_config(base_cfg.get("CLASSIFICATION-RESNET", "resnet.config"))
    cfg        = ResNetConfig(resnet_ini)
    print(cfg.backbone_lr, cfg.num_classes)

    # Subsequent import anywhere in the process — same instance, no re-parse:
    cfg2 = ResNetConfig(None)
    assert cfg is cfg2
"""


class ResNetConfig:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            if config is None:
                raise ValueError("config must be provided on first initialisation")
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
        self.num_classes    = train.getint("num_classes")
        self.epochs         = train.getint("epochs")
        self.batch_size     = train.getint("batch_size")
        # Pretrained backbone: lower LR — fine-tuning only.
        # Same rationale as encoder_lr / decoder_lr split in U-Net.
        self.backbone_lr    = train.getfloat("backbone_lr")
        # Freshly initialised FC head: higher LR — learning from scratch.
        self.head_lr        = train.getfloat("head_lr")
        self.momentum       = train.getfloat("momentum")
        self.weight_decay   = train.getfloat("weight_decay")
        self.gradient_clip  = train.getfloat("gradient_clip")
        self.val_split      = train.getfloat("val_split")
        self.test_split     = train.getfloat("test_split")
        self.lr_scheduler   = train.get("lr_scheduler")
        self.min_lr         = train.getfloat("min_lr")
        self.lr_patience    = train.getint("lr_patience")
        self.lr_factor      = train.getfloat("lr_factor")
        self.num_workers    = train.getint("num_workers")
        self.log_every      = train.getint("log_every")
        self.val_every      = train.getint("val_every")
        self.score_threshold = train.getfloat("score_threshold")

        # ── SAMPLER ───────────────────────────────────────────────────
        sampler = config["SAMPLER"]
        self.weighted_sampler    = sampler.getboolean("weighted_sampler")
        self.oversample_minority = sampler.getboolean("oversample_minority")

        # ── MODEL ─────────────────────────────────────────────────────
        model = config["MODEL"]
        self.backbone   = model.get("backbone")
        self.pretrained = model.getboolean("pretrained")
        self.dropout    = model.getfloat("dropout")
        inp             = model.get("input_size").split(",")
        self.input_size = (int(inp[0]), int(inp[1]))

        # ── SYSTEM ────────────────────────────────────────────────────
        system = config["SYSTEM"]
        self.device = system.get("device")
        self.seed   = system.getint("seed")

        # ── LOGGING ───────────────────────────────────────────────────
        logging_ = config["LOGGING"]
        self.save_best_model  = logging_.getboolean("save_best_model")
        self.save_every_epoch = logging_.getboolean("save_every_epoch")

        # ── PATHS ─────────────────────────────────────────────────────
        paths = config["PATHS"]
        self.model_dir        = paths.get("model_dir")
        self.output_dir       = paths.get("output_dir")
        self.infer_output_dir = paths.get("infer_output_dir")
        self.checkpoint_dir   = paths.get("checkpoint_dir")
        self.log_file         = paths.get("log_file")

        # ── DATASET ───────────────────────────────────────────────────
        dataset = config["DATASET"]
        self.dataset_val_split = dataset.getfloat("val_split")
        self.random_seed       = dataset.getint("random_seed")
        self.normal_dataset    = dataset.get("normal_dataset")

        # ── VISUAL ────────────────────────────────────────────────────
        self.visualize_predictions = config.getboolean(
            "VISUAL", "visualize_predictions"
        )

        # ── STAGES ────────────────────────────────────────────────────
        _stages     = config.get("STAGES", "stages")
        self.stages = [s.strip() for s in _stages.split(",")]
