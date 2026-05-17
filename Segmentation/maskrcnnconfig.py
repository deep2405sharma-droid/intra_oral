class MaskRCNNConfig:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            if config is None:
                raise ValueError("config must be provided on first initialization")
            instance = super().__new__(cls)
            instance._load(config)
            cls._instance = instance
        return cls._instance

    def _load(self, config):
        print(f"Config: {config}")
        # AUGMENTATION
        aug = config["AUGMENTATION"]
        self.include_augmented = aug.getboolean("include_augmented")
        self.smart_aug_csv = aug.get("smart_aug_csv")
        self.smartom_aug_csv = aug.get("smartom_aug_csv")

        # TRAINING
        train = config["TRAINING"]
        self.num_classes = train.getint("num_classes")
        self.epochs = train.getint("epochs")
        self.batch_size = train.getint("batch_size")
        self.backbone_lr = train.getfloat("backbone_lr")
        self.head_lr = train.getfloat("head_lr")
        self.momentum = train.getfloat("momentum")
        self.weight_decay = train.getfloat("weight_decay")
        self.gradient_clip = train.getfloat("gradient_clip")
        self.val_split = train.getfloat("val_split")
        self.test_split = train.getfloat("test_split")
        # self.patience = train.getint("patience")
        self.pretrained_weights = train.get("pretrained_weights")
        self.num_workers = train.getint("num_workers")
        self.images_per_batch = train.getint("images_per_batch")
        self.min_lr = train.getfloat("min_lr")
        self.max_iter = train.getint("max_iter")
        # self.batch_per_image = train.getint("batch_per_image")
        # Note: score_threshold is defined twice in the .ini — using the last value (0.70)
        self.score_threshold = train.getfloat("score_threshold")
        self.checkpoint_period = train.getint("checkpoint_period")
        self.log_every = train.getint("log_every")
        self.val_every = train.getint("val_every")
        self.mask_threshold = train.getfloat("mask_threshold")

        # SAMPLER
        sampler = config["SAMPLER"]
        self.weighted_sampler = sampler.getboolean("weighted_sampler")
        self.oversample_minority = sampler.getboolean("oversample_minority")

        # MODEL
        model = config["MODEL"]
        self.backbone = model.get("backbone")
        self.pretrained = model.getboolean("pretrained")

        # SYSTEM
        system = config["SYSTEM"]
        self.device = system.get("device")
        self.seed = system.getint("seed")

        # LOGGING
        logging_ = config["LOGGING"]
        self.save_best_model = logging_.getboolean("save_best_model")
        self.save_every_epoch = logging_.getboolean("save_every_epoch")

        # PATHS
        paths = config["PATHS"]
        self.images_dir = paths.get("images_dir")
        self.json_dir = paths.get("json_dir")
        self.coco_dir = paths.get("coco_dir")
        self.model_dir = paths.get("model_dir")
        self.roi_output_dir = paths.get("roi_output_dir")
        self.infer_output_dir = paths.get("infer_output_dir")
        self.log_file = paths.get("log_file")
        self.csv_path = paths.get("csv.path")
        self.output_dir = paths.get("output_dir")
        self.zeroshot_dir = paths.get("zeroshot_output_dir")
        self.checkpoint_dir = paths.get("checkpoint_dir")

        # DATASET
        dataset = config["DATASET"]
        self.dataset_val_split = dataset.getfloat("val_split")
        self.random_seed = dataset.getint("random_seed")
        self.min_area = dataset.getint("min_area")
        self.normal_dataset = dataset.get("normal_dataset")

        # ROI
        roi = config["ROI"]
        self.roi_size = roi.getint("roi_size")
        self.roi_background = roi.get("roi_background")
        self.save_mask_overlay = roi.getboolean("save_mask_overlay")
        self.score_min_roi = roi.getfloat("score_min_roi")

        # VISUAL
        self.visualize_coco = config.get("VISUAL", "visualize_coco")

        # STAGES
        _stages = config.get("STAGES", "stages")
        self.stages = _stages.split(",")
