import torch
import yaml
import argparse
import torch.multiprocessing as mp
from functools import partial
from data.data import RexailDataset
from utils.transforms import get_stages_pre_transforms, get_stages_transforms, collate_cutmix_or_mixup_transform, default_transform
from data.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_datasets
from utils.other import dirjoin, start_log, missing_keys, ConfigError
from engine.trainer import trainer


MODEL_CFG_DIR_PATH = "configs/architecture"
TRAINING_CFG_DIR_PATH = "configs/training/general"
STAGES_CFG_DIR_PATH = "configs/training/stages"
STATE_DICTS_DIR_PATH = "state_dicts"
LOGS_DIR_PATH = "logs"

TRAIN_CFG_REQ = ["train_dir", "test_dir", "batch_size", "dataloader_num_workers", "momentum", "weight_decay", "train_split"]
STAGES_CFG_REQ = ["resize", "res", "lr_min", "lr_max", "epochs"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train CNN on Rexail's data"
    )
    ap.add_argument("--model-cfg", type=str, required=True, help="File name of model architecture configuration (.yaml)")
    ap.add_argument("--training-cfg", type=str, required=True, help="File name of training settings configuration (.yaml)")
    ap.add_argument("--stages-cfg", type=str, required=True, help="File name of stages configuration (.yaml)")
    ap.add_argument("--version-name", type=str, default=None, help="Output's file name")
    ap.add_argument("--log-to", type=str, default="stdout-only", help="File name to log to (.txt / 'stdout-only')")
    args = ap.parse_args()
    WORLD_SIZE = torch.cuda.device_count()

    if args.version_name is None:
        VERSION_FNAME = f"{args.model_cfg}_{args.training_cfg}_{args.stages_cfg}.pth"
    else:
        VERSION_FNAME = f"{args.version_name}.pth"

    TRAINING_CFG_PATH = dirjoin(TRAINING_CFG_DIR_PATH, args.training_cfg)
    STAGES_CFG_PATH = dirjoin(STAGES_CFG_DIR_PATH, args.stages_cfg)

    log = start_log(LOGS_DIR_PATH, args.log_to)
    log(f"Logging - {VERSION_FNAME}. into - {args.log_to}")
    
    
    log("Setting up training env")
    with open(TRAINING_CFG_PATH, 'r') as tf:
        train_cfg = (yaml.safe_load(tf)).get('settings')[0]
    
    train_cfg_missing = missing_keys(train_cfg, TRAIN_CFG_REQ)
    if train_cfg_missing:
        log(f"{TRAINING_CFG_PATH} is missing the following required attributes: \n {train_cfg_missing}")
        raise ConfigError()
    
    TRAIN_DIR = train_cfg.get("train_dir")
    TEST_DIR = train_cfg.get("test_dir")
    HALF_PRECISION = train_cfg.get("half_precision", True)
    DTYPE = torch.float16 if HALF_PRECISION else torch.float32


    with open(STAGES_CFG_PATH, 'r') as sf:
        stages_cfg = (yaml.safe_load(sf)).get('stages')

    stages_cfg_missing = set()
    for stage in stages_cfg:
        stages_cfg_missing.update(missing_keys(stage, STAGES_CFG_REQ))
    if stages_cfg_missing:
        log(f"{STAGES_CFG_PATH} is missing the following required attributes: \n {stages_cfg_missing}")
        raise ConfigError()

    stage_pre_transforms = get_stages_pre_transforms(stages_cfg, DTYPE)

    pretrained = "slow.pth"
    log("Starting training:")
    for idx, stage in enumerate(stages_cfg):

        if stage.get('cutmix_alpha', 0.0) == 0.0 or stage.get('mixup_alpha', 0.0) == 0.0:
            external_collate_func_builder = None
        
        else:
            external_collate_func_builder=partial(collate_cutmix_or_mixup_transform,
                                                cutmix_alpha=stage.get('cutmix_alpha', 0.0),
                                                mixup_alpha=stage.get('mixup_alpha', 0.0))
        
        train_dataset = RexailDataset(root=TRAIN_DIR,
                                        transform=None,
                                        pre_transform=stage_pre_transforms[idx],
                                        max_class_size=train_cfg.get('max_class_size', -1),
                                        min_class_size=train_cfg.get('min_class_size', -1),
                                        earliest_timestamp_ms=train_cfg.get('earliest_timestamp_ms', 0),
                                        latest_timestamp_ms=train_cfg.get('latest_timestamp_ms', -1),
                                        ratio=train_cfg.get('train_split'),
                                        complement_ratio=False)

        test_dataset = RexailDataset(root=TEST_DIR,
                                        transform=None,
                                        pre_transform=stage_pre_transforms[idx],
                                        max_class_size=train_cfg.get('max_class_size', -1),
                                        min_class_size=train_cfg.get('min_class_size', -1),
                                        earliest_timestamp_ms=train_cfg.get('earliest_timestamp_ms', 0),
                                        latest_timestamp_ms=train_cfg.get('latest_timestamp_ms', -1),
                                        ratio=100,
                                        complement_ratio=False,
                                        force_classes=train_dataset.classes) #clean force classes input from ""user""

        print(len(train_dataset.classes))
        print(len(test_dataset.classes))

        mean, std = train_cfg.get('mean', None), train_cfg.get('std', None)
        if (mean is None) or (std is None):
            log("Calculating std and mean across training set..")
            mean, std = calculate_mean_std(train_dataset)
            log(f"mean and std calculated - mean: {mean}, std: {std}")

        
        if idx == 0:
            stage_transforms = get_stages_transforms(stages_cfg=stages_cfg,
                                                     mean=mean,
                                                     std=std,
                                                     dtype=DTYPE)
            test_transform = default_transform(mean, std)
            test_dataset.set_transform(test_transform)
        train_dataset.set_transform(stage_transforms[idx])


        if train_cfg.get('lazy_dataset', True):
            create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_datasets, 
                                                    train_dataset=train_dataset,
                                                    test_dataset=test_dataset,
                                                    batch_size=train_cfg.get('batch_size'),
                                                    num_workers=train_cfg.get('dataloader_num_workers'),
                                                    external_collate_func_builder=external_collate_func_builder)
        else:
            log("Loading datasets into memory..")
            train_dataset.load_into_memory(num_workers=train_cfg.get('dataset_num_workers'),
                                            dtype=DTYPE)
                
            test_dataset.load_into_memory(num_workers=train_cfg.get('dataset_num_workers'),
                                            dtype=DTYPE)
            log("Finished loading train/test datasets")

            create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_shared_datasets,
                                                        train_dataset=train_dataset,
                                                        test_dataset=test_dataset,
                                                        batch_size=train_cfg.get('batch_size'),
                                                        num_workers=train_cfg.get('dataloader_num_workers'),
                                                        external_collate_func_builder=external_collate_func_builder)

        log(f"Starting training stage #{str(idx)}")
        mp.spawn(
            trainer,
            args=(WORLD_SIZE, 
                args.model_cfg,
                create_dataloaders_per_process,
                train_cfg.get('momentum'),
                train_cfg.get('weight_decay'),
                stage.get('lr_min'),
                stage.get('lr_max'),
                stage.get('dropout_prob', 0),
                stage.get('warmup_epochs', 0),
                torch.optim.RMSprop,
                torch.nn.CrossEntropyLoss(label_smoothing=stage.get('label_smoothing', 0.0)),
                stage.get('epochs'),
                HALF_PRECISION,
                stage.get('decay_mode', None),
                stage.get('decay_factor', 0),
                VERSION_FNAME,
                args.log_to,
                LOGS_DIR_PATH,
                pretrained),
            nprocs=WORLD_SIZE,
            join=True
        )
        log(f"Finished training stage #{str(idx)} \n")
        pretrained = VERSION_FNAME
    
    log("FINISHED TRAINING")
