import torch
import logging
import yaml
import sys
import os
import torch.multiprocessing as mp
from functools import partial
from utils.data import RexailDataset
from models.model import CFGCNN
from utils.transforms import get_stages_image_transforms, collate_cutmix_or_mixup_transform
from utils.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_dirs, create_dataloaders_and_samplers_from_datasets
from utils.other import dirjoin, logp
from engine.trainer import trainer


if __name__ == "__main__":
        
        if len(sys.argv) < 5:
                raise ValueError("Not enough arguments provided. \n required: MODEL_CONFIG_FILENAME , TRAINING_CONFIG_FILENAME , STAGES_CONFIG_FILENAME , DESIRED_MODEL_NAME")

        TRAIN_DIR = "~/.cache/kagglehub/datasets/ethanelkayam/pretraindata/versions/1/Data/CLS-LOC/train"
        TEST_DIR = "~/.cache/kagglehub/datasets/ethanelkayam/pretraindata/versions/1/Data/CLS-LOC/train"

        STAGES_SETTINGS_DIR = "configs/training/stages"
        MODEL_CONFIG_DIR = "configs/architecture"
        TRAINING_CONFIG_DIR = "configs/training/general"
        STATE_DICTS_DIR = "state_dicts"

        MODEL_CONFIG_NAME, TRAINING_SETTINGS_NAME, STAGES_SETTINGS_NAME, MODEL_NAME = sys.argv[1] , sys.argv[2], sys.argv[3], sys.argv[4]
        assert MODEL_NAME.endswith(".pth") or MODEL_NAME.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        
        STAGES_SETTINGS_PATH = (dirjoin(STAGES_SETTINGS_DIR,STAGES_SETTINGS_NAME))
        TRAINING_SETTINGS_PATH = (dirjoin(TRAINING_CONFIG_DIR,TRAINING_SETTINGS_NAME))
        START_EPOCH = 1
        WORLD_SIZE = torch.cuda.device_count()
        SAVED_MODEL_PATH = None

        if not os.path.exists("logs/"): 
                os.makedirs("logs/")
        logpath = (f"logs/{sys.argv[5]}" if len(sys.argv) >= 6 else "logs/log.txt")

        logging.basicConfig(filename=logpath,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
        logger = logging.getLogger('log')
        log = partial(logp, logger=logger)
        
        log("------LOGGING: .(" + MODEL_NAME + "). ------")


        SAVED_MODEL_FNAME = input("If continuing from checkpoint, enter saved model's file name (else, leave null): ")
        if SAVED_MODEL_FNAME != "":
                SAVED_MODEL_PATH = dirjoin(STATE_DICTS_DIR,SAVED_MODEL_FNAME)

                START_EPOCH = input("How many epochs have this model been trained for under the specified settings (if settings accommodate for the loading, leave null): ")
                if START_EPOCH.isdigit():
                        START_EPOCH = int(START_EPOCH) + 1
                else:
                        START_EPOCH = 0

        log("---Model created---\n")

        log("Setting up training env")
        with open(TRAINING_SETTINGS_PATH, 'r') as training_settings_file:
                train_cfg = (yaml.safe_load(training_settings_file)).get('training_general')[0]
                HALF_PRECISION = train_cfg.get("half_precision", True)
                DTYPE = torch.float16 if HALF_PRECISION else torch.float32

        log("---Calculating std and mean across training set---")
        mean, std = train_cfg.get('ds_mean'), train_cfg.get('ds_std')

        if (mean is None) or (std is None):
                mean, std = calculate_mean_std(TRAIN_DIR)

        log(f"---mean and std calculated: mean : {mean}, std : {std} ---")

        log("---Creating stage transforms---")
        transforms = get_stages_image_transforms(settings_name=STAGES_SETTINGS_NAME, 
                                                 settings_dir=STAGES_SETTINGS_DIR, 
                                                 mean=mean, 
                                                 std=std, 
                                                 dtype=DTYPE, 
                                                 divide_crop_and_augment=True)
        log("---Stage transform created---\n")


        log("---STARTING TRAINING---")
        with open(STAGES_SETTINGS_PATH, 'r') as stages_settings_file:
                stages_cfg = (yaml.safe_load(stages_settings_file))


        for idx, stage in enumerate(stages_cfg.get('training_stages')):                
                
                if START_EPOCH < stage.get('epochs'):       
                    train_decider = partial(RexailDataset.sha256_modulo_split,ratio=75)
                    test_decider = partial(RexailDataset.sha256_modulo_split,ratio=75, complement=True)
                    
                    if stage.get('cutmix_alpha', 0.0) == 0.0 or stage.get('mixup_alpha', 0.0) == 0.0:
                            external_collate_func_builder = None
                    
                    else:
                        external_collate_func_builder=partial(collate_cutmix_or_mixup_transform,
                                                            cutmix_alpha=stage.get('cutmix_alpha', 0.0),
                                                            mixup_alpha=stage.get('mixup_alpha', 0.0))

                    if train_cfg.get('lazy_dataset', False):
                            train_dataset = RexailDataset(root=TRAIN_DIR,
                                                    transform=(transforms[idx])[1],
                                                    pre_transform=(transforms[idx])[0],
                                                    decider=train_decider,
                                                    load_into_memory=False,
                                                    dtype=DTYPE,
                                                    num_workers=0)
                    
                            test_dataset = RexailDataset(root=TEST_DIR,
                                                    transform=(transforms[idx])[1],
                                                    pre_transform=(transforms[idx])[0],
                                                    decider=test_decider,
                                                    load_into_memory=False,
                                                    dtype=DTYPE,
                                                    num_workers=0)
                            
                            create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_datasets,
                                                                train_dataset=train_dataset,
                                                                test_dataset=test_dataset,
                                                                batch_size=train_cfg.get('batch_size'),
                                                                num_workers=train_cfg.get('dataloader_num_workers', 0),
                                                                external_collate_func_builder=external_collate_func_builder)

                    else:
                        train_dataset = RexailDataset(root=TRAIN_DIR,
                                                    transform=(transforms[idx])[1],
                                                    pre_transform=(transforms[idx])[0],
                                                    decider=train_decider,
                                                    load_into_memory=True,
                                                    dtype=DTYPE,
                                                    num_workers=train_cfg.get('dataset_num_workers', 0))
                    
                        test_dataset = RexailDataset(root=TEST_DIR,
                                                    transform=(transforms[idx])[1],
                                                    pre_transform=(transforms[idx])[0],
                                                    decider=test_decider,
                                                    load_into_memory=True,
                                                    dtype=DTYPE,
                                                    num_workers=train_cfg.get('dataset_num_workers', 0))
                                        
                        create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_shared_datasets,
                                                                train_dataset=train_dataset,
                                                                test_dataset=test_dataset,
                                                                batch_size=train_cfg.get('batch_size'),
                                                                num_workers=train_cfg.get('dataloader_num_workers', 0),
                                                                external_collate_func_builder=external_collate_func_builder)

                    log(f"Starting training stage #{str(idx)}")
                    mp.spawn(
                        trainer,
                        args=(WORLD_SIZE, 
                            MODEL_CONFIG_NAME,
                            create_dataloaders_per_process,
                            train_cfg.get('momentum'),
                            train_cfg.get('weight_decay'),
                            stage.get('lr_min'),
                            stage.get('lr_max'),
                            stage.get('dropout_prob'),
                            stage.get('warmup_epochs'),
                            torch.optim.RMSprop,
                            torch.nn.CrossEntropyLoss(label_smoothing=stage.get('label_smoothing', 0.0)),
                            stage.get('epochs'),
                            HALF_PRECISION,
                            stage.get('decay_mode'),
                            stage.get('decay_factor', 0),
                            START_EPOCH,
                            MODEL_NAME,
                            SAVED_MODEL_PATH,
                            logpath),
                        nprocs=WORLD_SIZE,
                        join=True
                    )
                    
                    SAVED_MODEL_PATH = f"state_dicts/{MODEL_NAME}"

                    if (not train_cfg.get('lazy_dataset', False)):                       
                        del train_dataset
                        del test_dataset

                log(f"Finished training stage #{str(idx)} \n")
                START_EPOCH = max((START_EPOCH - stage.get('epochs')), 1)

        log("---FINISHED TRAINING---\n")
