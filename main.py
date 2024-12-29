import torch
import logging
import yaml
import sys
import os
from functools import partial
from utils.data import RexailDataset
from models.model import CFGCNN
from utils.transforms import get_stage_transforms, default_transform
from utils.data import create_dataloaders_and_samplers, calculate_mean_std, dirjoin
from engine.trainer import trainer

if __name__ == "__main__":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        TRAIN_DIR = "/workspace/dataset/train/"
        TEST_DIR = "/workspace/dataset/train/"

        STAGES_SETTINGS_DIR = "configs/training/stages"
        MODEL_CONFIG_DIR = "configs/architecture"
        TRAINING_CONFIG_DIR = "configs/training/general"
        STATE_DICTS_DIR = "engine/state_dicts"

        MODEL_CONFIG_NAME , STAGES_SETTINGS_NAME, TRAINING_SETTINGS_NAME, MODEL_NAME = sys.argv[1] , sys.argv[2], sys.argv[3], sys.argv[4]
        STAGES_SETTINGS_PATH = (dirjoin(STAGES_SETTINGS_DIR,STAGES_SETTINGS_NAME))
        TRAINING_SETTINGS_PATH = (dirjoin(TRAINING_CONFIG_DIR,TRAINING_SETTINGS_NAME))
        START_EPOCH = 1
        SAVED_MODEL_PATH = None

        if not os.path.exists("logs/"): 
                os.makedirs("logs/")
        logname = (f"logs/{sys.argv[5]}" if len(sys.argv) >= 6 else "logs/log.txt")

        logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
        logger = logging.getLogger('log')
        logger.info("------LOGGING: .(" + MODEL_CONFIG_NAME + "," + TRAINING_SETTINGS_NAME + "," + STAGES_SETTINGS_NAME + "). ------")


        SAVED_MODEL_FNAME = input("If continuing from checkpoint, enter saved model's file name (else, leave null): ")
        if SAVED_MODEL_FNAME != "":
                SAVED_MODEL_PATH = dirjoin(STATE_DICTS_DIR,SAVED_MODEL_FNAME)

                START_EPOCH = input("How many epochs have this model been trained for under the specified settings (if settings accommodate for the loading, leave null): ")
                if START_EPOCH.isdigit():
                        START_EPOCH = int(START_EPOCH) + 1
                else:
                        START_EPOCH = 1

        logger.info("---Model created---\n")

        logger.info("Setting up training env")
        with open(TRAINING_SETTINGS_PATH, 'r') as training_settings_file:
                train_cfg = (yaml.safe_load(training_settings_file)).get('training_general')[0]

        logger.info("---Calculating std and mean across training set---")
        mean, std = train_cfg.get('ds_mean'), train_cfg.get('ds_std')

        if (mean is None) or (std is None):
                mean, std = calculate_mean_std(TRAIN_DIR)

        logger.info(f"---mean and std calculated: mean : {mean}, std : {std} ---")

        logger.info("---Creating stage transforms---")
        transforms = get_stage_transforms(STAGES_SETTINGS_NAME, STAGES_SETTINGS_DIR, mean, std, True, logger)
        logger.info("---Stage transform created---\n")


        logger.info("---STARTING TRAINING---")
        with open(STAGES_SETTINGS_PATH, 'r') as stages_settings_file:
                stages_cfg = (yaml.safe_load(stages_settings_file))


        for idx, stage in enumerate(stages_cfg.get('training_stages')):
                create_dataloaders_per_process = partial(create_dataloaders_and_samplers, train_dir=TRAIN_DIR,
                                                                        test_dir=TEST_DIR,
                                                                        batch_size=train_cfg.get('batch_size'),
                                                                        num_workers=train_cfg.get('num_workers'),
                                                                        train_transform=(transforms[idx])[1],
                                                                        train_pre_transform=(transforms[idx])[0],
                                                                        train_decider=partial(RexailDataset.sha256_modulo_split,ratio=70),
                                                                        test_decider=partial(RexailDataset.sha256_modulo_split,ratio=70,complement=True),
                                                                        test_transform=(transforms[idx])[1],
                                                                        test_pre_transform=(transforms[idx])[0],
                                                                        load_into_memory=True)

                logger.info(f"Starting training stage #{str(idx)}")
                trainer(model=model,
                        create_dataloaders_and_samplers=create_dataloaders_per_process,
                        optimizer=optimizer,
                        lr_min=stage.get('lr_min',0),
                        lr_max=stage.get('lr_max'),
                        warmup_epochs=stage.get('warmup_epochs'),
                        loss_fn=loss_fn,
                        epochs=stage.get('epochs'),
                        device=device,
                        save_freq=10,
                        decay_mode=stage.get('decay_mode',"none"),
                        exp_decay_factor=stage.get('decay_factor',0),
                        curr_epoch=START_EPOCH,
                        model_name=(f"{MODEL_NAME}_{idx}"),
                        logger=logger)
                
                START_EPOCH = max((START_EPOCH - stage.get('epochs')), 1)
                logger.info(f"Finished training stage #{str(idx)} \n")

        logger.info("---FINISHED TRAINING---\n")
