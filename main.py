import torch
import logging
import yaml
import sys
import os
import torch.multiprocessing as mp
from functools import partial
from utils.data import RexailDataset
from models.model import CFGCNN
from utils.transforms import get_stage_transforms, default_transform
from utils.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, dirjoin
from engine.trainer import trainer

if __name__ == "__main__":

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
        WORLD_SIZE = torch.cuda.device_count()
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
                
                train_dataset = RexailDataset(root=TRAIN_DIR,
                                              transform=(transforms[idx])[1],
                                              pre_transform=(transforms[idx])[0],
                                              decider=partial(RexailDataset.sha256_modulo_split,ratio=70),
                                              load_into_memory=True,
                                              num_workers=train_cfg.get('num_workers'))
                
                test_dataset = RexailDataset(root=TEST_DIR,
                                              transform=(transforms[idx])[1],
                                              pre_transform=(transforms[idx])[0],
                                              decider=partial(RexailDataset.sha256_modulo_split,ratio=70,complement=True),
                                              load_into_memory=True,
                                              num_workers=train_cfg.get('num_workers'))

                create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_shared_datasets,
                                                         train_dataset=train_dataset,
                                                         test_dataset=test_dataset,
                                                         batch_size=train_cfg.get('batch_size'),
                                                         num_workers=train_cfg.get('num_workers'))

                logger.info(f"Starting training stage #{str(idx)}")
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
                          torch.nn.CrossEntropyLoss(),
                          stage.get('epochs'),
                          stage.get('decay_mode'),
                          stage.get('decay_factor', 0),
                          START_EPOCH,
                          MODEL_NAME,
                          SAVED_MODEL_PATH,
                          logger),
                    nprocs=WORLD_SIZE,
                    join=True
                )
                
                START_EPOCH = max((START_EPOCH - stage.get('epochs')), 1)
                logger.info(f"Finished training stage #{str(idx)} \n")

        logger.info("---FINISHED TRAINING---\n")
