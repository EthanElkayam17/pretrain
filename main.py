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
from utils.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_dirs
from utils.other import dirjoin
from engine.trainer import trainer

CLASSES_TO_IGNORE_IN_DEBUGGING = ["b0001",  "b0009",  "b0017",  "b0025",  "b0033",  "b0041",  "b0049", "b0057",  "b0065",  "b0073",  "b0081",  "b0089",  "b0097",  "b0105",  "b0113",  "b0121",  "b0129",  "b0137",  "b0145",  "b0153",  "b0161",  "b0169",  "b0177",  "b0185",  "b0193",
"b0002",  "b0010",  "b0018",  "b0026",  "b0034",  "b0042",  "b0050",  "b0058",  "b0066",  "b0074",  "b0082",  "b0090",  "b0098",  "b0106",  "b0114",  "b0122",  "b0130",  "b0138",  "b0146",  "b0154",  "b0162",  "b0170",  "b0178",  "b0186",  "b0194",
"b0003",  "b0011",  "b0019",  "b0027",  "b0035",  "b0043",  "b0051",  "b0059",  "b0067",  "b0075",  "b0083", "b0091",  "b0099",  "b0107",  "b0115",  "b0123",  "b0131",  "b0139",  "b0147",  "b0155",  "b0163",  "b0171",  "b0179",  "b0187",  "b0195",
"b0004",  "b0012",  "b0020", "b0028",  "b0036",  "b0044",  "b0052",  "b0060",  "b0068",  "b0076",  "b0084",  "b0092",  "b0100",  "b0108",  "b0116",  "b0124",  "b0132",  "b0140",  "b0148",  "b0156",  "b0164",  "b0172",  "b0180",  "b0188",  "b0196",
"b0005",  "b0013",  "b0021",  "b0029",  "b0037",  "b0045",  "b0053",  "b0061",  "b0069",  "b0077",  "b0085",  "b0093",  "b0101",  "b0109",  "b0117",  "b0125",  "b0133",  "b0141",  "b0149",  "b0157",  "b0165",  "b0173",  "b0181",  "b0189",  "b0197",
"b0006",  "b0014",  "b0022",  "b0030",  "b0038",  "b0046",  "b0054",  "b0062",  "b0070",  "b0078",  "b0086", "b0094",  "b0102",  "b0110",  "b0118",  "b0126",  "b0134"]

if __name__ == "__main__":
        
        if len(sys.argv) < 5:
                raise ValueError("Not enough arguments provided. \n required: MODEL_CONFIG_FILENAME TRAINING_CONFIG_FILENAME STAGES_CONFIG_FILENAME DESIRED_MODEL_NAME")

        TRAIN_DIR = "/workspace/train/"
        TEST_DIR = "/workspace/train/"

        STAGES_SETTINGS_DIR = "configs/training/stages"
        MODEL_CONFIG_DIR = "configs/architecture"
        TRAINING_CONFIG_DIR = "configs/training/general"
        STATE_DICTS_DIR = "engine/state_dicts"

        MODEL_CONFIG_NAME, TRAINING_SETTINGS_NAME, STAGES_SETTINGS_NAME, MODEL_NAME = sys.argv[1] , sys.argv[2], sys.argv[3], sys.argv[4]
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
        logger.info("------LOGGING: .(" + MODEL_NAME + "). ------")


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
                
                train_decider = partial(RexailDataset.sha256_modulo_split,ratio=70)
                test_decider = partial(RexailDataset.sha256_modulo_split,ratio=70, complement=True)
                
                if train_cfg.get('lazy_dataset', False):
                        create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_dirs,
                                                                 train_dir=TRAIN_DIR,
                                                                 test_dir=TEST_DIR,
                                                                 batch_size=train_cfg.get('batch_size'),
                                                                 num_workers=train_cfg.get('dataloader_num_workers'),
                                                                 train_transform=(transforms[idx])[1],
                                                                 train_pre_transform=(transforms[idx])[0],
                                                                 test_transform=(transforms[idx])[1],
                                                                 test_pre_transform=(transforms[idx])[0],
                                                                 train_decider=train_decider,
                                                                 test_decider=test_decider)

                else:
                    train_dataset = RexailDataset(root=TRAIN_DIR,
                                                transform=(transforms[idx])[1],
                                                pre_transform=(transforms[idx])[0],
                                                decider=train_decider,
                                                load_into_memory=True,
                                                num_workers=train_cfg.get('dataset_num_workers'))
                
                    test_dataset = RexailDataset(root=TEST_DIR,
                                                transform=(transforms[idx])[1],
                                                pre_transform=(transforms[idx])[0],
                                                decider=test_decider,
                                                load_into_memory=True,
                                                num_workers=train_cfg.get('dataset_num_workers'))
                
                    create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_shared_datasets,
                                                            train_dataset=train_dataset,
                                                            test_dataset=test_dataset,
                                                            batch_size=train_cfg.get('batch_size'),
                                                            num_workers=train_cfg.get('dataloader_num_workers'))

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
