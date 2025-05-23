import torch
import logging
import yaml
import sys
import os
import torch.multiprocessing as mp
from functools import partial
from data.data import RexailDataset
from utils.transforms import get_stages_image_transforms, collate_cutmix_or_mixup_transform
from data.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_datasets
from utils.other import dirjoin, logp
from engine.trainer import trainer

CLASSES_TO_IGNORE_IN_DEBUGGING = ["b0001",  "b0009",  "b0017",  "b0025",  "b0033",  "b0041",  "b0049", "b0057",  "b0065",  "b0073",  "b0081",  "b0089",  "b0097",  "b0105",  "b0113",  "b0121",  "b0129",  "b0137",  "b0145",  "b0153",  "b0161",  "b0169",  "b0177",  "b0185",  "b0193",
"b0002",  "b0010",  "b0018",  "b0026",  "b0034",  "b0042",  "b0050",  "b0058",  "b0066",  "b0074",  "b0082",  "b0090",  "b0098",  "b0106",  "b0114",  "b0122",  "b0130",  "b0138",  "b0146",  "b0154",  "b0162",  "b0170",  "b0178",  "b0186",  "b0194",
"b0003",  "b0011",  "b0019",  "b0027",  "b0035",  "b0043",  "b0051",  "b0059",  "b0067",  "b0075",  "b0083", "b0091",  "b0099",  "b0107",  "b0115",  "b0123",  "b0131",  "b0139",  "b0147",  "b0155",  "b0163",  "b0171",  "b0179",  "b0187",  "b0195",
"b0004",  "b0012",  "b0020", "b0028",  "b0036",  "b0044",  "b0052",  "b0060",  "b0068",  "b0076",  "b0084",  "b0092",  "b0100",  "b0108",  "b0116",  "b0124",  "b0132",  "b0140",  "b0148",  "b0156",  "b0164",  "b0172",  "b0180",  "b0188",  "b0196",
"b0005",  "b0013",  "b0021",  "b0029",  "b0037",  "b0045",  "b0053",  "b0061",  "b0069",  "b0077",  "b0085",  "b0093",  "b0101",  "b0109",  "b0117",  "b0125",  "b0133",  "b0141",  "b0149",  "b0157",  "b0165",  "b0173",  "b0181",  "b0189",  "b0197",
"b0206",  "b0214",  "b0022",  "b0030",  "b0038",  "b0046",  "b0054",  "b0062",  "b0070",  "b0078",  "b0086", "b0094",  "b0102",  "b0110",  "b0118",  "b0126",  "b0134", "b0001",  "b0009",  "b0017",  "b0025",  "b0033",  "b0041",  "b0049", "b0057",  "b0065",  "b0073",  "b0081",  "b0089",  "b0297",  "b0205",  "b0213",  "b0221",  "b0229",  "b0237",  "b0245",  "b0253",  "b0261",  "b0269",  "b0277",  "b0285",  "b0293",
"b0022",  "b0210",  "b0218",  "b0226",  "b0234",  "b0242",  "b0250",  "b0258",  "b0266",  "b0274",  "b0282",  "b0290",  "b0298",  "b0206",  "b0214",  "b0222",  "b0230",  "b0238",  "b0246",  "b0254",  "b0262",  "b0270",  "b0278",  "b0286",  "b0294",
"b0203",  "b0211",  "b0219",  "b0227",  "b0235",  "b0243",  "b0251",  "b0259",  "b0267",  "b0275",  "b0283", "b0291"]

if __name__ == "__main__":
        
    if len(sys.argv) < 5:
        raise ValueError("Not enough arguments provided. \n required: MODEL_CONFIG_FILENAME , TRAINING_CONFIG_FILENAME , STAGES_CONFIG_FILENAME , DESIRED_MODEL_NAME")

    TRAIN_DIR = "~/.cache/kagglehub/datasets/ethanelkayam/example2/versions/1/newdata"
    TEST_DIR = "~/.cache/kagglehub/datasets/ethanelkayam/example2/versions/1/newdata"

    STAGES_SETTINGS_DIR = "configs/training/stages"
    MODEL_CONFIG_DIR = "configs/architecture"
    TRAINING_CONFIG_DIR = "configs/training/general"
    STATE_DICTS_DIR = "state_dicts"

    MODEL_CONFIG_NAME, TRAINING_SETTINGS_NAME, STAGES_SETTINGS_NAME, MODEL_NAME = sys.argv[1] , sys.argv[2], sys.argv[3], sys.argv[4]
    assert MODEL_NAME.endswith(".pth") or MODEL_NAME.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    STAGES_SETTINGS_PATH = (dirjoin(STAGES_SETTINGS_DIR,STAGES_SETTINGS_NAME))
    TRAINING_SETTINGS_PATH = (dirjoin(TRAINING_CONFIG_DIR,TRAINING_SETTINGS_NAME))
    WORLD_SIZE = torch.cuda.device_count()
    SAVED_MODEL_PATH = None
    START_EPOCH = 1

    SAVED_MODEL_FNAME = input("If continuing from checkpoint, enter saved model's file name (else, leave null): ")
    if SAVED_MODEL_FNAME != "":
        SAVED_MODEL_PATH = dirjoin(STATE_DICTS_DIR,SAVED_MODEL_FNAME)

        START_EPOCH = input("How many epochs have this model been trained for under the specified settings (if settings accommodate for the loading, leave null): ")
        if START_EPOCH.isdigit():
            START_EPOCH = int(START_EPOCH) + 1
        else:
            START_EPOCH = 1


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

    log("---Model created---\n")

    log("Setting up training env")
    with open(TRAINING_SETTINGS_PATH, 'r') as training_settings_file:
        train_cfg = (yaml.safe_load(training_settings_file)).get('training_general')[0]
    
    HALF_PRECISION = train_cfg.get("half_precision", True)
    DTYPE = torch.float16 if HALF_PRECISION else torch.float32
    mean, std = train_cfg.get('ds_mean'), train_cfg.get('ds_std')
    

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
            
            if stage.get('cutmix_alpha', 0.0) == 0.0 or stage.get('mixup_alpha', 0.0) == 0.0:
                external_collate_func_builder = None
            
            else:
                external_collate_func_builder=partial(collate_cutmix_or_mixup_transform,
                                                    cutmix_alpha=stage.get('cutmix_alpha', 0.0),
                                                    mixup_alpha=stage.get('mixup_alpha', 0.0))

            train_dataset = RexailDataset(root=TRAIN_DIR,
                                        transform=(transforms[idx])[1],
                                        pre_transform=(transforms[idx])[0],
                                        class_decider=partial(RexailDataset.filter_by_min,
                                                              threshold=750),
                                        max_class_size=750,
                                        ratio=90,
                                        complement_ratio=False,
                                        storewise=False)
            
            test_dataset = RexailDataset(root=TEST_DIR,
                                        transform=(transforms[idx])[1],
                                        pre_transform=(transforms[idx])[0],
                                        class_decider=partial(RexailDataset.filter_by_min,
                                                              threshold=750),
                                        max_class_size=750,
                                        ratio=90,
                                        complement_ratio=True,
                                        storewise=False)
            
            if (mean is None) or (std is None):
                log("---Calculating std and mean across training set---")
                mean, std = calculate_mean_std(train_dataset)
                log(f"---mean and std calculated: mean : {mean}, std : {std} ---")
            
            if train_cfg.get('lazy_dataset', False):
                create_dataloaders_per_process = partial(create_dataloaders_and_samplers_from_datasets,
                                                        train_dataset=train_dataset,
                                                        test_dataset=test_dataset,
                                                        batch_size=train_cfg.get('batch_size'),
                                                        num_workers=train_cfg.get('dataloader_num_workers', 0),
                                                        external_collate_func_builder=external_collate_func_builder)

            else:

                train_dataset.load_into_memory(num_workers=train_cfg.get('dataset_num_workers', 0),
                                                dtype=DTYPE)
                
                test_dataset.load_into_memory(num_workers=train_cfg.get('dataset_num_workers', 0),
                                                dtype=DTYPE)
                                
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
                    stage.get('warmup_epochs', 0),
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
