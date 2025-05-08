import torch
import logging
import yaml
import sys
import os
import torch.multiprocessing as mp
from functools import partial
from utils.data import RexailDataset
from utils.transforms import get_stages_image_transforms, collate_cutmix_or_mixup_transform
from utils.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_datasets
from utils.other import dirjoin, logp
from engine.trainer import trainer
from collections import Counter

TRAIN_DIR = "~/data"


train_dataset = RexailDataset(root=TRAIN_DIR,
                                        transform=None,
                                        pre_transform=None,
                                        class_decider=partial(RexailDataset.filter_by_min,
                                                              threshold=500),
                                        max_class_size=500,
                                        ratio=90,
                                        complement_ratio=False,
                                        storewise=True)

print("calculating")

mean, std = calculate_mean_std(train_dataset)
print(mean)
print(std)
