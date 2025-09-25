from __future__ import annotations
import torch
import logging
import yaml
import sys
import os
import torch.multiprocessing as mp
import shutil
from pathlib import Path
from typing import Iterable, List, Optional
from functools import partial
from data.data import RexailDataset
from utils.transforms import get_stages_image_transforms, collate_cutmix_or_mixup_transform
from data.data import create_dataloaders_and_samplers_from_shared_datasets, calculate_mean_std, create_dataloaders_and_samplers_from_datasets
from utils.other import dirjoin, logp
from engine.trainer import trainer

a = set()
b = ["1","2","3"]


if a:
    print("nigga!!")
