import torch
import numpy as np
import os
import copy
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.sampler import Sampler
from functools import partial
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler, default_collate
from utils.transforms import default_transform
from utils.other import dirjoin
from torch import Tensor
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

t = RexailDataset(root="~/newdata2", min_class_size=800)
print((t))


