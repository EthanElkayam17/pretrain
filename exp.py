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
from collections import Counter
import pandas as pd
import shutil

"""ORIGINAL_DIR = '~/data'
ORIGINAL_DIR = os.path.expanduser(ORIGINAL_DIR)

NEW_DIR = '~/newdata'
NEW_DIR = os.path.expanduser(NEW_DIR)

df = pd.read_csv('stores_products.csv', index_col=0)

classes, _ = RexailDataset.find_classes(directory=ORIGINAL_DIR,
                                     storewise=True)


lenned = len(classes)
for idx, store_product in enumerate(classes):
    full_path = dirjoin(ORIGINAL_DIR, store_product)
    
    store_product_id = int(store_product.split('/')[1])
    skeleton_id = df.loc[store_product_id].values[0]
    new_subdir_path = dirjoin(NEW_DIR,str(skeleton_id))

    if not os.path.isdir(new_subdir_path):
        os.mkdir(new_subdir_path)

    files = os.listdir(full_path)

    for fname in files:
        shutil.copy2(dirjoin(full_path,fname), new_subdir_path)
    
    print(f"copied {idx}/{lenned}")"""

DIR = '~/newdata'

new_dataset = RexailDataset(root=DIR,
                            class_decider=partial(RexailDataset.filter_by_min,
                                                    threshold=750),
                            storewise=False,
                            max_class_size=750,
                            ratio=90,
                            complement_ratio=False)

print(f"new: {len(new_dataset)}")



