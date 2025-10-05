from typing import List
import os
import logging
import torch
import torchvision.transforms.v2 as v2
import yaml
from utils.other import dirjoin

def get_stages_pre_transforms(stages_cfg: List[dict], dtype: torch.dtype) -> List[v2.Transform]:
    
    """Create custom pre_transforms based on each training stage.
    
    Args:
        stages_cfg: list of dictionaries containing stages configuration content.
        dtype: desired data type.
    
    Returns:
        List[v2.Transform] - [pre_transform_stage_1, ... , pre_transform_stage_n]
    """
     
    transforms: List[v2.Transform] = []

    for stage in stages_cfg:
        if stage.get('centered', False):
            cropper_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(stage.get('resize'), stage.get('resize')), antialias=True),
                v2.CenterCrop(size=(stage.get('res'), stage.get('res'))),
                v2.ToDtype(dtype=dtype, scale=True)
            ])
        else:
            cropper_transform = v2.Compose([ 
                v2.ToImage(),
                v2.Resize(size=(stage.get('resize'), stage.get('resize')), antialias=True),
                v2.ToDtype(dtype=dtype, scale=True)
            ])    
    
        transforms.append(cropper_transform)
    return transforms


def get_stages_transforms(stages_cfg: List[dict], mean: list, std: list, dtype: torch.dtype) -> List[v2.Transform]:
    
    """Create custom transforms based on each training stage.
    
    Args:
        stages_cfg: list of dictionaries containing stages configuration content.
        mean: mean value of each channel.
        std: standard deviation of each channel.
        dtype: desired dtype.

    Returns:
        List[v2.Transform] - [transform_stage_1, ... , transform_stage_n]
    """
     
    transforms: List[v2.Transform] = []

    for stage in stages_cfg:
        if stage.get('centered', False):
            augmentation_transform = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandAugment(magnitude=stage.get('randAugment_magnitude', 0)),
                v2.RandomHorizontalFlip(p=stage.get('horiz_flip_prob', 0)),
                v2.RandomVerticalFlip(p=stage.get('vert_flip_prob', 0)),
                v2.ToDtype(dtype=dtype, scale=True),
                v2.Normalize(mean=mean, std=std)
            ])
        else:
            augmentation_transform = v2.Compose([ 
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomResizedCrop(size=(stage.get('res'), stage.get('res'))),
                v2.RandAugment(magnitude=stage.get('randAugment_magnitude', 0)),
                v2.RandomHorizontalFlip(p=stage.get('horiz_flip_prob', 0)),
                v2.RandomVerticalFlip(p=stage.get('vert_flip_prob', 0)),
                v2.ToDtype(dtype=dtype, scale=True),
                v2.Normalize(mean=mean, std=std)
            ])
    
        transforms.append(augmentation_transform)
    return transforms


def default_transform(mean: list = None, 
                    std: list = None):
    """Returns testing transform
     
    Args:
        mean: mean rgb value to normalize by.
        std: standard deviation to normalize by.
    
    Returns:
        Callable - default transform
    """

    if mean is None:
        mean = [0,0,0]
    
    if std is None:
        std = [1,1,1]
     
    res = v2.Compose([
                v2.Normalize(mean=mean, std=std)
           ])
    return res


def collate_cutmix_or_mixup_transform(numclasses: int,
                                      mixup_alpha: float,
                                      cutmix_alpha: float):
    """Apply cutmix or mixup on batches fetched from dataloader
    
    Args:
        numclasses: number of classes
        cutmix_alpha: alpha coefficient for cutmix
        mixup_alpha: alpha coefficient for mixup
    
    Returns:
        Callable - collate function with cutmix/mixup
    """
    
    mixup = v2.MixUp(num_classes=numclasses, alpha=mixup_alpha)
    cutmix = v2.CutMix(num_classes=numclasses, alpha=cutmix_alpha)

    func = v2.RandomChoice([cutmix,mixup])
    
    return func
