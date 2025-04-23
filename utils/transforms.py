from typing import List
import os
import logging
import torch
import torchvision.transforms.v2 as v2
import yaml
from utils.other import dirjoin

def get_stages_image_transforms(settings_name: str,
                                settings_dir: str,
                                mean: list,
                                std: list,
                                dtype: torch.dtype = torch.float16,
                                divide_crop_and_augment: bool = False) -> List[v2.Transform]:
    
    """Create custom transform based on each training stage in settings.yaml file.
    
    Args:
        settings_name: path to settings.yaml file.
        settings_dir: path to settings directory.
        mean: mean rgb value to normalize by.
        std: standard deviation to normalize by.
        dtype: data type for resulting tensor. 
        divide_crop_and_augment: whether to divide each transform into a tuple of cropper transform and augmentation transform
    
    Returns:
        List[v2.Transform] - [batched_transform_stage_1, ... , batched_transform_stage_n]
    """
     
    SETTINGS_PATH = dirjoin(settings_dir,settings_name)
    transforms: List[v2.Transform] = []

    with open(SETTINGS_PATH, 'r') as settings_file:
        settings = (yaml.safe_load(settings_file))
    

    for idx, stage in enumerate(settings.get('training_stages')):
        cropper = v2.CenterCrop(size=(stage.get('res'),stage.get('res'))) if stage.get('centered') else v2.RandomResizedCrop(size=(stage.get('res'),stage.get('res')), antialias=True)
        

        if divide_crop_and_augment:
            cropper_transform = v2.Compose([ 
                v2.Resize(size=(stage.get('resize'),stage.get('resize')), antialias=True),
                v2.Compose([v2.ToImage(), v2.ToDtype(dtype=dtype, scale=True)])
            ])

            augmentation_transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                cropper,
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandAugment(magnitude=stage.get('RandAugment_magnitude')),
                v2.ToDtype(dtype=dtype,scale=True),
                v2.Normalize(mean=mean, std=std)
            ])
            
            transforms.append(tuple([cropper_transform,augmentation_transform]))
        else:
            transforms.append(
                v2.Compose([ 
                    v2.Resize(stage.get('resize')),
                    cropper,
                    v2.RandomHorizontalFlip(0.5),
                    v2.RandomVerticalFlip(0.5),
                    v2.RandAugment(magnitude=stage.get('RandAugment_magnitude')),
                    v2.ToImage(),
                    v2.ToDtype(dtype=dtype,scale=True),
                    v2.Normalize(mean=mean, std=std)
                ])
            )

    return transforms


def default_transform(resize: tuple = (224,224),
                    crop_size: tuple = (224,224), 
                    mean: list = None, 
                    std: list = None,
                    dtype: torch.dtype = torch.float16):
    """Returns testing transform
     
    Args:
        resize: resize size before crop.
        crop_size: final output image-size.
        mean: mean rgb value to normalize by.
        std: standard deviation to normalize by.
        dtype: data type for resulting tensor.
    
    Returns:
        Callable - default transform
    """

    if mean is None:
        mean = [0,0,0]
    
    if std is None:
        std = [1,1,1]
     
    res = v2.Compose([
                v2.Resize(size=resize),
                v2.CenterCrop(size=crop_size),
                v2.ToImage(),
                v2.ToDtype(dtype=dtype,scale=True),
                v2.Normalize(mean=mean, std=std)
           ])
    return res


def collate_cutmix_or_mixup_transform(numclasses: int,
                                      mixup_alpha: float,
                                      cutmix_alpha: float):
    """Apply cutmix or mixup on batches fetched from dataloader
    
    Args:
        cutmix_alpha: alpha coefficient for cutmix
        mixup_alpha: alpha coefficient for mixup
        numclasses: number of classes
    
    Returns:
        Callable - collate function with cutmix/mixup
    """
    
    mixup = v2.MixUp(num_classes=numclasses, alpha=mixup_alpha)
    cutmix = v2.CutMix(num_classes=numclasses, alpha=cutmix_alpha)

    func = v2.RandomChoice([cutmix,mixup])
    
    return func
