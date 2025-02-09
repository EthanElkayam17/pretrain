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
                         divide_crop_and_augment: bool = False):
    
    """Create custom transform based on each training stage in settings.yaml file.
    
    Args:
        settings_name: path to settings.yaml file.
        settings_dir: path to settings directory.
        mean: mean rgb value to normalize by.
        std: standard deviation to normalize by.
        divide_crop_and_augment: whether to divide each transform into a tuple of cropper transform and augmentation transform
    
    Returns: [batched_transform_stage_1, ... , batched_transform_stage_n]
    """
     
    SETTINGS_PATH = dirjoin(settings_dir,settings_name)
    transforms: List[v2.Transform] = []

    with open(SETTINGS_PATH, 'r') as settings_file:
        settings = (yaml.safe_load(settings_file))
    

    for idx, stage in enumerate(settings.get('training_stages')):
        cropper = v2.CenterCrop(size=stage.get('res')) if stage.get('centered') else v2.RandomResizedCrop(size=stage.get('res'))
        
        if divide_crop_and_augment:
            cropper_transform = v2.Compose([ 
                v2.Resize(stage.get('resize')),
                cropper,
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float16, scale=True)])
            ])

            augmentation_transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandAugment(magnitude=stage.get('RandAugment_magnitude')),
                v2.ToDtype(torch.float32,scale=True),
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
                    v2.ToDtype(torch.float32,scale=True),
                    v2.Normalize(mean=mean, std=std)
                ])
            )

    return transforms


def default_transform(resize: tuple = (224,224),
                   crop_size: tuple = (224,224), 
                   mean: list = [0,0,0], 
                   std: list = [1,1,1]):
     """Returns testing transform
     
     Args:
        resize: resize size before crop.
        crop_size: final output image-size.
        mean: mean rgb value to normalize by.
        std: standard deviation to normalize by.
    """
     
     res = v2.Compose([
                v2.Resize(size=resize),
                v2.CenterCrop(size=crop_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32,scale=True),
                v2.Normalize(mean=mean, std=std)
           ])
     return res
