import torch
import logging
import yaml
import sys
import os
import torchvision
from utils.data import RexailDataset
from models.model import CFGCNN
from utils.transforms import get_stage_transforms, default_transform
from utils.data import create_dataloaders, calculate_mean_std
from engine.trainer import trainer
from torchinfo import summary
from torchvision.models import efficientnet_v2_s
import time

model = CFGCNN("efficientnetv2-s.yaml")

summary(model=model,input_size=(1,3,224,224),depth=5)
