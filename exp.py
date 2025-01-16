import PIL.Image
import torch
import onnx
import onnxruntime
import logging
import yaml
import sys
import os
import torchvision
from utils.data import RexailDataset
from models.model import CFGCNN, InvertedResidualDWBlock
import torchvision.transforms.v2 as v2
from torchinfo import summary
from torchvision.models import efficientnet_v2_s
import PIL
import time
from pathlib import Path
from utils.checkpoint import save_to_onnx
from torchvision.ops import StochasticDepth


model = CFGCNN("RV1.1-1.yaml")
for name,module in model.named_modules():
        if isinstance(module, InvertedResidualDWBlock):
                print(module.stochastic_depth_prob)

