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
from models.model import CFGCNN
import torchvision.transforms.v2 as v2
from torchinfo import summary
from torchvision.models import efficientnet_v2_s
import PIL
import time
from pathlib import Path
from utils.checkpoint import save_to_onnx


pathtoonxx = "/Users/ethanelkayam/Downloads/RV1.1-1_SHELL_ONLY.onnx"

onnx_model = onnx.load(pathtoonxx)
onnx.checker.check_model(onnx_model, full_check=True)


