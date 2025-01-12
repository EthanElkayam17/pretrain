import torch
import os
from pathlib import Path
from utils.other import dirjoin

#REWRITE EVERYTHING HERE TO BE MORE COMPREHENSIVE

def save_state_dict(model: torch.nn.Module,
               dir: str,
               model_name: str):
  """Saves a model's state dict.

  Args:
    model: a model to save.
    dir: target directory.
    model_name: filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.
  """

  target_dir_path = Path(dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  torch.save(obj=model.state_dict(),
             f=model_save_path)


def save_to_onnx(model: torch.nn.Module,
              dir: str,
              input_res: tuple,
              model_name: str):
  """Saves model in onnx format
  
  Args:
    model: a model to save.
    dir: target directory.
    input_res: image resolution that the model expects.
    model_name: name for the saved file.
"""
  
  path = (dirjoin(dir,f"{model_name}.onnx"))
  model.eval()

  ex_input = torch.randn(1,3,input_res[0],input_res[1])
  
  torch.onnx.export(model, ex_input, path, verbose=True)
