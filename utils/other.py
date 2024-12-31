import os
import torch
from pathlib import Path
from typing import Union

def dirjoin(main_dir: Union[str,Path],
            other_dir: Union[str,Path]):
    """Joining paths with '/' instead of '\\'"""
    return (os.path.join(main_dir,other_dir)).replace("\\","/")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
