import os
import torch
from pathlib import Path
from typing import Union
from logging import Logger

def dirjoin(main_dir: Union[str,Path],
            other_dir: Union[str,Path]):
    """Joining paths with '/' instead of '\\'"""
    
    return (os.path.join(main_dir,other_dir)).replace("\\","/")

def set_random_seed(seed):
    """set random seed everywhere"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def logp(logger: Logger, 
         str: str):
    """log and print"""

    assert isinstance(logger, Logger), "Can't log with non-Logger type"
    
    logger.info(str)
    print(str)
