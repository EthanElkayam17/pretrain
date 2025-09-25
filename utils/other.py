import os
import torch
import logging
from pathlib import Path
from typing import Union, Callable, Any, Hashable, List
from logging import Logger
from functools import partial
from collections.abc import Mapping, Iterable

def dirjoin(main_dir: Union[str,Path], other_dir: Union[str,Path]):
    """Joining paths with '/' instead of '\\'"""
    
    return (os.path.join(main_dir,other_dir)).replace("\\","/")


def set_random_seed(seed):
    """set random seed everywhere"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logp(str: str, logger: Logger):
    """log and print"""

    assert isinstance(logger, Logger), "Can't log with non-Logger type"
    
    logger.info(str)
    print(str)


def start_log(log_path: str, log_fname: str) -> Callable[[str], None]:
    """Set up logging function"""

    if log_fname == "stdout-only":
        return print
    
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    logpath = dirjoin(log_path, log_fname)

    logging.basicConfig(filename=logpath,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    logger = logging.getLogger('log')
    log = partial(logp, logger=logger)

    return log
    

def missing_keys(d: Mapping[Hashable, Any], required: Iterable[Hashable]) -> List[Hashable]:
    """ Return a list of keys from `required` that are not/ present in dict-like d.
    Order of `required` is preserved in the output. """
    return [k for k in required if k not in d]


class ConfigError(Exception):
    pass
