import os
from pathlib import Path
from typing import Union

def dirjoin(main_dir: Union[str,Path],
            other_dir: Union[str,Path]):
    """Joining paths with '/' instead of '\\'"""
    return (os.path.join(main_dir,other_dir)).replace("\\","/")