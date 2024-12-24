import torch
import numpy as np
import os
from hashlib import sha256
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.transforms import default_transform
from utils.other import dirjoin

def default_decider(path: str) -> bool:
        """Default decider"""
        return True

class RexailDataset(datasets.VisionDataset):
    """A dataset class for loading and partitioning data from rexail's dataset,
    expects directory in one of the following structures: 
        root/{product_id}/{store_id}/{image_name}.png
        /
        root/{product_id}/{image_name}.png
    
    the flag 'storewise' can be true only when the first form is present,
    the flag 'weighed' can be true when including weight data in {image_name}
    such that {image_name} is in the form: '{image_id}${weight}$' 
    
    Data should be partitioned into train and test (if not partitioned already in the directory itself)
    by using the 'decider' parameter"""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        decider: Callable[[str], bool] = default_decider,
        extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        ignore_classes: List[str] = list(),
        storewise: bool = False,
        weighed: bool = False,
    ):
        """Args:
            root: Root directory path.
            transform: A transform for the PIL images.
            target_transform: A transform for the target.
            loader: A function to load an image given its path.
            decider: A function that takes path of an Image file and decides if its in the dataset.
            extensions: file extensions that are acceptable.
            ignore_classes: classes to ignore when making dataset.
            storewise: whether the directory is partitioned store-wise inside each class.
            weighed: whether the file names contain weight data."""
        
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )
        self.ignore_classes = ignore_classes
        classes, class_to_idx = RexailDataset.find_classes(self.root, self.ignore_classes)
        self.class_to_idx = class_to_idx

        self.storewise = storewise
        self.weighed = weighed
        self.decider = decider
        self.loader = loader
        self.extensions = extensions

        self.classes = classes

        self.samples = self.make_dataset()
        self.targets = [s[1] for s in self.samples]

        self.loaded = False
        self.whole = self._load_everything()
        self.loaded = True


    def __len__(self) -> int:
        """Returns the amount of items in the dataset"""

        return len(self.samples)


    def __getitem__(self, index: int) -> Tuple:
        """Returns item based on index, in a tuple of the form: (sample, target, weight, sID) 
            where:
                sample is the tensor representing the image.
                target is class_index of the target class.
                weight is the weight data of the product.
                sID is the id of the store where the image is from.
        
        Args:
            index: Index """
        
        if self.loaded:
            return self.whole[index]

        path, target = self.samples[index]
        sample = self.loader(path)
        res = []
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        res.extend([sample,target])

        if self.weighed:
            fname = path.split("/")[-1]
            weight = float(fname.split("$")[1])
            res.append(weight)
        if self.storewise:
            sID = path.split("/")[-2]
            res.append(sID)

        return tuple(res)


    def _load_everything(self):
        whole = []
        for index in range(len(self.samples)):
            whole.append(self.__getitem__(index))
        return whole


    @staticmethod
    def find_classes(directory: Union[str, Path], 
                    ignore_classes: List[str] = []) -> Tuple[List[str], Dict[str, int]]:
        """Creates (list_of_classes,map_class_to_idx)
        
        Args:
            directory: directory containing the classes
            ignore_classes: classes to ignore"""
        
        classes = sorted(entry.name for entry in os.scandir(directory) if (entry.is_dir() and (entry.name not in ignore_classes)))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def is_valid_file(self, path: str) -> bool:
            """Whether or not a file can be in the dataset
            
            Args:
                path: path to file"""
            
            return ((self.extensions is None) or (path.lower().endswith(self.extensions))) and self.decider(path)


    def make_dataset(self) -> List[Tuple[str, int]]:
        """Makes a list of pairs (sample,target) of the dataset"""

        directory = os.path.expanduser(self.root)
        
        res = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[target_class]
            target_dir = dirjoin(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = dirjoin(root, fname)
                    if self.is_valid_file(path):
                        item = path, class_idx
                        res.append(item)
        
        return res


    @staticmethod
    def sha256_modulo_split(path: str, ratio: int, complement: bool = False) -> bool:
        """A decider function that partitions the data according to the remainder of an hashed image_id
        
        Args:
            path: path to file
            ratio: ~percentage of samples to be included in the dataset
            complement: whether to produce the complement set of data"""

        if ratio > 100 or ratio < 0:
            raise ValueError("ratio needs to be between 0 and 100")

        fname = path.split('/')[-1]

        image_id = fname.split("$")[0]
        image_id_bytes = image_id.encode('utf-8')
    
        hashed_id = int(sha256(image_id_bytes).hexdigest(), 16)

        return (hashed_id%100 < ratio) if (not complement) else (not (hashed_id%100 < ratio))


    @staticmethod
    def filter_by_store(path:str, stores_lst: List[str]) -> bool:
        """A decider function that accepts data only from stores in {stores_lst}
        
        Args:
            path: path to file
            stores_lst: List of acceptable stores"""
        
        sID = path.split("/")[-2]
        return sID in stores_lst


def create_dataloaders(
        train_dir: Union[str, Path],
        test_dir: Union[str, Path],
        batch_size: int,
        num_workers: int = 0,        
        train_transform: Union[torch.nn.Sequential, transforms.Compose] = default_transform(),
        test_transform: Union[torch.nn.Sequential, transforms.Compose] = default_transform(),
        train_extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        train_decider: Callable[[str], bool] = (lambda x: True),
        test_extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        test_decider: Callable[[str], bool] = (lambda x: True),
        ignore_classes: List[str] = list(),
        storewise: bool = False,
        weighed: bool = False
    ) -> Tuple[DataLoader, DataLoader, List]:
    """Creates training/testing dataloaders from training/testing directories
    
    Args:
        train_dir: path of training data directory.
        test_dir: path of testing data directory.
        batch_size: amount per batch.
        train_transform: transforms to apply to training data.
        test_transform: transforms to apply to testing data.
        train_decider: A function that takes path of an Image file and decides if its in the training dataset.
        train_extensions: file extensions that are acceptable in the training dataset.
        test_decider: A function that takes path of an Image file and decides if its in the testing dataset.
        test_extensions: file extensions that are acceptable in the testing dataset.
        ignore_classes: classes to ignore when making training/testing sets.
        storewise: whether both training and testing directories are partitioned store-wise inside each class.
        weighed: whether the file names contain weight data.

    Returns: (train_dataloader, test_dataloader, class_names)
    """

    train_data = RexailDataset(root=train_dir,
                               transform=train_transform,
                               decider=train_decider,
                               extensions=train_extensions,
                               ignore_classes=ignore_classes,
                               storewise=storewise,
                               weighed=weighed)
    
    test_data = RexailDataset(root=test_dir, 
                              transform=test_transform,
                              decider=test_decider,
                              extensions=test_extensions,
                              ignore_classes=ignore_classes,
                              storewise=storewise,
                              weighed=weighed)
    
    class_names = train_data.classes
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        
        # Try to avoid costs of transfer between pageable and pinned memory
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names


def calculate_mean_std(dir: Union[str, Path]) -> Tuple[List, List]:
    """Calculates mean and standard deviation of each channel across a directory
    
    Args:
        dir: path to directory
    
    returns: mean,std"""

    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    total_pixels = 0
    
    for root, _, files in os.walk(dir):
        for image_name in files:
            image_path = dirjoin(root, image_name)
            try:
                image = Image.open(image_path).convert('RGB') 
                image_np = np.array(image) / 255.0 

                channel_sum += np.sum(image_np, axis=(0, 1))
                channel_sum_squared += np.sum(image_np ** 2, axis=(0, 1))

                total_pixels += image_np.shape[0] * image_np.shape[1]
            except Exception:
                print(f"Skipping {image_path}")
    
    if total_pixels == 0:
        return [0,0,0] , [1,1,1]

    mean = channel_sum / total_pixels
    std = np.sqrt(channel_sum_squared / total_pixels - mean ** 2)
    
    return mean.tolist(), std.tolist()
