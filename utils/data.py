import torch
import numpy as np
import os
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from hashlib import sha256
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
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
    
    Data should be filtered/partitioned by using the 'decider' parameter"""
    
    def __init__(
        self,
        root: str,
        transform: Union[Callable] = None,
        pre_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        decider: Callable[[str], bool] = default_decider,
        extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        ignore_classes: List[str] = list(),
        storewise: bool = False,
        weighed: bool = False,
        load_into_memory: bool = False,
        num_workers: int = 1,
    ):
        """Args:
            root: Root directory path.
            transform: A transform for the PIL images.
            pre_transform: a transform to be applied pre-fetch to all the data (usable only if load_into_memory = True)
            target_transform: A transform for the target.
            loader: A function to load an image given its path.
            decider: A function that takes path of an Image file and decides if its in the dataset.
            extensions: file extensions that are acceptable.
            ignore_classes: classes to ignore when making dataset.
            storewise: whether the directory is partitioned store-wise inside each class.
            weighed: whether the file names contain weight data.
            
            load_into_memory: whether to load the entire dataset into a shared in-memory block,
            by default only the paths are stored and the data itself is fetched lazily,
            this option will load all the required tensors into memory when creating the instance,
            VERIFY THAT THERE IS ENOUGH MEMORY AVAILABLE BEFORE SETTING TRUE,
            the samples of the dataset will be stored in a large contiguous tensor,
            one should utilize SharedDatasetWrapper to properly share the dataset across processes.

            num_workers: number of workers to be used to load dataset into memory"""
        
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )
        self.pre_transform = pre_transform
        self.ignore_classes = ignore_classes
        classes, class_to_idx = RexailDataset.find_classes(self.root, self.ignore_classes)
        self.class_to_idx = class_to_idx

        self.loaded_dataset = False
        self.storewise = storewise
        self.weighed = weighed
        self.decider = decider
        self.loader = loader
        self.extensions = extensions

        self.classes = classes

        self.samples = self.make_dataset()
        self.targets = [s[1] for s in self.samples]

        if load_into_memory:
            shape_sample = (self.__getitem__(0,only_pre_transform=(self.pre_transform is not None)))[0].shape
            self.data = torch.empty((len(self.samples), *shape_sample), dtype=torch.float32)
            self._load_everything(num_workers=num_workers)
            self.data.share_memory_()
            self.loaded_dataset = True


    def __len__(self) -> int:
        """Returns the amount of items in the dataset"""

        return len(self.samples)


    def __getitem__(self, 
                    index: int,
                    only_pre_transform: bool = False) -> Tuple:
        """Returns item based on index, in a tuple of the form: (sample, target) 
            where:
                sample is the tensor representing the image.
                target is class_index of the target class.
        
        Args:
            index: Index
            only_pre_transform: whether to apply only pre_transform instead of pre_transform+transform """
        
        if self.loaded_dataset:
            sample = self.data[index]
            target = self.targets[index]
            
            if self.transform is not None:
                sample = self.transform(sample)
            
            return tuple([sample,target])

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.pre_transform is not None:
            sample = self.pre_transform(sample)
        
        if (self.transform is not None) and (not only_pre_transform):
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        if index%1000 == 0:
            print(f"just got {index}")
        return tuple([sample,target])


    def _load_everything(self, num_workers: int):
        """Parallel loading of the dataset into memory"""

        def fill_index(index):
            self.data[index] = self.__getitem__(index=index,only_pre_transform=(self.pre_transform is not None))[0]
        
        indices = list(range(len(self.samples)))
        
        print("loading dataset into memory...")
        with Pool(num_workers) as pool:
            pool.map(fill_index,indices)


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


def create_dataloaders_and_samplers(
        world_size: int,
        rank: int,
        train_dir: Union[str, Path],
        test_dir: Union[str, Path],
        batch_size: int,
        num_workers: int = 1,        
        train_transform: Union[torch.nn.Sequential, transforms.Compose] = default_transform(),
        train_pre_transform: Optional[Callable] = None,
        test_transform: Union[torch.nn.Sequential, transforms.Compose] = default_transform(),
        test_pre_transform: Optional[Callable] = None,
        train_extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        train_decider: Callable[[str], bool] = (lambda x: True),
        test_extensions: Optional[Tuple[str, ...]] = (".png",".jpeg",".jpg"),
        test_decider: Callable[[str], bool] = (lambda x: True),
        ignore_classes: List[str] = list(),
        storewise: bool = False,
        weighed: bool = False,
        load_into_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader, List]:
    """Creates training/testing dataloaders from training/testing directories
    
    Args:
        world_size: number of total processes (GPU's),
        rank: identifier for current process (GPU).
        train_dir: path of training data directory.
        test_dir: path of testing data directory.
        batch_size: amount per batch.
        num_workers: int = 0,
        train_transform: transforms to apply to training data.
        train_pre_transform: pre_transform to apply to training data.
        test_transform: transforms to apply to testing data.
        test_pre_transform: pre_transform to apply to testing data.
        train_decider: A function that takes path of an Image file and decides if its in the training dataset.
        train_extensions: file extensions that are acceptable in the training dataset.
        test_decider: A function that takes path of an Image file and decides if its in the testing dataset.
        test_extensions: file extensions that are acceptable in the testing dataset.
        ignore_classes: classes to ignore when making training/testing sets.
        storewise: whether both training and testing directories are partitioned store-wise inside each class.
        weighed: whether the file names contain weight data.
        load_into_memory: whether to load datasets into memory.

    Returns: (train_dataloader, test_dataloader, class_names)
    """

    train_data = RexailDataset(root=train_dir,
                               transform=train_transform,
                               pre_transform=train_pre_transform,
                               decider=train_decider,
                               extensions=train_extensions,
                               ignore_classes=ignore_classes,
                               storewise=storewise,
                               weighed=weighed,
                               load_into_memory=load_into_memory,
                               num_workers=num_workers)
    
    test_data = RexailDataset(root=test_dir, 
                              transform=test_transform,
                              pre_transform=test_pre_transform,
                              decider=test_decider,
                              extensions=test_extensions,
                              ignore_classes=ignore_classes,
                              storewise=storewise,
                              weighed=weighed,
                              load_into_memory=load_into_memory,
                              num_workers=num_workers)
    
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    
    class_names = train_data.classes
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sampler=train_sampler,
        
        # Try to avoid costs of transfer between pageable and pinned memory
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True
    )

    return train_sampler, train_dataloader, test_sampler, test_dataloader, class_names


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
