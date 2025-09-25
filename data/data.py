import torch
import numpy as np
import os
import copy
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.sampler import Sampler
from functools import partial
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler, default_collate
from utils.transforms import default_transform
from utils.other import dirjoin
from torch import Tensor


class RexailDataset(datasets.VisionDataset):
    """A dataset class for loading and partitioning data from rexail's dataset,
    expects directory in one of the following structure: 
        root/{product_id}/{image_name}.jpeg
        
    the images file names are expected to be in the form: 
        {date_of_capture(ms)}_{quantity}_{weight(g)}.ext .

    Data should be filtered/partitioned by using the filter/partition parameters, 
    if not internally.
    
    When not utilizing 'load_into_memory' the dataset can be used as-is to access directly or create dataloaders,
    only the paths and targets will be stored and the actual data is fetched lazily on __getitem___.
    
    When 'load_into_memory' is called the entire dataset will be stored in a large shared tensor in shared-memory space,
    this option takes up significant memory (oftentimes more than the space taken up by the dataset on disk, because the tensor
    does not take advantage of compression), but effectively removes the bottleneck of reading the images from disk during access.
    When this method is called - one must have defined 'pre_transform' to transform the images into the tensors that will be stored (and
    hence it is recommended to keep 'pre_transform' deterministic), furthermore 'pre_transform' MUST produce a fixed-shape tensor
    regardless of the image that is being transformed. NOTICE: upon loading the dataset into memory the 'transform' must expect the
    stored tensor as input as opposed to the output of the loader (which is the expected input of 'transform' w/o loading into memory).

    when using this option with multiple processes (GPUs) one should create one instance of this class outside of the processes
    and use it to create instances of WrappedRexailDataset in each process, which will be used for samplers/dataloaders.
    """
    
    def __init__(
        self,
        root: str,
        transform: Callable = None,
        pre_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        max_class_size: int = -1,
        min_class_size: int = -1,
        earliest_timestamp_ms: int = 0,
        latest_timestamp_ms: int = -1,
        ratio: int = 100,
        complement_ratio: bool = False,
        extensions: Tuple[str, ...] = (".png",".jpeg",".jpg"),
        ignore_classes: List[str] = None,
        load_into_memory: bool = False,
        num_workers_load: int = 4,
        ):
        """Args:
            root: Root directory path.
            transform: A transform for the PIL images.
            pre_transform: a transform to be applied pre-fetch to all the data (usable only if load_into_memory = True) should return tensor.
            target_transform: A transform for the target.
            loader: A function to load an image given its path.
            max_class_size: maximum size of a single class (total images after filtering, including complement ratio), cuts uniformly-in-time to this amount if exceeds.
            min_class_size: minimum size of a single class (total images after filtering, including complement ratio).
            earliest_timestamp_ms: earliest date-of-capture (in ms) allowed for an image.
            latest_timestamp_ms: latest date=of=capture (in ms) allowed for an image.
            ratio: ratio of images in every class that should be in the database (0-100).
            complement_ratio: whether to take the complement set of pictures filtered by the ratio.
            extensions: file extensions that are acceptable.
            ignore_classes: classes to ignore when making dataset.
            load_into_memory: whether to load the dataset into shared-memory.
            num_workers_load: number of workers for parallel loading into memory (useful only when load_into_memory=True).
        """
        
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )
        assert max_class_size >= -1, "max_class_size must be positive integer (or -1 for no cap)"
        assert latest_timestamp_ms >= -1, "latest_timestamp_ms must be a postivie integer (or -1 for no cap)"
        
        if min_class_size != -1 and max_class_size != -1:
            assert max_class_size >= min_class_size, "max_class_size must be at least min_class_size"

        self.max_class_size = max_class_size
        self.min_class_size = min_class_size
        self.ratio = ratio
        self.complement_ratio = complement_ratio
        self.ignore_classes = ignore_classes
        self.pre_transform = pre_transform
        self.earliest_timestamp_ms = earliest_timestamp_ms
        self.latest_timestamp_ms = latest_timestamp_ms

        self.loaded_dataset = False
        self.loader = loader
        self.extensions = extensions
    
        if self.ignore_classes is None:
            self.ignore_classes = []

        classes, class_to_idx = RexailDataset.find_classes(directory=self.root, 
                                                           ignore_classes=self.ignore_classes)
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.num_classes = len(classes)

        self.samples = self.make_dataset()
        self.targets = [s[1] for s in self.samples]

        if load_into_memory:
            self.load_into_memory(num_workers=num_workers_load)
        

    def __len__(self) -> int:
        """Returns the amount of items in the dataset"""

        return len(self.samples)


    def __getitem__(self, 
                    index: int,
                    only_pre_transform = False) -> Tuple:
        """Finds item based on index
        
        Args:
            index: Index
            only_pre_transform: whether to disable the main transform.
        
        Returns: 
            Tuple - sample,target
            where: sample is the tensor representing the image, target is class_index of the target class.
        """
        
        if self.loaded_dataset:
            sample = self.data[index]
            target = self.targets[index]
            
            if self.transform is not None:
                sample = self.transform(sample)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return tuple([sample,target])

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.pre_transform is not None:
            sample = self.pre_transform(sample)
        
        if (self.transform is not None) and (not only_pre_transform):
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return tuple([sample,target])


    @staticmethod
    def _load_index(index: int,
                    samples: List,
                    data: torch.Tensor,
                    transform: Callable,
                    loader: Callable = datasets.folder.default_loader,
                    pbar = None) -> None:
        """Load a single index into memory"""

        path, _ = samples[index]
        sample = loader(path)

        if transform is not None:
            sample = transform(sample)
            
        data[index] = sample.detach().clone()

        if pbar is not None:
            pbar.update(1)


    def _load_everything(self, num_workers: int) -> None:
        """Parallel loading of the dataset into memory"""
        
        indices = list(range(0,len(self.samples)))
        
        print("loading dataset into memory...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(indices)) as pbar:
                filler = partial(RexailDataset._load_index, samples=self.samples, data=self.data, transform=self.pre_transform, loader=self.loader, pbar=pbar)
                list(executor.map(filler, indices))

    
    def load_into_memory(self,
                        num_workers: int = 0,
                        dtype: torch.dtype = torch.float16) -> None:
        """Load dataset into memory as a large tensor in shared-memory
        
        Args:
            num_workers: number of workers to work on the operation
            dtype: desired dtype for tensor

        Returns:
            None    
        """        

        assert self.pre_transform is not None, "pre_transform is required when loading dataset into memory"

        data_shape_sample = (self.__getitem__(0,only_pre_transform=True))[0].shape
        self.data = torch.zeros((len(self.samples), *data_shape_sample), dtype=dtype).share_memory_()

        self._load_everything(num_workers=num_workers)
        self.loaded_dataset = True
    

    def set_pre_transform(self, pre_transform: Callable) -> None:
        """Setter for transform parameter"""
        self.pre_transform = pre_transform
    
    
    def set_transform(self, transform: Callable) -> None:
        """Setter for transform parameter"""
        self.transform = transform
    

    def is_valid_file(self, 
                    path: str) -> bool:
        """Whether or not a file can be in the dataset by its path"""
        
        return ((self.extensions is None) or (path.lower().endswith(self.extensions)))


    @staticmethod
    def fname_time(fname: str) -> int:
        """Get time in ms from formatted fname"""
        time = (fname.split('_', 1)[0]).split('/')[-1]
        return int(time)


    @staticmethod
    def find_idx_by_time(sorted_fnames: List[str], time_ms: int, is_max: bool) -> int:
        """Finds the first/last index which fname's includes a time after the minimum / maximum.
        
        Args:
            sorted_fnames: list of file names sorted by time.
            min_time_ms: minimum time threshold (in ms)
            is_max: whether maximum or minimum.
        """
        if time_ms == -1:
            return int(is_max)*len(sorted_fnames)

        for idx, fname in enumerate(sorted_fnames):
            curr_time_ms = RexailDataset.fname_time(fname)

            if (curr_time_ms >= time_ms):
                return idx

        return len(sorted_fnames)


    def make_dataset(self) -> List[Tuple[str, int]]:
        """Makes a list of pairs (sample,target) of the dataset
        Note: dataset is partitioned and filtered using the partition/filter params, in deterministic way"""

        directory = os.path.expanduser(self.root)
        random.seed(42)
        
        res = []
        cl_idx_rm = []

        for target_class in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[target_class]
            target_dir = Path(dirjoin(directory, target_class))
            fnames = sorted((str(p) for p in target_dir.iterdir() if self.is_valid_file(str(p))), key=lambda p: RexailDataset.fname_time(str(p)))
            
            s = RexailDataset.find_idx_by_time(fnames, self.earliest_timestamp_ms, False)
            e = RexailDataset.find_idx_by_time(fnames, self.latest_timestamp_ms, True)
            valid_fnames = fnames[s:e]
            
            if len(valid_fnames) < self.min_class_size:
                cl_idx_rm.append(class_idx)

            else:
                if self.max_class_size == -1 or self.max_class_size >= len(valid_fnames):
                    universe = valid_fnames
                else:
                    indices = sorted(random.sample(range(len(valid_fnames)), self.max_class_size))
                    universe = [valid_fnames[i] for i in indices]

                cut_index = (len(universe) * self.ratio) // 100

                if self.complement_ratio:
                    filtered_fnames = universe[cut_index:]
                else:
                    filtered_fnames = universe[:cut_index]

                for fname in filtered_fnames:
                    if self.is_valid_file(fname):
                        item = fname, class_idx
                        res.append(item)
        
        for idx in sorted(cl_idx_rm, reverse=True):
            del self.class_to_idx[self.classes[idx]]
            del self.classes[idx]
        
        if not self.classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory} that adhere to the restrictions.")        

        return res


    @staticmethod
    def find_classes(directory: Union[str, Path], 
                    ignore_classes: List[str] = None) -> Tuple[List[str], Dict[str, int]]:
        """Creates a list of the classes and provides a mapping to indices.
        
        Args:
            directory: directory containing the classes
            ignore_classes: classes to ignore

        Returns:
            Tuple - (list_of_classes, map_class_to_idx)
        """

        if ignore_classes is None:
            ignore_classes = []

        classes = sorted((entry.name for entry in os.scandir(directory) if (entry.is_dir() and (entry.name not in ignore_classes))))
        
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory} that adhere to the restrictions.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class WrappedRexailDataset(Dataset):
    """Wrapper class for RexailDataset,
    
    Should be used with an in-memory RexailDataset"""

    def __init__(self,
                 shared_dataset: RexailDataset):
        """Args:
            shared_dataset: underlying RexailDataset instance"""
        
        super().__init__()
        
        assert shared_dataset.loaded_dataset is True, "underlying dataset should be loaded into memory"

        self.targets = copy.deepcopy(shared_dataset.targets)
        self.samples = copy.deepcopy(shared_dataset.samples)
        self.data = shared_dataset.data
        self.transform = shared_dataset.transform
        self.target_transform = shared_dataset.target_transform
        self.classes = shared_dataset.classes
        self.num_classes = shared_dataset.num_classes
    

    def __len__(self) -> int:
        """Returns the amount of items in the dataset"""

        return self.data.size(0)
    

    def __getitem__(self, index) -> Tuple:
        """Finds item based on index, similar form to RexailDataset's '__getitem__'"""

        sample = self.data[index]
        target = self.targets[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return tuple([sample,target])


def custom_collate_fn(batch: Tuple[Tensor, Tensor],
                      func: Callable[[Tensor,Tensor], Tuple[Tensor,Tensor]]):
    """Applies custom function to batch of data fetched from dataloader"""

    return func(*default_collate(batch))


def create_dataloaders_and_samplers_from_datasets(
        world_size: int,
        rank: int,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        external_collate_func_builder: Union[Callable, None] = None
    ) -> Tuple[DataLoader, Sampler, DataLoader, Sampler]:
        """Create dataloaders and samplers from train/test datasets
        
        Args:
            world_size: number of processes
            rank: current process id
            train_dataset: training dataset
            test_dataset: testing dataset
            batch_size: batch size
            num_workers: number of workers for dataloaders
            external_collate_func_builder: builder function for custom collate_fn
        
        Returns:
            (train_dataloader, train_smapler, test_dataloader, test_sampler)
        """

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        if external_collate_func_builder is not None:
            external_collate_func = external_collate_func_builder(train_dataset.num_classes)
            collate_fn = partial(custom_collate_fn,
                                func=external_collate_func)
        
        else:
            collate_fn = None

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=train_sampler,
            collate_fn=collate_fn,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  
            num_workers=num_workers,
            sampler=test_sampler,
            pin_memory=True
        )

        return train_dataloader, train_sampler, test_dataloader, test_sampler, 


def create_dataloaders_and_samplers_from_shared_datasets(
        world_size: int,
        rank: int,
        train_dataset: RexailDataset,
        test_dataset: RexailDataset,
        batch_size: int,
        num_workers: int = 0,
        external_collate_func_builder: Union[Callable, None] = None
    ) -> Tuple[DataLoader, Sampler, DataLoader, Sampler]:
        """Create dataloaders and samplers from shared RexailDatasets
        
        Args:
            world_size: number of processes
            rank: current process id
            train_dataset: shared training RexailDataset
            test_dataset: shared testing RexailDataset
            batch_size: batch size
            num_workers: number of workers for dataloaders
        
        Returns:
            (train_dataloader, train_smapler, test_dataloader, test_sampler)
        """

        train_data = WrappedRexailDataset(train_dataset)
        test_data = WrappedRexailDataset(test_dataset)

        return create_dataloaders_and_samplers_from_datasets(world_size, rank, train_data, test_data, batch_size, num_workers, external_collate_func_builder)


def calculate_mean_std(dataset: RexailDataset) -> Tuple[List, List]:
    """Calculates mean and standard deviation of each channel across a RexailDataset
    
    Args:
        dataset: a RexailDataset instance
    
    Returns:
        List - [mean,std]
    """

    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    total_pixels = 0
    
    for image_path, _ in dataset.samples:
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
