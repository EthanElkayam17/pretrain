import torch
import math
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import Tensor
from data.data import RexailDataset
from torch.utils.data import DataLoader, Subset
from models.model import CFGCNN
from torch.nn.parallel import DistributedDataParallel
from utils.checkpoint import save_state_dict
from utils.other import dirjoin

STATE_DICT_DIR = "state_dicts"

def setup(rank, world_size):
    """Set up a process"""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            init_method="env://", 
                            world_size=world_size)


def cleanup():
    """Clean up processes"""

    dist.destroy_process_group()


def load_predictions(rank: int,
                    world_size: int,
                    model_cfg_name: str,
                    state_dict_name: str,
                    dataset: RexailDataset,
                    shared_preds: Tensor,
                    shared_reals: Tensor,
                    batch_size: int,
                    half_precision: bool) -> None:
    """load predictions for each process
    
    Args:
        rank: process id
        world_size: number of processes
        model_cfg_name: name of model config file
        state_dict_name: name of state_dict file
        dataset: RexailDataset instance
        shared_preds: shared prediction tensor to fill up
        shared_reals: shared truth tensor to fill up
        batch_size: batch_size
        half_precision: whether to utilize half-precision
    
    Returns:
        None"""
    
    setup(world_size=world_size, rank=rank)
    
    model = CFGCNN(cfg_name=model_cfg_name).to(rank)
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    if half_precision: model.half()


    state_dict_path = dirjoin(STATE_DICT_DIR,state_dict_name)

    if (state_dict_path is not None) and (rank == 0):
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    
    if rank == 0:
        save_state_dict(model=model, 
                        dir="temp_state_dicts",
                        model_name="temp_state_dict_rank_0.pth")
    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.load_state_dict(torch.load("temp_state_dicts/temp_state_dict_rank_0.pth", map_location=map_location, weights_only=True))    
    dist.barrier()

    if rank == 0:
        os.remove("temp_state_dicts/temp_state_dict_rank_0.pth")
        os.rmdir("temp_state_dicts")


    N = len(dataset)
    per_proc = math.ceil(N / world_size)
    start = rank * per_proc
    end   = min(start + per_proc, N)
    subset = Subset(dataset, list(range(start, end)))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.inference_mode():
        offset = start
        for X,y in dataloader:
            X, y = X.to(rank), y.to(rank)
            
            if half_precision:
                with torch.cuda.amp.autocast():
                    preds = model(X).to(rank)

            else:
                preds = model(X).to(rank)

            shared_reals[offset:offset + len(y)] = y
            shared_preds[offset:offset + len(preds)] = preds
            offset += len(preds)
    
    dist.barrier()

    cleanup()


def load_all_predictions(model_cfg_name: str,
                        state_dict_name: str,
                        dataset: RexailDataset,
                        batch_size: int,
                        half_precision: bool = False,
                        save_tensor_with_prefix: str = "") -> Tensor:
    """load all of model's predictions onto a large shared-memory tensor.
    
    Args:
        model_cfg_name: config file name of model
        state_dict_path: name of state_dict of model
        dataset: RexailDataset instance
        batch_size: batch size
        half_precision: whether to utilize half-precision
        save_tensor_with_prefix: file name to save tensor (optional) 
    
    Returns:
        Tensor - all of the model's predictions on the dataset
        Tensor - truth values of each entry in the dataset"""
    
    dtype = torch.float16 if half_precision else torch.float32
    N = len(dataset)

    shared_preds = torch.zeros((N, dataset.num_classes), dtype=dtype).share_memory_()
    shared_reals = torch.zeros(N).share_memory_()
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(
        load_predictions,
        args=(WORLD_SIZE,
            model_cfg_name,
            state_dict_name,
            dataset,
            shared_preds,
            shared_reals,
            batch_size,
            half_precision),
        nprocs=WORLD_SIZE,
        join=True
    )

    if save_tensor_with_prefix != "":
        torch.save(shared_preds, f'{save_tensor_with_prefix}_preds.pt')
        torch.save(shared_reals, f'{save_tensor_with_prefix}_reals.pt')

    return shared_preds, shared_reals


def top_k_accuracy(y_preds: Tensor, 
                   y_real: Tensor, 
                   k: int) -> float:
    """
    Calculates the top-k accuracy of a batched prediction.

    Args:
        y_preds: The predicted tensor, should be of shape [batchsize, num_of_classes] or [batch_size]
        y_real: he real tensor to be predicted, should be of shape [batchsize] 
        k: k to use for top-k.

    Returns:
        float: top-k accuracy.
    """
    _, topk_idxs = y_preds.topk(k, dim=1, largest=True, sorted=True)
    
    hits = topk_idxs.eq(y_real.view(-1, 1))

    correct_any = hits.any(dim=1)

    return correct_any.float().mean().item()


class OverviewRV():
    """A class providing an overview of the performance of a given iteration/version.
    
    A version can be loaded using either:
        the appropriate model config file, its respective state dict and a RexailDataset instance (REQUIRES CUDA, SLOW).
        OR
        preds+reals Tensors, residing in storage (RECOMMENDED).
    
    In practice the former is used to calculate the latter and then the former is discarded, one can (and should) 
    choose to save the tensors to storage upon calculation by providing a 'save_tensor_with_prefix' so they can be
    easily loaded again (via the second method) if needed without requiring extensive re-computation.
     
    """

    #todo
