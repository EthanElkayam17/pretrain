import torch
import logging
import math
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.optimizer
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from functools import partial
from typing import Tuple, Union, Callable, Any
from utils.checkpoint import save_state_dict
from models.model import CFGCNN
from utils.other import set_random_seed


def warmup_to_cosine_decay(epoch: int,
                    lr_min: float,
                    lr_max: float,
                    warmup_epochs: int,
                    total_epochs: int):
    """Linearly warm up learning rate for warmup epochs then decay with cosine annealing.
     
     Args:
        epoch: current epoch.
        lr_min: minimum learning rate.
        lr_max: maximum learning rate.
        warmup_epochs: on how many epochs should the warmup span.
        total amount of epochs in training."""
     
    if epoch <= warmup_epochs:
        return (epoch*lr_max)/warmup_epochs
    
    else:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def warmup_to_exponential_decay(epoch: int,
                    lr_min: float,
                    lr_max: float,
                    warmup_epochs: int,
                    decay_factor: float,):
    """Linearly warm up learning rate for warmup epochs then decay exponentialy.
     
     Args:
        epoch: current epoch.
        lr_min: minimum learning rate.
        lr_max: maximum learning rate.
        decay_factor: constant to decay by.
        warmup_epochs: on how many epochs should the warmup span.
        """
    
    if epoch <= warmup_epochs:
        return lr_min + (epoch*(lr_max-lr_min))/warmup_epochs

    else:
        return max((lr_max * (decay_factor**(epoch-warmup_epochs))),lr_min)


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               rank: Any) -> Tuple[float, float]:
    
    """Basic train for a single epoch.

    Args:
        model: model to train.
        dataloader: dataloader containing data.
        loss_fn: loss function.
        optimizer: optimizer function.
        rank: device to compute on.
    
    Returns: (loss, accuracy, grad_norms)
    """

    model.train()

    train_loss, train_accuracy = 0 , 0
    print("just before da loop")
    for batch, (X,y) in enumerate(dataloader):
        print(batch)
        X, y = X.to(rank) , y.to(rank)

        y_res = model(X).to(rank)

        loss = loss_fn(y_res,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_accuracy += ((torch.argmax(torch.softmax(y_res, dim=1), dim=1) == y).sum().item() / len(y_res))

        if rank == 0 and (batch % 50 == 0):
            print(f"training batch number #{batch}")

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              rank: Any) -> Tuple[float, float]:
    
    """Basic test for a single epoch.

    Args:
        model: model to test.
        dataloader: dataloader to use for test.
        loss_fn: loss function.
        rank: device to compute on.
    
    Returns: (loss, accuracy)
    """

    model.eval()
    
    test_loss, test_accuracy = 0 , 0

    # disables autograd
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(rank) , y.to(rank)

            y_res = model(X).to(rank)

            test_loss += loss_fn(y_res,y).item()

            test_accuracy += ((torch.argmax(torch.softmax(y_res, dim=1), dim=1) == y).sum().item() / len(y_res))

            if rank == 0 and (batch % 50 == 0):
                print(f"testing batch number #{batch}")

    test_loss = test_loss / len(dataloader)
    test_accuracy = test_accuracy / len(dataloader)
    
    return test_loss,test_accuracy


def setup(rank, world_size):
    """Set up a process"""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up processes"""

    dist.destroy_process_group()


def trainer(rank: int,
          world_size: int,
          model_cfg_name: str,
          create_dataloaders_and_samplers: Callable[[int,int], Any],
          momentum: float,
          weight_decay: float,
          lr_min: Union[float,list],
          lr_max: Union[float,list],
          dropout_prob: float,
          warmup_epochs: int,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          decay_mode: str = "none",
          exp_decay_factor: float = 0,
          curr_epoch: int = 1,
          model_name: str = "model",
          load_state_dict_path: str = None,
          logger = None):

    """Train and test CFGCNN networks using multiple GPUs
    
    Args:
        rank: identifier of the gpu for this process.
        world_size: number of total processes (GPUs).
        model_cfg_name: model's config file name.
        create_dataloaders_and_samplers: function that will create the test/train dataloaders and samplers.
        momentum: momentum for optimizer.
        weight_decay: weight decay for optimizer.
        lr_min: initial learning rate, float when uniform or list[param_group_id] = lr_of_that_param_group.
        lr_max: final learning rate desired after warm up, float when uniform or list[param_group_id] = lr_of_that_param_group.
        dropout_prob: dropout probability.
        decay_mode: specifies how to decay learning rate ["exp","cos","none"]
        warmup_epochs: on how many epochs should the warmup span.
        optimizer: optimizer function for training.
        loss_fn: loss function.
        epochs: number of epochs for current stage.
        save_freq: how many epochs between model saves.
        exp_decay_factor: decay factor if using 'exp' decay.
        curr_epoch: current epoch (in case of model checkpoint loading).
        model_name: name of model's state_dict file that will be saved.
        load_state_dict_path: path to state dict to load at the start of training.
        logger: logging function.
    """

    if curr_epoch > epochs:
        return
    
    if logger is None or rank != 0:
        logger = logging.getLogger('null_logger')
        logger.addHandler(logging.NullHandler)


    DECAY_MODE_TO_FUNC = {
                    "exp": partial(warmup_to_exponential_decay, lr_min=lr_min,lr_max=lr_max,warmup_epochs=warmup_epochs, decay_factor=exp_decay_factor),
                    "cos": partial(warmup_to_cosine_decay, lr_min=lr_min, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epochs),
                    "none": None
                    }

    
    setup(world_size=world_size, rank=rank)
    model = CFGCNN(cfg_name=model_cfg_name, logger=logger, dropout_prob_override=dropout_prob).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])


    if (load_state_dict_path is not None) and (rank == 0):
        model.load_state_dict(torch.load(load_state_dict_path, weights_only=True))
    
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


    loss_fn.to(rank)
    optimizer = optimizer(params=model.parameters(),
                                lr=0, 
                                momentum=momentum, 
                                weight_decay=weight_decay)


    logger.info(f"---Creating dataloaders..---")
    train_sampler: DistributedSampler 
    test_sampler: DistributedSampler
    train_dataloader, train_sampler, test_dataloader, test_sampler = create_dataloaders_and_samplers(world_size=world_size,rank=rank)
    logger.info("---Dataloaders created---\n")

    
    if not (decay_mode in DECAY_MODE_TO_FUNC.keys()):
        raise ValueError(f"Invalid decay mode, should be one of {DECAY_MODE_TO_FUNC.keys()}")
    
    decay = DECAY_MODE_TO_FUNC.get(decay_mode)

    
    epoch = curr_epoch
    while (epoch <= epochs):
        print("epoch:" + str(epoch))
        
        for idx, param_group in enumerate(optimizer.param_groups):
            
            if decay is not None:
                param_group["lr"] = decay(epoch)
                
            if idx == 0:
                logger.info(f"Learning rate adjusted to: {param_group['lr']}")


        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        train_loss, train_acc = train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      rank=rank)
        
        test_loss, test_acc = test_step(model=model,
                                   dataloader=test_dataloader,
                                   loss_fn=loss_fn,
                                   rank=rank)
        

        logger.info(f"Epoch: {epoch}. \n Train loss: {train_loss}, Train Acc: {train_acc}. \n Test loss: {test_loss}, Test acc: {test_acc}. \n")
        
        epoch += 1
    
    if rank == 0:
        save_state_dict(model=model, 
                        dir="state_dicts",
                        model_name=model_name)
    dist.barrier()

    cleanup()
