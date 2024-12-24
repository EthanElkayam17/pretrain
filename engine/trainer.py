import torch
import logging
import math
from functools import partial
from typing import Dict, List, Tuple, Union
from utils.checkpoint import save_state_dict
from models.model import CFGCNN


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
               device: torch.device) -> Tuple[float, float]:
    
    """Basic train for a single epoch.

    Args:
        model: model to train.
        dataloader: dataloader containing data.
        loss_fn: loss function.
        optimizer: optimizer function.
        device: device to compute on.
    
    Returns: (loss, accuracy, grad_norms)
    """

    model.train()

    train_loss, train_accuracy = 0 , 0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device) , y.to(device)

        y_res = model(X).to(device)

        loss = loss_fn(y_res,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_accuracy += ((torch.argmax(torch.softmax(y_res, dim=1), dim=1) == y).sum().item() / len(y_res))

        print(f"training batch number #{batch}")

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    """Basic test for a single epoch.

    Args:
        model: model to test.
        dataloader: dataloader to use for test.
        loss_fn: loss function.
        device: device to compute on.
    
    Returns: (loss, accuracy)
    """

    model.eval()
    
    test_loss, test_accuracy = 0 , 0

    # disables autograd
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device) , y.to(device)

            y_res = model(X).to(device)

            test_loss += loss_fn(y_res,y).item()

            test_accuracy += ((torch.argmax(torch.softmax(y_res, dim=1), dim=1) == y).sum().item() / len(y_res))

            print(f"testing batch number #{batch}")

    test_loss = test_loss / len(dataloader)
    test_accuracy = test_accuracy / len(dataloader)
    
    return test_loss,test_accuracy


def trainer(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          lr_min: Union[float,list],
          lr_max: Union[float,list],
          warmup_epochs: int,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_freq: int,
          decay_mode: str = "none",
          exp_decay_factor: float = 0,
          curr_epoch: int = 1,
          model_name: str = "model",
          logger = None):

    """Train and test CFGCNN networks
    
    Args:
        model: model to train and test.
        train_dataloader: dataloader for training.
        test_dataloader: dataloader for testing.
        optimizer: optimizer function for training.
        lr_min: initial learning rate, float when uniform or list[param_group_id] = lr_of_that_param_group.
        lr_max: final learning rate desired after warm up, float when uniform or list[param_group_id] = lr_of_that_param_group.
        decay_mode: specifies how to decay learning rate ["exp","cos","none"]
        warmup_epochs: on how many epochs should the warmup span.
        loss_fn: loss function.
        epochs: number of epochs for current stage.
        device: device to compute on.
        save_freq: how many epochs between model saves.
        exp_decay_factor: decay factor if using 'exp' decay.
        curr_epoch: current epoch (in case of model checkpoint loading).
        model_name: name of model's state_dict file.
        logger: logging function.
    """

    decay_mode_to_func = {
                    "exp": partial(warmup_to_exponential_decay, lr_min=lr_min,lr_max=lr_max,warmup_epochs=warmup_epochs, decay_factor=exp_decay_factor),
                    "cos": partial(warmup_to_cosine_decay, lr_min=lr_min, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epochs),
                    "none": None
                    }

    if logger is None:
            logger = logging.getLogger('null_logger')
            logger.addHandler(logging.NullHandler)
    
    if not (decay_mode in decay_mode_to_func.keys()):
        raise ValueError(f"Invalid decay mode, should be one of {decay_mode_to_func.keys()}")
    
    decay = decay_mode_to_func.get(decay_mode)

    
    epoch = curr_epoch
    while (epoch <= epochs):
        print("epoch:" + str(epoch))
        
        for idx, param_group in enumerate(optimizer.param_groups):
            
            if decay is not None:
                param_group["lr"] = decay(epoch)
                
            if idx == 0:
                logger.info(f"Learning rate adjusted to: {param_group["lr"]}")


        train_loss, train_acc = train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)
        
        test_loss, test_acc = test_step(model=model,
                                   dataloader=test_dataloader,
                                   loss_fn=loss_fn,
                                   device=device)
        

        logger.info(f"Epoch: {epoch}. \n Train loss: {train_loss}, Train Acc: {train_acc}. \n Test loss: {test_loss}, Test acc: {test_acc}. \n")

        if (epoch-curr_epoch)%save_freq == 0:
             logger.info("---Saving current model's state dict...---")
             
             save_state_dict(model=model,
                        dir="state_dicts",
                        model_name=(f"{model_name}_" + str(epoch) + ".pth"))
             
             logger.info("---model saved!---")
        
        epoch += 1
