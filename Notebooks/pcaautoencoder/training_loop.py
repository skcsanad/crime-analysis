from .model import PCAAutoencoder, PCAAE_Loss
import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import operator
from .callbacks import *


def __shared_eval_step(model: nn.Module, 
                       X: torch.Tensor, 
                       y: torch.Tensor, 
                       loss_func: Union[nn.Module, Callable], 
                       device: torch.device) -> Tuple[torch.Tensor, dict]:
    
    X, y = X.to(device), y.to(device)
    y_hat, hidden = model(X)

    if isinstance(loss_func, PCAAE_Loss):
        loss, recon_loss, cov_loss = loss_func(y_hat, y, hidden)
        metrics = {
            "loss": loss.item(), 
            "recon_loss": recon_loss.item(), 
            "cov_loss": cov_loss.item()
            }
    else:
        loss = loss_func(y_hat, y)
        metrics = {
            "loss": loss.item()
            }
    
    return loss, metrics


def add_metrics(running: dict,
                current: dict,
                multiply_by: Optional[int]=None):
    
    if isinstance(multiply_by, int):
        for key in current:
            current[key] *= multiply_by

    if not running:
        running.update(current)
    else:
        for key, value in current.items():
            running[key] += value


def log_metrics(metrics: dict,
                logs: Optional[dict]=None, 
                divide_by: Optional[int]=None,
                verbose: bool=True,
                mode: Optional[str]=None,
                timestep: Optional[str]=None,
                n_timestep: Optional[int]=None,
                writer: Optional[SummaryWriter]=None,
                total_timesteps: Optional[int]=None):
    
    #TODO: print all metrics to one line
    if verbose: 
        for key, value in metrics.items():
            print(f"{timestep}: {n_timestep} {mode} {key}: {value / divide_by}")

    if isinstance(writer, SummaryWriter) and total_timesteps is not None:
        for key, value in metrics.items():
            writer.add_scalar(f"{key} x {timestep}/{mode}", value / divide_by, total_timesteps)

    if isinstance(logs, dict):
        if not logs:
            logs.update({key: [value / divide_by] for key, value in metrics.items()})
        else:
            for key, value in metrics.items():
                logs[key].append(value / divide_by)
    
    metrics.clear()


def combine_logs(**kwargs: dict):
    combined = {}
    for dict_name, dict_contents in kwargs.items():
        for key, value in dict_contents.items():
            combined[f"{dict_name}_{key}"] = value
    
    return combined



def validation_loop(model: nn.Module, 
                    testloader: DataLoader, 
                    loss_func: Union[nn.Module, Callable], 
                    device: str) -> dict:
    
    model.eval()
    running_metrics = {}
    with torch.no_grad():
        for X, y in testloader:
            loss, metrics = __shared_eval_step(model, X, y, loss_func, device)
            add_metrics(running=running_metrics,
                        current=metrics,
                        multiply_by=X.size(0))
    
    return running_metrics


def training_loop(model: nn.Module, 
                  trainloader: DataLoader, 
                  loss_func: Union[nn.Module, Callable], 
                  device: torch.device,
                  optimizer: torch.optim.Optimizer, 
                  verbose: int,
                  print_every: Optional[int]=None,
                  step_logs: Optional[dict]=None,
                  writer: Optional[SummaryWriter]=None,
                  total_steps: Optional[List[int]]=None) -> dict:
    
    model.train()
    epoch_metrics = {}
    if isinstance(print_every, int):
        step_metrics = {}
        steps = 0

    for X, y in trainloader:
        steps += 1
        optimizer.zero_grad()
        loss, metrics = __shared_eval_step(model, X, y, loss_func, device)
        loss.backward()
        optimizer.step()

        #print(metrics)
        add_metrics(running=epoch_metrics,
                    current=metrics, 
                    multiply_by=X.size(0))
        #print(epoch_metrics)
        
        if isinstance(print_every, int):
            add_metrics(running=step_metrics,
                    current=metrics)
            steps += 1
            if isinstance(total_steps, list):
                total_steps[0] += 1
            if steps % print_every == 0:
                log_metrics(metrics=step_metrics,
                            logs=step_logs,
                            divide_by=print_every,
                            verbose=True if verbose == 2 else False,
                            mode="train",
                            timestep="step",
                            n_timestep=steps,
                            writer=writer,
                            total_timesteps=total_steps[0])
        
    return epoch_metrics


def train_model(model: PCAAutoencoder,
                goal_hidden_dim: int,
                optimizer: torch.optim.Optimizer,
                loss_func: Union[nn.Module, Callable],
                epochs: int,
                trainloader: DataLoader,
                testloader: DataLoader,
                verbose: int=2,
                print_every: Optional[int]=40,
                log_to_tensorboard: bool=True,
                callbacks: List[CallBack]=[],
                device: torch.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')) -> dict:
    
    model.to(device)
    if log_to_tensorboard:
        writer = SummaryWriter()

    training_logs = {}
    eval_logs = {}
    total_epochs = 0
    hidden_dim = 1
    stepwise_training_logs = {} if isinstance(print_every, int) else None
    total_steps = [0] if isinstance(print_every, int) else None


    while hidden_dim < goal_hidden_dim:

        if not total_epochs == 0:
            model.increase_latentdim()
            model.to(device)
            hidden_dim += 1

        if verbose > 0:
            print(f"Training with hidden dim: {hidden_dim}")
        
        for e in range(epochs):
            epoch_metrics = training_loop(model=model,
                                            trainloader=trainloader,
                                            loss_func=loss_func,
                                            device=device,
                                            optimizer=optimizer,
                                            verbose=verbose,
                                            print_every=print_every,
                                            step_logs=stepwise_training_logs,
                                            writer=writer,
                                            total_steps=total_steps)
            
            log_metrics(metrics=epoch_metrics,
                        logs=training_logs,
                        divide_by=len(trainloader.dataset),
                        verbose=True if verbose > 0 else False,
                        mode="train",
                        timestep="epoch",
                        n_timestep=e,
                        writer=writer,
                        total_timesteps=total_epochs)
            
            eval_metrics = validation_loop(model=model, 
                                            testloader=testloader, 
                                            loss_func=loss_func, 
                                            device=device)
            
            log_metrics(metrics=eval_metrics,
                        logs=eval_logs,
                        divide_by=len(testloader.dataset),
                        verbose=True if verbose > 0 else False,
                        mode="test",
                        timestep="epoch",
                        n_timestep=e,
                        writer=writer,
                        total_timesteps=total_epochs)
            
            total_epochs += 1

            # Calling callbacks that might stop the training or save the model
            should_break = False
            for callback in callbacks:
                should_break =  callback.on_epoch_end(metrics=eval_metrics,
                                            model=model,
                                            filename=f"PCAAE_hidden_dim{hidden_dim}",
                                            verbose=True if verbose > 0 else False)
                if should_break:
                    break

            if should_break:
                break

        for callback in callbacks:
            callback.on_training_end(model)

    return combine_logs(train=training_logs,
                        test=eval_logs,
                        stepwise=stepwise_training_logs)

