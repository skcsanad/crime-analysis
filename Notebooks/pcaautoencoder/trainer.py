import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from .callbacks import *

class ModelWithTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward()")

    def __shared_eval_step(self,
                           X: torch.Tensor,
                           y: torch.Tensor,
                           loss_func: Union[nn.Module, Callable]) -> torch.Tensor:
    
        y_hat = self.forward(X)
        loss = loss_func(y_hat, y)
        return loss
    
    def train_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        loss = self.__shared_eval_step(X, y, loss_func)
        metrics = {"train loss": loss.item()}
        return loss, metrics
    
    def eval_step(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        loss = self.__shared_eval_step(X, y, loss_func)
        metrics = {"test loss": loss.item()}
        return loss, metrics

    
class Trainer():

    @staticmethod
    def add_metrics(running: dict,
                    current: dict,
                    multiply_by: Optional[int]=None):
    
        if isinstance(multiply_by, int):
            for key in current:
                current[key] *= multiply_by

        for key, value in current.items():
            if key not in running:
                running[key] = value
            else:
                running[key] += value


    @staticmethod
    def log_metrics(metrics: dict,
                    logs: Optional[dict]=None, 
                    divide_by: Optional[int]=None,
                    verbose: bool=True,
                    mode: Optional[str]=None,
                    timestep: Optional[str]=None,
                    n_timestep: Optional[int]=None,
                    writer: Optional[SummaryWriter]=None,
                    total_timesteps: Optional[int]=None,
                    timestep_name: bool=False):
    
        #TODO: print all metrics to one line
        if verbose: 
            for key, value in metrics.items():
                print(f"{timestep}: {n_timestep} {mode} {key}: {value / divide_by}")

        if isinstance(writer, SummaryWriter) and total_timesteps is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"{key} x {timestep}/{mode}", value / divide_by, total_timesteps)

        if isinstance(logs, dict):
            for key, value in logs.items():
                to_append = value / divide_by if divide_by is not None else value
                if timestep_name:
                    key = f"{key} {timestep}"
                if not key in logs:
                    logs[key] = [to_append]
                else:
                    logs[key].append(to_append)


    def validation_loop(self,
                        model: ModelWithTrainer, 
                        testloader: DataLoader, 
                        loss_func: Union[nn.Module, Callable],
                        verbose: int,
                        epoch: int, 
                        device: torch.device):
    
        model.eval()
        running_metrics = {}
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                loss, metrics = model.eval_step(X, y, loss_func, device)
                self.add_metrics(running=running_metrics,
                            current=metrics,
                            multiply_by=X.size(0))
        
        self.log_metrics(metrics=running_metrics,
                        logs=self.logs,
                        divide_by=len(testloader.dataset),
                        verbose=True if verbose > 0 else False,
                        mode="test",
                        timestep="epoch",
                        n_timestep=epoch,
                        writer=self.writer,
                        total_timesteps=self.epochs)
    

    def training_loop(self,
                      model: ModelWithTrainer, 
                      trainloader: DataLoader, 
                      loss_func: Union[nn.Module, Callable], 
                      device: torch.device,
                      optimizer: torch.optim.Optimizer, 
                      verbose: int,
                      epoch: int,
                      print_every: Optional[int]=None):
    
        model.train()
        epoch_metrics = {}
        if isinstance(print_every, int):
            step_metrics = {}
            steps = 0

        for X, y in trainloader:
            steps += 1
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss, metrics = model.training_step(X, y, loss_func, device)
            loss.backward()
            optimizer.step()

            self.add_metrics(running=epoch_metrics,
                             current=metrics, 
                             multiply_by=X.size(0))
            
            if isinstance(print_every, int):
                self.add_metrics(running=step_metrics,
                        current=metrics)
                steps += 1
                self.steps +=1
                if steps % print_every == 0:
                    self.log_metrics(metrics=step_metrics,
                                     logs=self.logs,
                                     divide_by=print_every,
                                     verbose=True if verbose == 2 else False,
                                     mode="train",
                                     timestep="step",
                                     n_timestep=steps,
                                     writer=self.writer,
                                     total_timesteps=self.steps)
                    step_metrics.clear()
            
        self.log_metrics(metrics=epoch_metrics,
                         logs=self.logs,
                         divide_by=len(trainloader.dataset),
                         verbose=True if verbose > 0 else False,
                         mode="train",
                         timestep="epoch",
                         n_timestep=epoch,
                         writer=self.writer,
                         total_timesteps=self.epochs)
        
    
    def train(self,
              model: ModelWithTrainer,
              optimizer: torch.optim.Optimizer,
              loss_func: Union[nn.Module, Callable],
              epochs: int,
              trainloader: DataLoader,
              testloader: DataLoader,
              verbose: int=2,
              print_every: Optional[int]=40,
              log_to_tensorboard: bool=True,
              callbacks: List[CallBack]=[],
              device: torch.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
              save_filename: Optional[str]=None,
              close_writer_on_end: bool=True,
              **kwargs) -> dict:
        
        
        if hasattr(model, "custom_train"):
            return model.custom_train(self,
                                    optimizer=optimizer,
                                    loss_func=loss_func,
                                    epochs=epochs,
                                    trainloader=trainloader,
                                    testloader=testloader,
                                    verbose=verbose,
                                    print_every=print_every,
                                    callbacks=callbacks,
                                    device=device,
                                    **kwargs)
        
        else:
            self.logs = {}
            self.epochs = 0
            self.steps = 0

        if log_to_tensorboard:
            self.writer = SummaryWriter()

        for e in range(epochs):
            self.training_loop(model=model, trainloader=trainloader, loss_func=loss_func,
                                device=device, optimizer=optimizer, verbose=verbose, epoch=e,
                                print_every=print_every)
            self.validation_loop(model=model, testloader=testloader, loss_func=loss_func, verbose=verbose,
                                    epoch=e, device=device)
                        # Calling callbacks that might stop the training or save the model
            should_break = False
            for callback in callbacks:
                should_break =  callback.on_epoch_end(metrics=self.logs,
                                            model=model,
                                            filename=save_filename,
                                            verbose=True if verbose > 0 else False)
                if should_break:
                    break

            if should_break:
                break

        for callback in callbacks:
            callback.on_training_end(model)
        if close_writer_on_end:
            self.writer.close()
        return self.logs
            



    

