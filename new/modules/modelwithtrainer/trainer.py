import torch
import torch.nn as nn
from typing import Union, Callable, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from .callbacks import *
import socket
import subprocess

class ModelWithTrainer(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self):
        raise NotImplementedError("Subclasses must implement forward()")


    def _shared_eval_step(self,
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
        
        loss = self._shared_eval_step(X, y, loss_func)
        metrics = {"train loss": loss.item()}
        return loss, metrics
    
    def eval_step(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        loss = self._shared_eval_step(X, y, loss_func)
        metrics = {"test loss": loss.item()}
        return loss, metrics


    def fit(self,
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
        
        self.to(device)
        trainer = Trainer(log_to_tensorboard=log_to_tensorboard, device=device)
        for e in range(epochs):
            trainer.training_loop(model=self, trainloader=trainloader, loss_func=loss_func,
                                  optimizer=optimizer, verbose=verbose, epoch=e,
                                  print_every=print_every)
            trainer.validation_loop(model=self, testloader=testloader, loss_func=loss_func, verbose=verbose,
                                    epoch=e)
            if trainer.on_epoch_end(model=self, callbacks=callbacks, save_filename=save_filename, verbose=verbose):
                break
        trainer.on_training_end(model=self, callbacks=callbacks)
        if close_writer_on_end:
            trainer.writer.close()
        return trainer.logs

    
class Trainer():
    def __init__(self, log_to_tensorboard: bool, device: torch.device):
        self.logs = {}
        self.epochs = 0
        self.steps = 0
        if log_to_tensorboard:
            self.writer = SummaryWriter()
            if not self._is_port_in_use():
                subprocess.Popen(["tensorboard", "--logdir=runs/", "--port=6006", "--reload_multifile=true"])
                print("TensorBoard started at http://localhost:6006")
            else:
                print("TensorBoard is already running at http://localhost:6006")
        else:
            self.writer = None
        self.device = device


    @staticmethod
    def add_metrics(running: dict,
                    current: dict,
                    multiply_by: Optional[int]=None):
    
        for key, value in current.items():
            if key not in running:
                if isinstance(multiply_by, int):
                    running[key] = multiply_by * value
                else:
                    running[key] = value
            else:
                if isinstance(multiply_by, int):
                   running[key] += multiply_by * value
                else: 
                    running[key] += value


    @staticmethod
    def log_metrics(metrics: dict,
                    logs: Optional[dict]=None, 
                    divide_by: Optional[int]=None,
                    verbose: bool=True,
                    #mode: Optional[str]=None,
                    timestep: Optional[str]=None,
                    n_timestep: Optional[int]=None,
                    writer: Optional[SummaryWriter]=None,
                    total_timesteps: Optional[int]=None,
                    timestep_name: bool=True):
    
        if verbose:
            # Create a single formatted string with all metrics
            metrics_str = ", ".join([f"{key}: {value/divide_by:.4f}" for key, value in metrics.items()])
            print(f"{timestep} {n_timestep} - {metrics_str}")

        if isinstance(writer, SummaryWriter) and total_timesteps is not None:
            for key, value in metrics.items():
                propname = " ".join(key.split(" ")[1:])
                split = key.split(" ")[0]
                writer.add_scalar(f"{propname} x {timestep}/{split}", value / divide_by, total_timesteps)

        if isinstance(logs, dict):
            for key, value in metrics.items():
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
                        epoch: int):
    
        model.eval()
        running_metrics = {}
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(self.device), y.to(self.device)
                loss, metrics = model.eval_step(X, y, loss_func)
                self.add_metrics(running=running_metrics, 
                                 current=metrics,
                                 multiply_by=X.size(0))
        
        self.log_metrics(metrics=running_metrics,
                        logs=self.logs,
                        divide_by=len(testloader.dataset),
                        verbose=True if verbose > 0 else False,
                        #mode="test",
                        timestep="epoch",
                        n_timestep=epoch,
                        writer=self.writer,
                        total_timesteps=self.epochs)
    

    def training_loop(self,
                      model: ModelWithTrainer, 
                      trainloader: DataLoader, 
                      loss_func: Union[nn.Module, Callable],
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
            self.steps += 1
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss, metrics = model.train_step(X, y, loss_func)
            loss.backward()
            optimizer.step()

            self.add_metrics(running=epoch_metrics,
                             current=metrics, 
                             multiply_by=X.size(0))
            
            if isinstance(print_every, int):
                self.add_metrics(running=step_metrics,
                        current=metrics)

                if steps % print_every == 0:
                    self.log_metrics(metrics=step_metrics,
                                     logs=self.logs,
                                     divide_by=print_every,
                                     verbose=True if verbose == 2 else False,
                                     #mode="train",
                                     timestep="step",
                                     n_timestep=steps,
                                     writer=self.writer,
                                     total_timesteps=self.steps)
                    step_metrics.clear()

        self.epochs += 1  
        self.log_metrics(metrics=epoch_metrics,
                         logs=self.logs,
                         divide_by=len(trainloader.dataset),
                         verbose=True if verbose > 0 else False,
                         #mode="train",
                         timestep="epoch",
                         n_timestep=epoch,
                         writer=self.writer,
                         total_timesteps=self.epochs)
        
         
    def on_epoch_end(self, model: ModelWithTrainer, callbacks: List[CallBack], save_filename: str, verbose: int) -> bool:
        should_break = False
        for callback in callbacks:
            if callback.on_epoch_end(metrics=self.logs, model=model, filename=save_filename,
                                     verbose=True if verbose > 0 else False):
                should_break = True
        return should_break
    

    def on_training_end(self, model: ModelWithTrainer, callbacks: List[CallBack]):
        for callback in callbacks:
            callback.on_training_end(model)

    
    def _is_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 6006)) == 0