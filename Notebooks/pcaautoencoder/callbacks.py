import operator
import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class CallBack(ABC):
    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_training_end(self):
        pass


class ModelCheckPoint(CallBack):
    def __init__(self,
                 monitored_metric: str,
                 minimize_metric: bool,
                 save_location: str="model_checkpoints",
                 load_best_weights: bool=True):

        self.best_value = float("inf") if minimize_metric else -float("inf")
        self.relation = operator.lt if minimize_metric else operator.gt
        self.to_print = "decreased" if minimize_metric else "increased"
        self.monitored_metric = monitored_metric
        self.load_best_weights = load_best_weights
        self.save_location = save_location


    def __save_model(self, 
                     model: nn.Module, 
                     filename: str):
        
        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)
            torch.save(model.state_dict(), f"{self.save_location}/{filename}.pt")
    

    def on_epoch_end(self,
                    metrics: dict,
                    model: nn.Module,
                    filename: str,
                    verbose: bool=True):
        
        if self.relation(metrics[self.monitored_metric], self.best_value):
            if verbose:
                print(f"{self.monitored_metric} {self.to_print} from {self.best_value} to {metrics[self.monitored_metric]}")
            self.__save_model(model, filename)
            self.best_value = metrics[self.monitored_metric]
            self.filename = filename

        return False

    
    def on_training_end(self,
                        model: nn.Module):
        if self.load_best_weights:
            model.load_state_dict(torch.load(self.filename))


class EarlyStopping(CallBack):
    def __init__(self,
                 monitored_metric: str,
                 minimize_metric: bool,
                 patience: int):
        
        self.relation = operator.lt if minimize_metric else operator.gt
        self.to_print = "decrease" if minimize_metric else "increase"
        self.monitored_metric = monitored_metric
        self.minimize_metric = minimize_metric
        self.patience = patience
        self.epochs = 0


    def on_epoch_end(self,
                metrics: dict,
                model: nn.Module,
                filename: str,
                verbose: bool=True):

        if self.relation(metrics[self.monitored_metric], self.best_value):
            self.epochs = 0
            return False
        else:
            self.epochs += 1
            if self.epochs > self.patience:
                if verbose:
                    print(f"{self.monitored_metric} did not {self.to_print} in {self.epochs} epochs, training stopped")
                return True
            else:
                return False
            

    def on_training_end(self,
                        model: nn.Module):
        pass

