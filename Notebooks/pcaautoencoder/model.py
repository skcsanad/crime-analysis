import torch
import torch.nn as nn
from typing import Tuple, Union, Callable, Optional, List
from .trainer import ModelWithTrainer, Trainer
from .callbacks import CallBack
from torch.utils.data import DataLoader

class PCAAutoencoder(ModelWithTrainer):
    def __init__(self, encoder: nn.ModuleList, decoder: nn.ModuleList, last_hidden_shape: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.last_hidden_shape = last_hidden_shape
        self.bottleneck = nn.ModuleList([nn.Linear(in_features=self.last_hidden_shape, out_features=1),
                                         nn.BatchNorm1d(num_features=1, affine=False)])

    def increase_latentdim(self):
        # Create new bottleneck expansion layer
        new_bottleneck = nn.ModuleList([nn.Linear(in_features=self.last_hidden_shape, out_features=self.bottleneck[0].out_features + 1),
                                        nn.BatchNorm1d(num_features=self.bottleneck[0].out_features + 1, affine=False)])
        # Copying weights while freezing old neurons
        with torch.no_grad():
            new_bottleneck[0].weight[: self.bottleneck[0].out_features] = self.bottleneck[0].weight
            new_bottleneck[0].bias[: self.bottleneck[0].out_features] = self.bottleneck[0].bias

        self.bottleneck = new_bottleneck  # Replace the layer
        self.bottleneck[0].requires_grad_(True)  # Allow gradients

        # Freeze the old neurons using a hook
        self.bottleneck[0].weight.register_hook(self._freeze_old_neurons_hook)
        self.bottleneck[0].bias.register_hook(self._freeze_old_neurons_hook)

        # Turn off gradients for all layers in the encoder (just in case)
        for layer in self.encoder:
            for param in layer.parameters():
                param.requires_grad = False

        self._recreate_decoder()

    def _freeze_old_neurons_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook: Freeze gradients for old neurons, allowing updates only for new ones"""
        grad[: -1] = 0  # Zero out gradients for old neurons
        return grad

    def _recreate_decoder(self):
        # Copying old decoder to new
        new_decoder = nn.ModuleList()
        for i, layer in enumerate(self.decoder):
            if i == 0 and isinstance(layer, nn.Linear):
                new_layer = nn.Linear(layer.in_features + 1, layer.out_features)
                nn.init.xavier_uniform_(new_layer.weight)
                if new_layer.bias is not None:
                    nn.init.zeros_(new_layer.bias)
                new_decoder.append(new_layer)
            else:
                new_decoder.append(layer)
        
        self.decoder = new_decoder  # Ensure it's still a ModuleList

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        for layer in self.bottleneck:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x  # Return the output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encode(x)
        out = self.decode(enc)
        return out, enc
    
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss, recon_loss, cov_loss = self.__shared_eval_step(X, y, loss_func)
        if isinstance(loss_func, PCAAE_Loss):
            metrics = {"train loss": loss.item(),
                       "train reconstruction loss": recon_loss.item(),
                       "train covariance loss": cov_loss.item()}
        else:
            metrics = {"train loss": loss.item()}

        return loss, metrics


    def eval_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss, recon_loss, cov_loss = self.__shared_eval_step(X, y, loss_func)
        if isinstance(loss_func, PCAAE_Loss):
            metrics = {"val loss": loss.item(),
                       "val reconstruction loss": recon_loss.item(),
                       "val covariance loss": cov_loss.item()}
        else:
            metrics = {"val loss": loss.item()}

        return loss, metrics       
        

    def fit(self,
            optimizer: torch.optim.Optimizer,
            loss_func: Union[nn.Module, Callable],
            epochs: int,
            trainloader: DataLoader,
            testloader: DataLoader,
            goal_hidden_dim: int,
            verbose: int=2,
            print_every: Optional[int]=40,
            log_to_tensorboard: bool=True,
            callbacks: List[CallBack]=[],
            device: torch.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
            save_filename: Optional[str]=None,
            close_writer_on_end: bool=True,):
        
        self.to(device)
        trainer = Trainer(log_to_tensorboard=log_to_tensorboard, device=device)
        hidden_dim = 1
        while hidden_dim < goal_hidden_dim:
            if not trainer.epochs == 0:
                self.increase_latentdim()
                self.to(device)
                hidden_dim += 1
            if verbose > 0:
                print(f"Training with hidden dim: {hidden_dim}")
            for e in range(epochs):
                trainer.training_loop(model=self, trainloader=trainloader, loss_func=loss_func,
                                      optimizer=optimizer, verbose=verbose, epoch=e, print_every=print_every)
                trainer.validation_loop(model=self, testloader=testloader, loss_func=loss_func, verbose=verbose,
                                        epoch=e)
                if trainer.on_epoch_end(model=self, callbacks=callbacks, save_filename=f"{save_filename}_{hidden_dim}", verbose=verbose):
                    break
            trainer.on_training_end(model=self, callbacks=callbacks)
        if close_writer_on_end:
            trainer.writer.close()
        return trainer.logs
    
    
    def __shared_eval_step(self, 
                        X: torch.Tensor, 
                        y: torch.Tensor, 
                        loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        y_hat, hidden = self.forward(X)

        if isinstance(loss_func, PCAAE_Loss):
            loss, recon_loss, cov_loss = loss_func(y_hat, y, hidden)
        else:
            loss = loss_func(y_hat, y)

        return loss, recon_loss, cov_loss 



class PCAAE_Loss(nn.Module):
    def __init__(self, loss_func: Union[nn.Module, Callable], lambda_cov=0.01):
        super().__init__()
        self.loss_func = loss_func
        self.lambda_cov = lambda_cov

    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.loss_func(y_hat, y)

        batch_size, latent_dim = z.shape
        z_mean = torch.mean(z, dim=0, keepdim=True)
        z_centered = z - z_mean

        covariance_matrix = (z_centered.T @ z_centered) / batch_size
        covariance_loss = torch.sum(covariance_matrix**2) - torch.sum(torch.diagonal(covariance_matrix)**2)

        total_loss = recon_loss + self.lambda_cov * covariance_loss
        return total_loss, recon_loss, covariance_loss