import torch
import torch.nn as nn
from typing import Tuple, Union, Callable, Optional, List
from modules.modelwithtrainer.trainer import ModelWithTrainer, Trainer
from modules.modelwithtrainer.callbacks import CallBack
from torch.utils.data import DataLoader
from .ae_utils import remap_metadata_pt


class AutoEncoder(ModelWithTrainer):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
    
    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
    
    def forward(self, x):
        enc = self.encode(x)
        out = self.decode(enc)
        return out, enc
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss = self._shared_eval_step(X, y, loss_func)
        metrics = {"train loss": loss.item()}
        return loss, metrics


    def eval_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss = self._shared_eval_step(X, y, loss_func)
        metrics = {"val loss": loss.item()}
        return loss, metrics
    

    def create_hidden(self,
                      dataloader: DataLoader,
                      categorymapper: Optional[dict],
                      device: str) -> Tuple[list, list, list]:
        
        self.eval()
        all_hidden_states = []
        all_metadata = {}
        with torch.no_grad():
            for X, _ in dataloader:
                metadata = remap_metadata_pt(X, categorymapper)
                for key, value in metadata.items():
                    if key in all_metadata:
                        all_metadata[key] += (value)
                    else:
                        all_metadata[key] = value
                X = X.to(device)
                embedding = self.encode(X)
                all_hidden_states.append(embedding.cpu().detach())
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
            metadata_headers = list(all_metadata.keys())
            metadata_values = list(zip(*all_metadata.values()))
            return all_hidden_states.clone().detach(), metadata_headers, metadata_values


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
            embed_hidden: List[DataLoader]=[],
            categorymapper: Optional[dict]=None,
            device: torch.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
            save_filename: Optional[str]=None,
            close_writer_on_end: bool=True) -> dict:
            
            self.to(device)
            trainer = Trainer(log_to_tensorboard=log_to_tensorboard, device=device)
            if not trainer.epochs == 0:
                self.to(device)
            for e in range(epochs):
                trainer.training_loop(model=self, trainloader=trainloader, loss_func=loss_func,
                                    optimizer=optimizer, verbose=verbose, epoch=e, print_every=print_every)
                trainer.validation_loop(model=self, testloader=testloader, loss_func=loss_func, verbose=verbose,
                                        epoch=e)
                if trainer.on_epoch_end(model=self, callbacks=callbacks, save_filename=save_filename, verbose=verbose):
                    break

            trainer.on_training_end(model=self, callbacks=callbacks)

            if embed_hidden and categorymapper is not None:
                all_hidden_states = []
                all_metadata_values = []
                for loader in embed_hidden:
                    hidden_states, metadata_headers, metadata_values = self.create_hidden(loader, categorymapper, device)
                    all_hidden_states.append(hidden_states)
                    all_metadata_values += metadata_values

                all_hidden_states = torch.cat(all_hidden_states, dim=0)
                trainer.writer.add_embedding(all_hidden_states, metadata=all_metadata_values,
                                            global_step=1, metadata_header=metadata_headers,
                                            tag=f"autoencoder")
                
            if close_writer_on_end:
                trainer.writer.close()
            return trainer.logs
        

    def _shared_eval_step(self, 
                          X: torch.Tensor, 
                          y: torch.Tensor, 
                          loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        y_hat, hidden = self.forward(X)
        loss = loss_func(y_hat, y)
        return loss
    


class PCAAutoEncoder(AutoEncoder):
    def __init__(self, encoder: nn.ModuleList, decoder: nn.ModuleList, last_hidden_shape: int):
        super().__init__(encoder, decoder)
        self.last_hidden_shape = last_hidden_shape
        self.bottleneck = nn.ModuleList([nn.Linear(in_features=self.last_hidden_shape, out_features=1),
                                         nn.BatchNorm1d(num_features=1, affine=False)])
        

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        X = super().encode(X)
        for layer in self.bottleneck:
            X = layer(X)
        return X
    

    def train_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss, recon_loss, cov_loss = self._shared_eval_step(X, y, loss_func)
        if isinstance(loss_func, PCAAE_Loss):
            metrics = {"train loss": loss.item(),
                       "train reconstruction loss": recon_loss.item(),
                       "train covariance loss": cov_loss.item()}
        else:
            metrics = {"train loss": loss.item()}

        return loss, metrics


    def eval_step(self, X: torch.Tensor, y: torch.Tensor, loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        loss, recon_loss, cov_loss = self._shared_eval_step(X, y, loss_func)
        if isinstance(loss_func, PCAAE_Loss):
            metrics = {"val loss": loss.item(),
                       "val reconstruction loss": recon_loss.item(),
                       "val covariance loss": cov_loss.item()}
        else:
            metrics = {"val loss": loss.item()}

        return loss, metrics
    

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
            embed_hidden: List[DataLoader]=[],
            categorymapper: Optional[dict]=None,
            device: torch.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
            save_filename: Optional[str]=None,
            close_writer_on_end: bool=True):
        
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

                if embed_hidden and categorymapper is not None:
                    all_hidden_states = []
                    all_metadata_values = []
                    for loader in embed_hidden:
                        hidden_states, metadata_headers, metadata_values = self.create_hidden(loader, categorymapper, device)
                        all_hidden_states.append(hidden_states)
                        all_metadata_values += metadata_values
                    all_hidden_states = torch.cat(all_hidden_states, dim=0)
                    trainer.writer.add_embedding(all_hidden_states, metadata=all_metadata_values,
                                                 global_step=hidden_dim, metadata_header=metadata_headers,
                                                 tag=f"pcaautoencoder_hidden_dim_{hidden_dim}")
                
            if close_writer_on_end:
                trainer.writer.close()
            return trainer.logs
    

    def _shared_eval_step(self, 
                          X: torch.Tensor, 
                          y: torch.Tensor, 
                          loss_func: Union[nn.Module, Callable]) -> Tuple[torch.Tensor, dict]:
        
        y_hat, hidden = self.forward(X)

        if isinstance(loss_func, PCAAE_Loss):
            loss, recon_loss, cov_loss = loss_func(y_hat, y, hidden)
        else:
            loss = loss_func(y_hat, y)
            recon_loss = 0
            cov_loss = 0

        return loss, recon_loss, cov_loss
    

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