import torch
import torch.nn as nn
from typing import Tuple, Union, Callable

class PCAAutoencoder(nn.Module):
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