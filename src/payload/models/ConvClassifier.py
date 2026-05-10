"""
Convolutional neural network architectures used in this analysis
"""

import numpy as np

import torch
import torch.nn as nn

# O = floor((I - K + 2*P)/S) + 1

def smallCNN(activation: nn.Module = nn.ReLU, activation_kwargs: dict = None): # for testing purposes
    model = nn.Sequential( # input: (bs, 3, 32, 32)
        nn.Conv2d(3, 8, 5, padding="full"), # (bs, 3, 32, 32)
        activation(**activation_kwargs),
        nn.MaxPool2d(2, 2), # (bs, 8, 16, 16)
        nn.Conv2d(8, 16, 5, padding="full"), # (bs, 16, 16, 16)
        activation(**activation_kwargs),
        nn.MaxPool2d(2, 2) # (bs, 16, 8, 8)
    )
    return model

def largeCNN(activation: nn.Module = nn.ReLU, activation_kwargs: dict = None):
    model = nn.Sequential( # input: (bs, 3, 32, 32)
        nn.Sequential(
            nn.Conv2d(3, 24, 3, padding="same"), # (bs, 24, 32, 32)
            activation(**activation_kwargs)
        ),
        nn.Sequential(
            nn.Conv2d(24, 64, 3, padding="same"), # (bs, 64, 32, 32)
            nn.MaxPool2d(2, 2), # (bs, 64, 16, 16)
            nn.BatchNorm2d(64),
            activation(**activation_kwargs),
            nn.Conv2d(64, 256, 3, padding="same"), # (bs, 256, 16, 16)
            nn.MaxPool2d(2, 2), # (bs, 256, 8, 8)
            nn.BatchNorm2d(64),
            activation(**activation_kwargs)
        ),
        nn.Sequential(
            nn.Conv2d(64, 256, 3, padding="same"), # (bs, 256, 8, 8)
            nn.MaxPool2d(2, 2), # (bs, 256, 8, 8)
            nn.BatchNorm2d(256),
            activation(**activation_kwargs),
            nn.Conv2d(256, 256, 3, padding="same"), # (bs, 256, 4, 4)
            nn.BatchNorm2d(256),
            activation(**activation_kwargs)
        )
    )
    return model

def build_mlp(layer_dims: list[int] | np.array, activation: str = "relu"):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
    }
    if activation.lower() not in activations:
        raise ValueError("Activation must be one of 'relu', 'sigmoid', or 'tanh'")
    activation_fn = activations[activation.lower()]

    layers = []
    layer_dims = np.array(layer_dims)
    for l in len(layer_dims):
        layers.append(nn.Linear(layer_dims[l], layer_dims[l+1]))
        if l < len(layers) - 2:
            layers.append(activation_fn())

    return nn.Sequential(*layers)

class ConvClassifier(nn.Module):
    """ConvClassifier class is the wrapper for Conv and FC layers
    
    Keyword arguments:
    """
    def __init__(
        self,
        conv_model: nn.Module = smallCNN,
        fc_model: nn.Module = build_mlp([1024, 10])
    ):
        self._conv_model = conv_model
        self._fc_model = fc_model

    def forward(self, x):
        r = self.conv_model(x).flatten(1)
        return self.fc_model(r)
    
    @property
    def conv_model(self):
        return self._conv_model
    
    @property
    def fc_model(self):
        return self._fc_model

