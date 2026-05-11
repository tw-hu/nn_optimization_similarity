"""
Convolutional neural network architectures used in this analysis
"""
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

def get_activation(activation: str = "relu"):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
    }
    if activation.lower() not in activations:
        raise ValueError("Activation must be one of 'relu', 'sigmoid', or 'tanh'")
    return activations[activation.lower()]

# O = floor((I - K + 2*P)/S) + 1

def build_small_cnn(act: str = "relu"): # for testing purposes
    activation = get_activation(act)
    model = nn.Sequential(OrderedDict([ # input: (bs, 3, 32, 32)
        ("conv1", nn.Conv2d(3, 16, 3, padding="same")), # (bs, 16, 32, 32)
        ("relu1", activation()),
        ("pool1", nn.MaxPool2d(4, 4)), # (bs, 16, 8, 8)
        ("conv2", nn.Conv2d(16, 32, 3, padding="same")), # (bs, 32, 8, 8)
        ("relu2", activation()),
        ("pool2", nn.MaxPool2d(4, 4))# (bs, 32, 2, 2)
    ]))
    return model

def build_large_cnn(act: str = "relu"):
    activation = get_activation(act)
    model = nn.Sequential(OrderedDict([ # input: (bs, 3, 32, 32)
        ("conv_block1", nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 24, 3, padding="same")), # (bs, 24, 32, 32)
                ("actv1", activation())
            ])
        )),
        ("conv_block2", nn.Sequential(
            OrderedDict([
                ("conv2", nn.Conv2d(24, 64, 3, padding="same")), # (bs, 64, 32, 32)
                ("pool2", nn.MaxPool2d(2, 2)), # (bs, 64, 16, 16)
                ("bnor2", nn.BatchNorm2d(64)),
                ("actv2", activation()),
                ("conv3", nn.Conv2d(64, 128, 3, padding="same")), # (bs, 128, 16, 16)
                ("pool3", nn.MaxPool2d(2, 2)), # (bs, 128, 8, 8)
                ("bnor3", nn.BatchNorm2d(128)),
                ("actv3", activation())
            ])
        )),
        ("conv_block3", nn.Sequential(
            OrderedDict([
                ("conv4", nn.Conv2d(128, 256, 3, padding="same")), # (bs, 256, 8, 8)
                ("pool4", nn.MaxPool2d(2, 2)), # (bs, 256, 4, 4)
                ("bnor4", nn.BatchNorm2d(256)),
                ("actv4", activation()),
                ("conv5", nn.Conv2d(256, 256, 3, padding="same")), # (bs, 256, 4, 4)
                ("pool5", nn.MaxPool2d(2, 2)), # (bs, 256, 2, 2)
                ("bnor5", nn.BatchNorm2d(256)),
                ("actv5", activation())
            ])
        ))])
    )
    return model

def build_mlp(layer_dims: list[int], act: str = "relu"):
    activation = get_activation(act)
    layers = OrderedDict()
    layer_dims = np.array(layer_dims)
    for l in range(len(layer_dims)-1):
        layers[f"fc{l}"] = nn.Linear(layer_dims[l], layer_dims[l+1])
        if l < len(layers) - 2:
            layers[f"fc_actv{l}"] = activation()
    return nn.Sequential(layers)

class ConvClassifier(nn.Module):
    """ConvClassifier class is the wrapper for Conv and FC layers
    
    Keyword arguments:
    """
    def __init__(
        self,
        conv_model: nn.Module = build_small_cnn(),
        fc_model: nn.Module = build_mlp([1024, 10])
    ):
        super().__init__()
        self._conv_model = conv_model
        self._fc_model = fc_model

    def forward(self, x):
        r = self.conv_model(x).flatten(1)
        return self.fc_model(r)
    
    def load_conv_weights(self, conv_weights: OrderedDict) -> None:
        self._assert_state_dict(self._conv_model, conv_weights)
        self._conv_model.load_state_dict(conv_weights)

    def load_fc_weights(self, fc_weights: OrderedDict) -> None:
        self._assert_state_dict(self._fc_model, fc_weights)
        self._fc_model.load_state_dict(fc_weights)
    
    @property
    def conv_model(self):
        return self._conv_model
    
    @property
    def fc_model(self):
        return self._fc_model
    
    def _assert_state_dict(model, state_dict) -> None:
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                assert value.shape == model_state[key].shape, \
                    f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
            else:
                print(f"Warning: {key} not found in model state_dict")