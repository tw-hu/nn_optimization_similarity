"""
Convolutional neural network architectures used in this analysis
"""
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

# def get_activation(activation: str = 'relu'):
#     activations = {
#         'relu': nn.ReLU,
#         'sigmoid': nn.Sigmoid,
#         'tanh': nn.Tanh,
#     }
#     if activation.lower() not in activations:
#         raise ValueError("Activation must be one of 'relu', 'sigmoid', or 'tanh'")
#     return activations[activation.lower()]

# O = floor((I - K + 2*P)/S) + 1
def build_conv_encoder():
    # activation = get_activation(act)
    model = nn.Sequential(OrderedDict([ # (bs, 3, 32, 32)
        ("enc_block1", nn.Sequential(
            OrderedDict([
                ("conv1_1", nn.Conv2d(3, 64, 3, padding=1)), # (bs, 64, 32, 32)
                ("bnor1_1", nn.BatchNorm2d(64)),
                ("relu1_1", nn.ReLU(inplace=True)),
                ("conv1_2", nn.Conv2d(64, 64, 3, padding=1)), # (bs, 64, 32, 32)
                ("bnor1_2", nn.BatchNorm2d(64)),
                ("relu1_2", nn.ReLU(inplace=True)),
            ])
        )),
        ("enc_block2", nn.Sequential(
            OrderedDict([
                ("conv2_1", nn.Conv2d(64, 128, 3, padding=1, stride=2)), # (bs, 128, 16, 16)
                ("bnor2_1", nn.BatchNorm2d(128)),
                ("relu2_1", nn.ReLU(inplace=True)),
                ("conv2_2", nn.Conv2d(128, 128, 3, padding=1)), # (bs, 128, 16, 16)
                ("bnor2_2", nn.BatchNorm2d(128)),
                ("relu2_2", nn.ReLU(inplace=True)),
                ("conv2_3", nn.Conv2d(128, 128, 3, padding=1)), # (bs, 128, 16, 16)
                ("bnor2_3", nn.BatchNorm2d(128)),
                ("relu2_3", nn.ReLU(inplace=True))
            ])
        )),
        ("enc_block3", nn.Sequential(
            OrderedDict([
                ("conv3_1", nn.Conv2d(128, 256, 3, padding=1, stride=2)), # (bs, 256, 8, 8)
                ("bnor3_1", nn.BatchNorm2d(256)),
                ("relu3_1", nn.ReLU(inplace=True)),
                ("conv3_2", nn.Conv2d(256, 256, 3, padding=1)), # (bs, 256, 8, 8)
                ("bnor3_2", nn.BatchNorm2d(256)),
                ("relu3_2", nn.ReLU(inplace=True)),
                ("conv3_3", nn.Conv2d(256, 256, 3, padding=1)), # (bs, 256, 8, 8)
                ("bnor3_3", nn.BatchNorm2d(256)),
                ("relu3_3", nn.ReLU(inplace=True))
            ])
        )),
        ("enc_block4", nn.Sequential(
            OrderedDict([
                ("conv4_1", nn.Conv2d(256, 512, 3, padding=1, stride=2)), # (bs, 512, 4, 4)
                ("bnor4_1", nn.BatchNorm2d(512)),
                ("relu4_1", nn.ReLU(inplace=True)),
                ("conv4_2", nn.Conv2d(512, 512, 3, padding=1)), # (bs, 512, 4, 4)
                ("bnor4_2", nn.BatchNorm2d(512)),
                ("relu4_2", nn.ReLU(inplace=True)),
                ("conv4_3", nn.Conv2d(512, 512, 3, padding=1)), # (bs, 512, 4, 4)
                ("bnor4_3", nn.BatchNorm2d(512)),
                ("relu4_3", nn.ReLU(inplace=True))
            ])
        ))
    ]))
    return model

def build_conv_decoder():
    # activation = get_activation(act)
    model = nn.Sequential(OrderedDict([ # (bs, 512, 3, 3)
        ("dec_block1", nn.Sequential(
            OrderedDict([
                ("conv5_1", nn.ConvTranspose2d(512, 512, 3, padding=1)), # (bs, 512, 3, 3)
                ("bnor5_1", nn.BatchNorm2d(512)),
                ("relu5_1", nn.ReLU(inplace=True)),
                ("conv5_2", nn.ConvTranspose2d(512, 512, 3, padding=1)), # (bs, 512, 3, 3)
                ("bnor5_2", nn.BatchNorm2d(512)),
                ("relu5_2", nn.ReLU(inplace=True)),
                ("conv5_3", nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)), # (bs, 256, 7, 7)
                ("bnor5_3", nn.BatchNorm2d(256)),
                ("relu5_3", nn.ReLU(inplace=True))
            ])
        )),
        ("dec_block2", nn.Sequential(
            OrderedDict([
                ("conv6_1", nn.ConvTranspose2d(256, 256, 3, padding=1)), # (bs, 256, 7, 7)
                ("bnor6_1", nn.BatchNorm2d(256)),
                ("relu6_1", nn.ReLU(inplace=True)),
                ("conv6_2", nn.ConvTranspose2d(256, 256, 3, padding=1)), # (bs, 256, 7, 7)
                ("bnor6_2", nn.BatchNorm2d(256)),
                ("relu6_2", nn.ReLU(inplace=True)),
                ("conv6_3", nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)), # (bs, 256, 7, 7)
                ("bnor6_3", nn.BatchNorm2d(128)),
                ("relu6_3", nn.ReLU(inplace=True))
            ])
        )),
        ("dec_block3", nn.Sequential(
            OrderedDict([
                ("conv7_1", nn.ConvTranspose2d(128, 128, 3, padding=1)), # (bs, 512, 3, 3)
                ("bnor7_1", nn.BatchNorm2d(128)),
                ("relu7_1", nn.ReLU(inplace=True)),
                ("conv7_2", nn.ConvTranspose2d(128, 128, 3, padding=1)),
                ("bnor7_2", nn.BatchNorm2d(128)),
                ("relu7_2", nn.ReLU(inplace=True)),
                ("conv7_3", nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                ("bnor7_3", nn.BatchNorm2d(64)),
                ("relu7_3", nn.ReLU(inplace=True))
            ])
        )),
        ("dec_block4", nn.Sequential(
            OrderedDict([
                ("conv8_1", nn.ConvTranspose2d(64, 64, 3, padding=1)), # (bs, 64, 32, 32)
                ("bnor8_1", nn.BatchNorm2d(64)),
                ("relu8_1", nn.ReLU(inplace=True)),
                ("conv8_2", nn.ConvTranspose2d(64, 3, 3, padding=1)), # (bs, 64, 32, 32)
            ])
        ))
    ]))
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

def build_model():
    enc_model = build_conv_encoder()
    dec_model = build_conv_decoder()
    return ConvAutoencoder(enc_model, dec_model)

class ConvAutoencoder(nn.Module):
    """ConvAutoencoder class is the wrapper for convolutional encoder and decoder layers"""

    def __init__(
        self,
        enc_model: nn.Module,
        dec_model: nn.Module
    ):
        super().__init__()
        self._enc_model = enc_model
        self._dec_model = dec_model

    def forward(self, x):
        r = self._enc_model(x)
        return self._dec_model(r)
    
    def load_enc_weights(self, enc_weights: OrderedDict) -> None:
        self._assert_state_dict(self._enc_model, enc_weights)
        self._enc_model.load_state_dict(enc_weights)

    def load_fc_weights(self, dec_weights: OrderedDict) -> None:
        self._assert_state_dict(self._dec_model, dec_weights)
        self._dec_model.load_state_dict(dec_weights)
    
    @property
    def enc_model(self):
        return self._enc_model
    
    @property
    def dec_model(self):
        return self._dec_model
    
    def _assert_state_dict(model, state_dict) -> None:
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                assert value.shape == model_state[key].shape, \
                    f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
            else:
                print(f"Warning: {key} not found in model state_dict")