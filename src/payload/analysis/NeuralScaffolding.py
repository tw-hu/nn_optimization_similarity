"""
Class which handles the weights, metrics, etc. of a trained classifier
"""

from pathlib import Path

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset

class NeuralScaffolding():
    """
    A class which handles i/o of model weights, sets up hooks, and analyzes activations
    """
    def __init__(
            self,
            model: nn.Module,
            dataset: Dataset
    ):
        self.model = model
        self.dataset = dataset
        self.activations = {}

        # set up forward hooks in self.model

    def import_weights(self, input_dir: str | Path):
        model_weights = self._load_pt(input_dir)["model_state"]
        self._assert_state_dict(self.model, model_weights)
        self.model.load_state_dict(model_weights)

    def export_weights(self, name: str, output_dir: str | Path):
        model_weights = self.model.state_dict()
        output_path = Path(output_dir) / name
        torch.save(model_weights, output_path)

    ## maybe set up hooks automatically when importing model, then just have a method to do forward pass through neural network
    def compute_activation_at_layer(self, layer_num: int = None, layer_name: str = None):
        assert (layer_num is None) != (layer_name is None), "Provide either 'layer_num' or 'layer_name' but not both."
        if layer_num is not None:
            activation = 0
        if layer_name is not None:
            layer = getattr(self.model, layer_name)
            layer.register_forward_hook(get_activation(f"{layer_name}_out"))
        return activation
    
    def get_activation_at_layer(self, layer_num: int = None, layer_name: str = None):
        assert (layer_num is None) != (layer_name is None), "Provide either 'layer_num' or 'layer_name' but not both."
        if layer_num is not None:
            return list(self.activations.items())[layer_num]
        if layer_name is not None:
            return self.activations[layer_name]

    @property
    def get_activations(self):
        return self.activations

    def _assert_state_dict(model, state_dict) -> None:
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                assert value.shape == model_state[key].shape, \
                    f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
            else:
                print(f"Warning: {key} not found in model state_dict")

    def _get_activation(name: str):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook


    def _load_pt(self, input_dir: str | Path) -> OrderedDict:
        path = Path(input_dir)
        contents = torch.load(path, weights_only=True)
        return contents