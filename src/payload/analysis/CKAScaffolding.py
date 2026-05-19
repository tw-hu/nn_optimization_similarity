"""
Class which handles the weights, metrics, etc. of a trained classifier
"""

from pathlib import Path

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset

from payload.analysis.similarity import compute_cka

class CKAScaffolding():
    """
    A class which handles i/o of model weights, sets up hooks, and analyzes activations
    """
    def __init__(
            self,
            model: nn.Module,
            model_name: str,
            dataset: Dataset
    ):
        self.model = model
        self.model_name = model_name
        self.dataset = dataset

        self._hooks = []
        self._layer_dict = model.named_children()
        self._activations = {}

    def import_weights(self, input_dir: str | Path):
        model_weights = self._load_pt(input_dir)["model_state"]
        self._assert_state_dict(self.model, model_weights)
        self.model.load_state_dict(model_weights)

    def export_weights(self, name: str, output_dir: str | Path):
        model_weights = self.model.state_dict()
        output_path = Path(output_dir) / name
        torch.save(model_weights, output_path)

    def deploy_hooks(self) -> None:
        for name, layer in self.model.named_modules():
            handle = layer.register_forward_hook(self._get_activation(f"{name}_out"))
            self._hooks.append(handle)
    
    def get_activation_at_layer(self, layer_num: int = None, layer_name: str = None):
        assert (layer_num is None) != (layer_name is None), "Provide either 'layer_num' or 'layer_name' but not both."
        if layer_num is not None:
            return list(self._activations.items())[layer_num]
        if layer_name is not None:
            return self._activations[layer_name]
        
    def get_activations(self, dataset: Dataset = None):
        data = self.dataset or dataset
        input = x
        with torch.no_grad():
            output = self.model(input)
        
    def save_activations(self, output_dir: str | Path):
        path = Path(output_dir) / self.model_name
        torch.save(self._activations, path)

    @property
    def activations(self):
        return self._activations
    
    @property
    def layer_dict(self):
        return self._layer_dict

    def _assert_state_dict(self, model, state_dict) -> None:
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                assert value.shape == model_state[key].shape, \
                    f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
            else:
                print(f"Warning: {key} not found in model state_dict")

    def _get_activation(self, name: str):
        def hook(model, input, output):
            self._activations[name] = output.detach()
        return hook

    def _load_pt(self, input_dir: str | Path) -> OrderedDict:
        path = Path(input_dir)
        contents = torch.load(path, weights_only=True)
        return contents