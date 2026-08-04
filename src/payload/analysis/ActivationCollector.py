"""
Class which handles the weights, metrics, etc. of a trained classifier
"""

from pathlib import Path
import logging

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset

from payload.data.cifar10 import build_dataloader

logger = logging.getLogger(__name__)

class ActivationCollector():
    """
    A class which handles i/o of model weights, sets up hooks, and analyzes activations
    """
    def __init__(
            self,
            model: nn.Module,
            model_name: str,
            dataset: Dataset,
            device: str = "cpu",
            num_samples: int = 512
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self.probe_loader = build_dataloader(dataset, batch_size=1)
        self.activations = {} # {layer_name: [n_samples, n_channels, dim_x, dim_y]}

        self._hooks = []
        self._layer_dict = model.named_children()
        self._epoch = None
        self._lr = None
        self._temp_activations = {} # {layer_name: [n_samples, n_channels, dim_x, dim_y]}

    def import_pt(self, input_dir: str | Path):
        pt_dict = torch.load(Path(input_dir), map_location=self.device)
        model_weights = pt_dict["model_state"]
        self._epoch = pt_dict["metrics"]["epoch"]
        self._lr = pt_dict["metrics"]["lr"]
        self._assert_state_dict(self.model, model_weights)
        self.model.load_state_dict(model_weights)

    def compute_activations(self):
        self.model.to(self.device)
        self.model.eval()
        self._deploy_hooks()

        logger.info("beginning forward pass of data...")
        with torch.no_grad():
            for n, (x, _) in enumerate(self.probe_loader):
                if n > self.num_samples:
                    continue
                x = x.to(self.device)
                self.model(x)
                
        logger.info("collecting activations...")
        for name, actv_list in list(self._temp_activations.items()):
            if actv_list:
                self.activations[name] = torch.cat(actv_list, dim=0)
                del self._temp_activations[name] # deletes cache to free up memory

        self._remove_hooks()
        self._temp_activations.clear()
        
        return self.activations
    
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
        
    def _get_hook(self, name: str):
        def hook(model, input, output):
            self._temp_activations[name].append(output.detach().cpu())
        return hook
    
    def _deploy_hooks(self, layer_type: nn.Module = nn.Conv2d) -> None:
        for name, layer in self.model.named_modules():
            if isinstance(layer, layer_type):
                self._temp_activations[name] = []
                hook = layer.register_forward_hook(self._get_hook(name))
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
