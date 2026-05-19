"""
Script to extract activations from trained neural networks
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from payload.analysis.CKAScaffolding import CKAScaffolding
from payload.data.cifar10 import build_cifar, build_dataloader
from payload.models.ConvClassifier import build_model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset = build_cifar(Path(cfg.data_dir), "val", mode=cfg.mode.mode)
    dataloader = build_dataloader(dataset, batch_size=len(dataset))
    model = build_model(cfg.mode.mode, activation=cfg.activation)
    scaffolding = CKAScaffolding(model, dataset)

    input_dir = ""
    scaffolding.import_weights(input_dir)
    scaffolding.deploy_hooks()
    
    # for name, layer in model.layer_dict:



    scaffolding.save_activations()

if __name__ == "__main__":
    main()