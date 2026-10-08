"""--model name -> "module:Class", imported lazily so missing mamba-ssm only breaks the Mamba models."""
import importlib
import json
from pathlib import Path

REGISTRY = {
    "mamba3": "src.models.mamba3:Mamba3Detector",
    "mamba2": "src.models.mamba2:Mamba2Detector",
    "transformer": "src.models.transformer:TransformerDetector",
}


def get(name: str):
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)


def load(run_dir: Path, n_features: int, device: str):
    """Rebuild a trained model from its run folder (config.json + saved weights), ready to score."""
    config = json.loads((run_dir / "config.json").read_text())
    model = get(config["model"])(n_features=n_features, device=device, **config["hparams"])
    model.load(run_dir)
    return model
