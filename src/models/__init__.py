"""--model name -> "module:Class", imported lazily so missing mamba-ssm only breaks the Mamba models."""
import importlib

REGISTRY = {
    "mamba3": "src.models.mamba3:Mamba3Detector",
    "mamba2": "src.models.mamba2:Mamba2Detector",
    "transformer": "src.models.transformer:TransformerDetector",
}


def get(name: str):
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)
