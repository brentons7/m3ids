"""Model registry: maps a --model name to its class.

Entries are "module:Class" strings, imported only when that model is used, so a machine
without e.g. mamba-ssm installed can still run every other model.

To add a model: write a Detector subclass (see base.py) and add a line to REGISTRY.
"""
import importlib

REGISTRY = {
    "mamba3": "src.models.mamba3:Mamba3Detector",
}


def get(name: str):
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)
