"""Dataset registry: maps a --dataset name to the module that turns data/raw/<RAW_DIRNAME> into data/processed/<name>.

Each dataset module provides:
    RAW_DIRNAME                                  folder name under data/raw/
    prepare(raw_dir, out_dir, seed, val_frac)    raw -> processed (see common.py for the output format)

To add a dataset: write a module next to this one and add a line to REGISTRY.
"""
from . import ciciomt2024

REGISTRY = {
    "ciciomt2024": ciciomt2024,
}
