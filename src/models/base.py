"""The blueprint every model follows: supervised flag, default hyperparameters, fit(), score().

experiment.py only talks to models through this interface, so any model can be swapped in
with --model. Hyperparameters listed in default_hparams automatically become CLI flags:
    default_hparams = {"hidden_dim": 64}   ->   --hidden-dim 128
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Split:
    """One split (train/val/test), already scaled and ready for a model."""
    X: np.ndarray         # (n_rows, n_features) float32
    y: np.ndarray         # (n_rows,) 0 = benign, 1 = attack
    attack: np.ndarray    # (n_rows,) attack name, for per-attack reporting
    stream: np.ndarray    # (n_rows,) id of the capture the row came from
    position: np.ndarray  # (n_rows,) row's position within its capture (time order)


class Detector:
    # True: trains on benign + attack rows with labels.
    # False: trains on benign rows only and flags whatever looks unlike them.
    supervised: bool = True
    default_hparams: dict = {}

    def __init__(self, n_features: int, device: str, **hparams):
        self.n_features = n_features
        self.device = device
        self.hp = {**self.default_hparams, **hparams}

    def fit(self, train: Split, val: Split) -> None:
        raise NotImplementedError

    def score(self, data: Split) -> np.ndarray:
        """One anomaly score per row; higher = more likely an attack."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Optional: write the trained model into the run's results folder."""
