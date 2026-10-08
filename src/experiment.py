"""One experiment = for each seed, train a model (src/train.py) then test it (src/test.py, which scores with
src/evaluate.py). No logic of its own: it only runs those steps in order.

    python3 run.py --task experiment --dataset ciciomt2024 --model mamba2 --seeds 1 2 3 --tag final --lr 3e-3
"""
from pathlib import Path

from src.test import test
from src.train import train


def run_experiment(dataset: str, model_name: str, hparams: dict, seeds: list[int], device: str, tag: str,
                   processed_dir: Path, results_dir: Path) -> list[Path]:
    """Returns the new run folders, one per seed."""
    run_dirs = []
    for seed in seeds:
        print(f"\n=== {dataset}, {model_name}, seed {seed} ===")
        run_dir = train(dataset, model_name, hparams, seed, device, tag, processed_dir, results_dir)
        print()
        test(run_dir, device, processed_dir)
        run_dirs.append(run_dir)
    return run_dirs
