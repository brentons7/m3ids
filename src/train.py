"""Train one model: load the processed splits -> scale -> train -> save to a new results/<run>/ folder.

    python3 run.py train --dataset ciciomt2024 --model mamba2 --seed 1 --tag final --lr 3e-3

Writes config.json (dataset, model, seed, every hyperparameter), model.pt (the trained weights) and metrics.json
with the training facts (n_params, train_seconds). src/test.py then scores the run and adds its metrics.
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import models
from src.models.sequence import Split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signed_log(x: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed features (rates in the millions next to 0/1 flags) before standardizing."""
    return np.sign(x) * np.log1p(np.abs(x))


def to_split(df: pd.DataFrame, features: list[str], scaler: StandardScaler) -> Split:
    return Split(
        X=scaler.transform(signed_log(df[features].to_numpy())).astype(np.float32),
        y=df["label"].to_numpy().astype(np.int64),
        attack=df["attack"].astype(str).to_numpy(),
        stream=df["source_file"].cat.codes.to_numpy(),
        position=df["row_in_file"].to_numpy(),
    )


def load_splits(processed_dir: Path, dataset: str,
                names: tuple[str, ...] = ("train", "val", "test")) -> tuple[dict[str, Split], dict]:
    """The requested splits, scaled. Scaling is always fitted on the training rows only, never on val/test,
    so train and test (which refits it) get exactly the same scaling."""
    data_dir = processed_dir / dataset
    meta = json.loads((data_dir / "meta.json").read_text())
    features = meta["features"]
    frames = {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in {"train", *names}}
    scaler = StandardScaler().fit(signed_log(frames["train"][features].to_numpy()))
    splits = {name: to_split(frames[name], features, scaler) for name in names}
    print(f"{dataset}: " + ", ".join(f"{k} {len(s.y):,} rows" for k, s in splits.items()))
    return splits, meta


def train(dataset: str, model_name: str, hparams: dict, seed: int, device: str, tag: str,
          processed_dir: Path, results_dir: Path) -> Path:
    """Train and save one run; returns its folder."""
    set_seed(seed)
    Model = models.get(model_name)
    splits, meta = load_splits(processed_dir, dataset)

    model = Model(n_features=len(meta["features"]), device=device, **hparams)
    print(f"Training {model_name} with {model.hp}")
    t0 = time.time()
    model.fit(splits["train"], splits["val"])
    train_seconds = time.time() - t0

    # Save everything needed to trace a number in the paper back to exactly how it was produced.
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{dataset}_{model_name}" + (f"_{tag}" if tag else "")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    config = {"dataset": dataset, "model": model_name, "seed": seed, "device": device, "tag": tag, "hparams": model.hp}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    model.save(run_dir)
    metrics = {"n_params": model.n_params(), "train_seconds": round(train_seconds, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTrained in {train_seconds / 60:.1f} min ({metrics['n_params']:,} params). Saved to {run_dir}")
    return run_dir
