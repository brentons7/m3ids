"""One experiment: load processed splits -> scale -> build model -> train -> score -> evaluate -> save to results/."""
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import evaluate, models
from src.models.base import Split


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


def run_experiment(
    dataset: str, model_name: str, hparams: dict, seed: int, device: str, tag: str,
    processed_dir: Path, results_dir: Path,
) -> dict:
    set_seed(seed)
    data_dir = processed_dir / dataset
    features = json.loads((data_dir / "meta.json").read_text())["features"]
    frames = {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in ["train", "val", "test"]}

    Model = models.get(model_name)
    # Unsupervised models learn "normal" from benign rows only.
    if not Model.supervised:
        frames["train"] = frames["train"][frames["train"]["label"] == 0]
    # Scaling is fitted on the rows the model trains on, never on val/test.
    scaler = StandardScaler().fit(signed_log(frames["train"][features].to_numpy()))
    splits = {name: to_split(df, features, scaler) for name, df in frames.items()}
    print(f"{dataset}: " + ", ".join(f"{k} {len(s.y):,} rows" for k, s in splits.items()))

    model = Model(n_features=len(features), device=device, **hparams)
    print(f"Training {model_name} with {model.hp}")
    t0 = time.time()
    model.fit(splits["train"], splits["val"])
    train_seconds = time.time() - t0

    val_scores = model.score(splits["val"])
    t0 = time.time()
    test_scores = model.score(splits["test"])
    infer_us_per_row = (time.time() - t0) / len(test_scores) * 1e6

    metrics = evaluate.evaluate(val_scores, splits["val"].y, test_scores, splits["test"].y, splits["test"].attack)
    metrics["train_seconds"] = round(train_seconds, 1)
    metrics["infer_us_per_row"] = round(infer_us_per_row, 3)

    # Save everything needed to trace a number in the paper back to exactly how it was produced.
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{dataset}_{model_name}" + (f"_{tag}" if tag else "")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    config = {"dataset": dataset, "model": model_name, "seed": seed, "device": device, "tag": tag, "hparams": model.hp}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(run_dir / "test_scores.npy", test_scores)
    model.save(run_dir)
    append_summary(results_dir / "summary.csv", run_id, config, metrics)

    print(f"\nResults saved to {run_dir}")
    for k in ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr", "balanced_accuracy", "train_seconds", "infer_us_per_row"]:
        print(f"  {k:18s} {metrics[k]:.4f}")
    print("  flagged as attack, per class:")
    for a, rate in metrics["flagged_rate_per_class"].items():
        print(f"    {a:26s} {rate:.4f}")
    return metrics


SUMMARY_COLUMNS = ["run_id", "dataset", "model", "seed", "tag", "roc_auc", "pr_auc", "f1", "precision",
                   "recall", "fpr", "balanced_accuracy", "train_seconds", "infer_us_per_row", "hparams"]


def append_summary(path: Path, run_id: str, config: dict, metrics: dict) -> None:
    """One row per run in results/summary.csv, for comparing models at a glance."""
    row = {"run_id": run_id, **config, **metrics, "hparams": json.dumps(config["hparams"])}
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
