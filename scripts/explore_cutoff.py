"""Compare alarm-cutoff rules (picked on val) on finished runs: python3 scripts/explore_cutoff.py --tag TAG"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def val_scores_for(run_dir: Path, config: dict, cache: dict) -> np.ndarray:
    """Load cached val scores, or rebuild the trained model and score val once."""
    path = run_dir / "val_scores.npy"
    if path.exists():
        return np.load(path)

    import torch
    from sklearn.preprocessing import StandardScaler
    from src import models
    from src.experiment import signed_log, to_split

    dataset = config["dataset"]
    if dataset not in cache:  # the same scaled val split serves every run on this dataset
        data_dir = ROOT / "data" / "processed" / dataset
        features = json.loads((data_dir / "meta.json").read_text())["features"]
        train = pd.read_parquet(data_dir / "train.parquet", columns=features)
        scaler = StandardScaler().fit(signed_log(train.to_numpy()))  # same scaling as in training
        del train
        cache[dataset] = (features, to_split(pd.read_parquet(data_dir / "val.parquet"), features, scaler))
    features, val = cache[dataset]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = models.get(config["model"])
    model = Model(n_features=len(features), device=device, **config["hparams"])
    model.net.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["state_dict"])
    scores = model.score(val)
    np.save(path, scores)
    return scores


def cutoffs(val_scores: np.ndarray, val_y: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(val_y, val_scores)
    f1 = (2 * precision * recall / np.maximum(precision + recall, 1e-12))[:-1]
    tied = thresholds[f1 >= f1.max() - 1e-9]
    benign = val_scores[val_y == 0]
    return {
        "current": float(tied[0]),
        "mid-tie": float((tied[0] + tied[-1]) / 2),
        "val-b99": float(np.quantile(benign, 0.99)),
        "val-b99.9": float(np.quantile(benign, 0.999)),
    }


def caught_at(test_scores: np.ndarray, test_y: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(test_y, test_scores)
    return float(tpr[fpr <= max_fpr].max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="runs whose folder ends in _<tag>")
    args = parser.parse_args()

    run_dirs = sorted((ROOT / "results").glob(f"*_{args.tag}"))
    if not run_dirs:
        raise SystemExit(f"No runs tagged {args.tag!r} in results/")

    cache, labels, rows = {}, {}, []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        dataset = config["dataset"]
        if dataset not in labels:
            d = ROOT / "data" / "processed" / dataset
            labels[dataset] = {s: pd.read_parquet(d / f"{s}.parquet", columns=["label"])["label"].to_numpy() for s in ["val", "test"]}
        val_y, test_y = labels[dataset]["val"], labels[dataset]["test"]

        print(f"{run_dir.name}")
        val_scores = val_scores_for(run_dir, config, cache)
        test_scores = np.load(run_dir / "test_scores.npy")

        row = {"model": config["model"], "seed": config["seed"],
               "caught@1%": caught_at(test_scores, test_y, 0.01), "caught@0.1%": caught_at(test_scores, test_y, 0.001)}
        for rule, cut in cutoffs(val_scores, val_y).items():
            pred = test_scores >= cut
            row[f"{rule} cutoff"] = cut
            row[f"{rule} false alarms %"] = 100 * pred[test_y == 0].mean()
            row[f"{rule} missed %"] = 100 * (1 - pred[test_y == 1].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    order = [m for m in ["mamba3", "mamba2", "transformer"] if m in set(df["model"])]
    means = df.drop(columns="seed").groupby("model").mean().reindex(order)
    pd.set_option("display.width", 200)

    print("\nSorting quality (no cutoff rule involved), mean over seeds, % of test attacks caught:")
    print((100 * means[["caught@1%", "caught@0.1%"]]).round(2).to_string())

    for rule in ["current", "mid-tie", "val-b99", "val-b99.9"]:
        cols = [f"{rule} cutoff", f"{rule} false alarms %", f"{rule} missed %"]
        print(f"\nRule '{rule}', mean over seeds:")
        print(means[cols].rename(columns=lambda c: c.removeprefix(f"{rule} ")).round(4).to_string())

    out = ROOT / "results" / "figures" / args.tag / "cutoff_rules.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPer-run numbers saved to {out}")


if __name__ == "__main__":
    main()
