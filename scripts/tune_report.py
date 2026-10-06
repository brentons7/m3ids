"""Compare tuning runs on VAL (never test): python3 scripts/tune_report.py --tag tune_lr

Per model, ranks configs by val balanced accuracy and prints the winner. Only the hyperparameters that
differ between runs are shown.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    rows = []
    for d in sorted((ROOT / "results").glob(f"*_{args.tag}")):
        cfg = json.loads((d / "config.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        if "val" not in m:
            continue
        model = cfg["model"] + ("_mimo" if cfg["hparams"].get("is_mimo") else "")
        rows.append({"model": model, **cfg["hparams"], "val_bal_acc": m["val"]["balanced_accuracy"],
                     "val_fpr": m["val"]["fpr"], "val_miss": 1 - m["val"]["recall"],
                     "train_min": m["train_seconds"] / 60, "run": d.name})
    if not rows:
        raise SystemExit(f"No finished runs tagged {args.tag!r} with val metrics")
    df = pd.DataFrame(rows)
    varying = [c for c in df.columns if c not in ("model", "val_bal_acc", "val_fpr", "val_miss", "train_min", "run")
               and df[c].astype(str).nunique() > 1]

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        for model, g in df.groupby("model"):
            g = g.sort_values("val_bal_acc", ascending=False)
            print(f"\n=== {model} ===")
            print(g[varying + ["val_bal_acc", "val_fpr", "val_miss", "train_min"]].to_string(index=False))
            best = g.iloc[0]
            print("best: " + ", ".join(f"{c}={best[c]}" for c in varying))


if __name__ == "__main__":
    main()
