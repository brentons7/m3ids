"""Single entry point. Run from the repo root:

    python -m src.run list                                             models + their hyperparameters
    python -m src.run prepare --dataset ciciomt2024                    raw -> processed
    python -m src.run experiment --dataset ciciomt2024 --model mamba3  train + evaluate

Any extra --flag on `experiment` sets a model hyperparameter, e.g. --epochs 5 --seq-len 64.
"""
import argparse
from pathlib import Path

from src import datasets, models

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def parse_hparams(extra: list[str], defaults: dict) -> dict:
    """Turn leftover CLI args like ["--seq-len", "64", "--lr=3e-4"] into {"seq_len": 64, "lr": 0.0003}.

    Values are converted to the type of the model's default, and unknown names are an
    error so a typo can't silently fall back to the default.
    """
    hparams, i = {}, 0
    while i < len(extra):
        if not extra[i].startswith("--"):
            raise SystemExit(f"Expected a --flag, got {extra[i]!r}")
        key, has_eq, value = extra[i][2:].partition("=")
        if not has_eq:
            i += 1
            if i >= len(extra):
                raise SystemExit(f"--{key} needs a value")
            value = extra[i]
        name = key.replace("-", "_")
        if name not in defaults:
            valid = ", ".join("--" + k.replace("_", "-") for k in defaults)
            raise SystemExit(f"Unknown hyperparameter --{key}. This model accepts: {valid}")
        default = defaults[name]
        if isinstance(default, bool):
            hparams[name] = value.lower() in ("1", "true", "yes")
        else:
            hparams[name] = type(default)(value)
        i += 1
    return hparams


def cmd_list(args, extra) -> None:
    print("Datasets:", ", ".join(datasets.REGISTRY))
    print("\nModels:")
    for name in models.REGISTRY:
        try:
            Model = models.get(name)
        except ImportError as e:
            print(f"  {name}  (unavailable: {e})")
            continue
        kind = "supervised" if Model.supervised else "unsupervised"
        print(f"  {name}  ({kind})")
        for k, v in Model.default_hparams.items():
            print(f"      --{k.replace('_', '-'):16s} {v}")


def cmd_prepare(args, extra) -> None:
    module = datasets.REGISTRY[args.dataset]
    module.prepare(
        raw_dir=RAW_DIR / module.RAW_DIRNAME,
        out_dir=PROCESSED_DIR / args.dataset,
        seed=args.seed,
        val_frac=args.val_frac,
    )


def cmd_experiment(args, extra) -> None:
    import torch
    from src.experiment import run_experiment

    Model = models.get(args.model)
    hparams = parse_hparams(extra, Model.default_hparams)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(args.dataset, args.model, hparams, args.seed, device, args.tag, PROCESSED_DIR, RESULTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.run")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show datasets, models and their hyperparameters").set_defaults(func=cmd_list)

    p = sub.add_parser("prepare", help="turn data/raw/<dataset> into data/processed/<dataset>")
    p.add_argument("--dataset", required=True, choices=datasets.REGISTRY)
    p.add_argument("--seed", type=int, default=42, help="seed for the train/val split")
    p.add_argument("--val-frac", type=float, default=0.1, help="share of train held out as validation")
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("experiment", help="train + evaluate one model on one dataset; extra --flags set hyperparameters")
    e.add_argument("--dataset", required=True, choices=datasets.REGISTRY)
    e.add_argument("--model", required=True, choices=models.REGISTRY)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    e.add_argument("--tag", default="", help="optional label appended to the run folder name")
    e.set_defaults(func=cmd_experiment)

    args, extra = parser.parse_known_args()
    if extra and args.command != "experiment":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.func(args, extra)


if __name__ == "__main__":
    main()
