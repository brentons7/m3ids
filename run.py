"""The one entry point: python3 run.py --task {list,prepare,train,test,experiment,benchmark}.

    python3 run.py --task list                                  # datasets, models and every model's hyperparameters
    python3 run.py --task prepare                               # data/raw/CICIoMT2024 -> data/processed/ciciomt2024
    python3 run.py --task experiment --model mamba2 --seeds 1 2 3 --tag final --lr 3e-3 --train-frac 0.3
    python3 run.py --task train --model mamba2 --seeds 1 --tag try          # train + save only
    python3 run.py --task test --tag try                                    # (re-)test saved runs
    python3 run.py --task benchmark --tag final ablation_siso               # Jetson cost of saved runs

experiment = train then test, per seed. Extra --flags (--lr, --seq-len, ...) set hyperparameters for train and
experiment. --tag labels new runs (train, experiment) or picks existing runs by tag (test, benchmark); test and
benchmark also take explicit --runs results/<run> ...
"""
import argparse
from pathlib import Path

from src import datasets, models

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def parse_hparams(extra: list[str], defaults: dict) -> dict:
    """["--seq-len", "64"] -> {"seq_len": 64}, typed like the defaults; unknown names are an error."""
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


def cmd_list(args) -> None:
    print("Datasets:", ", ".join(datasets.REGISTRY))
    print("\nModels:")
    for name in models.REGISTRY:
        try:
            Model = models.get(name)
        except ImportError as e:
            print(f"  {name}  (unavailable: {e})")
            continue
        print(f"  {name}")
        for k, v in Model.default_hparams.items():
            print(f"      --{k.replace('_', '-'):16s} {v}")


def cmd_prepare(args) -> None:
    module = datasets.REGISTRY[args.dataset]
    module.prepare(raw_dir=RAW_DIR / module.RAW_DIRNAME, out_dir=PROCESSED_DIR / args.dataset)


def pick_device(device: str) -> str:
    import torch
    return device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(args, hparams: dict) -> None:
    from src.train import train

    for seed in args.seeds:
        train(args.dataset, args.model, hparams, seed, pick_device(args.device), args.tag[0] if args.tag else "",
              PROCESSED_DIR, RESULTS_DIR)


def cmd_experiment(args, hparams: dict) -> None:
    from src.experiment import run_experiment

    run_experiment(args.dataset, args.model, hparams, args.seeds, pick_device(args.device),
                   args.tag[0] if args.tag else "", PROCESSED_DIR, RESULTS_DIR)


def selected_runs(args) -> list[Path]:
    """--runs as given, plus every results/ run whose folder ends in _<tag> for each --tag."""
    runs = [r.resolve() for r in args.runs]
    for tag in args.tag:
        runs += sorted(d for d in RESULTS_DIR.glob(f"*_{tag}") if (d / "config.json").exists())
    if not runs:
        raise SystemExit("No runs selected: give --tag <tag> or --runs results/<run> ...")
    return runs


def cmd_test(args) -> None:
    from src.test import test

    for run_dir in selected_runs(args):
        test(run_dir, pick_device(args.device), PROCESSED_DIR)


def cmd_benchmark(args) -> None:
    from src.benchmark import benchmark

    benchmark(selected_runs(args))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 run.py", description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True,
                        choices=["list", "prepare", "train", "test", "experiment", "benchmark"])
    parser.add_argument("--dataset", default="ciciomt2024", choices=datasets.REGISTRY)
    parser.add_argument("--model", choices=models.REGISTRY, help="train / experiment: which model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="train / experiment: one run per seed")
    parser.add_argument("--tag", nargs="+", default=[],
                        help="train / experiment: label for the new runs (one); test / benchmark: runs to use")
    parser.add_argument("--runs", type=Path, nargs="+", default=[], help="test / benchmark: results/<run> folders")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    args, extra = parser.parse_known_args()

    if args.task in ("train", "experiment"):
        if args.model is None:
            parser.error(f"--task {args.task} needs --model")
        if len(args.tag) > 1:
            parser.error(f"--task {args.task} takes one --tag (it labels the new runs)")
        hparams = parse_hparams(extra, models.get(args.model).default_hparams)
        (cmd_train if args.task == "train" else cmd_experiment)(args, hparams)
        return
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)} "
                     "(hyperparameter flags only go with train / experiment)")
    {"list": cmd_list, "prepare": cmd_prepare, "test": cmd_test, "benchmark": cmd_benchmark}[args.task](args)


if __name__ == "__main__":
    main()
