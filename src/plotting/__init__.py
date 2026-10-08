"""Figures for the paper. Shared pieces live here (run loading, model lineups, colors, style); each module
makes one family of figures:

    python3 -m src.plotting.results  --tags untuned                                  # Transformer, Mamba-2, Mamba-3
    python3 -m src.plotting.ablation --tags final ablation_siso --name ablation_tuned  # Mamba-3 SISO vs MIMO
    python3 -m src.plotting.hardware --tags final                                    # Jetson cost (benchmark.json)

Each writes PNG + PDF to results/figures/<name>/ (<name> defaults to the tags joined by '+').
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"

# Model "variant" = model name, plus "_mimo" for Mamba-3 with MIMO (a different model at the same size).
# The paper's lineup calls the MIMO model just "Mamba-3"; plain (SISO) Mamba-3 only appears in the ablation.
MAIN = {"transformer": "Transformer", "mamba2": "Mamba-2", "mamba3_mimo": "Mamba-3"}
ABLATION = {"mamba3": "Mamba-3 SISO", "mamba3_mimo": "Mamba-3 MIMO"}
# Fixed color per model in every figure; checked for colorblind separation.
COLORS = {"transformer": "#1baf7a", "mamba2": "#eb6834", "mamba3_mimo": "#eda100", "mamba3": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DATASET_NAMES = {"ciciomt2024": "CICIoMT2024"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "pdf.fonttype": 42,
})


def parse_args(doc: str, default_name=lambda tags: "+".join(tags)) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", required=True, help="include runs whose folder ends in _<tag>")
    parser.add_argument("--name", help="output folder under results/figures/")
    parser.add_argument("--dataset", default="ciciomt2024")
    args = parser.parse_args()
    args.out = FIGURES / (args.name or default_name(args.tags))
    return args


def load_runs(tags: list[str], dataset: str) -> pd.DataFrame:
    """One row per run: config basics plus everything in metrics.json; `full_test` metrics get a full_ prefix."""
    rows = []
    for tag in tags:
        for run_dir in sorted((ROOT / "results").glob(f"*_{tag}")):
            config = json.loads((run_dir / "config.json").read_text())
            if config["dataset"] != dataset:
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text())
            variant = config["model"] + ("_mimo" if config["hparams"].get("is_mimo") else "")
            full = {f"full_{k}": v for k, v in metrics.get("full_test", {}).items()}
            rows.append({"run_id": run_dir.name, "model": variant, "seed": config["seed"], "dataset": dataset,
                         **metrics, **full})
    if not rows:
        raise SystemExit(f"No {dataset} runs tagged {tags} in results/")
    df = pd.DataFrame(rows)
    df["miss_rate"] = 1 - df["recall"]
    return df


def lineup(df: pd.DataFrame, names: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Keep only the runs of the lineup's models; returns them and the models present, in lineup order."""
    models = [m for m in names if m in set(df["model"])]
    if not models:
        raise SystemExit(f"None of {list(names.values())} in these runs")
    return df[df["model"].isin(models)].reset_index(drop=True), models


def tint(hex_color: str, amount: float = 0.55) -> str:
    """Mix a color toward white (lighter version of a model's color)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.to_hex(rgb + (1 - rgb) * amount)


def style_rows(ax, models: list[str], names: dict[str, str]) -> None:
    """Horizontal-bar axes: one row per model, top to bottom in lineup order."""
    ax.set_yticks(range(len(models)), [names[m] for m in models])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  {name}.png / .pdf")
