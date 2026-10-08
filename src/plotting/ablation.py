"""MIMO ablation: Mamba-3 SISO vs Mamba-3 MIMO at equal size, nothing else.

    python3 -m src.plotting.ablation --tags untuned --name ablation_untuned
    python3 -m src.plotting.ablation --tags final ablation_siso --name ablation_tuned

Same detection figures as src.plotting.results (ml_metrics, metrics_by_model, errors_by_class, summary_table.csv),
plus the Jetson hardware figure when the runs have a benchmark.json. Writes to results/figures/<name>/.
"""
from . import ABLATION, lineup, load_runs, parse_args
from .hardware import load_benchmarks, plot_hardware
from .results import make_all


def main() -> None:
    args = parse_args(__doc__, default_name=lambda tags: "ablation_" + "+".join(tags))
    df, models = lineup(load_runs(args.tags, args.dataset), ABLATION)
    if len(models) < 2:
        raise SystemExit(f"Only {models} in these runs: the ablation needs both Mamba-3 SISO and MIMO")
    make_all(df, models, ABLATION, args.out)
    try:
        plot_hardware(load_benchmarks(df), models, ABLATION, args.out)
    except SystemExit as e:
        print(f"  hardware skipped ({e})")


if __name__ == "__main__":
    main()
