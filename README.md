# m3ids

Mamba-3 intrusion detection for the Internet of Medical Things (IoMT), benchmarked against Mamba-2 and a Transformer on an NVIDIA Jetson. EagleCyberNest, Fall 2026.

## Repository structure

```
m3ids/
├── run.py                  # entry point: list, prepare, experiment
├── requirements.txt
├── data/
│   ├── raw/                # downloaded datasets go here (not in git)
│   └── processed/          # output of `run.py prepare` (not in git)
├── results/                # one folder per experiment run, plus summary.csv (not in git)
│   └── figures/            # plots from scripts/plot_results.py
├── scripts/
│   ├── plot_results.py     # figures and summary tables for a group of runs
│   └── run_all.sh
└── src/
    ├── datasets/           # one loader per dataset, plus common.py (shared cleaning and splitting)
    ├── models/             # mamba3.py, mamba2.py, transformer.py; sequence.py (shared training code)
    ├── experiment.py       # train, score, evaluate, save one run
    ├── evaluate.py         # metrics
    └── power.py            # Jetson power, GPU and temperature sampling (tegrastats)
```

## Setup

PyTorch has to be installed first, because mamba-ssm compiles against it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # pick the build for your GPU: https://pytorch.org/get-started/locally/
pip install --no-build-isolation -r requirements.txt
```

## Datasets

| Name (`--dataset`) | Dataset | Level | Put in `data/raw/` |
|---|---|---|---|
| `ciciomt2024` | [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) | Feature windows | `CICIoMT2024/WiFI_and_MQTT/attacks/csv/{train,test}/` |
| `xiomt` | [X-IoMT](https://github.com/RuiPintoUBI/X-IoMTDataset) v1.1 | Packets | `X-IoMTDataset/OriginalDatasetUpdated/` |
| `iomttrafficdata` | IoMT-TrafficData <!-- TODO: add source link --> | Flows | `IoMT-TrafficData/output.csv` |

Only the folders listed are used; the rest of each download can be deleted. Each loader's docstring in `src/datasets/` explains its labels, dropped columns and train/test split.

Prepare each dataset once (writes `data/processed/<name>/`):

```bash
python3 run.py prepare --dataset ciciomt2024
python3 run.py prepare --dataset xiomt
python3 run.py prepare --dataset iomttrafficdata
```

## Models

| Name (`--model`) | Model |
|---|---|
| `mamba3` | Mamba-3 |
| `mamba2` | Mamba-2 |
| `transformer` | Transformer encoder |

All three read a window of consecutive rows (32 by default) and predict whether the last row is an attack.

## Running experiments

One model on one dataset:

```bash
python3 run.py experiment --dataset xiomt --model mamba3 --seed 1 --tag baseline
```

Any hyperparameter can be changed with a flag, for example `--seq-len 64 --epochs 5`. `python3 run.py list` shows every dataset, model and hyperparameter with its default.

Each run is saved to `results/<time>_<dataset>_<model>_<tag>/` (config, metrics, scores, model weights) and added as a row to `results/summary.csv`. Metrics include ROC-AUC, F1, false-alarm rate and per-attack detection rates; on a Jetson they also include power draw, latency and GPU use.

To plot all runs with a given tag:

```bash
python3 scripts/plot_results.py --tag baseline
```

Figures and tables are written to `results/figures/<tag>/<dataset>/`.
