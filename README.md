# m3ids

Mamba-3 intrusion detection for the Internet of Medical Things (IoMT), benchmarked against Mamba-2 and a Transformer on CICIoMT2024, with inference cost measured on an NVIDIA Jetson Orin Nano.

## Repository structure

```
m3ids/
├── run.py                  # the one entry point: prepare, train, test, experiment, benchmark
├── requirements.txt
├── data/
│   ├── raw/                # the CICIoMT2024 CSVs go here (not in git)
│   └── processed/          # output of `--task prepare` (not in git)
├── results/                # one folder per run (not in git)
│   └── figures/            # figures (PNGs in git)
└── src/
    ├── datasets/           # ciciomt2024.py (loading, labels, split) and common.py (cleaning helpers)
    ├── models/             # mamba3.py, mamba2.py, transformer.py; sequence.py (shared windowing and training)
    ├── train.py            # train one model and save it to results/<run>/
    ├── test.py             # score a saved run on val + test, write its metrics
    ├── evaluate.py         # alarm threshold (picked on val) and metrics
    ├── experiment.py       # train, then test, for each seed
    ├── benchmark.py        # latency, throughput, GPU memory and power on the Jetson
    └── plotting/           # results.py, ablation.py, hardware.py
```

## Setup

PyTorch has to be installed first, because mamba-ssm compiles against it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # pick the build for your GPU: https://pytorch.org/get-started/locally/
pip install --no-build-isolation -r requirements.txt
```

Mamba-3 with MIMO uses TileLang kernels, which need an NVIDIA GPU.

## Data

[CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html), WiFi and MQTT attacks. The full download is about 60 GB, but only one folder is used: copy `WiFI_and_MQTT/attacks/csv/` from it to `data/raw/CICIoMT2024/csv/`, so the CSVs sit at `data/raw/CICIoMT2024/csv/{train,test}/`. Then prepare it once (writes `data/processed/ciciomt2024/`):

```bash
python3 run.py --task prepare
```

Labels come from the file names, and the task is binary (benign vs. attack). The published train set is used for training. Validation is the first 30% (in time) of each published test recording, and test is the remaining 70%. `src/datasets/ciciomt2024.py` explains why, and which column is dropped.

## Models

| `--model` | Model | Parameters |
|---|---|---|
| `transformer` | Causal Transformer encoder | 72,001 |
| `mamba2` | Mamba-2 | 72,025 |
| `mamba3` with `--is-mimo true --mimo-rank 2 --d-state 32` | Mamba-3 (MIMO) | 73,937 |
| `mamba3` | Mamba-3 without MIMO (SISO), for the ablation | 73,553 |

All of them read a window of the last 32 rows of the same recording and predict whether the last row is an attack. Only the middle layers differ. In the results and figures, "Mamba-3" means the MIMO model.

## Running

```bash
python3 run.py --task list                                                  # models and every hyperparameter
python3 run.py --task experiment --model mamba2 --seeds 1 2 3 --tag mytag   # train + test, one run per seed
python3 run.py --task train --model mamba2 --seeds 1 --tag mytag            # train and save only
python3 run.py --task test --tag mytag                                      # (re-)test saved runs
python3 run.py --task benchmark --tag mytag                                 # inference cost (on the Jetson)
```

Any hyperparameter can be changed with a flag, for example `--lr 3e-3 --seq-len 64`. Each run is saved to `results/<time>_ciciomt2024_<model>_<tag>/`, which holds `config.json`, `model.pt`, `metrics.json`, the scores, and `benchmark.json` once benchmarked.

Figures are made from the runs with a given tag and written to `results/figures/<name>/`:

```bash
python3 -m src.plotting.results --tags mytag                     # Transformer, Mamba-2, Mamba-3 (MIMO)
python3 -m src.plotting.ablation --tags mytag --name my_ablation  # Mamba-3 SISO vs. MIMO
python3 -m src.plotting.hardware --tags mytag                    # Jetson cost, from benchmark.json
```

## Results

Tuned models (`--lr` 1e-3 Transformer, 3e-3 Mamba-2, 3e-4 Mamba-3; all with `--train-frac 0.3`), mean of 3 seeds. False alarms, missed attacks and balanced accuracy are on the held-out 70% test split. The other columns are on the full published test set, so they can be compared with other papers.

| Model | False alarms | Missed attacks | Balanced acc. | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Transformer | 0.37 ± 0.12% | 0.098% | 0.9977 | 0.9991 | 0.9999 | 0.9992 | 0.9995 | 0.99992 |
| Mamba-2 | 0.52 ± 0.07% | 0.104% | 0.9969 | 0.9990 | 0.9999 | 0.9991 | 0.9995 | 0.99992 |
| Mamba-3 | 0.78 ± 0.39% | 0.069% | 0.9958 | 0.9992 | 0.9998 | 0.9994 | 0.9996 | 0.99995 |

Inference on the Jetson Orin Nano (MAXN_SUPER, clocks locked), mean of 3 seeds:

| Model | Latency, 1 window | Throughput, batch 512 | GPU memory, batch 512 | Board power, batch 512 |
|---|---|---|---|---|
| Transformer | 4.01 ms | 54k windows/s | 37.3 MB | 15.0 W |
| Mamba-2 | 6.88 ms | 36k windows/s | 83.9 MB | 15.3 W |
| Mamba-3 | 6.85 ms | 57k windows/s | 61.8 MB | 15.1 W |

Mamba-3 with MIMO was trained on the Jetson, because its kernels need an NVIDIA GPU. The Transformer and Mamba-2 were trained on an AMD workstation GPU. All inference numbers come from the Jetson. Figures for all of the above are in `results/figures/`.
