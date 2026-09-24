"""CICIoMT2024 (Canadian Institute for Cybersecurity, UNB): Wi-Fi + MQTT attack traffic.

Download it from https://www.unb.ca/cic/datasets/iomt-dataset-2024.html and extract the
WiFi_and_MQTT/attacks/csv folder so the layout is:

    data/raw/CICIoMT2024/csv/train/*.csv
    data/raw/CICIoMT2024/csv/test/*.csv

Each CSV holds one traffic type, and each row summarizes a small window of packets
(45 numeric features). There is no label column: the filename is the label, e.g.
    TCP_IP-DDoS-SYN3_train.pcap.csv  ->  attack "DDoS-SYN", category "DDoS"
where "3" is just the capture chunk number.

We keep the authors' train/test split so results are comparable with published work,
and carve our validation set out of train.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import common

RAW_DIRNAME = "CICIoMT2024"

# Attack-name prefix -> category (the dataset paper's 5 attack categories, plus Benign).
CATEGORY_PREFIXES = {
    "Benign": "Benign",
    "ARP_Spoofing": "Spoofing",
    "Recon-": "Recon",
    "MQTT-": "MQTT",
    "DDoS-": "DDoS",
    "DoS-": "DoS",
}


def attack_from_filename(path: Path) -> str:
    """TCP_IP-DDoS-SYN3_train.pcap.csv -> DDoS-SYN"""
    name = path.name.removesuffix(".pcap.csv")
    name = re.sub(r"_(train|test)$", "", name)
    name = re.sub(r"\d+$", "", name)  # capture chunk number
    return name.removeprefix("TCP_IP-")


def category_of(attack: str) -> str:
    for prefix, category in CATEGORY_PREFIXES.items():
        if attack.startswith(prefix):
            return category
    raise ValueError(f"Unknown attack type {attack!r}; add it to CATEGORY_PREFIXES")


def load_split(csv_dir: Path) -> pd.DataFrame:
    """Read every CSV in one split folder and label each row from its filename."""
    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs in {csv_dir}. See this module's docstring for the expected layout.")

    frames = []
    for path in paths:
        df = pd.read_csv(path, engine="pyarrow", dtype=np.float32)
        # "Protocol Type" -> "protocol_type", so features are easy to reference in code
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        attack = attack_from_filename(path)
        df["attack"] = attack
        df["category"] = category_of(attack)
        df["label"] = np.int8(attack != "Benign")
        df["source_file"] = path.name
        df["row_in_file"] = np.arange(len(df), dtype=np.int32)
        frames.append(df)
        print(f"  {path.name:45s} {len(df):>9,} rows  -> {attack}")

    df = pd.concat(frames, ignore_index=True)
    for col in ["attack", "category", "source_file"]:
        df[col] = df[col].astype("category")  # far less memory than millions of repeated strings
    return df


def prepare(raw_dir: Path, out_dir: Path, seed: int, val_frac: float) -> None:
    print(f"Loading train CSVs from {raw_dir / 'csv/train'}")
    train = load_split(raw_dir / "csv" / "train")
    print(f"Loading test CSVs from {raw_dir / 'csv/test'}")
    test = load_split(raw_dir / "csv" / "test")

    features = [c for c in train.columns if c not in common.META_COLUMNS]
    assert list(test.columns) == list(train.columns), "train and test CSVs have different columns"

    train, train_clean = common.clean(train, features)
    test, test_clean = common.clean(test, features)

    # Duplicates are removed from train only. The test set stays exactly as published,
    # so our numbers stay comparable to other papers; the overlap count goes in meta.json.
    train, train_dups = common.drop_duplicates(train, features)
    test_rows_also_in_train = common.count_overlap(test, train, features)

    train, val = common.split_val(train, val_frac, seed, stratify="attack")

    info = {
        "dataset": "CICIoMT2024 (WiFi_and_MQTT attacks, CSV features)",
        "source": "https://www.unb.ca/cic/datasets/iomt-dataset-2024.html",
        "seed": seed,
        "val_frac": val_frac,
        "cleaning": {
            "train": {**train_clean, **train_dups},
            "test": {**test_clean, "rows_with_features_also_in_train": test_rows_also_in_train},
        },
    }
    common.save(out_dir, {"train": train, "val": val, "test": test}, features, info)
    for split, stats in info["cleaning"].items():
        print(f"\nCleaning ({split}): {stats}")
