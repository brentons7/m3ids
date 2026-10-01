"""IoMT-IND2026: expects data/raw/IoMT-IND2026/original_data_iot.csv. Split by time; last TEST_FRAC is test."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import common

RAW_DIRNAME = "IoMT-IND2026"
FILENAME = "original_data_iot.csv"
TEST_FRAC = 0.2

NUMERIC = [
    "Packet_Size", "Flow_Duration", "Packet_Rate", "TTL",
    "SYN_Flag", "ACK_Flag", "RST_Flag", "Window_Size", "Payload_Entropy",
]
# One-hot encoded.
CATEGORICAL = ["Device_Type", "Protocol"]
# Timestamp is only for ordering; IPs would let the model memorize hosts.
DROPPED = ["Timestamp", "Source_IP", "Destination_IP"]


def load(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the CSV, sort by time, and return the rows plus the feature column names."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See this module's docstring for the expected layout.")
    raw = pd.read_csv(path)
    raw = raw.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    print(f"  {path.name}: {len(raw):,} rows, columns: {list(raw.columns)}")

    onehot = pd.get_dummies(raw[CATEGORICAL], dtype=np.float32)
    df = pd.concat([raw[NUMERIC].astype(np.float32), onehot], axis=1)
    # "Device_Type_ECG_Monitor" -> "device_type_ecg_monitor", matching CICIoMT2024's style
    df.columns = [c.lower() for c in df.columns]
    features = list(df.columns)

    attack = raw["Attack_Label"].replace({"Normal": "Benign"})
    df["attack"] = attack.astype("category")
    df["category"] = df["attack"]  # this dataset has no finer attack names, so both are the same
    df["label"] = (attack != "Benign").astype(np.int8)
    df["source_file"] = pd.Categorical([path.name] * len(df))
    df["row_in_file"] = np.arange(len(df), dtype=np.int32)
    return df, features


def prepare(raw_dir: Path, out_dir: Path, val_frac: float) -> None:
    print(f"Loading {raw_dir / FILENAME}")
    df, features = load(raw_dir / FILENAME)

    df, clean_stats = common.clean(df, features)
    n_test = int(len(df) * TEST_FRAC)
    train, test = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)

    train, train_dups = common.drop_duplicates(train, features)
    test_rows_also_in_train = common.count_overlap(test, train, features)

    train, val = common.split_val(train, val_frac)

    info = {
        "dataset": "IoMT-IND2026 (single CSV, time-ordered split)",
        "val_frac": val_frac,
        "test_frac": TEST_FRAC,
        "dropped_columns": DROPPED,
        "cleaning": {
            "all": clean_stats,
            "train": train_dups,
            "test": {"rows_with_features_also_in_train": test_rows_also_in_train},
        },
    }
    common.save(out_dir, {"train": train, "val": val, "test": test}, features, info)
    for split, stats in info["cleaning"].items():
        print(f"\nCleaning ({split}): {stats}")
