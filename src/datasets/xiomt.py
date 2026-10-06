"""X-IoMT (https://github.com/RuiPintoUBI/X-IoMTDataset), v1.1 packet-level CSVs.

Expects data/raw/X-IoMTDataset/OriginalDatasetUpdated/*_v11.csv. DeprecatedDataset/ (v1.0) and
AugmentedDataset/ (FGSM/PGD adversarial copies) aren't used. Labels come from each row's anomaly.* columns,
since some files mix several anomaly types. No published split: the last TEST_FRAC (in time) of each file is test.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import common

RAW_DIRNAME = "X-IoMTDataset"
CSV_SUBDIR = "OriginalDatasetUpdated"
TEST_FRAC = 0.2

NUMERIC = [
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset", "tcp.flags.push", "tcp.len",
    "ip.ttl", "tcp.window_size", "frame.len", "http.content_length",
    "mqtt.msgtype", "mqtt.topic_len", "mqtt.conflag.cleansess", "mqtt.len", "mqtt.proto_len",
    "mqtt.ver", "mqtt.qos", "mqtt.retain", "mqtt.dupflag",
]
# Device readings, blank on packets that carry none (always blank on attacker traffic). Blanks become 0,
# and each group gets a has_* flag so 0 isn't mistaken for a real reading.
SENSOR_GROUPS = {
    "has_temperature": ["Temperature"],
    "has_device_status": ["Memory", "In_Temperature", "CPU", "RSSI"],
    "has_vitals": ["HR", "PULSE", "RESP", "SpO2"],
}
# One-hot encoded.
CATEGORICAL = ["frame.protocols"]
# Not used as features: frame.time only orders rows; IPs, ports, MACs, client IDs and HTTP host/agent/referer
# identify hosts or attack tools; seq/ack are per-connection counters; message text repeats the sensor
# columns; tcp.flags.urg, mqtt.conflags, mqtt.hdrflags and mqtt.msg_decoded_as never change.
LABELS = ["anomaly.layer", "anomaly.name", "anomaly.label"]
LAYERS = {0: "Benign", 1: "Perception", 2: "Network", 3: "Application"}


def load_file(path: Path) -> pd.DataFrame:
    """One CSV -> numeric features + meta columns, in time order. Columns are read by name (their order varies)."""
    sensors = [c for cols in SENSOR_GROUPS.values() for c in cols]
    raw = pd.read_csv(path, usecols=["frame.time", *NUMERIC, *sensors, *CATEGORICAL, *LABELS], low_memory=False)
    raw = raw.dropna(subset=LABELS)  # a few Normal MQTT rows have no label at all
    raw = raw.sort_values("frame.time", kind="stable").reset_index(drop=True)

    df = raw[NUMERIC].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    for flag, cols in SENSOR_GROUPS.items():
        values = raw[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        df[flag] = values.notna().any(axis=1).astype(np.float32)
        df[cols] = values.fillna(0)
    df[CATEGORICAL] = raw[CATEGORICAL]

    attack = raw["anomaly.name"].replace({"Normal": "Benign"})
    df["attack"] = attack
    df["category"] = raw["anomaly.layer"].astype(float).astype(int).map(LAYERS)
    df["label"] = raw["anomaly.label"].astype(float).astype(np.int8)
    assert ((df["label"] == 0) == (attack == "Benign")).all(), f"{path.name}: anomaly.label disagrees with anomaly.name"
    df["source_file"] = path.name
    df["row_in_file"] = np.arange(len(df), dtype=np.int32)
    print(f"  {path.name:55s} {len(df):>8,} rows  {dict(attack.value_counts())}")
    return df


def prepare(raw_dir: Path, out_dir: Path, val_frac: float) -> None:
    paths = sorted((raw_dir / CSV_SUBDIR).glob("*_v11.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_v11.csv in {raw_dir / CSV_SUBDIR}. See this module's docstring for the expected layout.")
    print(f"Loading {len(paths)} CSVs from {raw_dir / CSV_SUBDIR}")
    df = pd.concat([load_file(p) for p in paths], ignore_index=True)

    onehot = pd.get_dummies(df[CATEGORICAL], dtype=np.float32)
    df = pd.concat([df.drop(columns=CATEGORICAL), onehot], axis=1)
    # "frame.protocols_eth:ethertype:ip:tcp" -> "frame_protocols_eth_ethertype_ip_tcp", like the other datasets
    df.columns = [c if c in common.META_COLUMNS else c.lower().replace(".", "_").replace(":", "_").replace("-", "_")
                  for c in df.columns]
    for col in ["attack", "category", "source_file"]:
        df[col] = df[col].astype("category")
    features = [c for c in df.columns if c not in common.META_COLUMNS]

    df, clean_stats = common.clean(df, features)
    # Same per-file "last share in time" cut as validation, applied first to carve out test.
    train, test = common.split_val(df, TEST_FRAC)
    del df

    # Kept, not dropped: flood packets are identical once hosts are removed (FIN_Flood would lose every train row).
    train, train_dups = common.drop_duplicates(train, features, drop=False)
    test_rows_also_in_train = common.count_overlap(test, train, features)

    train, val = common.split_val(train, val_frac)

    info = {
        "dataset": "X-IoMT v1.1 (OriginalDatasetUpdated, packet-level; last test_frac of each file in time is test)",
        "source": "https://github.com/RuiPintoUBI/X-IoMTDataset",
        "val_frac": val_frac,
        "test_frac": TEST_FRAC,
        "cleaning": {
            "all": clean_stats,
            "train": train_dups,
            "test": {"rows_with_features_also_in_train": test_rows_also_in_train},
        },
    }
    common.save(out_dir, {"train": train, "val": val, "test": test}, features, info)
    for split, stats in info["cleaning"].items():
        print(f"\nCleaning ({split}): {stats}")
