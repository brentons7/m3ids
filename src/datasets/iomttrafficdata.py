"""IoMT-TrafficData, flow-level (Zeek conn log + flow statistics).

Expects data/raw/IoMT-TrafficData/output.csv. The file has no timestamps and holds one block of rows per
traffic class, so each class is treated as its own capture (rows assumed in capture order within it).
No published split: the last TEST_FRAC of each class block is test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from . import common

RAW_DIRNAME = "IoMT-TrafficData"
FILENAME = "output.csv"
TEST_FRAC = 0.2

# One-hot encoded.
CATEGORICAL = ["proto", "service", "conn_state"]
# Not used as features: the unnamed first column is a row index; IPs and ports identify hosts (the attacker
# is one fixed IP); local_orig/local_resp follow from the IPs; tunnel_parents is always "-"; history is a
# free-form flag string covered by the flag counts; duration/orig_bytes/resp_bytes are "-" on most flows
# and duplicated by flow_duration and the payload totals.
DROPPED = [
    "", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "local_orig", "local_resp",
    "tunnel_parents", "history", "duration", "orig_bytes", "resp_bytes",
]
LABELS = ["traffic", "is_attack"]


def load(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the CSV and return the rows plus the feature column names."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See this module's docstring for the expected layout.")
    header = pd.read_csv(path, nrows=0).columns
    header = ["" if c.startswith("Unnamed: 0") else c for c in header]
    numeric = [c for c in header if c not in DROPPED + CATEGORICAL + LABELS]
    # Read in chunks: the pyarrow engine peaks around 4.5 GB on this 1.5 GB file, too close to the Jetson's RAM.
    chunks = list(pd.read_csv(
        path, usecols=numeric + CATEGORICAL + LABELS, chunksize=250_000,
        dtype={**{c: np.float32 for c in numeric}, **{c: "category" for c in CATEGORICAL + ["traffic"]}},
    ))
    # Same categories in every chunk, so one-hot columns line up and concat keeps them categorical.
    categories = {col: union_categoricals([chunk[col] for chunk in chunks]).categories for col in CATEGORICAL + ["traffic"]}
    parts = []
    while chunks:  # pop as we go so the raw chunks and the finished parts are never both fully in memory
        chunk = chunks.pop(0)
        for col, cats in categories.items():
            chunk[col] = pd.Categorical(chunk[col], categories=cats)
        attack = chunk["traffic"].cat.rename_categories({"normal": "Benign"})
        parts.append(pd.concat([
            chunk[numeric],
            pd.get_dummies(chunk[CATEGORICAL], dtype=np.float32),
            pd.DataFrame({
                "attack": attack,
                "category": attack,  # this dataset has no attack families, so both are the same
                "label": chunk["is_attack"].astype(np.int8),
            }),
        ], axis=1))
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"  {path.name}: {len(df):,} rows, {len(numeric)} numeric columns")

    # "fwd_pkts_payload.min" -> "fwd_pkts_payload_min", "flow_FIN_flag_count" -> "flow_fin_flag_count"
    df.columns = [c if c in common.META_COLUMNS else c.lower().replace(".", "_").replace("-", "_") for c in df.columns]
    features = [c for c in df.columns if c not in common.META_COLUMNS]
    assert ((df["label"] == 0) == (df["attack"] == "Benign")).all(), "is_attack disagrees with traffic"
    # Each class block is its own capture: windows and per-capture splits stay inside one class.
    df["source_file"] = df["attack"]
    df["row_in_file"] = df.groupby("source_file", observed=True).cumcount().astype(np.int32)
    return df, features


def prepare(raw_dir: Path, out_dir: Path, val_frac: float) -> None:
    print(f"Loading {raw_dir / FILENAME}")
    df, features = load(raw_dir / FILENAME)

    df, clean_stats = common.clean(df, features)
    # Same per-capture "last share in time" cut as validation, applied first to carve out test.
    train, test = common.split_val(df, TEST_FRAC)
    del df

    # Kept, not dropped: most flood flows are identical (camoverflow would fall from ~1.3M train rows to 84).
    train, train_dups = common.drop_duplicates(train, features, drop=False)
    test_rows_also_in_train = common.count_overlap(test, train, features)

    train, val = common.split_val(train, val_frac)

    info = {
        "dataset": "IoMT-TrafficData (flow-level; one capture per class, last test_frac of each is test)",
        "val_frac": val_frac,
        "test_frac": TEST_FRAC,
        "dropped_columns": [c or "<row index>" for c in DROPPED],
        "cleaning": {
            "all": clean_stats,
            "train": train_dups,
            "test": {"rows_with_features_also_in_train": test_rows_also_in_train},
        },
    }
    common.save(out_dir, {"train": train, "val": val, "test": test}, features, info)
    for split, stats in info["cleaning"].items():
        print(f"\nCleaning ({split}): {stats}")
