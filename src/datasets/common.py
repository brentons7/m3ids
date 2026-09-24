"""Shared preprocessing helpers: cleaning features, the train/val split, and saving processed data.

Every dataset module produces the same processed format, so the rest of the pipeline
never needs to know which dataset it's looking at:

    data/processed/<dataset>/
        train.parquet, val.parquet, test.parquet   feature columns + META_COLUMNS
        meta.json                                  feature list, class counts, cleaning stats
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Columns that describe a row but are never fed to a model as features.
#   label        0 = benign, 1 = attack (the anomaly-detection target)
#   category     coarse attack family, e.g. "DDoS"
#   attack       specific attack, e.g. "DDoS-SYN"
#   source_file  raw file the row came from
#   row_in_file  original row position, kept so time order can be rebuilt for sequence models
META_COLUMNS = ["label", "category", "attack", "source_file", "row_in_file"]


def clean(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN or +/-inf in any feature column."""
    values = df[features].to_numpy()
    bad = ~np.isfinite(values).all(axis=1)
    stats = {"rows_dropped_nan_or_inf": int(bad.sum())}
    return df[~bad].reset_index(drop=True), stats


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One 64-bit hash per row, so duplicate checks on millions of rows stay cheap."""
    return pd.util.hash_pandas_object(df[cols], index=False)


def drop_duplicates(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Drop rows whose features AND attack label repeat an earlier row.

    Also counts "conflicting" duplicates: identical features but different labels.
    No model can get those right, so a high count means the features can't fully separate
    the classes.
    """
    feat_hash = row_hashes(df, features)
    full_hash = row_hashes(df, features + ["attack"])
    dup = full_hash.duplicated()
    conflicting = (
        pd.DataFrame({"h": feat_hash, "label": df["label"]}).groupby("h")["label"].nunique() > 1
    ).sum()
    stats = {
        "duplicate_rows_dropped": int(dup.sum()),
        "feature_vectors_with_conflicting_labels": int(conflicting),
    }
    return df[~dup].reset_index(drop=True), stats


def count_overlap(df: pd.DataFrame, reference: pd.DataFrame, features: list[str]) -> int:
    """How many rows of df have a feature vector that also appears in reference.

    Rows shared between train and test let a model score well by memorizing, so this is
    worth reporting even when we don't remove them.
    """
    return int(row_hashes(df, features).isin(set(row_hashes(reference, features))).sum())


def split_val(df: pd.DataFrame, val_frac: float, seed: int, stratify: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of train, keeping each class's share the same in both."""
    train, val = train_test_split(df, test_size=val_frac, stratify=df[stratify], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def class_counts(splits: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Table of rows per class (rows) per split (columns)."""
    return pd.DataFrame({name: df[col].value_counts() for name, df in splits.items()}).fillna(0).astype(int)


def save(out_dir: Path, splits: dict[str, pd.DataFrame], features: list[str], info: dict) -> None:
    """Write each split to parquet plus a meta.json describing them, and print a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "rows": {name: len(df) for name, df in splits.items()},
        "class_counts": {
            col: class_counts(splits, col).to_dict(orient="index") for col in ["label", "category", "attack"]
        },
        **info,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to {out_dir}")
    print(f"{len(features)} features, rows: {meta['rows']}")
    for col in ["category", "attack"]:
        print(f"\nRows per {col}:\n{class_counts(splits, col).to_string()}")
