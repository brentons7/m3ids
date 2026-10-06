"""Preprocessing shared by all datasets. Output: data/processed/<dataset>/{train,val,test}.parquet + meta.json."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Non-feature columns. label: 0 = benign, 1 = attack; row_in_file keeps time order.
META_COLUMNS = ["label", "category", "attack", "source_file", "row_in_file"]


def clean(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Drop rows with NaN or +/-inf in any feature column."""
    bad = np.zeros(len(df), dtype=bool)
    for col in features:  # column by column, so a multi-GB table is never copied whole
        bad |= ~np.isfinite(df[col].to_numpy())
    stats = {"rows_dropped_nan_or_inf": int(bad.sum())}
    if not bad.any():
        return df, stats
    return df[~bad].reset_index(drop=True), stats


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One 64-bit hash per row, so duplicate checks on millions of rows stay cheap."""
    return pd.util.hash_pandas_object(df[cols], index=False)


def drop_duplicates(df: pd.DataFrame, features: list[str], drop: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop repeated (features, attack) rows; also count identical features with conflicting labels.

    drop=False only counts them: in packet- or flow-level data, floods are runs of identical rows, and
    removing them would delete most of an attack and break the time order the sequence models read.
    """
    feat_hash = row_hashes(df, features)
    full_hash = row_hashes(df, features + ["attack"])
    dup = full_hash.duplicated()
    conflicting = (
        pd.DataFrame({"h": feat_hash, "label": df["label"]}).groupby("h")["label"].nunique() > 1
    ).sum()
    stats = {
        "duplicate_rows_dropped" if drop else "duplicate_rows_kept": int(dup.sum()),
        "feature_vectors_with_conflicting_labels": int(conflicting),
    }
    if not drop:
        return df, stats
    return df[~dup].reset_index(drop=True), stats


def count_overlap(df: pd.DataFrame, reference: pd.DataFrame, features: list[str]) -> int:
    """How many rows of df have a feature vector that also appears in reference."""
    return int(row_hashes(df, features).isin(set(row_hashes(reference, features))).sum())


def split_val(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Val = the last val_frac (in time) of each capture, so val windows don't overlap train windows."""
    frac_in_file = df.groupby("source_file", observed=True)["row_in_file"].rank(pct=True, method="first")
    is_val = frac_in_file > 1 - val_frac
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def split_head(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(first frac in time of each capture, the rest). Used to carve val out of the test recordings."""
    frac_in_file = df.groupby("source_file", observed=True)["row_in_file"].rank(pct=True, method="first")
    is_head = frac_in_file <= frac
    return df[is_head].reset_index(drop=True), df[~is_head].reset_index(drop=True)


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
