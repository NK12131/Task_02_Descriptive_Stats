"""
pandas_stats.py  —  Task 02 / Milestone A
Descriptive statistics + grouped analysis using Pandas.
"""

import sys
import os
import ast
import argparse
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════

def load_dataframe(filepath: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns  [{enc}]")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode: {filepath}")


# ═══════════════════════════════════════════════════════════
# DICT-STRING EXPANSION
# ═══════════════════════════════════════════════════════════

def _parse_bound(s, bound="lower_bound"):
    if pd.isna(s) or not str(s).strip().startswith("{"):
        return np.nan
    try:
        d = ast.literal_eval(str(s).strip())
        val = d.get(bound) or d.get("lower_bound")
        return float(val) if val else np.nan
    except Exception:
        return np.nan


def expand_dict_cols(df: pd.DataFrame) -> pd.DataFrame:
    DICT_COLS = [c for c in df.columns if c in ("spend", "impressions", "estimated_audience_size")]
    for col in DICT_COLS:
        sample = df[col].dropna().astype(str).head(50)
        if sample.str.strip().str.startswith("{").mean() >= 0.5:
            df[f"{col}_lower"] = df[col].apply(lambda v: _parse_bound(v, "lower_bound"))
            print(f"  Expanded dict-string column: '{col}'")
    return df


# ═══════════════════════════════════════════════════════════
# COLUMN CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def classify_columns(df: pd.DataFrame):
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


# ═══════════════════════════════════════════════════════════
# SECTIONS
# ═══════════════════════════════════════════════════════════

SEP  = "─" * 72
SEP2 = "═" * 72


def section_overview(df: pd.DataFrame):
    print(f"\n{SEP2}")
    print("  DATASET OVERVIEW")
    print(f"{SEP2}")
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print()

    null_sum = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)

    print(f"  {'Column':<45} {'Dtype':<12} {'Missing':>9}  {'%':>5}")
    print(f"  {'─'*45} {'─'*12} {'─'*9}  {'─'*5}")
    for col in df.columns:
        disp = (col[:43] + "…") if len(col) > 44 else col
        print(f"  {disp:<45} {str(df[col].dtype):<12} "
              f"{null_sum[col]:>9,}  {null_pct[col]:>4.1f}%")


def section_numeric(df: pd.DataFrame, numeric_cols: list):
    if not numeric_cols:
        return
    print(f"\n{SEP2}")
    print(f"  NUMERIC COLUMNS  ({len(numeric_cols)})")
    print(f"{SEP2}")

    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        miss = df[col].isna().sum()
        miss_pct = miss / len(df) * 100

        print(f"\n  ┌ {col}  [numeric]")
        print(f"  │  non-null : {n:>10,}   missing  : {miss:>8,}  ({miss_pct:.1f}%)")
        print(f"  │  mean     : {s.mean():>14.4f}   std (pop): {s.std(ddof=0):>12.4f}")
        print(f"  │  std(smp) : {s.std(ddof=1):>14.4f}   (Pandas default is ddof=1)")
        print(f"  │  min      : {s.min():>14.4f}   max      : {s.max():>12.4f}")
        print(f"  │  median   : {s.median():>14.4f}   skew     : {s.skew():>12.4f}")
        print(f"  └{'─'*60}")


def section_categorical(df: pd.DataFrame, categorical_cols: list):
    if not categorical_cols:
        return
    print(f"\n{SEP2}")
    print(f"  CATEGORICAL COLUMNS  ({len(categorical_cols)})")
    print(f"{SEP2}")

    for col in categorical_cols:
        series = df[col].dropna().astype(str).str.strip()
        n_valid  = len(series)
        n_miss   = df[col].isna().sum()
        n_unique = series.nunique()
        miss_pct = n_miss / len(df) * 100
        vc = series.value_counts()

        print(f"\n  ┌ {col}  [categorical]")
        print(f"  │  non-null : {n_valid:>10,}   missing : {n_miss:>8,}  ({miss_pct:.1f}%)")
        print(f"  │  unique   : {n_unique:>10,}   mode    : {str(vc.index[0])[:42] if len(vc) else 'N/A'}")
        print(f"  │  top 5 values:")
        for val, cnt in vc.head(5).items():
            pct = cnt / n_valid * 100 if n_valid else 0
            disp = (str(val)[:52] + "…") if len(str(val)) > 53 else str(val)
            print(f"  │      {cnt:>8,}  ({pct:5.1f}%)  {disp}")
        print(f"  └{'─'*60}")


def section_grouped(df: pd.DataFrame, key_cols: list, numeric_cols: list, label: str):
    print(f"\n{SEP}")
    print(f"  GROUPED ANALYSIS — {label}")
    print(f"  Key columns : {key_cols}")
    print(SEP)

    if not numeric_cols:
        print("  No numeric columns to aggregate.")
        return

    # Build aggregation: for each numeric col compute these functions
    agg_dict = {col: ["count", "mean", "min", "max",
                      lambda x: x.std(ddof=0),   # population std
                      "median"]
                for col in numeric_cols}

    # Rename lambdas so the multi-index is readable
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ["count", "mean", "min", "max", "std", "median"]

    grouped = (df.groupby(key_cols, dropna=False)
                 .agg(agg_dict))

    # Also add a population std column for comparison
    for col in numeric_cols:
        grouped[(col, "std_pop")] = (
            df.groupby(key_cols, dropna=False)[col]
              .std(ddof=0)
        )

    n_groups = len(grouped)
    sizes    = df.groupby(key_cols, dropna=False).size()

    print(f"\n  Groups found: {n_groups:,}")
    print(f"  Group size  —  min={sizes.min()}  max={sizes.max()}  "
          f"mean={sizes.mean():.1f}  median={sizes.median()}")

    print(f"\n  Preview (top 10 groups by size):")
    top_keys = sizes.sort_values(ascending=False).head(10).index
    preview = grouped.loc[top_keys]
    print(preview.to_string())
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pandas descriptive stats + grouped analysis (Milestone A)"
    )
    parser.add_argument("file", help="Path to the CSV file")
    args = parser.parse_args()

    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    print(f"\n{SEP2}")
    print("  PANDAS DESCRIPTIVE STATISTICS  —  Milestone A")
    print(f"  File : {os.path.basename(filepath)}")
    print(f"{SEP2}\n")

    print("  Loading...")
    df = load_dataframe(filepath)
    df = expand_dict_cols(df)

    numeric_cols, categorical_cols = classify_columns(df)

    section_overview(df)
    section_numeric(df, numeric_cols)
    section_categorical(df, categorical_cols)

    # ── Grouped analysis ─────────────────────────────────────
    print(f"\n{SEP2}")
    print("  GROUPED ANALYSIS")
    print(f"{SEP2}")

    h_lower = {c.lower(): c for c in df.columns}
    group_configs = []

    if "page_id" in h_lower:
        pid = h_lower["page_id"]
        group_configs.append(([pid], "by page_id"))
        if "ad_id" in h_lower:
            aid = h_lower["ad_id"]
            group_configs.append(([pid, aid], "by page_id + ad_id"))

    if not group_configs:
        print("  No page_id / ad_id columns found — skipping grouped analysis.")
    else:
        for key_cols, label in group_configs:
            section_grouped(df, key_cols, numeric_cols, label)

    print(f"\n{SEP2}")
    print("  PANDAS ANALYSIS COMPLETE")
    print(f"{SEP2}\n")


if __name__ == "__main__":
    main()
