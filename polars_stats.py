"""
polars_stats.py  —  Task 02 / Milestone A
Descriptive statistics + grouped analysis using Polars.

"""

import sys
import os
import ast
import argparse

import polars as pl


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════

def load_dataframe(filepath: str) -> pl.DataFrame:
    """
    Read CSV with Polars.
    - infer_schema_length=10000  : scan more rows for type inference
    - null_values               : treat these strings as null
    - ignore_errors=True        : skip cells that cannot be parsed
    - encoding='utf8-lossy'     : handle non-UTF-8 bytes gracefully
    """
    try:
        df = pl.read_csv(
            filepath,
            infer_schema_length=10_000,
            null_values=["", "NA", "N/A", "nan", "None", "null", "NULL"],
            ignore_errors=True,
            encoding="utf8-lossy",
        )
    except Exception:
        df = pl.read_csv(
            filepath,
            infer_schema_length=10_000,
            null_values=["", "NA", "N/A", "nan", "None", "null", "NULL"],
            ignore_errors=True,
            encoding="utf8-lossy",
            truncate_ragged_lines=True,
        )
    print(f"  Loaded {df.height:,} rows x {df.width} columns")
    return df


# ═══════════════════════════════════════════════════════════
# DICT-STRING EXPANSION
# ═══════════════════════════════════════════════════════════

def _parse_bound(s, bound="lower_bound"):
    if s is None:
        return None
    try:
        d = ast.literal_eval(str(s).strip())
        val = d.get(bound) or d.get("lower_bound")
        return float(val) if val else None
    except Exception:
        return None


def expand_dict_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add _lower Float64 columns for spend / impressions / estimated_audience_size."""
    DICT_COLS = [c for c in df.columns if c in ("spend", "impressions", "estimated_audience_size")]
    for col in DICT_COLS:
        lowers = [_parse_bound(v) for v in df[col].to_list()]
        df = df.with_columns(
            pl.Series(name=f"{col}_lower", values=lowers, dtype=pl.Float64)
        )
    if DICT_COLS:
        print(f"  Expanded dict-string columns: {DICT_COLS}")
    return df


# ═══════════════════════════════════════════════════════════
# COLUMN CLASSIFICATION
# ═══════════════════════════════════════════════════════════

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


def classify_columns(df: pl.DataFrame):
    numeric_cols     = [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]
    categorical_cols = [c for c in df.columns if df[c].dtype not in NUMERIC_DTYPES]
    return numeric_cols, categorical_cols


# ═══════════════════════════════════════════════════════════
# SECTIONS
# ═══════════════════════════════════════════════════════════

SEP  = "─" * 72
SEP2 = "═" * 72


def section_overview(df: pl.DataFrame):
    print(f"\n{SEP2}")
    print("  DATASET OVERVIEW")
    print(f"{SEP2}")
    print(f"  Rows   : {df.height:,}")
    print(f"  Columns: {df.width}")
    print()

    null_counts = df.null_count()

    print(f"  {'Column':<45} {'Dtype':<15} {'Nulls':>9}  {'%':>5}")
    print(f"  {'─'*45} {'─'*15} {'─'*9}  {'─'*5}")
    for col in df.columns:
        nulls = null_counts[col][0]
        pct = nulls / df.height * 100 if df.height else 0
        disp = (col[:43] + "…") if len(col) > 44 else col
        print(f"  {disp:<45} {str(df[col].dtype):<15} {nulls:>9,}  {pct:>4.1f}%")


def section_numeric(df: pl.DataFrame, numeric_cols: list):
    if not numeric_cols:
        return
    print(f"\n{SEP2}")
    print(f"  NUMERIC COLUMNS  ({len(numeric_cols)})  — Polars expression-based stats")
    print(f"{SEP2}")

    # Build all stats in ONE select call — this is the Polars idiomatic pattern.
    # Instead of looping and calling methods, we describe the computation as
    # a list of expressions and let Polars execute them in parallel.
    exprs = []
    for col in numeric_cols:
        exprs += [
            pl.col(col).count().alias(f"{col}__count"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
            pl.col(col).std(ddof=0).alias(f"{col}__std_pop"),
            pl.col(col).std(ddof=1).alias(f"{col}__std_smp"),
            pl.col(col).median().alias(f"{col}__median"),
        ]

    result = df.select(exprs)

    for col in numeric_cols:
        count   = result[f"{col}__count"][0]
        mean_v  = result[f"{col}__mean"][0]
        min_v   = result[f"{col}__min"][0]
        max_v   = result[f"{col}__max"][0]
        std_pop = result[f"{col}__std_pop"][0]
        std_smp = result[f"{col}__std_smp"][0]
        median  = result[f"{col}__median"][0]
        nulls   = df.height - (count if count else 0)
        miss_pct = nulls / df.height * 100 if df.height else 0

        def f(v): return f"{v:,.4f}" if v is not None else "N/A"

        print(f"\n  ┌ {col}  [numeric]")
        print(f"  │  non-null : {count:>10,}   nulls    : {nulls:>8,}  ({miss_pct:.1f}%)")
        print(f"  │  mean     : {f(mean_v):>14}   std (pop): {f(std_pop):>12}")
        print(f"  │  std(smp) : {f(std_smp):>14}   (use ddof=0 to match pure Python)")
        print(f"  │  min      : {f(min_v):>14}   max      : {f(max_v):>12}")
        print(f"  │  median   : {f(median):>14}")
        print(f"  └{'─'*60}")


def section_categorical(df: pl.DataFrame, categorical_cols: list):
    if not categorical_cols:
        return
    print(f"\n{SEP2}")
    print(f"  CATEGORICAL COLUMNS  ({len(categorical_cols)})")
    print(f"{SEP2}")

    for col in categorical_cols:
        series = df[col].cast(pl.Utf8).drop_nulls()
        n_valid  = len(series)
        n_nulls  = df.height - n_valid
        n_unique = series.n_unique()
        miss_pct = n_nulls / df.height * 100 if df.height else 0

        # value_counts() — Polars returns a DataFrame with (value, count) columns
        vc = series.value_counts(sort=True)
        # Column names differ slightly across Polars versions
        val_col = vc.columns[0]
        cnt_col = vc.columns[1]

        mode_val  = vc[val_col][0]  if len(vc) else "N/A"
        mode_freq = vc[cnt_col][0]  if len(vc) else 0
        top5 = vc.head(5)

        print(f"\n  ┌ {col}  [categorical]")
        print(f"  │  non-null : {n_valid:>10,}   nulls   : {n_nulls:>8,}  ({miss_pct:.1f}%)")
        print(f"  │  unique   : {n_unique:>10,}   mode    : {str(mode_val)[:42]}")
        print(f"  │  top 5 values  (via .value_counts(sort=True)):")
        for row in top5.iter_rows():
            v, c = row[0], row[1]
            pct = c / n_valid * 100 if n_valid else 0
            disp = (str(v)[:52] + "…") if len(str(v)) > 53 else str(v)
            print(f"  │      {c:>8,}  ({pct:5.1f}%)  {disp}")
        print(f"  └{'─'*60}")


def section_grouped(df: pl.DataFrame, key_cols: list, numeric_cols: list, label: str):
    print(f"\n{SEP}")
    print(f"  GROUPED ANALYSIS — {label}  (Polars)")
    print(f"  Key columns : {key_cols}")
    print(SEP)

    if not numeric_cols:
        print("  No numeric columns to aggregate.")
        return

    # Polars group_by + agg using expressions — the idiomatic pattern.
    # Each expression is independent; Polars may execute them in parallel.
    agg_exprs = []
    for col in numeric_cols:
        agg_exprs += [
            pl.col(col).count().alias(f"{col}__count"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
            pl.col(col).std(ddof=0).alias(f"{col}__std_pop"),
            pl.col(col).median().alias(f"{col}__median"),
        ]

    grouped = (df.group_by(key_cols)
                 .agg(agg_exprs)
                 .sort(key_cols))

    # Group sizes
    sizes = (df.group_by(key_cols)
               .agg(pl.len().alias("n"))
               .sort("n", descending=True))

    n_groups = grouped.height
    size_col = sizes["n"]

    print(f"\n  Groups found: {n_groups:,}")
    print(f"  Group size  —  min={size_col.min()}  max={size_col.max()}  "
          f"mean={size_col.mean():.1f}  median={size_col.median()}")

    print(f"\n  Preview (top 10 groups by size):")
    # Get top-10 keys by size, then show their aggregated stats
    top_keys = sizes.head(10).select(key_cols)
    preview = grouped.join(top_keys, on=key_cols, how="inner")
    print(preview.head(10))
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Polars descriptive stats + grouped analysis (Milestone A)"
    )
    parser.add_argument("file", help="Path to the CSV file")
    args = parser.parse_args()

    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    print(f"\n{SEP2}")
    print("  POLARS DESCRIPTIVE STATISTICS  —  Milestone A")
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

    cols_lower = {c.lower(): c for c in df.columns}
    group_configs = []

    if "page_id" in cols_lower:
        pid = cols_lower["page_id"]
        group_configs.append(([pid], "by page_id"))
        if "ad_id" in cols_lower:
            aid = cols_lower["ad_id"]
            group_configs.append(([pid, aid], "by page_id + ad_id"))

    if not group_configs:
        print("  No page_id / ad_id columns found — skipping grouped analysis.")
    else:
        for key_cols, label in group_configs:
            section_grouped(df, key_cols, numeric_cols, label)

    print(f"\n{SEP2}")
    print("  POLARS ANALYSIS COMPLETE")
    print(f"{SEP2}\n")


if __name__ == "__main__":
    main()
