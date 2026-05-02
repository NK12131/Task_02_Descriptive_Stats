"""
pure_python_stats.py  —  Task 02 / Milestone A
Descriptive statistics + grouped analysis using ONLY the Python standard library.

"""

import csv
import math
import sys
import os
import ast
import argparse
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════

def load_csv(filepath: str):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            col_data = {}
            with open(filepath, newline="", encoding=enc) as fh:
                reader = csv.DictReader(fh)
                headers = list(reader.fieldnames or [])
                for h in headers:
                    col_data[h] = []
                n_rows = 0
                for row in reader:
                    n_rows += 1
                    for h in headers:
                        col_data[h].append(row.get(h, ""))
                    if n_rows % 50_000 == 0:
                        print(f"    ... {n_rows:,} rows loaded")
            print(f"  Loaded {n_rows:,} rows x {len(headers)} columns  [{enc}]")
            return headers, col_data, n_rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file: {filepath}")


# ═══════════════════════════════════════════════════════════
# TYPE HELPERS
# ═══════════════════════════════════════════════════════════

_MISSING_SENTINELS = {"", "nan", "none", "n/a", "na", "null", "undefined"}

def is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in _MISSING_SENTINELS


def try_float(v):
    if is_missing(v):
        return None, False
    cleaned = str(v).strip().replace(",", "").replace("$", "")
    try:
        return float(cleaned), True
    except ValueError:
        return None, False


def infer_type(values: list) -> str:
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        return "categorical"
    numeric_count = sum(1 for v in non_missing if try_float(v)[1])
    return "numeric" if numeric_count / len(non_missing) >= 0.80 else "categorical"


# ═══════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════

def compute_numeric_stats(raw_values: list) -> dict:
    nums = []
    missing = 0
    for v in raw_values:
        f, ok = try_float(v)
        if ok:
            nums.append(f)
        else:
            missing += 1

    if not nums:
        return dict(count=0, missing=missing,
                    mean=None, min=None, max=None, std=None, median=None)

    n = len(nums)
    mean_val = sum(nums) / n
    variance = sum((x - mean_val) ** 2 for x in nums) / n   # population std
    std_val  = math.sqrt(variance)
    s = sorted(nums)
    mid = n // 2
    median_val = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

    return dict(
        count=n, missing=missing,
        mean=round(mean_val, 4),
        min=min(nums), max=max(nums),
        std=round(std_val, 4),
        median=median_val,
    )


def compute_categorical_stats(raw_values: list, top_n: int = 5) -> dict:
    non_missing = [str(v).strip() for v in raw_values if not is_missing(v)]
    missing = len(raw_values) - len(non_missing)
    if not non_missing:
        return dict(count=0, missing=missing, unique=0,
                    mode=None, mode_freq=0, top5=[])
    freq = Counter(non_missing)
    most_common = freq.most_common()
    return dict(
        count=len(non_missing),
        missing=missing,
        unique=len(freq),
        mode=most_common[0][0],
        mode_freq=most_common[0][1],
        top5=most_common[:top_n],
    )


# ═══════════════════════════════════════════════════════════
# DICT-STRING EXPANSION
# ═══════════════════════════════════════════════════════════

def parse_dict_string(s):
    if is_missing(s):
        return None
    try:
        return ast.literal_eval(str(s).strip())
    except Exception:
        return None


def expand_dict_col(raw_values: list, bound: str = "lower_bound") -> list:
    """
    Return a list of float strings (or '') for the requested bound.
    Handles both plain numeric values AND Meta-style dict-strings.
    """
    result = []
    for v in raw_values:
        # Already a plain number — return it directly
        f, ok = try_float(v)
        if ok:
            result.append(str(f))
            continue
        # Try to parse as dict-string e.g. {'lower_bound': '45000', ...}
        d = parse_dict_string(v)
        if d is None or not isinstance(d, dict):
            result.append("")
        else:
            val = d.get(bound) or d.get("lower_bound")
            f, ok = try_float(val)
            result.append(str(f) if ok else "")
    return result


# ═══════════════════════════════════════════════════════════
# GROUPED ANALYSIS
# ═══════════════════════════════════════════════════════════

def build_groups(col_data: dict, n_rows: int, key_cols: list) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    all_cols = list(col_data.keys())
    for i in range(n_rows):
        key = tuple(str(col_data[c][i]).strip() for c in key_cols)
        for col in all_cols:
            groups[key][col].append(col_data[col][i])
    return dict(groups)


def summarise_groups(groups: dict, numeric_cols: list, key_cols: list) -> list:
    rows = []
    for key, grp in groups.items():
        entry = {key_cols[i]: key[i] for i in range(len(key_cols))}
        entry["group_size"] = len(next(iter(grp.values())))
        for col in numeric_cols:
            s = compute_numeric_stats(grp.get(col, []))
            for stat_name, stat_val in s.items():
                entry[f"{col}__{stat_name}"] = stat_val
        rows.append(entry)
    return rows


# ═══════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════

SEP  = "─" * 72
SEP2 = "═" * 72

def _fmt(v, dec=2):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:,.{dec}f}"
    return str(v)


def print_numeric_col(col, s, n_rows):
    miss_pct = s["missing"] / n_rows * 100 if n_rows else 0
    print(f"\n  ┌ {col}  [numeric]")
    print(f"  │  non-null : {s['count']:>10,}   missing : {s['missing']:>8,}  ({miss_pct:.1f}%)")
    print(f"  │  mean     : {_fmt(s['mean']):>14}   std (pop) : {_fmt(s['std']):>12}")
    print(f"  │  min      : {_fmt(s['min']):>14}   max       : {_fmt(s['max']):>12}")
    print(f"  │  median   : {_fmt(s['median']):>14}")
    print(f"  └{'─'*60}")


def print_categorical_col(col, s, n_rows):
    miss_pct = s["missing"] / n_rows * 100 if n_rows else 0
    print(f"\n  ┌ {col}  [categorical]")
    print(f"  │  non-null : {s['count']:>10,}   missing : {s['missing']:>8,}  ({miss_pct:.1f}%)")
    print(f"  │  unique   : {s['unique']:>10,}   mode    : {str(s['mode'])[:42]}")
    print(f"  │  top 5 values:")
    for val, cnt in s["top5"]:
        pct = cnt / s["count"] * 100 if s["count"] else 0
        disp = (str(val)[:52] + "…") if len(str(val)) > 53 else str(val)
        print(f"  │      {cnt:>8,}  ({pct:5.1f}%)  {disp}")
    print(f"  └{'─'*60}")


def print_group_results(group_results, key_cols, label, numeric_cols, top_n=10):
    print(f"\n{SEP}")
    print(f"  GROUPED ANALYSIS — {label}")
    print(f"  Key columns : {key_cols}")
    print(f"  Groups found: {len(group_results):,}")
    print(SEP)

    sorted_results = sorted(
        group_results, key=lambda r: r.get("group_size", 0), reverse=True
    )

    print(f"\n  Top {min(top_n, len(sorted_results))} groups by size:")
    for row in sorted_results[:top_n]:
        key_str = "  |  ".join(f"{c}={str(row.get(c,''))[:30]}" for c in key_cols)
        size = row.get("group_size", "?")
        sample = []
        for col in numeric_cols[:2]:
            mv = row.get(f"{col}__mean")
            if mv is not None:
                sample.append(f"{col}_mean={_fmt(mv)}")
        extra = "   ".join(sample)
        print(f"    [{key_str}]   n={size:,}   {extra}")

    sizes = [r.get("group_size", 0) for r in group_results]
    if sizes:
        n = len(sizes)
        s = sorted(sizes)
        mid = n // 2
        med = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        print(f"\n  Group size stats —  "
              f"min={min(sizes):,}  max={max(sizes):,}  "
              f"mean={sum(sizes)/n:.1f}  median={med}")
    print()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pure Python descriptive stats + grouped analysis (Milestone A)"
    )
    parser.add_argument("file", help="Path to the CSV file")
    args = parser.parse_args()

    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    print(f"\n{SEP2}")
    print("  PURE PYTHON DESCRIPTIVE STATISTICS  —  Milestone A")
    print(f"  File : {os.path.basename(filepath)}")
    print(f"{SEP2}\n")

    print("  Loading...")
    headers, col_data, n_rows = load_csv(filepath)

    # Expand dict-string columns if present (handles both plain ints and dict-strings)
    DICT_COLS = [c for c in headers if c in ("spend", "impressions", "estimated_audience_size")]
    for dc in DICT_COLS:
        new_col = f"{dc}_lower"
        col_data[new_col] = expand_dict_col(col_data[dc], "lower_bound")
        if new_col not in headers:
            headers.append(new_col)
    if DICT_COLS:
        print(f"  Expanded dict-string columns: {DICT_COLS}")

    # Dataset overview
    print(f"\n{SEP2}")
    print("  DATASET OVERVIEW")
    print(f"{SEP2}")
    print(f"  Rows   : {n_rows:,}")
    print(f"  Columns: {len(headers)}")
    print()
    print(f"  {'Column':<45} {'Type':<12} {'Missing':>9}  {'%':>5}")
    print(f"  {'─'*45} {'─'*12} {'─'*9}  {'─'*5}")
    col_types = {}
    for col in headers:
        t = infer_type(col_data[col])
        col_types[col] = t
        miss = sum(1 for v in col_data[col] if is_missing(v))
        miss_pct = miss / n_rows * 100 if n_rows else 0
        disp = (col[:43] + "…") if len(col) > 44 else col
        print(f"  {disp:<45} {t:<12} {miss:>9,}  {miss_pct:>4.1f}%")

    numeric_cols     = [c for c in headers if col_types[c] == "numeric"]
    categorical_cols = [c for c in headers if col_types[c] == "categorical"]

    # Per-column statistics
    print(f"\n{SEP2}")
    print(f"  NUMERIC COLUMNS  ({len(numeric_cols)})")
    print(f"{SEP2}")
    for col in numeric_cols:
        s = compute_numeric_stats(col_data[col])
        print_numeric_col(col, s, n_rows)

    print(f"\n{SEP2}")
    print(f"  CATEGORICAL COLUMNS  ({len(categorical_cols)})")
    print(f"{SEP2}")
    for col in categorical_cols:
        s = compute_categorical_stats(col_data[col])
        print_categorical_col(col, s, n_rows)

    # Grouped analysis
    print(f"\n{SEP2}")
    print("  GROUPED ANALYSIS")
    print(f"{SEP2}")

    h_lower = {h.lower(): h for h in headers}
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
            print(f"\n  Building groups [{label}] ...")
            groups = build_groups(col_data, n_rows, key_cols)
            group_results = summarise_groups(groups, numeric_cols, key_cols)
            print_group_results(group_results, key_cols, label, numeric_cols)

    print(f"\n{SEP2}")
    print("  PURE PYTHON ANALYSIS COMPLETE")
    print(f"{SEP2}\n")


if __name__ == "__main__":
    main()