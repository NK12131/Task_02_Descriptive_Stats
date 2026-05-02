# Task 02 — Descriptive Statistics with Polars + Grouped Analysis (Milestone A)

Three independent scripts producing identical descriptive statistics — at dataset level
and grouped level — using pure Python, Pandas, and Polars.

---

## Dataset

**Source:** [2024 Facebook Political Ads — Google Drive](#) 

Each row is a Facebook ad purchase by an organisation whose creative content references
one or more 2024 presidential candidates. This is a **different file** from Task 01 —
do not assume the schema is identical.

---

## Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

`pure_python_stats.py` has zero external dependencies.

---

## Running the Scripts

All three scripts accept the CSV path as a positional argument:

```bash
python pure_python_stats.py  path/to/facebook_ads.csv
python pandas_stats.py       path/to/facebook_ads.csv
python polars_stats.py       path/to/facebook_ads.csv
```

Each script prints:
1. Dataset overview (row count, column count, missing values, inferred types)
2. Per-column statistics
   - Numeric: count, mean, min, max, std (population), median
   - Categorical: count, unique, mode, top-5 values
3. Grouped analysis
   - By `page_id`
   - By `page_id` + `ad_id`

---

## What Each Script Computes

| Statistic | Pure Python | Pandas | Polars |
|---|:---:|:---:|:---:|
| Row / column count | ✓ | ✓ | ✓ |
| Missing values per column | ✓ | ✓ | ✓ |
| Numeric: count, mean, min, max | ✓ | ✓ | ✓ |
| Numeric: std (population, ddof=0) | ✓ | ✓ | ✓ |
| Numeric: median | ✓ | ✓ | ✓ |
| Categorical: count, unique, mode | ✓ | ✓ | ✓ |
| Categorical: top-5 values | ✓ | ✓ | ✓ |
| Grouped by `page_id` | ✓ | ✓ | ✓ |
| Grouped by `page_id` + `ad_id` | ✓ | ✓ | ✓ |

**Note on std deviation:** All three scripts use population std (ddof=0) for
cross-script comparability. Pandas and Polars default to sample std (ddof=1);
this is overridden explicitly in both library scripts.

---

## Key Design Decisions

### Dict-string columns
`spend`, `impressions`, and `estimated_audience_size` are stored as
`{'lower_bound': '45000', 'upper_bound': '49999'}` strings. All three scripts
parse these with `ast.literal_eval()` and add a `_lower` numeric column before
computing any statistics. Without this step, Pandas would silently exclude these
columns from numeric summaries — a meaningful silent failure.

### Grouped analysis (pure Python)
The `build_groups()` function manually partitions every row into a dict keyed by
the grouping columns, then runs `compute_numeric_stats()` on each group. This is
the literal equivalent of what `pandas.groupby()` does internally — writing it
once makes the abstraction visible.

---

## Approach Comparison

See [REFLECTION.md](REFLECTION.md) for a detailed comparison covering:
- Numerical consistency (std deviation, null handling)
- Performance differences
- Developer experience and learning curve
- AI tool observations
- Data cleaning requirements

---

## Project Structure

```
Task_02_Descriptive_Stats/
├── pure_python_stats.py   # Standard library only
├── pandas_stats.py        # Pandas-based analysis
├── polars_stats.py        # Polars-based analysis
├── requirements.txt       # pandas, polars
├── REFLECTION.md          # Comparative analysis + research question responses
└── README.md
```

---

## Reproducibility

Running any of the three scripts against the same input file produces the same
numerical results (within floating-point precision). The only cross-script
divergence point is std deviation if `ddof` is not normalised — all three scripts
in this project explicitly use population std (ddof=0).
