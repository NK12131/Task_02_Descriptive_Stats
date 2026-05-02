# REFLECTION.md — Task 02 / Milestone A

## Comparing Three Approaches to Descriptive Statistics

---

### Was it a challenge to produce identical numerical results across all three approaches?

Yes, in two specific ways that are worth understanding.

**Standard deviation.** Pure Python computes population std (divide by N) because that is the most natural implementation from first principles. Pandas `.std()` and Polars `.std()` both default to *sample* std (divide by N−1, Bessel's correction), which is what you want when estimating a population parameter from a sample. At N = 246,745 rows the difference is negligible numerically, but it represents a real conceptual choice. All three scripts in this project explicitly use `ddof=0` (population) in the library scripts so that outputs match the pure Python version  and both values are printed side by side so the difference is visible.

**Null handling.** Pure Python requires every null check to be written by hand (`is_missing()`). Pandas silently drops NaN values from most aggregations (convenient but invisible). Polars is the most explicit: null semantics are documented per-function, and operations on null-containing columns fail loudly unless you specify what to do. The practical effect is that Polars is most likely to catch data quality issues early, Pandas is most likely to produce a plausible-looking result that quietly excludes rows, and pure Python does whatever you coded.

---

### Do you find one approach easier or more performant?

**Ease of use:** Pandas wins for day-to-day work. `describe()`, `value_counts()`, `groupby()` — the API maps closely to how analysts think about tables and has the widest documentation coverage.

**Performance:** Running all three scripts against the same 246,745-row dataset made the performance differences concrete:
- **Pure Python** took several minutes for the grouped analysis. The `build_groups()` loop iterates over every row and builds a dict of lists by hand exactly what both libraries replace with optimised C/Rust implementations.
- **Pandas** completed the full analysis in roughly 3–5 seconds.
- **Polars** was the fastest, finishing in under 2 seconds including the grouped analysis.

The gap between pure Python and the libraries is large even at this dataset size. At 10x or 100x the rows, pure Python would become impractical.

---

### If you were coaching a junior data analyst who had never used any of these tools, what approach would you recommend they learn first? Why?

I would start a junior analyst with Pandas, without hesitation. When I first approached this task, Pandas was the most natural fit the mental model of a table with named rows and columns maps directly to how anyone already thinks about spreadsheets, and when something went wrong there was always a Stack Overflow answer within the first three search results. That kind of friction reduction matters a lot when you are learning.

That said, working through all three approaches in this project changed how I think about Pandas. Writing the pure Python version first forced me to make every decision explicitly what counts as missing, what threshold makes a column numeric, how to partition rows into groups. When I then wrote the Pandas version, I could see exactly what `groupby()` and `describe()` were doing for me, because I had just done it by hand. I would recommend that sequence to anyone: write the manual version first, then appreciate what the library gives you.

I would introduce Polars as a second tool, not a first. The expression syntax takes adjustment if you are coming from Pandas, and the strict type system produces errors that Pandas would silently absorb which is actually a feature, but it feels like friction until you understand why. Once a junior analyst is fluent in Pandas, Polars starts to feel like a more disciplined version of the same ideas. Its lazy evaluation model is also the right mental model for eventually working with files too large to load into memory, which comes up sooner than most people expect.

The one thing I would tell a junior analyst about all three tools: do not trust output you have not verified. The most dangerous moment in this project was when Polars showed `estimated_audience_size_lower` as 100% null a column that should have had real values. Without the pure Python output to compare against, that failure could have gone unnoticed. No tool tells you when its output is wrong.

---

### Can AI tools produce useful starter code for each approach?

Yes, with important caveats. When prompted to "compute descriptive statistics for a CSV in Python," most AI tools default to Pandas + `describe()`, which is reasonable. Generated code is usually correct for the happy path (clean types, no nulls, standard encodings).

It tends to break silently on:
- Mixed-type columns (e.g. mostly integers but some `"N/A//NULL"` strings)
- Non-UTF-8 encodings
- The population vs. sample std distinction (tools rarely surface this)
- Columns whose names conflict with Pandas method names

For Polars specifically, AI tools sometimes generate deprecated API calls — Polars has moved fast and broken backward compatibility in several places. Always verify generated Polars code against the current user guide (`https://docs.pola.rs`).

The most useful AI output is not the finished script but the scaffold: the structure of the analysis, the list of statistics to compute, and the basic API calls. Plan to read and verify every line, especially null-handling and type-coercion code.

---

### What data cleaning was required? Did the three approaches handle it differently?

Unlike the Task 01 dataset where `spend`, `impressions`, and `estimated_audience_size` were stored as Meta Ad Library dict-strings like `{'lower_bound': '45000', 'upper_bound': '49999'}`, **this dataset stores those columns as plain integers**. No dict-string parsing was required. The scripts were written to handle both formats gracefully — if a value is already a number it is returned directly, and dict-string parsing is only attempted if the plain number conversion fails.

The main cleaning challenge in this dataset was correctly identifying column types. Several columns that look numeric at first glance (`page_id`, `ad_id`) are actually categorical identifiers and should never be averaged or summed. The 80% threshold in `infer_type()` handles this correctly — a column is only classified as numeric if at least 80% of its non-missing values parse as floats.

The 28 binary illuminating columns (0/1 flags) are a borderline case: they are numeric in the sense that their means are interpretable as prevalence rates, but treating their min/max/std as meaningful statistics requires understanding what they represent. All three scripts compute the stats correctly; the interpretation is left to the analyst.

**Key difference across approaches:**
- **Pure Python** — type inference is fully visible and controllable. Every decision about what counts as numeric or missing is explicit code.
- **Pandas** — loads columns as `int64` or `object` based on its own inference, which matched the actual types well for this dataset. Would silently exclude `object` columns from `describe()`.
- **Polars** — strict typing catches mismatches immediately. It committed to `Int64` for the numeric columns and `String` for categoricals, with no ambiguous `object` dtype in sight.

---

### Summary comparison table

| Dimension | Pure Python | Pandas | Polars |
|---|---|---|---|
| Lines of code | Most (200) | Medium (150) | Medium (160) |
| Learning curve | Low (familiar syntax) | Low | Higher |
| Type safety | Manual | Implicit | Strict |
| Null safety | Manual | Implicit (silent) | Explicit |
| Performance (this dataset) | Several minutes | 3–5 seconds | 1–2 seconds |
| Std dev default | Population (ddof=0) | Sample (ddof=1) | Sample (ddof=1) |
| Best for | Learning, no-dep environments | Most day-to-day analysis | Large files, production pipelines |