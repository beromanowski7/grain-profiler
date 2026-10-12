# GRAIN Profiler v0.1

Local-first exploratory profiling for unfamiliar Excel and CSV datasets.

## Features
- Column traits: data types, cardinality, missingness, entropy, concentration, numeric ranges, zeros and negatives, and dates
- Directional categorical relationship heatmap using Theil's U
- Exact functional-dependency audit, conflicting keys, and in-sample mapping agreement
- Minimal candidate composite-key discovery (bounded search)
- Held-out categorical lookup validation with baseline and coverage
- Exportable multi-sheet Excel evidence workbook

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests with `python -m pytest -q`.

## How to interpret results
- **Theil's U** is directional entropy reduction, not a probability of correctness.
- **Exact dependency** means no violations *in the observed dataset*, not proof of a universal business rule.
- **Row agreement** is in-sample majority-mapping accuracy and can be optimistic.
- **Held-out accuracy** is a simple lookup-model test, not a confidence score. Compare against baseline and key coverage.
- Unique IDs can trivially determine all attributes; watch predictor uniqueness.
- Random splitting can leak repeated entity information. Select a datetime column for chronological validation where appropriate.
- Constant target columns have undefined normalized uncertainty (shown blank).
- Numeric-coded categories (<=20 distinct values) are included; continuous numeric and datetime columns are excluded from the dependency heatmap by default.
- This initial release does not automatically infer numeric arithmetic identities, comprehensive temporal event semantics, or generalize to unseen categorical keys.
- Pairwise categorical contingency tables can be expensive on high-cardinality datasets. Use a small column limit first.

## Security
Run locally for internal data. Never commit sensitive, proprietary, personal, or export-controlled data to public GitHub. `.gitignore` excludes common local data formats but cannot replace a data-handling review.

## Roadmap
Numeric relationships; time-series/snapshot detection; permutation baselines; composite functional dependencies; exceptions drill-down; configurable split strategies; HTML report.
