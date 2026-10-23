# HKUST protocol-readiness inventory

This generated record joins the mechanical audit, metadata parser and conservative system map. It contains no observations and is not a model result.

## Counts

- Site-level target files: **60**; provisional pass: **34**; review: **26**.
- Shared weather files: **20**; inverter exports excluded: **44**; rainfall workbooks: **1**.
- Median target cadence distribution: `{15.0: 37, 60.0: 23}`. The analysis grid is explicitly hourly.

## Mapping and overlap

- Mapping rows: **60**, mapped: **60**, unique Brick systems: **60**, unresolved: **0**.
- Passing-candidate overlap: `2021-08-01T00:00:00+08:00` to `2023-12-31T10:00:00+08:00`; non-empty: **True**.

## Locked protocol record

Development = 2021, calibration = 2022, untouched test = 2023; primary horizon = 6 h; secondary horizons = 2, 4 and 12 h. Naive timestamps are provisionally interpreted as Asia/Hong_Kong until the provider convention is confirmed.

## Gate

**REVIEW — model fitting is not authorized yet.** Complete unit/capacity checks, confirm timezone, decide rainfall treatment, build a split-aware weather join, and write the causal preprocessing manifest before fitting any model.
