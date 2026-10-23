# HKUST LightGBM h=4 pilot

Pooled LightGBM trained on 2021 (116,486 rows), early-stopped on 2022 (205,294 rows), and scored once on untouched 2023 (200,126 rows) across 24 systems.

| Aggregate | N systems | N rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| pooled | — | 200126 | 0.055519 | 0.101919 | 0.010175 |
| system-balanced mean | 24 | 200126 | 0.055553 | 0.101360 | 0.010062 |

No 2023 model choice, calibration or threshold selection was performed. Per-system predictions remain in the private output directory.
