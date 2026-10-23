# HKUST LightGBM h=2 pilot

Pooled LightGBM trained on 2021 (116,587 rows), early-stopped on 2022 (205,357 rows), and scored once on untouched 2023 (200,403 rows) across 24 systems.

| Aggregate | N systems | N rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| pooled | — | 200403 | 0.047944 | 0.088841 | 0.021633 |
| system-balanced mean | 24 | 200403 | 0.047887 | 0.088279 | 0.021465 |

No 2023 model choice, calibration or threshold selection was performed. Per-system predictions remain in the private output directory.
