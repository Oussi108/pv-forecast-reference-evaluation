# HKUST LightGBM h=12 pilot

Pooled LightGBM trained on 2021 (116,556 rows), early-stopped on 2022 (205,359 rows), and scored once on untouched 2023 (199,556 rows) across 24 systems.

| Aggregate | N systems | N rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| pooled | — | 199556 | 0.049694 | 0.100934 | -0.007490 |
| system-balanced mean | 24 | 199556 | 0.049723 | 0.099927 | -0.007498 |

No 2023 model choice, calibration or threshold selection was performed. Per-system predictions remain in the private output directory.
