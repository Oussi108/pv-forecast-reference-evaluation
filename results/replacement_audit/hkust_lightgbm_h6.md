# HKUST LightGBM h=6 pilot

Pooled LightGBM trained on 2021 (116,377 rows), early-stopped on 2022 (205,236 rows), and scored once on untouched 2023 (199,832 rows) across 24 systems.

| Aggregate | N systems | N rows | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| pooled | — | 199832 | 0.059660 | 0.111301 | 0.010231 |
| system-balanced mean | 24 | 199832 | 0.059732 | 0.110898 | 0.010209 |

No 2023 model choice, calibration or threshold selection was performed. Per-system predictions remain in the private output directory.
