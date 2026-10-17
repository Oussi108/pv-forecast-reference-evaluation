# HKUST horizon comparison

System-balanced 2023 test metrics for the fit-free persistence reference and the pooled LightGBM model. LightGBM is trained on 2021 and early-stopped on 2022; 2023 is untouched.

| Horizon | Systems | Persistence MAE | LightGBM MAE | MAE skill | Persistence RMSE | LightGBM RMSE | Bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 h | 24 | 0.078124 | 0.047887 | 38.70% | 0.138985 | 0.088279 | 0.021465 |
| 4 h | 24 | 0.137808 | 0.055553 | 59.69% | 0.227034 | 0.101360 | 0.010062 |
| 6 h | 24 | 0.183454 | 0.059732 | 67.44% | 0.282111 | 0.110898 | 0.010209 |
| 12 h | 24 | 0.231294 | 0.049723 | 78.50% | 0.317488 | 0.099927 | -0.007498 |

Skill is `(persistence MAE − LightGBM MAE) / persistence MAE`; this is descriptive and has no confidence interval yet.
