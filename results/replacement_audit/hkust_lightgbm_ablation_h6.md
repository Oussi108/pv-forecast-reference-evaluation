# HKUST LightGBM feature ablation (h=6)

System-balanced metrics on untouched 2023. Each row removes one feature group from the full issue-time contract.

| Variant | Features | MAE | Δ MAE vs full | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| full | 16 | 0.059019 | +0.000000 | 0.109982 | 0.009281 |
| without_target_history | 10 | 0.060924 | +0.001904 | 0.109590 | 0.009486 |
| without_weather | 10 | 0.072726 | +0.013707 | 0.122147 | 0.013026 |
| without_calendar | 12 | 0.071600 | +0.012581 | 0.124334 | 0.008106 |

The gain-importance map is descriptive (LightGBM split gain), not a causal contribution claim. No 2023 tuning or calibration was performed.
