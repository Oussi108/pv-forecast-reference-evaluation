# HKUST regime-conditional conformal intervals (h=6)

Pooled quantile LightGBM models are trained on 2021, calibrated on 2022, and scored on untouched 2023. Regimes use issue-time capacity factor terciles frozen from 2021.

| Regime | Interval | Coverage | Mean width | N | Correction |
|---|---|---:|---:|---:|---:|
| overall | raw_80 | 81.04% | 0.170078 | 199780 | 0.000000 |
| overall | global_conformal_80 | 92.48% | 0.204224 | 199780 | 0.017073 |
| overall | regime_conformal_80 | 92.48% | 0.204224 | 199780 | 0.017073 |
| overall | raw_90 | 92.15% | 0.207225 | 199780 | 0.000000 |
| overall | global_conformal_90 | 92.15% | 0.207225 | 199780 | 0.000000 |
| overall | regime_conformal_90 | 92.15% | 0.207225 | 199780 | 0.000000 |
| Normal | raw_80 | 80.54% | 0.169013 | 97426 | 0.000000 |
| Normal | global_conformal_80 | 92.29% | 0.203159 | 97426 | 0.017073 |
| Normal | regime_conformal_80 | 93.56% | 0.216049 | 97426 | 0.023518 |
| Normal | raw_90 | 92.43% | 0.208132 | 97426 | 0.000000 |
| Normal | global_conformal_90 | 92.43% | 0.208132 | 97426 | 0.000000 |
| Normal | regime_conformal_90 | 93.14% | 0.208681 | 97426 | 0.000275 |
| Moderate | raw_80 | 83.45% | 0.209772 | 41372 | 0.000000 |
| Moderate | global_conformal_80 | 90.74% | 0.243918 | 41372 | 0.017073 |
| Moderate | regime_conformal_80 | 91.45% | 0.251999 | 41372 | 0.021113 |
| Moderate | raw_90 | 91.49% | 0.254136 | 41372 | 0.000000 |
| Moderate | global_conformal_90 | 91.49% | 0.254136 | 41372 | 0.000000 |
| Moderate | regime_conformal_90 | 91.49% | 0.254136 | 41372 | 0.000000 |
| Extreme | raw_80 | 80.20% | 0.144849 | 60982 | 0.000000 |
| Extreme | global_conformal_80 | 93.95% | 0.178995 | 60982 | 0.017073 |
| Extreme | regime_conformal_80 | 91.16% | 0.152961 | 60982 | 0.004056 |
| Extreme | raw_90 | 92.14% | 0.173952 | 60982 | 0.000000 |
| Extreme | global_conformal_90 | 92.14% | 0.173952 | 60982 | 0.000000 |
| Extreme | regime_conformal_90 | 92.14% | 0.173952 | 60982 | 0.000000 |

Corrections are estimated without accessing 2023 labels; regime-specific corrections fall back to the global correction only when the calibration stratum has fewer than 100 rows.
