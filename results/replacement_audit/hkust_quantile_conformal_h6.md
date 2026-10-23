# HKUST quantile and conformal pilot (h=6)

Five LightGBM quantiles are trained on 2021, calibrated on 2022 and scored on untouched 2023.

| Interval | Nominal | Coverage | Mean width | N |
|---|---:|---:|---:|---:|
| raw_80 | 80% | 82.21% | 0.167020 | 199832 |
| conformal_80 | 80% | 92.42% | 0.202511 | 199832 |
| raw_90 | 90% | 92.51% | 0.208441 | 199832 |
| conformal_90 | 90% | 92.51% | 0.208441 | 199832 |

Approximate CRPS: **0.033221**. Conformal corrections are learned from 2022 only; this is a preliminary global interval check, not the final regime-conditional ladder.
