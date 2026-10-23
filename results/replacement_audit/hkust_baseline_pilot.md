# HKUST baseline pilot

Fit-free metrics on untouched 2023 rows from the 24 eligible systems. Hourly climatology is estimated only from pre-2023 rows; no learned model or test-year calibration is used.

| Horizon | Reference | Systems | MAE | RMSE | Bias |
|---:|---|---:|---:|---:|---:|
| 2 h | day_persistence | 24 | 0.088998 | 0.158797 | 0.000299 |
| 2 h | hourly_climatology | 24 | 0.092795 | 0.156067 | 0.008122 |
| 2 h | persistence | 24 | 0.078124 | 0.138985 | -0.000043 |
| 4 h | day_persistence | 24 | 0.141058 | 0.232032 | 0.000276 |
| 4 h | hourly_climatology | 24 | 0.144510 | 0.222987 | 0.008963 |
| 4 h | persistence | 24 | 0.137808 | 0.227034 | -0.000067 |
| 6 h | day_persistence | 24 | 0.184825 | 0.283781 | 0.000217 |
| 6 h | hourly_climatology | 24 | 0.190063 | 0.272544 | 0.010052 |
| 6 h | persistence | 24 | 0.183454 | 0.282111 | -0.000041 |
| 12 h | day_persistence | 24 | 0.231381 | 0.317573 | 0.000008 |
| 12 h | hourly_climatology | 24 | 0.239856 | 0.305286 | 0.011345 |
| 12 h | persistence | 24 | 0.231294 | 0.317488 | -0.000066 |

The pilot is a sanity check and does not establish that a learned model adds value. The next stage requires the pinned LightGBM dependency and a predeclared training/validation contract.
