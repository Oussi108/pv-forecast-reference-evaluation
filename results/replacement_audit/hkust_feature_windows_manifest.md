# HKUST feature and target-window manifest

Private derived feature/label tables were generated from the hourly joins. No observations are included in the public release.

- Systems: **24**
- Total hourly rows: **542894**
- Window totals (h=2,4,6,12): `{'2': 522347, '4': 521906, '6': 521445, '12': 521471}`

Features use only issue-time weather and target lags. Labels are shifted to y(t+h), and rows with missing required features or labels are dropped without interpolation. Rainfall is excluded from the primary contract.

**REVIEW — model fitting remains blocked** until the generated window counts and target-key rules are reviewed and the development/calibration/test transforms are fit separately.
