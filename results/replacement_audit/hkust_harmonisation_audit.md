# HKUST causal-harmonisation audit

This generated manifest checks hourly target keys for the provisional-pass site files. It contains no observations and is not a model result.

- Mechanical-pass candidates: **34**
- Systems harmonised: **34**
- Systems meeting all year-level coverage checks: **24**
- Systems with capacity-factor range flags: **6**
- File errors: **0**
- Detected weather fields: `{'irradiance': 3, 'rainfall': 2, 'humidity': 3, 'pressure': 3, 'temperature': 3, 'visibility': 3, 'wind': 3}`

## Declared rules

Naive timestamps are provisionally localised as `Asia/Hong_Kong`. Site-level power is aggregated to hourly means, requiring at least 75% of the expected source samples in an hour (minimum one sample for hourly exports). Capacity factor uses Brick nameplate capacity; values outside [-0.05, 1.20] are flagged and excluded from the eligible set until investigated. No interpolation or weather join is performed here; those operations must be fitted within each chronological split.

## Decision

**REVIEW — model fitting remains blocked.** This manifest is the target-key and physical-range checkpoint before implementing split-local weather joining and interpolation.
