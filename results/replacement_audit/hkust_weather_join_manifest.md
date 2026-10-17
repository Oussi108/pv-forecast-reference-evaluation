# HKUST weather-join readiness

This generated manifest records source-field availability for the shared HKUST weather station. It contains no weather observations and does not perform the causal join.

- Core fields available in all three years: **True**
- Rainfall treatment decision required: **False** (primary policy: excluded)

## Required join rule

Join weather at the issue timestamp (or an explicitly declared past-only aggregate), resample to the hourly grid, and interpolate only inside each chronological split. The shared station means uncertainty must be reported as fleet-level dependence, not as 24 independent weather sources.

## Current gate

**REVIEW — join implementation is not yet authorized.** Rainfall is excluded from the primary contract; produce a split-local joined-feature manifest for the core fields before fitting models.
