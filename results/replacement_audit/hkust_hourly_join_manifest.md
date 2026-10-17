# HKUST private hourly join manifest

This manifest records a private derived hourly join. No observations or derived tables are included in the public release.

- Eligible systems joined: **24**
- Private joined files: **24**
- Complete core-weather hourly keys: **26133 / 26280** (99.44%)

## Contract

Target power is aggregated to hourly means with at least 75% source support. One-minute core weather is averaged hourly and left-joined at the issue timestamp. No forward-fill or cross-split interpolation is applied. Rainfall is excluded from the primary contract.

## Gate

**REVIEW — model fitting remains blocked.** The next checkpoint is feature engineering and target-window construction with explicit issue-time cutoffs, followed by a locked baseline pilot.
