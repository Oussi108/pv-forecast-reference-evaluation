# Replacement-dataset audit outputs

This directory is reserved for the machine-readable screening report produced by
`scripts/audit_hkust_dataset.py` after the user obtains and extracts the HKUST
archive locally. The third-party archive and all raw or derived observations must
remain outside version control.

Expected outputs after an audit:

- `hkust_audit.json` — complete file-level report and thresholds;
- `hkust_audit_files.csv` — compact table for manual review.
- `hkust_metadata.csv` and `hkust_metadata_summary.json` — parsed Brick metadata;
- `hkust_system_map.csv` — conservative site-file-to-system mapping with audit status.
- `hkust_protocol_inventory.json` and `.md` — protocol-readiness decision joining
  the audit, mapping and metadata without selecting systems by model performance.
- `hkust_harmonisation_audit.json` and `.md` — split-aware hourly target-key,
  source-support and capacity-factor range checkpoint; no observations are saved.
- `hkust_weather_join_manifest.json` and `.md` — shared-weather field/year
  availability and leakage-safe join contract; no observations are saved.
- `hkust_hourly_join_manifest.json` and `.md` — private hourly target/weather
  joins for the eligible systems; only counts and hashes are public.
- `hkust_feature_windows_manifest.json` and `.md` — past-only feature schema and
  multi-horizon target-key counts; derived tables remain private.
- `hkust_baseline_pilot.json` and `.csv` — fit-free 2023 baseline metrics only.

The supplied archive has been audited outside the release tree. The current record
contains 60 site targets (34 provisional passes, 26 review), 20 weather files and
44 inverter exports excluded from the independent-site count. The protocol
inventory is intentionally `REVIEW`; an audit pass is not permission to fit models
until timezone, units, rainfall handling, causal joins and split-specific coverage
are documented.

The harmonisation checkpoint currently reports 24 systems meeting all year-level
coverage and physical-range checks. Six additional mechanical-pass systems contain
capacity-factor range flags; the raw values are preserved privately for review and
are not clipped automatically.

The weather manifest confirms all core fields across 2021–2023 but flags that
rainfall CSV coverage begins in 2022. Rainfall is therefore excluded from the
primary matched protocol; a 2021 workbook parser is required before any rainfall
sensitivity is reported.

The baseline pilot is a locked sanity check over the 24 eligible systems and four
horizons; it does not tune or fit a learned model.

The LightGBM runner is included and the pinned dependency was installed for this
run. h=2/4/6/12 metrics plus the h=6 grouped ablation are now present. The
ablation's gain-importance values are descriptive, not causal.

The consolidated horizon table is `hkust_horizon_comparison.md`; individual
LightGBM metrics are `hkust_lightgbm_h2.json`, `h4.json`, `h6.json` and `h12.json`
with matching CSV/Markdown summaries.

The preliminary probabilistic rung is recorded in
`hkust_quantile_conformal_h6.md` and `hkust_quantile_intervals_h6.csv`. The
regime-conditional sensitivity is recorded in
`hkust_regime_conformal_h6.{csv,json,md}`; it uses issue-time capacity-factor
terciles frozen from 2021 and 2022-only corrections.

`hkust_h6_bootstrap.md` reports a 5,000-replicate system-cluster bootstrap for
the h=6 LightGBM skill.

`hkust_h6_moving_block_bootstrap.md` adds a 2,000-replicate daily-block temporal
sensitivity. It is descriptive and should not be labelled as a formal DM test.

`cross_testbed_context_h6.{csv,json,md}` gives a traceable contextual comparison
with the existing locked DKASC/PVOD aggregate scores. Because the HKUST and
original contracts use different reference arms and test partitions, this is not
an apples-to-apples leaderboard.
