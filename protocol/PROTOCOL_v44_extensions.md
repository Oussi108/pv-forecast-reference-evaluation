# Protocol v44: method extensions A–E (draft, not frozen)

Status: **draft**. It becomes binding only when its SHA-256 is recorded in `protocol/FREEZE_v44.json` and that commit is
public, before any A–E result on 2023 or 2024 data is computed. The v43 protocol, its freeze and the E1–E5 results are
unchanged. Results for A–E on the 2023 and 2024 data of Array 1B, HKUST and PVDAQ 9068 are **exploratory**, because
those years have already been scored for other contrasts. They become confirmatory only on data not yet scored by anyone.

Common rules (as v43 unless stated): fitting on 2021, every choice on 2022, testing on 2023 and 2024; skill =
1 − MAE_model / MAE_reference with seed-wise losses averaged; 2,000 synchronized non-circular seven-day calendar-block
resamples (seed 42); Holm within comparison type × testbed × year × lead. **One evidence rule for every claim:** a
contrast is "supported" when the 95% interval lies on the stated side of zero; the Holm-adjusted HAC p-value is
reported beside it, and every supported contrast with Holm p ≥ 0.05 is named as such in the text.

## A. Timestamp-alignment audit (`analysis/x01_alignment_audit.py`)

On the 2021 mean diurnal profile of each power series: `offset_h`, the whole-hour shift in {−2, …, +2} that maximizes
the correlation with the clear-sky profile, and `lag_h`, the power-weighted minus clear-sky-weighted mean hour. A series
is flagged when `offset_h ≠ 0` and `|lag_h| ≥ 0.5`; it is shifted by `offset_h` before windows are built, and the
shift is logged. Pass test: the audit must flag the HKUST hourly systems found late by the archive review and must not
flag DKASC Array 1B or PVDAQ 9068.

Disclosure: this rule was written after the misalignment was found, and its pass test has already been run on the
released 2021 profiles (`results/v44_extensions/x01_alignment_audit.csv`): it flags exactly 14 of 32 series, all
hourly HKUST files, and no 15-minute file or Array 1B. PVDAQ 9068 has `lag_h` = −0.17 h (`a02_audit.json`), so it is
not flagged. The raw-file check of the stamp convention (`check_hkust_timestamps.py`) decides how the fix is applied.

## B. Fitted optimal convex reference (`analysis/x02_optimal_reference.py`)

The plane-of-array smart persistence of the D01 audit is not used, because it was already scored on the 2023 keys
and was worse than the GHI forms. Instead, per lead and per target hour, non-negative weights summing to one over
(persistence, previous day, hour climatology, smart persistence) are fitted to 2022 daylight rows by least absolute
error (a linear program; at most 4,000 rows per hour drawn with seed 42; hours with fewer than 50 rows use the
all-hour weights). Contrasts: every frozen and updated model vs this reference, at 2, 4, 6 and 12 h.

## C. Drift-triggered updating (`analysis/x03_drift_updates.py`)

Page–Hinkley monitor (increase direction, δ = 0.5) on daily daylight MAE of the deployed 6-h model, standardized by the
mean and SD of the daily errors in the 2021 validation quarter. A day enters the monitor only after it has ended. On an
alarm the model is refitted on the targets of the preceding 365 days and serves every row issued from the next midnight;
refits are at least 30 days apart; the monitor restarts. λ ∈ {2, 4, 8, 16} is chosen by the lowest 2022 daylight MAE,
running the same policy through 2022 from the frozen 2021 fit. Families: LightGBM, GRU and Transformer with the
development-selected settings, five seeds. Contrasts at 6 h: drift vs frozen, drift vs annual, drift vs monthly, and
monthly vs annual. Also reported: number and dates of refits per testbed and year.

## D. Seed-aware inference (`analysis/x04_seed_bootstrap.py`)

Every resample draws the blocks exactly as the v43 bootstrap and, from a separate generator (seed 4242), draws the
five seeds with replacement; the model loss is the mean over the drawn seeds. Reported for every v43 and v44 contrast
beside the v43 interval, with the ratio of interval widths.

## E. Condition-aware calibration (`analysis/x05_conditional_calibration.py`)

LightGBM 80% and 90% central intervals. (1) Mondrian split-conformal correction by solar-elevation band of the target
hour (night; below 10°; 10–30°; at least 30°), calibrated on 2022, with a fallback to the global correction for a
band with fewer than 100 rows. (2) Adaptive conformal inference with γ = 0.005, separately for day and night, where
an error is used only after its target time. Compared with raw, all-hour and day/night corrections on daylight
coverage and on interval-score skill against day/night.

## What is reported whatever the outcome

All contrasts above, for every testbed and both test years, including those that favour the v43 methods.
