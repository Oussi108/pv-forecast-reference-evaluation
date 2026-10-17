# Paper 1 v43 upgrade — pre-specified protocol (frozen before any forecast is scored)

Written 26 September 2026, before any forecast for the new testbed was produced or scored. The only data
inspected beforehand are file headers, provider metadata, per-year coverage counts and a 2021 (development-year)
mean-power-by-hour profile used to confirm the local-time convention. The SHA-256 of this file and of
`protocol_v43_pvdaq9068.json` are recorded in `protocol/FREEZE.json` before training starts.

## 1. Purpose

Address the three main weaknesses of v42: (i) only two testbeds, (ii) untuned neural configurations, and
(iii) a retrospective (previously examined) 2023 test year. This upgrade adds an independent third testbed whose
2023 and 2024 labels have never been examined, and a pre-specified neural-configuration sensitivity analysis.

## 2. New testbed

- **PVDAQ system 9068** (NREL PVDAQ, 2023 Solar Data Prize), Colorado, USA; 40.3864 N, 104.5512 W, 1407 m;
  Köppen BSk; single-axis tracking utility-scale array; provider licence CC BY 4.0 (OEDI submission 4568,
  DOI 10.25984/1846021).
- **Power**: sum of the two inverters' AC power (both required in an hour), normalized by the 4,738 kW DC nameplate.
  The revenue meter was not used because it fails in 2024 (daytime ratio to the inverter sum ~0.5 from January, then
  zero or negative from August), whereas the two agree closely in 2021–2023 (correlation ≥ 0.99, ratio 0.97);
  see `results/a02_power_source.json`. This decision was made from data-quality checks only.
- **Irradiance channel**: mean of the two measured plane-of-array pyranometers (not GHI; the site has no GHI
  sensor). This is a documented deviation from the two original testbeds.
- **Temperature**: site weather-station ambient temperature.
- **Humidity**: NOAA ISD station 724768-24051 (Greeley–Weld County Airport, 8.8 km), relative humidity from
  hourly temperature and dew point (Magnus form); public-domain US Government data. A METAR near :53 is
  assigned to the completed hour it precedes, so it is available at the issue time.
- **Time**: local civil time **with daylight saving (America/Denver)**. The provider metadata code (7) suggested fixed
  UTC−7, but the 2021 mean-power centroid jumps by about one hour exactly at the March and November clock changes
  and aligns with the clear-sky centroid only under America/Denver. Previous-day persistence uses the absolute
  24-hour lag, so it is one clock hour off on the two transition days each year. Completed-hour bins `[t−1 h, t)`, 75% native-support rule,
  no interpolation, identical range checks except the lower temperature bound (−40 °C instead of −30 °C,
  because the site's winters fall below −30 °C; decided before any forecast).

## 3. Chronology and populations

2021 development (Jan–Sep selection fit, Oct–Dec validation), 2022 reference selection and interval
calibration, **2023 primary evaluation** (data end 16 Nov 2023 in the main provider files) and
**2024 confirmation**. The analysis populations are all valid daylight target windows (clear-sky GHI > 20 W m−2)
in each year; no month or target is removed on the basis of forecast outcomes.

## 4. Models, references and policies (unchanged code)

LightGBM, GRU and Transformer with the v28 configuration and seeds 11, 23, 37, 51, 73, trained with the
unchanged `train_corrected_study.py`; quantile models (LightGBM, Transformer) at 6 h. References: persistence,
previous day, hour climatology, smart persistence (any-daylight form (a)), specified smart persistence (choice among
the v42 candidates (a)–(c) by lowest 2022 six-hour daylight MAE; for this site "clear-sky GHI" in (a)/(b) remains
the pvlib Ineichen GHI), convex previous-day–climatology reference (weight selected on Oct–Dec 2021).
Update policies: frozen 2021; annual (previous calendar year); monthly (trailing 365 days), with the development-
selected rounds/epochs, for all three families at 6 h and for LightGBM at 2, 4, 6 and 12 h. For 2024 the annual
fit uses 2023 and monthly fits use the trailing 365 days available at each monthly cutoff.

## 5. Pre-specified contrasts and inference

Skill = 1 − MAE_model/MAE_reference, seed-wise absolute losses averaged before aggregation; 2000 synchronized
non-circular seven-day calendar-block resamples (seed 42); Holm within comparison type × year × lead.

- **C1** frozen model vs previous day at 6 h (each family).
- **C2** frozen model vs smart (a) and vs specified smart at 6 h.
- **C3** monthly vs frozen, monthly vs previous day, monthly vs convex at 6 h (each family).
- **C4** LightGBM nominal-80% intervals: daylight coverage raw vs all-hour vs day/night; interval-score skill of
  variability grouping vs day/night.
- **C5** frozen models vs previous day at 2, 4, 6 and 12 h on matched daylight keys.

Each contrast is reported for 2023 and 2024, whatever the outcome.

## 6. Pre-stated expectations (derived from DKASC and HKUST, to be confirmed or refuted)

- **E1** No frozen family has a 6-h skill interval above zero against previous-day persistence.
- **E2** Specified smart persistence has lower 6-h daylight MAE than form (a).
- **E3** Monthly refitting lowers 6-h MAE relative to the frozen fit for every family.
- **E4** An all-hour correction changes LightGBM daylight coverage by less than 1 percentage point, whereas the
  day/night correction increases it.
- **E5** Variability grouping has no resolved interval-score benefit over day/night correction.

An expectation is "confirmed" only if the stated direction holds and, where an interval is involved, the 95%
interval supports it; otherwise it is reported as not confirmed.

## 7. Neural-configuration sensitivity (all three testbeds)

Purpose: test whether the v42 conclusions depend on the compact neural configuration (at DKASC the default
runs reached the 40-epoch cap). For GRU and Transformer at 6 h, the grid {width 32, 64} × {1, 2 layers} with
maximum 150 epochs and patience 10 (learning rate 0.001, dropout 0.1, batch 1024) is fitted with seed 11 on
Jan–Sep 2021 and the configuration with the lowest Oct–Dec 2021 validation MAE is selected per family and testbed.
The selected configuration is refitted on all of 2021 for seeds 11, 23, 37, 51, 73 and evaluated on the same
2023 daylight keys. Reported: tuned vs default MAE, tuned skill vs previous-day and specified smart persistence.
No test-period information is used for selection. For DKASC and HKUST this analysis is retrospective (2023 already
examined); for PVDAQ 9068 it is confirmatory.

## 8. What would change the paper's claims

If E1–E5 fail at the new site, the manuscript will say so and narrow its claims to the two original testbeds.
If tuned neural models beat previous-day persistence where default ones did not, the "no frozen model beats
previous day" statement will be restricted to the default configurations.
