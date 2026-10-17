# Publication control evidence

These aggregate results support the v29 manuscript. The experiment is defined
in `../../configs/publication_revision_v29.json` and adds controls to the
preserved v28 Array 1B/HKUST study. It is retrospective. The 2024 file had
already been source-audited and is not an untouched independent holdout.

## Read the evidence

| File | Content |
|---|---|
| `verification.json` | Completed-bundle, temporal cutoff, source-hash and reproduction checks |
| `control_tests.json` | Finite-sample rank and test-label-poisoning checks |
| `calibration_metrics.csv` | Coverage, width and interval score for four forecasts and four correction policies at 80% and 90% |
| `calibration_inference.csv` | Paired calibration-control and raw-distribution comparisons with pointwise intervals and Holm HAC p-values |
| `calibration_groups.csv` | Calibration-group sizes, corrections and fallback flags |
| `distribution_metrics.csv` | Raw truncated CRPS7 and mean pinball scores |
| `updating_metrics_2023.csv` | Frozen, annual and monthly LightGBM policies at 2 and 6 h on both testbeds, plus point references |
| `updating_metrics_2024.csv` | Same comparisons on the retained 2024 Array 1B targets |
| `updating_inference_2023.csv`, `updating_inference_2024.csv` | Paired daylight MAE skill against previous-day persistence and the frozen model |
| `updating_monthly_2023.csv`, `updating_monthly_2024.csv` | Descriptive monthly scores without month-specific confidence claims |
| `temporal_coverage.csv` | All 12 months, calendar daylight denominators and retained counts |
| `temporal_preparation.json` | Reproduction of earlier arrays before extending the source period |
| `update_runs/*.json` | Thirty signed seed/horizon/year bundles; 380 additional fits, no minimum-support fallbacks |
| `execution.jsonl` | Append-only execution events and timestamps |

`daylight` means deterministic target clear-sky GHI above 20 W m⁻². `all`
includes nights. Do not compare the two populations as if their denominators
were identical. Each point score averages seed-wise absolute losses and then
system means equally. Quantile-model outputs instead average quantiles across
seeds before scoring and calibration.

Skill is `1 - loss / reference_loss`. CSV skill and confidence bounds are
fractions: `0.06925` is approximately **6.9%**, not 0.06925%. Zero skill for a
reference compared with itself is definitional. Negative skill favors the
reference. Intervals use 2,000 synchronized non-circular seven-day calendar-block
resamples and remain pointwise. HAC p-values are Holm-adjusted within the
declared year/testbed/analysis/metric family; they are not the same procedure as
the percentile intervals.

## Main interpretation

- On 2023 HKUST six-hour daylight targets, monthly-refit LightGBM MAE is
  0.09062 versus 0.09736 for previous-day persistence: +6.9% skill, pointwise
  95% interval 2.5% to 10.9%, Holm HAC p = 0.0083.
- On the corresponding DKASC population, monthly-refit skill against the same
  reference is -1.2%, interval -12.5% to 6.3%. The original frozen-model result
  cannot be generalized to all update policies, and updating is not uniformly beneficial.
- Day/night correction reproduces the principal HKUST LightGBM 80% interval
  improvement. Further grouping adds -0.020% score skill, interval -0.161% to
  0.104%. A secondary HKUST Transformer comparison at 90% does show a small
  positive incremental group effect: +0.622%, interval 0.444% to 0.814%.
- Previous-day residual distributions narrow the learned forecasts' apparent
  advantage over smart-residual persistence. Both uncertain and unfavorable
  learned-model comparisons remain in the tables.
- The 2024 six-hour extension retains 1,136 daylight targets on 118 dates,
  only 26.6% of calendar daylight hours; 84.4% occur during August-November.
  It does not repair incomplete annual coverage or add an independent location.

Raw observations, prediction arrays and saved models remain in private
workspace storage outside this release. This directory contains aggregate
scientific evidence, not a redistribution of the source datasets.
