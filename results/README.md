# Results: where each number in the paper comes from

All files here are aggregates: metrics, paired comparisons, interval records, run records and training curves. They contain
no observation series, prediction arrays or model files. MAE values are in normalized-power units (fraction of nominal
capacity). Skill is in percent relative to the named reference.

## Folders

| Folder | Produced by | Contents |
|---|---|---|
| `replacement_audit/` | `scripts/audit_hkust_dataset.py`, `parse_hkust_metadata.py`, `build_hkust_system_map.py` | HKUST screening: files, columns, system map, eligibility |
| `corrected_v28_array1b/` | `scripts/corrected_study.py`, `train_corrected_study.py`, `evaluate_corrected_study.py` | Array 1B and HKUST preparation record (`preparation.json`), frozen 2021 fits (`runs/`, `curves/`), point and interval metrics |
| `publication_v29/` | `scripts/publication_revision_v29.py` | Fixed-map interval calibration, LightGBM annual and monthly updates (2023 and Array 1B 2024), 2024 temporal support |
| `updates_2023/` | `updates/*.py` | Neural and LightGBM update fits for 2023 (`p03_updates/`, `p03_lightgbm_updates/`), input-removal fits (`p06_runs/`), reference diagnostics (`reference_*.csv`) |
| `reference_and_update_inference/` | `analysis/d01`–`d05` | Smart-persistence audit and specification, paired comparisons with Holm adjustment, block-length sensitivity, figure data; `tables/` is written by `d03_build_tables.py` |
| `v43_pvdaq9068/` | `scripts/train_corrected_study.py` under `protocol/protocol_v43_pvdaq9068.json` | PVDAQ 9068 frozen 2021 fits (`runs/`, `curves/`) |
| `pvdaq9068_evaluation/` | `analysis/a01`, `a02`, `t01`, `d06`–`d09` | PVDAQ 9068 download manifest, data audit, prepared-window counts, pre-specified evaluation, expectations, configuration sensitivity |
| `figures/` | `figures/*.py` | Figures 1–6 of the paper; `figures/jrse/Fig1`–`Fig6` are the journal versions (`figures/jrse_figures_1_2.py`, `figures/jrse_figures_3_to_6.py`) |

## Main paper

Table numbers follow the manuscript submitted to the *Journal of Renewable and Sustainable Energy* (Roman numerals); the file names keep Arabic numbers (`table2.csv` is Table II). The journal versions of Figures 1–6 are `figures/jrse/Fig1`–`Fig6`.

| Item | File(s) |
|---|---|
| Figure 1, Figure 2 | `figures/Figure1_study_workflow_v44.*`, `figures/Figure2_model_paths_v44.*` (`figures/build_figures_1_2.py`) |
| Table I | Array 1B and HKUST: `corrected_v28_array1b/preparation.json` (`populations`). Array 1B 2024: `publication_v29/temporal_preparation.json`. PVDAQ 9068: `pvdaq9068_evaluation/a02_prepared_windows.json` |
| Table II | `reference_and_update_inference/tables/table2.csv`; PVDAQ 9068 rows: `pvdaq9068_evaluation/d06_metrics_9068.csv` |
| Figure 3 | `reference_and_update_inference/d01_smart_persistence_audit_h6.json`, `d03_inference_holm_by_lead.csv` |
| Figure 4 | `reference_and_update_inference/d05_skill_by_lead.csv` |
| Table III | `reference_and_update_inference/tables/table3.csv`; PVDAQ 9068 rows: `pvdaq9068_evaluation/d06_metrics_9068.csv`, `d06_inference_9068.csv` |
| Table IV | `reference_and_update_inference/tables/table4_inference.json` (from `publication_v29/calibration_inference.csv`); PVDAQ 9068: `pvdaq9068_evaluation/d06_calibration_9068.csv` |
| Figure 5 | `reference_and_update_inference/d05_coverage_by_hour.csv` |
| Table V | `pvdaq9068_evaluation/d08_expectations.json` |
| Figure 6 | `pvdaq9068_evaluation/d06_inference_9068.csv`, `d09_matched_c5_9068.csv`, `d07_tuning_evaluation.csv` |
| Table VI | Reporting checklist (text); its evidence column points to the items above |
| Prose numbers | `reference_and_update_inference/tables/text_numbers.json`, `d02_summary.json` and the files above |

## Supplementary material

| Item | File(s) |
|---|---|
| Table S1 | `reference_and_update_inference/tables/tableS1.csv` (`d04_matched_h6.csv`) |
| Table S2 | `reference_and_update_inference/tables/tableS2.csv` |
| Table S3 | `reference_and_update_inference/tables/tableS3.csv` |
| Table S4 | `reference_and_update_inference/tables/tableS3b.csv` (updated models against the convex reference) |
| Table S5a | `reference_and_update_inference/tables/tableS5_smart_selection_2022.csv` |
| Table S5b | `reference_and_update_inference/d02_metrics.csv`; seven-day median row: `updates_2023/reference_metrics.csv` |
| Table S6 | `reference_and_update_inference/tables/tableS6_block_length.csv` |
| Table S7 | `updates_2023/p06_runs/p06_summary.csv` |
| Fig. S1 | `publication_v29/temporal_coverage.csv` |
| Table S9 | `pvdaq9068_evaluation/a02_audit.json`, `a02_power_source.json`, `a02_prepared_windows.json` |
| Tables S10a, S10b | `pvdaq9068_evaluation/d06_metrics_9068.csv`, `d06_inference_9068.csv` |
| Tables S11a, S11b | `pvdaq9068_evaluation/d09_matched_c5_9068.csv`, `d06_metrics_9068.csv`, `d06_inference_9068.csv` |
| Tables S12a, S12b | `pvdaq9068_evaluation/d06_smart_selection_9068.csv`, `d06_calibration_9068.csv` |
| Tables S13a, S13b | `pvdaq9068_evaluation/t01_selection.json`, `d07_tuning_evaluation.csv` |

## Notes

- In three run-record files the original workspace folder name was replaced by `<workspace>`; no value was changed
  (`provenance/redacted_paths.json`).
- Run records (`runs/*.json`) store the SHA-256 of the protocol, the prepared data and the training code used for each fit.
