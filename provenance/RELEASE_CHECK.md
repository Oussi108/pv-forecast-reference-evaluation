# Release check (17 October 2026)

**Setup.** A copy of this release folder was run against the author's private prepared arrays and fitted predictions
(`PV_PRIVATE_ROOT`). It used CPython 3.13.9, numpy 2.3.4, pandas 3.0.2, scipy 1.16.2, lightgbm 4.6.0, torch 2.9.1,
pvlib 0.15.1 and matplotlib 3.10.8, on Windows 11 and CPU only. The regenerated files were compared with the released
`results/`.

## Scripts re-run (all finished without error)

| Script | Run time |
|---|---|
| `analysis/d01_smart_persistence_audit.py 6` | 77 s |
| `analysis/d02_reference_and_update_inference.py` | 234 s |
| `analysis/d04_matched_h6.py` | 12 s |
| `analysis/d05_figure_data.py` | 72 s |
| `analysis/d03_build_tables.py` | 3 s |
| `analysis/d06_evaluate_9068.py` | 29 s |
| `analysis/d09_matched_9068.py` | 7 s |
| `analysis/d08_expectations.py` | 7 s |
| `analysis/d07_evaluate_tuning.py` | 50 s |
| `updates/aggregate_p06.py` | 2 s |
| `figures/f02_result_figures.py` | 8 s |
| `figures/f03_figure6.py` | 5 s |
| `figures/build_figures_1_2.py` | 11 s |

## Outcome

- **33 result files identical** to the released ones, cell by cell. They are:
  - every `d01`–`d05` output (`d05_skill_by_lead.csv` is cached by `d05` and was deleted first, so it was recomputed);
  - all table files (`tables/table2.csv` … `tableS6_block_length.csv`, `table4_inference.json`, `text_numbers.json`);
  - all PVDAQ 9068 evaluation outputs (`d06`–`d09`, including the expectations E1–E5);
  - the input-removal summaries (`p06_summary.csv`, `p06_seed_spread.csv`, `ablation_paired_inference.csv`, and
    `progress.json` apart from its completion time).
- **Figures pixel-identical to the published PNGs:** Figures 1–2 (`*_v44.png`), Figures 3–5 (`*_v42.png`) and Figure 6
  (`*_v43.png`). The PDF files differ only in their embedded creation dates.
- **Expected differences, no value changed:**
  - `p06_seed_records.csv` and `ablation_seed_metrics.csv` differ only in the `record_path` column, which holds the
    folder of the run.
  - `feature_map.json` differs in the stored file paths and in the hash of `scripts/corrected_study.py`, whose default
    protocol line was changed for the release (`path_changes.diff`).

## Not re-run for the release

The model-fitting and data-preparation steps were not re-run. These are `scripts/corrected_study.py`,
`train_corrected_study.py`, `publication_revision_v29.py`, `updates/run_p03_*.py`, `run_p06_ablation.py`,
`analysis/a01`, `a02`, `u01` and `t01`. Their released versions differ from the versions that produced the fits only in
the path lines listed in `path_changes.diff`. Run records in `results/*/runs/` store the hashes of the data, protocol and
code used for each fit.
