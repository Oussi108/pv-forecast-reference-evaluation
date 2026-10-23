# Reproducing the study

Run every command from the repository root. Each step lists the files it writes.

- **Private files** go under `$PV_PRIVATE_ROOT`: prepared arrays, fitted models and predictions. They are never published.
- **Public files** go under `results/`.

Fitting steps skip runs that are already complete, so an interrupted step can be restarted.

## 0. Environment

The study ran with CPython 3.13 on Windows 11 (CPU only); `requirements.txt` lists the package versions.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PV_PRIVATE_ROOT=/path/to/pv_private             # outside the repository; see DATA_ACCESS.md
```

On Windows, keep the repository and `PV_PRIVATE_ROOT` at short paths (for example `C:\pv\repo` and `C:\pv\private`). Some
output paths are long, and the default Windows path limit is 260 characters.

## 1. Array 1B and HKUST: hourly preparation (2021–2023)

```bash
python scripts/corrected_study.py --private-root "$PV_PRIVATE_ROOT/corrected_v28_array1b" \
    --hkust-root "$PV_PRIVATE_ROOT/raw/HKUST" \
    --dkasc-file "$PV_PRIVATE_ROOT/raw/DKASC/87-Site_DKA-M9_A+C-Phases.csv"
```

- **Reads** `configs/corrected_study_v28_array1b.json`. `PV_FORECAST_PROTOCOL_PATH` can point to another protocol file.
- **Reads** the HKUST screening in `results/replacement_audit/`.
- **Writes (private)** `h2.npz`, `h4.npz`, `h6.npz` and `h12.npz` for each testbed.
- **Writes (public)** `results/corrected_v28_array1b/preparation.json` and `system_screen.csv`. Compare the array hashes
  in `preparation.json` with the released file.

## 2. Array 1B and HKUST: frozen 2021 fits and their evaluation

```bash
python scripts/train_corrected_study.py --private-root "$PV_PRIVATE_ROOT/corrected_v28_array1b" --stage point
python scripts/train_corrected_study.py --private-root "$PV_PRIVATE_ROOT/corrected_v28_array1b" --stage quantile
python scripts/evaluate_corrected_study.py --private-root "$PV_PRIVATE_ROOT/corrected_v28_array1b"
```

The point stage fits 3 models × 4 lead times × 5 seeds (11, 23, 37, 51, 73) for each testbed. The quantile stage fits
LightGBM and the Transformer at 6 h. Run records and training curves go to `results/corrected_v28_array1b/`.

## 3. Interval calibration, LightGBM updates and the 2024 Array 1B extension

```bash
python scripts/publication_revision_v29.py --stage calibration   # fixed-map interval corrections (Table IV, Figure 5)
python scripts/publication_revision_v29.py --stage updating      # LightGBM annual/monthly updates, 2023
python scripts/publication_revision_v29.py --stage temporal      # Array 1B 2024 (reads raw/DKASC for 2024)
```

Outputs go to `results/publication_v29/` and `$PV_PRIVATE_ROOT/publication_v29/`.

## 4. Update fits for all model families (2023) and input removal

```bash
python updates/run_p03_updates.py --horizons 2 4 6 12          # GRU and Transformer annual/monthly updates
python updates/run_p03_lightgbm_updates.py --horizons 4 12      # LightGBM updates at 4 and 12 h (2 and 6 h: step 3)
python updates/run_p06_ablation.py                              # five-seed input-removal fits at 6 h
python updates/aggregate_p06.py                                 # Table S7 -> results/updates_2023/p06_runs/
```

- **Parallel launchers.** `run_p03_h4_h12_neural_parallel.py`, `run_p03_h2_transformer_parallel.py` and
  `run_p03_lightgbm_parallel.py` launch the same fits in parallel processes. They were used for parts of this grid and
  change no setting.
- **Seven-day median reference (Table S5b).** This was scored by `updates/strengthening_runner.py --task P02`.
  - The runner's `P00` task records the original workspace and must run first.
  - It is kept as run. It audits files of the original research workspace that are not part of this release, so it
    may need adaptation.

## 5. Reference specification and update inference (Array 1B, HKUST)

```bash
python analysis/d01_smart_persistence_audit.py 6    # how the any-daylight smart reference picks its ratio (Figure 3a)
python analysis/d02_reference_and_update_inference.py
python analysis/d04_matched_h6.py
python analysis/d05_figure_data.py
python analysis/d03_build_tables.py                 # Holm families, Tables II-IV, S1-S6, text numbers
```

`d02` does three jobs:

- selects the smart-persistence specification on 2022;
- scores all references and update policies with paired seven-day calendar-block intervals (2,000 resamples);
- runs the block-length sensitivity.

All outputs go to `results/reference_and_update_inference/`.

## 6. PVDAQ 9068 (pre-specified evaluation)

```bash
python analysis/a01_download.py           # PVDAQ 9068 + NOAA ISD raw files -> $PV_PRIVATE_ROOT/pvdaq9068/raw
python analysis/a02_prepare_9068.py       # hourly frame, data audit, h{2,4,6,12}.npz (no forecast is scored)
export PV_FORECAST_PROTOCOL_PATH="$PWD/protocol/protocol_v43_pvdaq9068.json"
python scripts/train_corrected_study.py --private-root "$PV_PRIVATE_ROOT/pvdaq9068/prepared" --stage point --beds PVDAQ9068
python scripts/train_corrected_study.py --private-root "$PV_PRIVATE_ROOT/pvdaq9068/prepared" --stage quantile --beds PVDAQ9068
python analysis/u01_updates_9068.py       # annual and monthly updates, 2023 and 2024
python analysis/t01_neural_tuning.py PVDAQ9068 DKASC HKUST   # neural-configuration grid (all three testbeds)
python analysis/d06_evaluate_9068.py      # pre-specified contrasts, references, calibration
python analysis/d09_matched_9068.py       # contrast C5 on matched keys
python analysis/d08_expectations.py       # expectations E1-E5 (Table V)
python analysis/d07_evaluate_tuning.py    # configuration sensitivity (Tables S13a, S13b)
```

- **Do not run `analysis/p01_write_protocol.py`.** The protocol in `protocol/` was written by it and frozen before any
  PVDAQ 9068 forecast existed. Re-running it would overwrite the frozen file.
- **Check the freeze.** `protocol/FREEZE.json` records the hashes of the protocol files, of `a02`, `u01` and `t01` as
  run (`provenance/as_run/v43_upgrade/code/`), and of the audit outputs. `tools/verify_release.py` checks them.
- **Original chain.** `provenance/as_run/v43_upgrade/code/run_chain.sh` is the chain as it was run.

## 7. Figures

```bash
python figures/build_figures_1_2.py   # Figures 1-2 (needs the prepared PVDAQ 9068 frame for the example series)
python figures/f02_result_figures.py  # Figures 3-5 (from results/ only)
python figures/f03_figure6.py         # Figure 6   (from results/ only)
python figures/jrse_figures_1_2.py    # journal versions of Figures 1-2 (needs the same PVDAQ 9068 frame)
python figures/jrse_figures_3_to_6.py # journal versions of Figures 3-6 (from results/ only)
```

All figures are written to `results/figures/`; the journal versions go to `results/figures/jrse/`.
