# Reference specification, model updating and calibration population in hourly photovoltaic power forecasting

Code and aggregate results for the paper of the same title (El Houcine Oussi and Ling Tan, Nanjing University of
Information Science and Technology). The study compares LightGBM, a GRU network and a compact Transformer at lead times of 2, 4, 6 and 12 h on
three testbeds:

- DKASC Array 1B, Alice Springs (23.4 kW dual-axis tracker);
- 31 HKUST rooftop systems, Hong Kong;
- NREL PVDAQ system 9068, Colorado (4.7 MW single-axis tracker), evaluated under a protocol fixed before scoring.

It asks how a learned model's advantage changes with three choices:

- the reference forecast it is compared against;
- how recently the model was trained;
- the population used to calibrate its prediction intervals.

## What is in this folder

| Folder | Contents |
|---|---|
| `scripts/` | Shared pipeline: completed-hour data preparation (`corrected_study.py`), model fitting (`train_corrected_study.py`), evaluation and calendar-block intervals (`evaluate_corrected_study.py`), interval calibration, LightGBM updates and the 2024 Array 1B extension (`publication_revision_v29.py`), HKUST screening scripts, and `pv_paths.py` (all file locations). |
| `updates/` | 2023 annual and monthly update fits for all three model families, input-removal fits (Table S7) and the reference diagnostics. |
| `analysis/` | Reference specification and update inference for Array 1B and HKUST (`d01`–`d05`). PVDAQ 9068 download, preparation, protocol, update fits, neural-configuration grid and pre-specified evaluation (`a01`, `a02`, `p01`, `u01`, `t01`, `d06`–`d09`). |
| `figures/` | Figures 1–2 (`build_figures_1_2.py`), Figures 3–5 (`f02_result_figures.py`), Figure 6 (`f03_figure6.py`). The journal versions submitted to the *Journal of Renewable and Sustainable Energy* (170 mm wide, all text at least 8 pt) are drawn by `jrse_figures_1_2.py` and `jrse_figures_3_to_6.py` from the same data. All are drawn with matplotlib. |
| `configs/` | Study protocol for Array 1B and HKUST (`corrected_study_v28_array1b.json`) and the calibration and update settings (`publication_revision_v29.json`). |
| `protocol/` | The PVDAQ 9068 protocol (`PREREGISTRATION_v43.md`, `protocol_v43_pvdaq9068.json`). `FREEZE.json` holds the file hashes, recorded on 26 Sep 2026 at 07:54 UTC, before any PVDAQ 9068 forecast was produced. |
| `results/` | Aggregate results only: metrics, paired comparisons, run records, training curves, tables and the figures. `results/README.md` maps every table and figure of the paper to its file. |
| `provenance/` | Byte-identical copies of the scripts as they were run (`as_run/`), the path-only differences to the released scripts (`path_changes.diff`, `code_provenance.json`), and records of excluded and redacted files. |
| `tools/verify_release.py` | Offline check of this folder (see below). |

## What is not in this folder

The following are **not** included:

- **Provider observations.** The DKASC terms limit redistribution of unaltered data. The other sources can be downloaded directly from their providers.
- **Prepared hourly arrays.**
- **Fitted models and target-level predictions.** These contain or reproduce the observations.

`DATA_ACCESS.md` explains how to obtain each dataset. The code reads private files only from the folder named by the
environment variable `PV_PRIVATE_ROOT`. That folder must lie outside the repository, and the code refuses to write
observations inside it.

## Quick check (no data needed)

```bash
pip install -r requirements.txt
python tools/verify_release.py
```

On Windows, enable long paths before cloning (`git config --global core.longpaths true`); some result paths are long.

The check runs four tests:

- every file matches `MANIFEST.sha256`;
- the protocol and script hashes in `protocol/FREEZE.json` match the files in `protocol/` and `provenance/as_run/`;
- no data arrays, model files or archives are present;
- every script compiles.

The tables and Figures 3–6 can be rebuilt from the released aggregate results alone:

```bash
python analysis/d03_build_tables.py        # Tables II-IV and S1-S6 -> results/reference_and_update_inference/tables/
python figures/f02_result_figures.py       # Figures 3-5            -> results/figures/
python figures/f03_figure6.py              # Figure 6               -> results/figures/
python figures/jrse_figures_3_to_6.py      # journal versions of Figures 3-6 -> results/figures/jrse/
```

## Full reproduction

`REPRODUCE.md` lists every step in order, from the raw downloads to the figures, with the file each step writes. Fitting all
models takes many CPU hours. The evaluation steps take minutes once the fitted predictions exist.

## Verification of this release

On 17 October 2026 the evaluation, table and figure scripts were re-run from this folder against the author's private
predictions. They reproduced every released table and evaluation file exactly, and Figures 1–6 pixel for pixel. The
details are in `provenance/RELEASE_CHECK.md`.

The scripts differ from the versions that produced the paper only in the lines that name file locations
(`provenance/path_changes.diff`). Model-fitting scripts were not re-run for the release.

On 23 October 2026 the journal versions of the figures were added (`figures/jrse_figures_1_2.py`,
`figures/jrse_figures_3_to_6.py`, `results/figures/jrse/`); `jrse_figures_3_to_6.py` reproduces the journal Figures 3–6
pixel for pixel from the released results alone. A `.gitattributes` file stops git from converting line endings, so every
file stays byte-identical to the hashes in `MANIFEST.sha256` and `protocol/FREEZE.json` on any platform.

## Licence

- **Code:** MIT licence (`LICENSE`).
- **Aggregate results** in `results/`: Creative Commons Attribution 4.0 (`results/LICENSE.md`).
- **Source data** (not included): each provider's own terms (`DATA_ACCESS.md`).

## Citation

Please cite the paper and this archive; `CITATION.cff` holds the archive metadata. The DOI will be added when the archive is
deposited.
