"""Where the code finds its inputs and writes its outputs.

Provider observations, prepared arrays, fitted models and target-level predictions are kept OUTSIDE this repository, under
the folder named by the environment variable PV_PRIVATE_ROOT (default: a folder ``pv_private`` next to the repository).
They are never published. Aggregate results, tables and figures are written inside the repository under ``results/``.

The private sub-folder names are those of the original research workspace, so the run records in ``results/`` stay
comparable with a re-run.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / 'configs'
PROTOCOL_DIR = REPO / 'protocol'
RESULTS = REPO / 'results'
PRIVATE = Path(os.environ.get('PV_PRIVATE_ROOT', REPO.parent / 'pv_private')).resolve()

PREP_V28 = PRIVATE / 'corrected_v28_array1b'              # DKASC Array 1B and HKUST: h{2,4,6,12}.npz, frozen fits in runs/
PRIV_V29 = PRIVATE / 'publication_v29'                    # Array 1B 2024 arrays and LightGBM update fits
PRIV_UPDATES = PRIVATE / 'paper1_strengthening_20260914'  # 2023 update fits (all families) and input-removal fits
PRIV_PVDAQ = PRIVATE / 'paper1_v43_private'               # PVDAQ 9068 and NOAA ISD raw files, prepared arrays, fits, tuning
RAW_DKASC = PRIVATE / 'raw' / 'DKASC'                     # DKASC provider CSV files (Array 1B)

if REPO == PRIVATE or REPO in PRIVATE.parents:
    raise ValueError('PV_PRIVATE_ROOT must be outside the repository: it holds provider observations')
