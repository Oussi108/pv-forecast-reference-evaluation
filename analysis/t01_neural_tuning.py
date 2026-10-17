"""T01: pre-specified neural-configuration sensitivity (PREREGISTRATION_v43.md section 7).

For each testbed and family (GRU, Transformer) at 6 h: fit the 4-configuration grid with seed 11 using the unchanged
train_corrected_study.neural_fit (selection on Jan-Sep 2021, validation Oct-Dec 2021, refit on 2021), pick the lowest
validation loss, then fit seeds 23, 37, 51, 73 with the selected configuration. No 2022-2024 label is used.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parent  # repository root
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(HERE.parent / 'protocol/protocol_v43_pvdaq9068.json')
sys.path.insert(0, str(REL / 'scripts'))
from pv_paths import PREP_V28, PRIV_PVDAQ  # noqa: E402
import train_corrected_study as tcs  # noqa: E402

OUTR = PRIV_PVDAQ / 'tuning'
LOG = HERE.parent / 'logs/t01_events.jsonl'
DATA = {'PVDAQ9068': PRIV_PVDAQ / 'prepared/PVDAQ9068/h6.npz',
        'DKASC': PREP_V28 / 'DKASC/h6.npz',
        'HKUST': PREP_V28 / 'HKUST/h6.npz'}
GRID = [dict(hidden=w, layers=l) for w in (32, 64) for l in (1, 2)]
BASE = copy.deepcopy(tcs.PROTOCOL['neural'])


def log(event, **kw):
    row = dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(), event=event, **kw)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print(json.dumps(row, default=str), flush=True)


def fit(bed, model, cfg, seed, data, masks):
    tag = f"w{cfg['hidden']}_l{cfg['layers']}"
    root = OUTR / f'{bed}_{model}_{tag}_s{seed}'
    done = root / 'result.json'
    if done.exists():
        return json.loads(done.read_text())
    root.mkdir(parents=True, exist_ok=True)
    tcs.PROTOCOL['neural'] = dict(BASE, hidden=cfg['hidden'], layers=cfg['layers'], max_epochs=150, patience=10)
    t0 = time.time()
    pred, settings, _ = tcs.neural_fit(data, masks, seed, model, False, 'full', root)
    np.savez_compressed(root / 'prediction.npz', prediction=pred)
    res = dict(bed=bed, model=model, config=tag, seed=seed, seconds=time.time() - t0, **{k: v for k, v in settings.items() if k != 'architecture'})
    done.write_text(json.dumps(res, indent=1))
    log('tuning_fit_done', **res)
    return res


def main(beds):
    summary = []
    for bed in beds:
        with np.load(DATA[bed]) as z:
            data = {k: z[k] for k in z.files}
        masks = tcs.split_masks(data)
        for model in ('gru', 'transformer'):
            grid = [fit(bed, model, c, 11, data, masks) for c in GRID]
            best = min(grid, key=lambda r: r['selection_validation_loss'])
            cfg = next(c for c in GRID if f"w{c['hidden']}_l{c['layers']}" == best['config'])
            for seed in (23, 37, 51, 73):
                fit(bed, model, cfg, seed, data, masks)
            summary.append(dict(bed=bed, model=model, selected=best['config'],
                                grid={r['config']: dict(val=r['selection_validation_loss'], epochs=r['selected_epochs']) for r in grid}))
            log('tuning_selected', bed=bed, model=model, selected=best['config'])
    (HERE.parent / 'results/pvdaq9068_evaluation/t01_selection.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['PVDAQ9068', 'DKASC', 'HKUST'])
