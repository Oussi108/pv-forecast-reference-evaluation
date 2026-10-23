"""D07: evaluate the pre-specified neural-configuration sensitivity (PREREGISTRATION_v43.md section 7).

For each testbed and neural family: default (v28) vs tuned configuration, 2023 six-hour daylight MAE, and skill of the
tuned model against previous-day persistence with the same block_interval engine. PVDAQ9068 also reports 2024.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
REL = WS / 'WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting'
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(HERE.parent / 'protocol/protocol_v43_pvdaq9068.json')
sys.path.insert(0, str(REL / 'scripts'))
from evaluate_corrected_study import balanced, block_interval, holm_adjust  # noqa: E402

SEEDS = [11, 23, 37, 51, 73]
TUN = WS / 'tmp/paper1_v43_private/tuning'
SRC = {'PVDAQ9068': (WS / 'tmp/paper1_v43_private/prepared/PVDAQ9068/h6.npz', WS / 'tmp/paper1_v43_private/prepared/runs'),
       'DKASC': (WS / 'tmp/corrected_v28_array1b/DKASC/h6.npz', WS / 'tmp/corrected_v28_array1b/runs'),
       'HKUST': (WS / 'tmp/corrected_v28_array1b/HKUST/h6.npz', WS / 'tmp/corrected_v28_array1b/runs')}
sys.path.insert(0, str(WS / 'reports/paper1_v41_repair_20260926/code'))
from d02_reference_and_update_inference import smart_variants  # noqa: E402
import d06_evaluate_9068 as d06  # noqa: E402

SMART_CHOICE = {'DKASC': 'ghi_min200', 'HKUST': 'envelope'}  # fixed on 2022 in v42 (d02_smart_selection_2022.csv)


def specified_smart(bed, d):
    if bed == 'PVDAQ9068':
        var, _, _ = d06.references(d, 6)
        choice = pd.read_csv(HERE.parent / 'results/d06_smart_selection_9068.csv').set_index('horizon').loc[6, 'choice']
        return var[choice]
    var, _ = smart_variants(bed, d, 6)
    return var[SMART_CHOICE[bed]]


sel = {(r['bed'], r['model']): r for r in json.loads((HERE.parent / 'results/t01_selection.json').read_text())}
rows = []
for bed, (npz, runs) in SRC.items():
    with np.load(npz) as z:
        d = {k: z[k] for k in z.files if k not in ('x', 'future')}
    y = d['y'].astype(float); t = pd.to_datetime(d['target_ns']); pdl = np.abs(d['references'][:, 1] - y)
    spl = np.abs(specified_smart(bed, d) - y)
    for model in ('gru', 'transformer'):
        cfg = sel[(bed, model)]['selected']
        default = np.mean([np.abs(np.load(runs / f'{bed}_h6_{model}_s{s}_point_full/prediction.npz')['prediction'][:, 0] - y) for s in SEEDS], axis=0)
        tuned = np.mean([np.abs(np.load(TUN / f'{bed}_{model}_{cfg}_s{s}/prediction.npz')['prediction'][:, 0] - y) for s in SEEDS], axis=0)
        epochs = [json.loads((TUN / f'{bed}_{model}_{cfg}_s{s}/result.json').read_text())['selected_epochs'] for s in SEEDS]
        for year in ((2023, 2024) if bed == 'PVDAQ9068' else (2023,)):
            m = (t.year == year) & d['daylight']
            for label, loss, ref, refname in (('tuned_vs_default', tuned, default, 'default'), ('tuned_vs_pd', tuned, pdl, 'previous_day'),
                                              ('default_vs_pd', default, pdl, 'previous_day'),
                                              ('tuned_vs_smart', tuned, spl, 'smart_specified'),
                                              ('default_vs_smart', default, spl, 'smart_specified')):
                r = block_interval(loss[m], ref[m], d['system'][m], d['target_ns'][m], days=7, resamples=2000)
                rows.append(dict(bed=bed, model=model, year=year, contrast=label, config=cfg, tuned_epochs=str(epochs),
                                 mae=balanced(loss[m], d['system'][m]), reference_mae=balanced(ref[m], d['system'][m]), **r))
    print('done', bed, flush=True)
df = holm_adjust(pd.DataFrame(rows), ['bed', 'year', 'contrast'])
df.to_csv(HERE.parent / 'results/d07_tuning_evaluation.csv', index=False)
print(df[['bed', 'model', 'year', 'contrast', 'config', 'mae', 'reference_mae', 'skill', 'ci_low', 'ci_high']].round(4).to_string())
