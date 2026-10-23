"""D05: data for the v42 mechanism figures (saved predictions only; no training).

(1) frozen and monthly skill vs previous day on each lead's 2023 daylight keys, all models (Figure 4);
(2) empirical coverage of nominal-80% six-hour intervals by target hour for raw, all-hour and
    day/night fixed maps (Figure 5), reusing the v29 correction-map code unchanged.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d02_reference_and_update_inference as d2  # noqa: E402  (sets protocol env + evaluator path)

REL = d2.REL
sys.path.insert(0, str(REL / 'scripts'))
from publication_revision_v29 import correction_maps, residual_quantiles  # noqa: E402
from evaluate_corrected_study import balanced  # noqa: E402

OUT = d2.OUT
rows = []
for bed in (() if (OUT / 'd05_skill_by_lead.csv').exists() else ('DKASC', 'HKUST')):
    for h in d2.HORIZONS:
        d = d2.load(bed, h); y = d['y'].astype(float); t = pd.to_datetime(d['target_ns'])
        mask = (t.year == 2023) & d['daylight']
        pd_loss = np.abs(d['references'][:, 1] - y)
        for model in d2.MODELS:
            up = d2.update_predictions(bed, h, model)
            for p in ('frozen_2021', 'monthly_trailing_365_days'):
                r = d2.paired(d2.seed_losses(y, up[p]), pd_loss, d, mask, testbed=bed, horizon=h, model=model, policy=p)
                rows.append(r)
        d2.log('d05_lead_done', testbed=bed, horizon=h)
if rows:
    pd.DataFrame(rows).to_csv(OUT / 'd05_skill_by_lead.csv', index=False)

cov = []
for bed in ('DKASC', 'HKUST'):
    with np.load(d2.PREP / bed / 'h6.npz') as z:
        d = {k: z[k] for k in z.files if k not in ('x', 'future')}
    t = pd.to_datetime(d['target_ns']); test = t.year == 2023
    models = {'lightgbm': np.mean([np.load(d2.PREP / f'runs/{bed}_h6_lightgbm_s{s}_quantile_full/prediction.npz')['prediction']
                                   for s in d2.SEEDS], axis=0),
              'previous_day_residual': residual_quantiles(d, d['references'][:, 1])}
    for model, q in models.items():
        maps, _, lo, hi = correction_maps(d, q, 0.8)
        for method in ('raw', 'global_all_hours', 'global_day_night'):
            a = maps[method]
            hit = (d['y'] >= q[:, lo] - a) & (d['y'] <= q[:, hi] + a)
            for hr in range(24):
                m = test & (t.hour == hr)
                if m.sum() == 0:
                    continue
                cov.append(dict(testbed=bed, model=model, method=method, target_hour=hr, n=int(m.sum()),
                                daylight_share=float(d['daylight'][m].mean()), coverage=balanced(hit[m].astype(float), d['system'][m])))
            for pop, m in (('daylight', test & d['daylight']), ('night', test & ~d['daylight'])):
                cov.append(dict(testbed=bed, model=model, method=method, target_hour=pop, n=int(m.sum()),
                                daylight_share=float(pop == 'daylight'), coverage=balanced(hit[m].astype(float), d['system'][m])))
pd.DataFrame(cov).to_csv(OUT / 'd05_coverage_by_hour.csv', index=False)
d2.log('d05_complete')
