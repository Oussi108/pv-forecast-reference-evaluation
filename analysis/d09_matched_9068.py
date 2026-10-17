"""D09: pre-specified contrast C5 at PVDAQ9068 on MATCHED daylight keys (PREREGISTRATION_v43.md section 5).

d06 reports C5 on each lead's own daylight keys; the protocol states matched keys, so this script restricts every lead
to the system-target keys valid at all of 2, 4, 6 and 12 h within each year. Holm within year x lead.
"""
import numpy as np
import pandas as pd

import d06_evaluate_9068 as d06

rows = []
data = {h: d06.load(h) for h in (2, 4, 6, 12)}
for year in d06.YEARS:
    keys = None
    for h, d in data.items():
        t = pd.to_datetime(d['target_ns']); m = (t.year == year) & d['daylight']
        k = set(zip(d['system'][m].tolist(), d['target_ns'][m].tolist())); keys = k if keys is None else keys & k
    for h, d in data.items():
        t = pd.to_datetime(d['target_ns']); y = d['y'].astype(float)
        mask = (t.year == year) & d['daylight'] & np.array([(int(s), int(x)) in keys for s, x in zip(d['system'], d['target_ns'])])
        pdl = np.abs(d['references'][:, 1].astype(float) - y)
        for m in d06.MODELS:
            loss = d06.seed_loss(y, [p[:, 0] for p in d06.frozen_preds(h, m)])
            rows.append(d06.paired(loss, pdl, d, mask, family='C5_matched', year=year, horizon=h, model=m, policy='frozen_2021',
                                   reference='previous_day', population='matched_daylight'))
    print('year done', year, len(keys), flush=True)
df = d06.holm_adjust(pd.DataFrame(rows), ['family', 'year', 'horizon'])
df.to_csv(d06.OUT / 'd09_matched_c5_9068.csv', index=False)
print(df[['year', 'horizon', 'model', 'n', 'skill', 'ci_low', 'ci_high', 'p_holm']].round(4).to_string())
