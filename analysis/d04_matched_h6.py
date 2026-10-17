"""D04: frozen-model skill vs previous day at 6 h on the matched daylight keys (Table S1 consistency)."""
import pandas as pd, numpy as np
import d02_reference_and_update_inference as d2
rows = []
for bed in ('DKASC', 'HKUST'):
    keys = None
    for h in d2.HORIZONS:
        d = d2.load(bed, h); t = pd.to_datetime(d['target_ns']); m = (t.year == 2023) & d['daylight']
        k = set(zip(d['system'][m].tolist(), d['target_ns'][m].tolist())); keys = k if keys is None else keys & k
    d = d2.load(bed, 6); t = pd.to_datetime(d['target_ns']); y = d['y'].astype(float)
    mask = (t.year == 2023) & d['daylight'] & np.array([(int(s), int(x)) in keys for s, x in zip(d['system'], d['target_ns'])])
    for model in d2.MODELS:
        up = d2.update_predictions(bed, 6, model)
        r = d2.paired(d2.seed_losses(y, up['frozen_2021']), np.abs(d['references'][:, 1] - y), d, mask,
                      family='F1_frozen_vs_reference', testbed=bed, horizon=6, population='matched_daylight', model=model, reference='previous_day')
        rows.append(r)
pd.DataFrame(rows).to_csv(d2.OUT / 'd04_matched_h6.csv', index=False)
print(pd.DataFrame(rows)[['testbed', 'model', 'n', 'skill', 'ci_low', 'ci_high', 'p_raw']].round(4))
