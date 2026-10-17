"""D08: score the pre-stated expectations E1-E5 (PREREGISTRATION_v43.md section 6) mechanically from d06 outputs.

Rule (section 6): confirmed only if the stated direction holds and, where an interval is involved, the 95% interval
supports it. E2 has no interval in its wording; a paired interval is added here for information only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import d06_evaluate_9068 as d06

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results/pvdaq9068_evaluation'
inf = pd.read_csv(RES / 'd06_inference_9068.csv')
met = pd.read_csv(RES / 'd06_metrics_9068.csv')
cal = pd.read_csv(RES / 'd06_calibration_9068.csv')
out = {}
for year in d06.YEARS:
    e = {}
    c1 = inf[(inf.family == 'C1') & (inf.year == year)]
    e['E1'] = dict(statement='No frozen family has a 6-h skill interval above zero against previous-day persistence',
                   confirmed=bool((c1.ci_low <= 0).all()),
                   detail={r.model: [round(r.skill, 4), round(r.ci_low, 4), round(r.ci_high, 4)] for r in c1.itertuples()})
    m6 = met[(met.horizon == 6) & (met.year == year) & (met.policy == 'reference')].set_index('forecast').mae
    d = d06.load(6); y = d['y'].astype(float); t = pd.to_datetime(d['target_ns'])
    var, _, _ = d06.references(d, 6)
    choice = pd.read_csv(RES / 'd06_smart_selection_9068.csv').set_index('horizon').loc[6, 'choice']
    day = (t.year == year) & d['daylight']
    r2 = d06.paired(np.abs(var[choice] - y), np.abs(var['ghi_any20'] - y), d, day)
    e['E2'] = dict(statement='Specified smart persistence has lower 6-h daylight MAE than form (a)',
                   confirmed=bool(m6['smart_specified'] < m6['smart_any']), specified_form=choice,
                   mae_specified=round(m6['smart_specified'], 4), mae_form_a=round(m6['smart_any'], 4),
                   paired_skill_for_information=[round(r2['skill'], 4), round(r2['ci_low'], 4), round(r2['ci_high'], 4)])
    c3 = inf[(inf.family == 'C3_vs_frozen') & (inf.year == year) & (inf.horizon == 6) & (inf.policy == 'monthly_trailing_365_days')]
    e['E3'] = dict(statement='Monthly refitting lowers 6-h MAE relative to the frozen fit for every family',
                   confirmed=bool(((c3.skill > 0) & (c3.ci_low > 0)).all() and len(c3) == 3),
                   detail={r.model: [round(r.skill, 4), round(r.ci_low, 4), round(r.ci_high, 4), round(r.p_holm, 4)] for r in c3.itertuples()})
    lc = cal[(cal.model == 'lightgbm') & (cal.year == year)].set_index('method').coverage
    e['E4'] = dict(statement='All-hour correction changes LightGBM daylight coverage by <1 pp; day/night correction increases it',
                   confirmed=bool(abs(lc['global_all_hours'] - lc['raw']) < 0.01 and lc['global_day_night'] > lc['raw']),
                   coverage={k: round(v, 4) for k, v in lc.items()})
    c4 = inf[(inf.family == 'C4_grouping') & (inf.year == year) & (inf.model == 'lightgbm')].iloc[0]
    e['E5'] = dict(statement='Variability grouping has no resolved interval-score benefit over day/night correction',
                   confirmed=bool(not (c4.ci_low > 0)),
                   detail=[round(c4.skill, 4), round(c4.ci_low, 4), round(c4.ci_high, 4), round(c4.p_holm, 4)],
                   note='interval entirely below zero = resolved small loss from grouping' if c4.ci_high < 0 else '')
    out[str(year)] = e
(RES / 'd08_expectations.json').write_text(json.dumps(out, indent=1))
for year, e in out.items():
    print(year, {k: v['confirmed'] for k, v in e.items()})
print(json.dumps(out, indent=1))
