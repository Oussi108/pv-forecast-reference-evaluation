"""D03: assemble v41 tables, text numbers and Figure 3 from D02 (and v29 calibration) outputs."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results/reference_and_update_inference'
REL = HERE.parent  # repository root
OUT = RES / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

inf = pd.read_csv(RES / 'd02_inference.csv')
met = pd.read_csv(RES / 'd02_metrics.csv')


def holm(df, by):
    df = df.copy(); df['p_holm_v41'] = np.nan
    for _, g in df.groupby(by):
        order = g.p_raw.sort_values().index
        adj = np.maximum.accumulate(df.loc[order, 'p_raw'].to_numpy() * np.arange(len(order), 0, -1))
        df.loc[order, 'p_holm_v41'] = np.minimum(adj, 1)
    return df


# One family per question x testbed x lead time (declared here, before reading adjusted values).
# Table S1 uses matched daylight keys at every lead; the 6-h matched rows come from D04.
m6 = pd.read_csv(RES / 'd04_matched_h6.csv')
inf = pd.concat([inf, m6.assign(family='F1s_frozen_vs_pd_matched6')], ignore_index=True)
inf = holm(inf, ['family', 'testbed', 'horizon'])
inf.to_csv(RES / 'd03_inference_holm_by_lead.csv', index=False)

pct = lambda v: f'{v*100:+.1f}'
ci = lambda r: f'{pct(r.skill)} [{pct(r.ci_low)}, {pct(r.ci_high)}]'
NAME = {'lightgbm': 'LightGBM', 'gru': 'GRU', 'transformer': 'Transformer'}


def row(**kw):
    q = inf
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) == 1, kw
    return q.iloc[0]


def mae(bed, forecast, policy='reference', h=6, pop='daylight'):
    q = met[(met.testbed == bed) & (met.horizon == h) & (met.population == pop) & (met.forecast == forecast) & (met.policy == policy)]
    assert len(q) == 1
    return float(q.mae.iloc[0])


# Table 2: six-hour daylight MAE, references and frozen models, plus frozen skill vs revised smart.
t2 = []
for label, f, p in [('Persistence', 'persistence', 'reference'), ('Smart persistence, any-daylight ratio', 'smart_v28', 'reference'),
                    ('Smart persistence, specified (2022-selected)', 'smart_revised', 'reference'),
                    ('Climatology', 'climatology', 'reference'), ('Previous day', 'previous_day', 'reference'),
                    ('Convex previous day–climatology', 'convex_pd_climatology', 'reference'),
                    ('LightGBM (frozen)', 'lightgbm', 'frozen_2021'), ('GRU (frozen)', 'gru', 'frozen_2021'),
                    ('Transformer (frozen)', 'transformer', 'frozen_2021')]:
    t2.append([label] + [f'{mae(b, f, p):.4f}' for b in ('DKASC', 'HKUST')])
pd.DataFrame(t2, columns=['Forecast', 'Array 1B MAE', 'HKUST MAE']).to_csv(OUT / 'table2.csv', index=False)

# Table 3: update policies with monthly skill vs previous day [95% CI]
t3 = []
for b in ('DKASC', 'HKUST'):
    for m in ('lightgbm', 'gru', 'transformer'):
        r = row(family='F2_update_vs_previous_day', testbed=b, horizon=6, model=m, policy='monthly_trailing_365_days')
        t3.append([b, NAME[m]] + [f"{mae(b, m, p):.4f}" for p in ('frozen_2021', 'annual_previous_year', 'monthly_trailing_365_days')] + [ci(r)])
pd.DataFrame(t3, columns=['Testbed', 'Model', 'Frozen', 'Annual', 'Monthly', 'Monthly skill vs PD, % [95% CI]']).to_csv(OUT / 'table3.csv', index=False)

# Table 4 extra column: grouped vs day/night interval-score skill [95% CI] (v29 calibration inference, level 0.8)
cal = pd.read_csv(REL / 'results/publication_v29/calibration_inference.csv')
cal = cal[(cal.analysis == 'calibration_control') & (cal.level == 0.8)]
t4 = {}
for b in ('DKASC', 'HKUST'):
    for m in ('previous_day_residual', 'lightgbm', 'transformer'):
        g = cal[(cal.testbed == b) & (cal.model == m) & (cal.method == 'variability_groups')].iloc[0]
        d = cal[(cal.testbed == b) & (cal.model == m) & (cal.method == 'global_day_night')].iloc[0]
        t4[f'{b}|{m}'] = dict(grouped_vs_daynight=f"{g.skill*100:+.2f} [{g.ci_low*100:+.2f}, {g.ci_high*100:+.2f}]",
                              daynight_vs_allhour=f"{d.skill*100:+.1f} [{d.ci_low*100:+.1f}, {d.ci_high*100:+.1f}]",
                              grouped_p_holm=float(g.p_holm))
(OUT / 'table4_inference.json').write_text(json.dumps(t4, indent=2))

# Supplement tables
s1 = []
for b in ('DKASC', 'HKUST'):
    for h in (2, 4, 6, 12):
        for m in ('lightgbm', 'gru', 'transformer'):
            r = row(family='F1s_frozen_vs_pd_matched6' if h == 6 else 'F1_frozen_vs_reference', testbed=b, horizon=h, model=m, reference='previous_day')
            s1.append([b, h, NAME[m], r.population.replace('_', ' '), f'{pct(r.skill)}', f'[{pct(r.ci_low)}, {pct(r.ci_high)}]', f'{r.p_holm_v41:.3f}'])
pd.DataFrame(s1, columns=['Testbed', 'Lead (h)', 'Model', 'Population', 'Skill vs PD (%)', '95% interval (%)', 'Holm p']).to_csv(OUT / 'tableS1.csv', index=False)

s2 = []
for b in ('DKASC', 'HKUST'):
    for m in ('lightgbm', 'gru', 'transformer'):
        for ref, lab in [('smart_v28', 'Smart, any-daylight ratio'), ('smart_revised', 'Smart, specified'), ('previous_day', 'Previous day'), ('convex_pd_climatology', 'Convex PD–climatology')]:
            r = row(family='F1_frozen_vs_reference', testbed=b, horizon=6, model=m, reference=ref)
            s2.append([b, NAME[m], lab, f'{pct(r.skill)}', f'[{pct(r.ci_low)}, {pct(r.ci_high)}]', f'{r.p_holm_v41:.3f}'])
pd.DataFrame(s2, columns=['Testbed', 'Frozen model', 'Reference', 'Skill (%)', '95% interval (%)', 'Holm p']).to_csv(OUT / 'tableS2.csv', index=False)

s3 = []
for b in ('DKASC', 'HKUST'):
    for h in (2, 4, 6, 12):
        for m in ('lightgbm', 'gru', 'transformer'):
            a = row(family='F2_update_vs_previous_day', testbed=b, horizon=h, model=m, policy='annual_previous_year')
            mo = row(family='F2_update_vs_previous_day', testbed=b, horizon=h, model=m, policy='monthly_trailing_365_days')
            fr = row(family='F3_update_vs_frozen', testbed=b, horizon=h, model=m, policy='monthly_trailing_365_days')
            s3.append([b, h, NAME[m], f"{mae(b, m, 'frozen_2021', h):.4f}", f"{mae(b, m, 'annual_previous_year', h):.4f}",
                       f"{mae(b, m, 'monthly_trailing_365_days', h):.4f}", ci(a), ci(mo), ci(fr)])
pd.DataFrame(s3, columns=['Testbed', 'h', 'Model', 'Frozen MAE', 'Annual MAE', 'Monthly MAE', 'Annual vs PD (%)',
                          'Monthly vs PD (%)', 'Monthly vs frozen (%)']).to_csv(OUT / 'tableS3.csv', index=False)

s3b = []
for b in ('DKASC', 'HKUST'):
    for m in ('lightgbm', 'gru', 'transformer'):
        for p, lab in [('annual_previous_year', 'Annual'), ('monthly_trailing_365_days', 'Monthly')]:
            r = row(family='F2b_update_vs_convex', testbed=b, horizon=6, model=m, policy=p)
            s3b.append([b, NAME[m], lab, ci(r), f'{r.p_holm_v41:.3f}'])
pd.DataFrame(s3b, columns=['Testbed', 'Model', 'Policy', 'Skill vs convex (%) [95% CI]', 'Holm p']).to_csv(OUT / 'tableS3b.csv', index=False)

sel = pd.read_csv(RES / 'd02_smart_selection_2022.csv')
sel = sel[sel.horizon == 6].pivot(index='candidate', columns='testbed', values='mae').round(4)
sel.to_csv(OUT / 'tableS5_smart_selection_2022.csv')
sens = pd.read_csv(RES / 'd02_block_length_sensitivity.csv')
sens['interval'] = sens.apply(lambda r: f'{pct(r.skill)} [{pct(r.ci_low)}, {pct(r.ci_high)}]', axis=1)
sens.pivot(index=['testbed', 'contrast'], columns='block_days', values='interval').to_csv(OUT / 'tableS6_block_length.csv')

# Numbers quoted in the text
num = {}
for b in ('DKASC', 'HKUST'):
    for m in ('lightgbm', 'gru', 'transformer'):
        for ref in ('smart_revised', 'smart_v28', 'previous_day', 'convex_pd_climatology'):
            num[f'{b}_{m}_frozen_vs_{ref}'] = ci(row(family='F1_frozen_vs_reference', testbed=b, horizon=6, model=m, reference=ref))
        for p in ('annual_previous_year', 'monthly_trailing_365_days'):
            r = row(family='F2_update_vs_previous_day', testbed=b, horizon=6, model=m, policy=p)
            num[f'{b}_{m}_{p}_vs_pd'] = ci(r); num[f'{b}_{m}_{p}_vs_pd_holm'] = round(float(r.p_holm_v41), 4)
            num[f'{b}_{m}_{p}_vs_convex'] = ci(row(family='F2b_update_vs_convex', testbed=b, horizon=6, model=m, policy=p))
            num[f'{b}_{m}_{p}_vs_frozen'] = ci(row(family='F3_update_vs_frozen', testbed=b, horizon=6, model=m, policy=p))
    for h in (2, 4, 12):
        for m in ('lightgbm', 'gru', 'transformer'):
            num[f'{b}_{m}_h{h}_matched_vs_pd'] = ci(row(family='F1_frozen_vs_reference', testbed=b, horizon=h, model=m, reference='previous_day'))
num['convex_vs_pd'] = {b: f"{(1-mae(b,'convex_pd_climatology')/mae(b,'previous_day'))*100:+.1f}" for b in ('DKASC', 'HKUST')}
num['smart_selection'] = json.loads((RES / 'd02_summary.json').read_text())['chosen_smart_reference']
(OUT / 'text_numbers.json').write_text(json.dumps(num, indent=2))

# Figure 3 (same renderer style as v31-v40 result charts), now with specified smart and convex references
COLORS = ['#346989']
font = lambda n: ImageFont.truetype('C:/Windows/Fonts/arial.ttf', n)
labels = ['Persistence', 'Smart (any-daylight ratio)', 'Smart (specified)', 'Climatology', 'Previous day', 'Convex reference',
          'LightGBM', 'GRU', 'Transformer']
keys = [('persistence', 'reference'), ('smart_v28', 'reference'), ('smart_revised', 'reference'), ('climatology', 'reference'),
        ('previous_day', 'reference'), ('convex_pd_climatology', 'reference'), ('lightgbm', 'frozen_2021'), ('gru', 'frozen_2021'),
        ('transformer', 'frozen_2021')]
im = Image.new('RGB', (1800, 820), 'white'); d = ImageDraw.Draw(im); w = 900
for panel, b in enumerate(('DKASC', 'HKUST')):
    vals = [mae(b, f, p) for f, p in keys]
    x0 = panel*w+330; x1 = (panel+1)*w-45; y0 = 100; y1 = 700
    maxv = math.ceil(max(vals)*100)/100
    d.text((panel*w+w/2, 45), b, font=font(33), fill='#222222', anchor='mm')
    for i in range(5):
        v = maxv*i/4; x = x0+(x1-x0)*i/4
        d.line((x, y0, x, y1), fill='#dedede', width=2); d.text((x, y1+22), f'{v:.2f}', font=font(23), fill='#222222', anchor='ma')
    step = (y1-y0)/len(vals)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = y0+step*(i+.5)
        d.text((x0-14, y), lab, font=font(23), fill='#222222', anchor='rm')
        d.rectangle((x0, y-19, x0+(x1-x0)*v/maxv, y+19), fill=COLORS[0] if i >= 6 else '#949494')
    d.text(((x0+x1)/2, 785), 'Daylight MAE (normalized power)', font=font(26), fill='#222222', anchor='mm')
im.save(OUT / 'Figure3_reference_mae_v41.png', dpi=(220, 220))
print(json.dumps(num, indent=1)[:3000])
