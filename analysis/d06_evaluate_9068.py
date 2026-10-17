"""D06: pre-specified evaluation at PVDAQ9068 (PREREGISTRATION_v43.md sections 4-6) for 2023 and 2024.

Uses the same block_interval engine, Holm adjustment, seed-wise loss aggregation, smart-persistence candidates,
convex-reference rule and fixed-map calibration code as the v41/v42 analyses of DKASC and HKUST.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parent  # repository root
V41 = HERE  # d02 is in the same folder
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(HERE.parent / 'protocol/protocol_v43_pvdaq9068.json')
sys.path.insert(0, str(REL / 'scripts')); sys.path.insert(0, str(V41))
from pv_paths import PRIV_PVDAQ  # noqa: E402
from evaluate_corrected_study import balanced, block_interval, holm_adjust, interval_score  # noqa: E402
from publication_revision_v29 import correction_maps  # noqa: E402
from d01_smart_persistence_audit import smart_from  # noqa: E402
from d02_reference_and_update_inference import envelope_smart  # noqa: E402

PRIV = PRIV_PVDAQ
BED = 'PVDAQ9068'
OUT = HERE.parent / 'results/pvdaq9068_evaluation'
SEEDS = [11, 23, 37, 51, 73]
MODELS = ['lightgbm', 'gru', 'transformer']
YEARS = (2023, 2024)
QUANT = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def load(h):
    with np.load(PRIV / f'prepared/{BED}/h{h}.npz') as z:
        return {k: z[k] for k in z.files if k not in ('x', 'future')}


def frozen_preds(h, model, quantile=False):
    kind = 'quantile' if quantile else 'point'
    return [np.load(PRIV / f'prepared/runs/{BED}_h{h}_{model}_s{s}_{kind}_full/prediction.npz')['prediction'] for s in SEEDS]


def updates(h, model, year):
    out = {}
    for s in SEEDS:
        p = PRIV / f'update_runs/{BED}_{year}_h{h}_{model}_s{s}/predictions.npz'
        if not p.exists():
            return None
        with np.load(p) as z:
            for k in z.files:
                out.setdefault(k, []).append(z[k].astype(float))
    return out


def seed_loss(y, preds):
    return np.mean([np.abs(p - y) for p in preds], axis=0)


def paired(loss, ref, d, mask, **lab):
    r = block_interval(loss[mask], ref[mask], d['system'][mask], d['target_ns'][mask], days=7, resamples=2000)
    return dict(**lab, n=int(mask.sum()), mae=balanced(loss[mask], d['system'][mask]), reference_mae=balanced(ref[mask], d['system'][mask]), **r)


def references(d, h):
    fr = pd.read_pickle(PRIV / f'prepared/{BED}/pvdaq_9068.pkl')
    issue = pd.to_datetime(d['issue_ns'])
    pos = fr.index.get_indexer(pd.DatetimeIndex(issue).tz_localize(fr.index.tz, ambiguous='NaT', nonexistent='NaT'))
    ok = pos >= 0
    cand = {'ghi_any20': smart_from(fr, fr.clear_ghi.to_numpy(), h, 20), 'ghi_min200': smart_from(fr, fr.clear_ghi.to_numpy(), h, 200),
            'envelope': envelope_smart(fr, h)}
    var = {}
    for k, v in cand.items():
        a = np.full(len(issue), np.nan); a[ok] = v[pos[ok]]
        bad = ~np.isfinite(a); a[bad] = d['references'][bad, 3]; var[k] = a
    # convex previous-day/climatology rule (strengthening_runner): weight on Oct-Dec 2021 with Jan-Sep climatology
    t = pd.to_datetime(d['target_ns']); y = d['y'].astype(float); prev = d['references'][:, 1].astype(float)

    def clim(mask):
        m = pd.Series(y[mask]).groupby(t.hour[mask]).mean()
        return pd.Series(t.hour).map(m).to_numpy(dtype=float)
    c_sel = clim((t.year == 2021) & (t.month <= 9)); c_fin = clim(t.year == 2021)
    val = (t >= pd.Timestamp('2021-10-01')) & (t < pd.Timestamp('2022-01-01')) & np.isfinite(c_sel)
    scores = [(w, balanced(np.abs(y[val] - (w * prev[val] + (1 - w) * c_sel[val])), d['system'][val])) for w in [i / 20 for i in range(21)]]
    best = min(s for _, s in scores); w = min(w for w, s in scores if abs(s - best) <= 1e-12)
    convex = np.clip(w * prev + (1 - w) * c_fin, 0, 1.2)
    return var, convex, w


def main():
    inf, met, sel = [], [], []
    for h in (6, 2, 4, 12):
        d = load(h); y = d['y'].astype(float); t = pd.to_datetime(d['target_ns'])
        var, convex, w = references(d, h)
        s22 = (t.year == 2022) & d['daylight']
        choice = min(var, key=lambda k: balanced(np.abs(var[k][s22] - y[s22]), d['system'][s22]))
        sel.append(dict(horizon=h, choice=choice, convex_weight=w, **{f'2022_{k}': balanced(np.abs(v[s22] - y[s22]), d['system'][s22]) for k, v in var.items()}))
        refs = {'persistence': d['references'][:, 0], 'previous_day': d['references'][:, 1], 'climatology': d['references'][:, 2],
                'smart_any': var['ghi_any20'], 'smart_specified': var[choice], 'convex': convex}
        rl = {k: np.abs(v.astype(float) - y) for k, v in refs.items()}
        frozen = {m: seed_loss(y, [p[:, 0] for p in frozen_preds(h, m)]) for m in MODELS}
        for year in YEARS:
            day = (t.year == year) & d['daylight']
            for k, l in rl.items():
                met.append(dict(horizon=h, year=year, forecast=k, policy='reference', n=int(day.sum()), mae=balanced(l[day], d['system'][day])))
            for m in MODELS:
                met.append(dict(horizon=h, year=year, forecast=m, policy='frozen_2021', n=int(day.sum()), mae=balanced(frozen[m][day], d['system'][day])))
                refset = ('previous_day', 'smart_any', 'smart_specified', 'convex') if h == 6 else ('previous_day',)
                for r in refset:
                    fam = {'previous_day': 'C1', 'smart_any': 'C2', 'smart_specified': 'C2', 'convex': 'C3b'}[r] if h == 6 else 'C5'
                    inf.append(paired(frozen[m], rl[r], d, day, family=fam, year=year, horizon=h, model=m, policy='frozen_2021', reference=r))
                up = updates(h, m, year)
                if up is None:
                    continue
                for p in ('annual_previous_year', 'monthly_trailing_365_days'):
                    l = seed_loss(y, up[p])
                    met.append(dict(horizon=h, year=year, forecast=m, policy=p, n=int(day.sum()), mae=balanced(l[day], d['system'][day])))
                    inf.append(paired(l, rl['previous_day'], d, day, family='C3_vs_pd', year=year, horizon=h, model=m, policy=p, reference='previous_day'))
                    inf.append(paired(l, frozen[m], d, day, family='C3_vs_frozen', year=year, horizon=h, model=m, policy=p, reference='frozen_2021'))
                    if h == 6:
                        inf.append(paired(l, rl['convex'], d, day, family='C3_vs_convex', year=year, horizon=h, model=m, policy=p, reference='convex'))
        print('horizon done', h, flush=True)
    # C4 calibration (6 h, fixed 2022 maps)
    d = load(6); y = d['y'].astype(float); t = pd.to_datetime(d['target_ns'])
    cal = []
    qsets = {'lightgbm': np.mean(frozen_preds(6, 'lightgbm', True), axis=0), 'transformer': np.mean(frozen_preds(6, 'transformer', True), axis=0)}
    for model, q in qsets.items():
        maps, groups, lo, hi = correction_maps(d, q, 0.8)
        losses = {}
        for meth, a in maps.items():
            lower, upper = q[:, lo] - a, q[:, hi] + a
            losses[meth] = interval_score(y, lower, upper, 0.8)
            for year in YEARS:
                day = (t.year == year) & d['daylight']
                cal.append(dict(model=model, method=meth, year=year, coverage=balanced(((y >= lower) & (y <= upper))[day].astype(float), d['system'][day]),
                                interval_score=balanced(losses[meth][day], d['system'][day]), fallback_groups=sum(g['fallback'] for g in groups)))
        for year in YEARS:
            day = (t.year == year) & d['daylight']
            inf.append(paired(losses['variability_groups'], losses['global_day_night'], d, day, family='C4_grouping', year=year, horizon=6,
                              model=model, policy='fixed_map', reference='global_day_night'))
    df = holm_adjust(pd.DataFrame(inf), ['family', 'year', 'horizon'])
    df.to_csv(OUT / 'd06_inference_9068.csv', index=False)
    pd.DataFrame(met).to_csv(OUT / 'd06_metrics_9068.csv', index=False)
    pd.DataFrame(sel).to_csv(OUT / 'd06_smart_selection_9068.csv', index=False)
    pd.DataFrame(cal).to_csv(OUT / 'd06_calibration_9068.csv', index=False)
    print('d06 complete')


if __name__ == '__main__':
    main()
