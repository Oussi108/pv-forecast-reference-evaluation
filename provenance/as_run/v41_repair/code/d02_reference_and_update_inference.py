"""D02: v41 repair analysis for Paper 1 (saved predictions only; no training).

1. Specifies the clear-sky (smart) persistence reference without using 2023:
   candidates are scored on 2022 six-hour daylight targets and the winner is fixed per testbed.
2. Scores frozen models against previous-day, revised smart and convex references with the
   inherited synchronized 7-day calendar-block engine (2000 draws, seed 42), Holm within family.
3. Completes the update-policy comparison for all three model families (average seed-wise
   losses; R01 aggregation) at all four lead times, with paired intervals.
4. Block-length sensitivity (1/7/14/28 days) for the headline contrasts.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
REL = WS / 'WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting'
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(REL / 'configs/corrected_study_v28_array1b.json')
sys.path.insert(0, str(REL / 'scripts'))
sys.dont_write_bytecode = True
from evaluate_corrected_study import balanced, block_interval, holm_adjust  # noqa: E402

sys.path.insert(0, str(HERE))
from d01_smart_persistence_audit import smart_from  # noqa: E402

PREP = WS / 'tmp/corrected_v28_array1b'
STR = WS / 'tmp/paper1_strengthening_20260914'
V29 = WS / 'tmp/publication_v29/update_runs'
OUT = HERE.parent / 'results'
LOG = HERE.parent / 'logs/d02_events.jsonl'
SEEDS = [11, 23, 37, 51, 73]
MODELS = ['lightgbm', 'gru', 'transformer']
HORIZONS = [2, 4, 6, 12]
POLICIES = ['frozen_2021', 'annual_previous_year', 'monthly_trailing_365_days']


def log(event, **kw):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(), event=event, **kw)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print(json.dumps(row, default=str), flush=True)


def envelope_smart(frame, horizon):
    """kcs persistence against an empirical 2021 clear-sky power envelope (95th pct of power by
    hour-of-day within a day-of-year window widened 15->45 days, then hour-only, until >=10 obs)."""
    dev = frame[frame.index.year == 2021].power.dropna()
    hod, doy, v = dev.index.hour.to_numpy(), dev.index.dayofyear.to_numpy(), dev.to_numpy()
    table = np.full((24, 367), np.nan)
    for h in range(24):
        mh = hod == h
        hour_all = np.quantile(v[mh], 0.95) if mh.sum() >= 10 else np.nan
        for d in range(1, 367):
            dist = np.abs(((doy[mh] - d + 183) % 366) - 183)
            for width in (15, 45):
                sel = dist <= width
                if sel.sum() >= 10:
                    table[h, d] = np.quantile(v[mh][sel], 0.95)
                    break
            else:
                table[h, d] = hour_all
    env = pd.Series(table[frame.index.hour, frame.index.dayofyear], index=frame.index).fillna(0).clip(0, 1.2)
    kcs = (frame.power / env).where(env >= 0.05).clip(upper=1.5).ffill(limit=24)
    tgt = env.reindex(frame.index + pd.Timedelta(hours=horizon)).to_numpy()
    return np.clip(kcs.to_numpy() * tgt, 0, 1.2)


def smart_variants(bed, d, horizon):
    systems = json.loads((PREP / bed / 'systems.json').read_text())
    issue = pd.to_datetime(d['issue_ns'])
    out = {k: np.full(len(issue), np.nan) for k in ('ghi_any20', 'ghi_min200', 'envelope')}
    for code, name in enumerate(systems):
        fr = pd.read_pickle(PREP / bed / f'{name}.pkl')
        cand = {'ghi_any20': smart_from(fr, fr.clear_ghi.to_numpy(), horizon, 20),
                'ghi_min200': smart_from(fr, fr.clear_ghi.to_numpy(), horizon, 200),
                'envelope': envelope_smart(fr, horizon)}
        rows = np.flatnonzero(d['system'] == code)
        pos = fr.index.get_indexer(pd.DatetimeIndex(issue[rows]).tz_localize(fr.index.tz, ambiguous='NaT', nonexistent='NaT'))
        ok = pos >= 0
        for k, p in cand.items():
            out[k][rows[ok]] = p[pos[ok]]
    # Keep the paper's key population: an unavailable candidate falls back to the v28 value.
    fallback = {k: ~np.isfinite(v) for k, v in out.items()}
    for k in out:
        out[k][fallback[k]] = d['references'][fallback[k], 3]
    assert np.allclose(out['ghi_any20'], d['references'][:, 3], atol=1e-6)
    return out, {k: float(v.mean()) for k, v in fallback.items()}


def load(bed, h):
    with np.load(PREP / bed / f'h{h}.npz') as z:
        return {k: z[k] for k in z.files if k not in ('x', 'future')}


def seed_losses(y, preds):
    """R01: mean of seed-wise absolute losses (never the loss of the seed-mean forecast)."""
    return np.mean([np.abs(p - y) for p in preds], axis=0)


def update_predictions(bed, h, model):
    out = {p: [] for p in POLICIES}
    for s in SEEDS:
        if model == 'lightgbm' and h in (2, 6):
            path = V29 / f'{bed}_2023_h{h}_s{s}/predictions.npz'
        elif model == 'lightgbm':
            path = STR / f'private_data/p03_lightgbm_updates/{bed}_2023_h{h}_lightgbm_s{s}/predictions.npz'
        else:
            path = STR / f'private_data/p03_updates/{bed}_2023_h{h}_{model}_s{s}/predictions.npz'
        with np.load(path) as z:
            for p in POLICIES:
                out[p].append(z[p].astype(float))
        # frozen bundle must equal the parent run exactly wherever the bundle stores it
        # (LightGBM update bundles store 2023 rows only; other rows are NaN by design)
        with np.load(PREP / f'runs/{bed}_h{h}_{model}_s{s}_point_full/prediction.npz') as z:
            parent = z['prediction'][:, 0].astype(float)
        stored = np.isfinite(out['frozen_2021'][-1])
        assert np.array_equal(parent[stored], out['frozen_2021'][-1][stored])
        test = pd.to_datetime(load(bed, h)['target_ns']).year == 2023
        assert all(np.isfinite(out[p][-1][test]).all() for p in POLICIES), (bed, h, model, s)
    return out


def paired(loss, ref, d, mask, days=7, **labels):
    r = block_interval(loss[mask], ref[mask], d['system'][mask], d['target_ns'][mask], days=days, resamples=2000)
    return dict(**labels, n=int(mask.sum()), mae=balanced(loss[mask], d['system'][mask]),
                reference_mae=balanced(ref[mask], d['system'][mask]), **r)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    selection, metrics, inference, sensitivity, fallback_rows = [], [], [], [], []
    chosen = {}
    cache = {}
    for bed in ('DKASC', 'HKUST'):
        for h in HORIZONS:
            d = load(bed, h)
            var, fb = smart_variants(bed, d, h)
            fallback_rows.append(dict(testbed=bed, horizon=h, **{f'fallback_share_{k}': v for k, v in fb.items()}))
            with np.load(STR / f'aggregates/p02_predictions/{bed}_h{h}.npz') as z:
                convex = z['convex_previous_day_climatology'].astype(float)
            cache[(bed, h)] = (d, var, convex)
            t = pd.to_datetime(d['target_ns'])
            sel = (t.year == 2022) & d['daylight']
            for k, v in var.items():
                selection.append(dict(testbed=bed, horizon=h, candidate=k, year=2022, population='daylight', n=int(sel.sum()),
                                      mae=balanced(np.abs(v[sel] - d['y'][sel]), d['system'][sel])))
            log('variants_ready', testbed=bed, horizon=h, rows=len(t))
        s6 = [r for r in selection if r['testbed'] == bed and r['horizon'] == 6]
        chosen[bed] = min(s6, key=lambda r: r['mae'])['candidate']
        log('smart_reference_selected_on_2022', testbed=bed, choice=chosen[bed],
            candidates={r['candidate']: r['mae'] for r in s6})

    matched = {}
    for bed in ('DKASC', 'HKUST'):
        keys = None
        for h in HORIZONS:
            d = cache[(bed, h)][0]; t = pd.to_datetime(d['target_ns'])
            m = (t.year == 2023) & d['daylight']
            k = set(zip(d['system'][m].tolist(), d['target_ns'][m].tolist()))
            keys = k if keys is None else keys & k
        matched[bed] = keys

    for bed in ('DKASC', 'HKUST'):
        for h in HORIZONS:
            d, var, convex = cache[(bed, h)]
            y = d['y'].astype(float); t = pd.to_datetime(d['target_ns'])
            test_day = (t.year == 2023) & d['daylight']
            mkeys = matched[bed]
            mmask = test_day & np.array([(int(s), int(tt)) in mkeys for s, tt in zip(d['system'], d['target_ns'])])
            refs = {'persistence': d['references'][:, 0], 'previous_day': d['references'][:, 1],
                    'climatology': d['references'][:, 2], 'smart_v28': d['references'][:, 3],
                    'smart_revised': var[chosen[bed]], 'convex_pd_climatology': convex}
            ref_loss = {k: np.abs(v.astype(float) - y) for k, v in refs.items()}
            losses = {}
            for model in MODELS:
                up = update_predictions(bed, h, model)
                for p in POLICIES:
                    losses[(model, p)] = seed_losses(y, up[p])
            for pop, mask in (('daylight', test_day), ('matched_daylight', mmask)):
                for k, l in ref_loss.items():
                    metrics.append(dict(testbed=bed, horizon=h, population=pop, forecast=k, policy='reference',
                                        n=int(mask.sum()), mae=balanced(l[mask], d['system'][mask])))
                for (model, p), l in losses.items():
                    metrics.append(dict(testbed=bed, horizon=h, population=pop, forecast=model, policy=p,
                                        n=int(mask.sum()), mae=balanced(l[mask], d['system'][mask])))
            # F1: frozen models vs references (h6 daylight primary; matched daylight at all leads)
            for model in MODELS:
                fl = losses[(model, 'frozen_2021')]
                refset = ('previous_day', 'smart_revised', 'smart_v28', 'convex_pd_climatology') if h == 6 else ('previous_day',)
                pop, mask = ('daylight', test_day) if h == 6 else ('matched_daylight', mmask)
                for r in refset:
                    inference.append(paired(fl, ref_loss[r], d, mask, family='F1_frozen_vs_reference', testbed=bed,
                                            horizon=h, population=pop, model=model, policy='frozen_2021', reference=r))
                if h != 6:
                    continue
            # F2: updating (all horizons, daylight population of that lead)
            for model in MODELS:
                for p in ('annual_previous_year', 'monthly_trailing_365_days'):
                    inference.append(paired(losses[(model, p)], ref_loss['previous_day'], d, test_day, family='F2_update_vs_previous_day',
                                            testbed=bed, horizon=h, population='daylight', model=model, policy=p, reference='previous_day'))
                    inference.append(paired(losses[(model, p)], losses[(model, 'frozen_2021')], d, test_day, family='F3_update_vs_frozen',
                                            testbed=bed, horizon=h, population='daylight', model=model, policy=p, reference='frozen_2021'))
                    if h == 6:
                        inference.append(paired(losses[(model, p)], ref_loss['convex_pd_climatology'], d, test_day, family='F2b_update_vs_convex',
                                                testbed=bed, horizon=h, population='daylight', model=model, policy=p, reference='convex_pd_climatology'))
            if h == 6:
                heads = [(losses[('lightgbm', 'monthly_trailing_365_days')], ref_loss['previous_day'], 'lightgbm_monthly_vs_previous_day'),
                         (losses[('lightgbm', 'frozen_2021')], ref_loss['smart_revised'], 'lightgbm_frozen_vs_smart_revised'),
                         (losses[('transformer', 'frozen_2021')], ref_loss['smart_revised'], 'transformer_frozen_vs_smart_revised')]
                for l, r, name in heads:
                    for days in (1, 7, 14, 28):
                        sensitivity.append(paired(l, r, d, test_day, days=days, testbed=bed, horizon=h, contrast=name))
            log('horizon_complete', testbed=bed, horizon=h, elapsed_s=round(time.time() - t0, 1))

    inf = pd.DataFrame(inference)
    inf = holm_adjust(inf, ['family', 'testbed'])
    inf.to_csv(OUT / 'd02_inference.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT / 'd02_metrics.csv', index=False)
    pd.DataFrame(selection).to_csv(OUT / 'd02_smart_selection_2022.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / 'd02_block_length_sensitivity.csv', index=False)
    pd.DataFrame(fallback_rows).to_csv(OUT / 'd02_smart_fallback_shares.csv', index=False)
    (OUT / 'd02_summary.json').write_text(json.dumps(dict(
        chosen_smart_reference=chosen, selection_year=2022, selection_population='daylight', selection_horizon=6,
        seeds=SEEDS, aggregation='mean of seed-wise absolute losses, then equal-system mean', block_engine=
        'evaluate_corrected_study.block_interval (synchronized non-circular calendar blocks, fixed system set, rejected draws dropped)',
        holm_families=['family', 'testbed'], elapsed_s=round(time.time() - t0, 1)), indent=2))
    log('d02_complete', elapsed_s=round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
