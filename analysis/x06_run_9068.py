"""X06: run the v44 extensions B-E at PVDAQ 9068 on the private prepared arrays and saved predictions.

Refuses to run unless protocol/FREEZE_v44.json records the SHA-256 of protocol/PROTOCOL_v44_extensions.md and of the
x02-x06 scripts as they are on disk (freeze before scoring). Reuses the v43 code paths: d06 for loading, references,
saved predictions and correction maps; u01 for refitting. Writes aggregates only, to results/v44_extensions/.

    python analysis/x06_run_9068.py            # all stages
    python analysis/x06_run_9068.py --stage B  # one of B, C, D, E (D is computed inside B and C)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d06_evaluate_9068 as d06  # noqa: E402  (sets the v43 protocol path and sys.path)
import u01_updates_9068 as u01  # noqa: E402
from evaluate_corrected_study import balanced, block_interval, holm_adjust, interval_score  # noqa: E402
from publication_revision_v29 import correction_maps  # noqa: E402
from x02_optimal_reference import fit_weights, predict  # noqa: E402
from x03_drift_updates import run_policy, select_lambda  # noqa: E402
from x04_seed_bootstrap import seed_block_interval  # noqa: E402
from x05_conditional_calibration import adaptive_correction, elevation_band, mondrian_correction  # noqa: E402

REPO = HERE.parent
OUT = REPO / 'results/v44_extensions'
FROZEN_FILES = ['protocol/PROTOCOL_v44_extensions.md', 'analysis/x02_optimal_reference.py', 'analysis/x03_drift_updates.py',
                'analysis/x04_seed_bootstrap.py', 'analysis/x05_conditional_calibration.py', 'analysis/x06_run_9068.py']
YEARS = (2023, 2024)
SITE = (40.3864, -104.5512, 'America/Denver', 1407)


def check_freeze():
    path = REPO / 'protocol/FREEZE_v44.json'
    if not path.exists():
        raise SystemExit('protocol/FREEZE_v44.json is missing: freeze v44 (and publish it) before scoring')
    frozen = json.loads(path.read_text())
    for name in FROZEN_FILES:
        if frozen.get(name) != hashlib.sha256((REPO / name).read_bytes()).hexdigest():
            raise SystemExit(f'{name} differs from its v44 freeze hash')


def contrast(model_losses, ref_losses, d, mask, **label):
    """v43 interval on seed-averaged losses, plus the seed-aware (D) interval."""
    m = np.mean(model_losses, axis=0)
    r = np.mean(ref_losses, axis=0)
    v43 = block_interval(m[mask], r[mask], d['system'][mask], d['target_ns'][mask])
    d_ = seed_block_interval(np.asarray(model_losses)[:, mask], np.asarray(ref_losses)[:, mask], d['system'][mask],
                             d['target_ns'][mask])
    return dict(**label, n=int(mask.sum()), mae=balanced(m[mask], d['system'][mask]),
                reference_mae=balanced(r[mask], d['system'][mask]), **v43,
                seed_ci_low=d_['ci_low'], seed_ci_high=d_['ci_high'],
                width_ratio=(d_['ci_high'] - d_['ci_low']) / (v43['ci_high'] - v43['ci_low'])
                if v43['ci_high'] > v43['ci_low'] else float('nan'))


def specified_references(d, h):
    var, convex, _ = d06.references(d, h)
    y = d['y'].astype(float)
    t = pd.to_datetime(d['target_ns'])
    s22 = (t.year == 2022) & d['daylight']
    choice = min(var, key=lambda k: balanced(np.abs(var[k][s22] - y[s22]), d['system'][s22]))
    refs = np.column_stack([d['references'][:, 0], d['references'][:, 1], d['references'][:, 2], var[choice]]).astype(float)
    return refs, convex, choice


def stage_b(rows, weights_out):
    for h in (6, 2, 4, 12):
        d = d06.load(h)
        y = d['y'].astype(float)
        t = pd.to_datetime(d['target_ns'])
        refs, convex, choice = specified_references(d, h)
        fit = (t.year == 2022) & d['daylight']
        w = fit_weights(refs, y, t.hour.to_numpy(), fit)
        weights_out.append(dict(horizon=h, smart_choice=choice, weights=w.round(4).tolist()))
        optimal = np.abs(predict(refs, t.hour.to_numpy(), w) - y)
        for year in YEARS:
            day = (t.year == year) & d['daylight']
            rows.append(contrast([optimal], [np.abs(convex - y)], d, day, family='B_optimal_vs_convex', year=year,
                                 horizon=h, model='optimal_convex', policy='reference', reference='convex'))
        for m in d06.MODELS:
            frozen = [np.abs(p[:, 0] - y) for p in d06.frozen_preds(h, m)]
            for year in YEARS:
                day = (t.year == year) & d['daylight']
                rows.append(contrast(frozen, [optimal], d, day, family='B_vs_optimal', year=year, horizon=h, model=m,
                                     policy='frozen_2021', reference='optimal_convex'))
                up = d06.updates(h, m, year)
                if up is None:
                    continue
                for p in ('annual_previous_year', 'monthly_trailing_365_days'):
                    rows.append(contrast([np.abs(q - y) for q in up[p]], [optimal], d, day, family='B_vs_optimal',
                                         year=year, horizon=h, model=m, policy=p, reference='optimal_convex'))


def drift_predictions(model, seed, d, lam_cache, log):
    y = d['y'].astype(float)
    t = pd.to_datetime(d['target_ns'])
    parent = json.loads((u01.PUBLIC / 'runs' / f'{u01.BED}_h6_{model}_s{seed}_point_full.json').read_text())
    if model == 'lightgbm':
        iters = int(parent['settings']['selected_iterations'][0])
        x = np.concatenate([d['x'].reshape(len(y), -1), d['future']], axis=1)

        def fit_predict(train, forward):
            return np.clip(u01.lgb_model(x, d['y'], train, seed, iters).predict(x[forward]), 0, 1.2)
    else:
        epochs = int(parent['settings']['selected_epochs'])

        def fit_predict(train, forward):
            return u01.neural_fit_predict(d, train, forward, model, seed, epochs)[forward]
    frozen = d06.frozen_preds(6, model)[d06.SEEDS.index(seed)][:, 0].astype(float)
    dev = (t.year == 2021) & d['daylight']
    daily = pd.Series(np.abs(frozen[dev] - y[dev])).groupby(t[dev].normalize()).mean()
    scale = (float(daily.mean()), float(daily.std()))
    issue, target = d['issue_ns'], d['target_ns']
    if model not in lam_cache:  # lambda chosen per family on seed 11 over 2022 only
        lam_seed = d06.SEEDS[0]
        if seed != lam_seed:
            raise RuntimeError('lambda must be chosen on the first seed')
        lam_cache[model], trials = select_lambda(issue, target, y, d['daylight'], fit_predict, frozen, scale)
        log.append(dict(model=model, stage='lambda', chosen=lam_cache[model], trials={str(k): v for k, v in trials.items()}))
    pred, refits = run_policy(issue, target, y, d['daylight'], fit_predict, frozen, pd.Timestamp('2022-01-01').value,
                              pd.Timestamp('2025-01-01').value, lam_cache[model], scale)
    log.append(dict(model=model, seed=seed, stage='run', refits=[str(pd.Timestamp(r)) for r in refits], scale=scale))
    return pred


def stage_c(rows, log):
    d = u01.load(6)
    y = d['y'].astype(float)
    t = pd.to_datetime(d['target_ns'])
    lam = {}
    for m in d06.MODELS:
        drift = [np.abs(drift_predictions(m, s, d, lam, log) - y) for s in d06.SEEDS]
        frozen = [np.abs(p[:, 0] - y) for p in d06.frozen_preds(6, m)]
        for year in YEARS:
            day = (t.year == year) & d['daylight']
            up = d06.updates(6, m, year)
            annual = [np.abs(q - y) for q in up['annual_previous_year']]
            monthly = [np.abs(q - y) for q in up['monthly_trailing_365_days']]
            for ref_name, ref in (('frozen_2021', frozen), ('annual_previous_year', annual),
                                  ('monthly_trailing_365_days', monthly)):
                rows.append(contrast(drift, ref, d, day, family=f'C_drift_vs_{ref_name}', year=year, horizon=6, model=m,
                                     policy='drift_page_hinkley', reference=ref_name))
            rows.append(contrast(monthly, annual, d, day, family='C_monthly_vs_annual', year=year, horizon=6, model=m,
                                 policy='monthly_trailing_365_days', reference='annual_previous_year'))


def stage_e(rows, metrics):
    d = d06.load(6)
    y = d['y'].astype(float)
    t = pd.to_datetime(d['target_ns'])
    q = np.mean(d06.frozen_preds(6, 'lightgbm', True), axis=0)
    band = elevation_band(d['target_ns'], d['daylight'], SITE[0], SITE[1], SITE[2], SITE[3])
    cal = t.year == 2022
    for level in (0.8, 0.9):
        maps, _, lo, hi = correction_maps(d, q, level)
        scores = np.maximum.reduce([q[:, lo] - y, y - q[:, hi], np.zeros(len(y))])
        maps['elevation_bands'] = mondrian_correction(scores, band, cal, level)
        for year in YEARS:
            test = t.year == year
            adaptive = adaptive_correction(scores, d['daylight'].astype(int), cal, test, d['issue_ns'], d['target_ns'],
                                           level)
            per_year = dict(maps, adaptive=np.where(test, adaptive, np.nan))
            day = test & d['daylight']
            losses = {}
            for name, a in per_year.items():
                lower, upper = q[:, lo] - a, q[:, hi] + a
                losses[name] = interval_score(y, lower, upper, level)
                metrics.append(dict(level=level, year=year, method=name,
                                    coverage=balanced(((y >= lower) & (y <= upper))[day].astype(float), d['system'][day]),
                                    interval_score=balanced(losses[name][day], d['system'][day])))
            for name in ('elevation_bands', 'adaptive'):
                rows.append(contrast([losses[name]], [losses['global_day_night']], d, day, family='E_vs_day_night',
                                     year=year, horizon=6, model='lightgbm', policy=f'{name}_{level}',
                                     reference='global_day_night'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['B', 'C', 'D', 'E', 'all'], default='all')
    a = ap.parse_args()
    check_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, weights, log, metrics = [], [], [], []
    if a.stage in ('B', 'D', 'all'):
        stage_b(rows, weights)
    if a.stage in ('C', 'D', 'all'):
        stage_c(rows, log)
    if a.stage in ('E', 'all'):
        stage_e(rows, metrics)
    table = holm_adjust(pd.DataFrame(rows), ['family', 'year', 'horizon'])
    table.to_csv(OUT / f'x06_inference_9068_{a.stage}.csv', index=False, lineterminator='\n')
    if metrics:
        pd.DataFrame(metrics).to_csv(OUT / 'x06_calibration_9068.csv', index=False, lineterminator='\n')
    (OUT / f'x06_record_9068_{a.stage}.json').write_text(json.dumps(dict(weights=weights, drift=log), indent=1, default=str))
    print(table.to_string(max_colwidth=24))


if __name__ == '__main__':
    main()
