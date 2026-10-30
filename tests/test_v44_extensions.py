"""Synthetic checks for the v44 extension code (no private data). Run: python -m pytest tests/test_v44_extensions.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'scripts')]


def _panel(seed=0, sites=3, days=120):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2023-01-01', periods=days * 24, freq='h')
    target_ns = np.tile(t.as_unit('ns').asi8, sites)
    system = np.repeat(np.arange(sites), len(t))
    ref = np.abs(rng.normal(0, 0.1, len(system)))
    return target_ns, system, ref, rng


def test_seed_bootstrap_equals_block_interval_when_seeds_identical():
    from evaluate_corrected_study import block_interval
    from x04_seed_bootstrap import seed_block_interval
    target_ns, system, ref, rng = _panel()
    model = np.abs(rng.normal(0, 0.09, len(system)))
    a = block_interval(model, ref, system, target_ns)
    b = seed_block_interval(np.stack([model] * 5), ref, system, target_ns)
    assert abs(a['skill'] - b['skill']) < 1e-12
    assert abs(a['ci_low'] - b['ci_low']) < 1e-12 and abs(a['ci_high'] - b['ci_high']) < 1e-12


def test_seed_bootstrap_widens_with_seed_spread():
    from x04_seed_bootstrap import seed_block_interval
    target_ns, system, ref, rng = _panel(1)
    base = np.abs(rng.normal(0, 0.09, len(system)))
    same = seed_block_interval(np.stack([base] * 5), ref, system, target_ns)
    spread = seed_block_interval(np.stack([base * f for f in (0.85, 0.95, 1.0, 1.05, 1.15)]), ref, system, target_ns)
    assert spread['ci_high'] - spread['ci_low'] > same['ci_high'] - same['ci_low']


def test_mondrian_bands_reach_nominal_coverage_per_band():
    from x05_conditional_calibration import mondrian_correction
    rng = np.random.default_rng(3)
    n = 40000
    groups = rng.integers(0, 4, n)
    scale = np.array([0.01, 0.05, 0.1, 0.2])[groups]
    scores = np.abs(rng.normal(0, 1, n)) * scale
    cal = np.arange(n) < n // 2
    c = mondrian_correction(scores, groups, cal, 0.8)
    for g in range(4):
        test = ~cal & (groups == g)
        assert abs(np.mean(scores[test] <= c[test]) - 0.8) < 0.02


def test_adaptive_tracks_drift_better_than_static():
    from x05_conditional_calibration import adaptive_correction, conformal_quantile
    rng = np.random.default_rng(4)
    n_cal, n_test = 5000, 8000
    scores = np.concatenate([np.abs(rng.normal(0, 1, n_cal)), np.abs(rng.normal(0, 1.6, n_test))])  # drift in test
    cal = np.arange(n_cal + n_test) < n_cal
    test = ~cal
    issue = np.arange(n_cal + n_test) * 3600 * 10**9
    target = issue + 6 * 3600 * 10**9
    groups = np.zeros(len(scores), int)
    adaptive = adaptive_correction(scores, groups, cal, test, issue, target, 0.8)
    static = conformal_quantile(scores[cal], 0.8)
    cov_adaptive = np.mean(scores[test] <= adaptive[test])
    cov_static = np.mean(scores[test] <= static)
    assert abs(cov_adaptive - 0.8) < abs(cov_static - 0.8)
    assert abs(cov_adaptive - 0.8) < 0.02


def test_adaptive_uses_no_future_label():
    from x05_conditional_calibration import adaptive_correction
    rng = np.random.default_rng(5)
    scores = np.abs(rng.normal(0, 1, 3000))
    cal = np.arange(3000) < 1000
    issue = np.arange(3000) * 3600 * 10**9
    target = issue + 6 * 3600 * 10**9
    a = adaptive_correction(scores, np.zeros(3000, int), cal, ~cal, issue, target, 0.8)
    changed = scores.copy(); changed[2500:] = 100.0  # later labels must not move earlier corrections
    b = adaptive_correction(changed, np.zeros(3000, int), cal, ~cal, issue, target, 0.8)
    assert np.array_equal(a[:2506], b[:2506], equal_nan=True)
    assert not np.array_equal(a, b, equal_nan=True)  # and they do move later ones


def _drift_data(shift_day=None, days=365, seed=6):
    rng = np.random.default_rng(seed)
    issue = pd.date_range('2022-01-01', periods=days * 24, freq='h')
    hist = pd.date_range('2021-01-01', '2021-12-31 23:00', freq='h')
    issue = hist.append(issue)
    issue_ns = issue.as_unit('ns').asi8
    target_ns = issue_ns + 6 * 3600 * 10**9
    level = np.where(pd.to_datetime(target_ns) >= pd.Timestamp('2022-01-01') + pd.Timedelta(days=shift_day), 0.8, 0.5) \
        if shift_day is not None else np.full(len(issue_ns), 0.5)
    y = level + rng.normal(0, 0.05, len(issue_ns))
    return issue_ns, target_ns, y, np.ones(len(y), bool)


def _fit_mean(y):
    def fit_predict(train, forward):
        return np.full(forward.sum(), y[train].mean())
    return fit_predict


def test_drift_policy_refits_after_a_level_shift_and_not_without_one():
    from x03_drift_updates import run_policy
    for shift, expect in ((120, True), (None, False)):
        issue_ns, target_ns, y, day = _drift_data(shift)
        frozen = np.full(len(y), y[pd.to_datetime(target_ns) < pd.Timestamp('2022-01-01')].mean())
        start, end = pd.Timestamp('2022-01-01').value, pd.Timestamp('2023-01-01').value
        pred, refits = run_policy(issue_ns, target_ns, y, day, _fit_mean(y), frozen, start, end, 4.0, (0.04, 0.01))
        if expect:
            assert refits and pd.Timestamp(refits[0]) <= pd.Timestamp('2022-01-01') + pd.Timedelta(days=125)
            late = pd.to_datetime(issue_ns) >= pd.Timestamp('2022-08-01')
            assert np.mean(np.abs(pred[late] - y[late])) < np.mean(np.abs(frozen[late] - y[late]))
        else:
            assert not refits


def test_optimal_reference_recovers_known_simplex_weights():
    from x02_optimal_reference import fit_weights, predict
    rng = np.random.default_rng(7)
    n = 6000
    refs = rng.uniform(0, 1, (n, 4))
    hour = rng.integers(0, 24, n)
    true = np.array([0.0, 0.7, 0.3, 0.0])
    y = refs @ true + rng.laplace(0, 0.01, n)
    fit = np.arange(n) < 4000
    w = fit_weights(refs, y, hour, fit)
    assert np.allclose(w.sum(axis=1), 1) and (w >= -1e-9).all()
    assert np.abs(w.mean(axis=0) - true).max() < 0.05
    mae_fit = np.mean(np.abs(predict(refs, hour, w)[~fit] - y[~fit]))
    assert mae_fit < min(np.mean(np.abs(refs[~fit, j] - y[~fit])) for j in range(4))
