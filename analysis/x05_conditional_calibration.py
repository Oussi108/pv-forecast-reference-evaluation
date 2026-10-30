"""X05 (v44 extension E): condition-aware and adaptive correction of the central quantile intervals.

Nonconformity score as in calibrate_intervals(): s = max(q_lo - y, y - q_hi, 0); the corrected interval is
[q_lo - c, q_hi + c].

  elevation_bands  Mondrian split-conformal correction by solar-elevation band of the target hour: night (not daylight),
                   below 10 deg, 10-30 deg, at least 30 deg. Bands are fixed in advance; corrections use 2022 only;
                   a band with fewer than 100 calibration rows falls back to the global correction.
  adaptive         Adaptive conformal inference (Gibbs and Candes, 2021) run separately in the day and night groups:
                   alpha_{t+1} = alpha_t + gamma * (alpha - err_t), gamma = 0.005 fixed in advance. The correction for a
                   forecast is the (1 - alpha_t) empirical quantile of that group's 2022 scores. An error is fed back
                   only once its target time has passed the current issue time, so no future label is used.
"""
import math

import numpy as np
import pandas as pd

GAMMA = 0.005
MIN_GROUP = 100
BAND_EDGES = (10.0, 30.0)


def conformal_quantile(values, level):
    values = np.sort(np.asarray(values, float)[np.isfinite(values)])
    rank = math.ceil((len(values) + 1) * level)
    if not len(values):
        raise ValueError('empty calibration group')
    if rank > len(values):
        return float('inf')
    if rank < 1:
        return 0.0
    return float(values[rank - 1])


def _sorted_quantile(sorted_values, level):
    rank = math.ceil((len(sorted_values) + 1) * level)
    if rank > len(sorted_values):
        return float('inf')
    return 0.0 if rank < 1 else float(sorted_values[rank - 1])


def elevation_band(target_ns, daylight, latitude, longitude, timezone, altitude=0):
    """Band of the completed target hour [t-1h, t), evaluated at its midpoint."""
    from pvlib.location import Location
    loc = Location(latitude, longitude, tz=timezone, altitude=altitude)
    mid = pd.DatetimeIndex(pd.to_datetime(target_ns)).tz_localize(timezone, ambiguous='NaT', nonexistent='NaT') \
        - pd.Timedelta(minutes=30)
    elevation = loc.get_solarposition(mid)['apparent_elevation'].to_numpy()
    band = np.where(elevation < BAND_EDGES[0], 1, np.where(elevation < BAND_EDGES[1], 2, 3))
    return np.where(np.asarray(daylight, bool), band, 0)


def mondrian_correction(scores, groups, cal_mask, level):
    global_c = conformal_quantile(scores[cal_mask], level)
    out = np.full(len(scores), global_c)
    for g in np.unique(groups):
        fit = cal_mask & (groups == g)
        if fit.sum() >= MIN_GROUP:
            out[groups == g] = conformal_quantile(scores[fit], level)
    return out


def adaptive_correction(scores, groups, cal_mask, test_mask, issue_ns, target_ns, level, gamma=GAMMA):
    """Corrections for the test rows (NaN elsewhere); groups processed independently, rows in issue-time order."""
    alpha = 1 - level
    out = np.full(len(scores), np.nan)
    for g in np.unique(groups[test_mask]):
        cal = np.sort(scores[cal_mask & (groups == g)])
        cal = cal[np.isfinite(cal)]
        rows = np.flatnonzero(test_mask & (groups == g))
        rows = rows[np.argsort(issue_ns[rows], kind='stable')]
        a = alpha
        pending = []  # (target_ns, err)
        for i in rows:
            ready = [p for p in pending if p[0] <= issue_ns[i]]
            if ready:
                pending = [p for p in pending if p[0] > issue_ns[i]]
                for _, err in sorted(ready):
                    a = a + gamma * (alpha - err)
            c = _sorted_quantile(cal, min(max(1 - a, 0.0), 1.0))
            out[i] = c
            pending.append((target_ns[i], float(scores[i] > c)))
    return out
