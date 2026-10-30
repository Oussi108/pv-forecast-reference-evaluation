"""X04 (v44 extension D): seed-aware synchronized block bootstrap for paired MAE skill.

block_interval() in scripts/evaluate_corrected_study.py averages the absolute losses of the five training seeds BEFORE
resampling, so its interval reflects test-period sampling only. seed_block_interval() adds a second level: in every
resample the seven-day calendar blocks are drawn exactly as in block_interval() (same generator, same seed, same order),
and, independently, the model's seeds are drawn with replacement from a separate generator. The model loss of a resample is
the mean over the drawn seeds. With identical seeds the interval is therefore identical to block_interval()'s.

The point estimate is unchanged (all seeds averaged). The reference may be seed-free (1-D loss) or seed-dependent (2-D).
No model is fitted; the inputs are the saved per-seed predictions.
"""
import math

import numpy as np
import pandas as pd

BLOCK_SEED = 42           # protocol inference seed (block draws), as in block_interval()
SEED_DRAW_SEED = 4242     # separate generator for the seed draws, fixed in PROTOCOL_v44_extensions.md


def _daily_site(loss, day, site, shape):
    out = np.zeros(shape)
    np.add.at(out, (day, site), loss)
    return out


def balanced(values, system):
    return float(np.mean([np.mean(values[system == s]) for s in np.unique(system)]))


def seed_block_interval(model_loss, ref_loss, system, target_ns, days=7, resamples=2000):
    """model_loss: (seeds, n) absolute errors; ref_loss: (n,) or (seeds_ref, n). Returns skill and 95% interval."""
    model_loss = np.atleast_2d(np.asarray(model_loss, float))
    ref_loss = np.atleast_2d(np.asarray(ref_loss, float))
    dates = pd.to_datetime(target_ns).normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq='D')
    if len(calendar) < days:
        raise ValueError('Block exceeds available calendar span')
    day = calendar.get_indexer(dates)
    sites = np.unique(system)
    site = np.searchsorted(sites, system)
    shape = (len(calendar), len(sites))
    n = np.zeros(shape)
    np.add.at(n, (day, site), 1)
    e = np.stack([_daily_site(row, day, site, shape) for row in model_loss])
    b = np.stack([_daily_site(row, day, site, shape) for row in ref_loss])
    rng = np.random.default_rng(BLOCK_SEED)
    srng = np.random.default_rng(SEED_DRAW_SEED)
    draws = []
    for _ in range(resamples):
        starts = rng.integers(0, max(1, len(calendar) - days + 1), size=math.ceil(len(calendar) / days))
        idx = (starts[:, None] + np.arange(days)[None, :]).ravel()[:len(calendar)]
        pick_m = srng.integers(0, len(e), size=len(e))
        pick_r = srng.integers(0, len(b), size=len(b))
        nn = n[idx].sum(axis=0)
        if np.any(nn == 0):
            continue
        ee = (e[pick_m][:, idx].sum(axis=1).mean(axis=0) / nn).mean()
        bb = (b[pick_r][:, idx].sum(axis=1).mean(axis=0) / nn).mean()
        draws.append(1 - ee / bb)
    estimate = 1 - balanced(model_loss.mean(axis=0), system) / balanced(ref_loss.mean(axis=0), system)
    ci = np.quantile(draws, [.025, .975])
    return dict(skill=estimate, ci_low=float(ci[0]), ci_high=float(ci[1]), resamples=len(draws),
                seeds=len(model_loss), calendar_days=len(calendar), block_days=days)
