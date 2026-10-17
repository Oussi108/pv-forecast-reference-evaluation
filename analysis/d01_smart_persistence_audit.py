"""D01: audit of the clear-sky ('smart') persistence reference used in Paper 1 v28-v40.

Reads the private prepared hourly frames and window arrays (read-only), reproduces
the Table 2 reference MAEs, diagnoses where the v28 smart reference fails, and scores
fairer smart-persistence variants on exactly the same 2023 six-hour daylight keys.
No model is trained; no source file is modified.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pv_paths import PREP_V28, RESULTS  # noqa: E402
PREP = PREP_V28
OUT = RESULTS / 'reference_and_update_inference'
H = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SITES = {'DKASC': (-23.7626, 133.8748, 580, 'Australia/Darwin'),
         'HKUST': (22.3363, 114.2634, 132, 'Asia/Hong_Kong')}


def balanced(err, system):
    s = pd.Series(err).groupby(system).mean()
    return float(s.mean())


def clear_components(index, testbed):
    """Hour-mean clear-sky GHI and dual-axis plane-of-array irradiance (4-point quadrature)."""
    from pvlib.location import Location
    lat, lon, alt, tz = SITES[testbed]
    loc = Location(lat, lon, tz=tz, altitude=alt)
    ghi, poa = [], []
    for m in (7.5, 22.5, 37.5, 52.5):
        t = index - pd.Timedelta(minutes=m)
        cs = loc.get_clearsky(t)
        zen = loc.get_solarposition(t)['apparent_zenith'].to_numpy()
        up = zen < 90
        # Dual-axis tracker: surface normal to the sun (tilt = zenith), isotropic sky, albedo 0.2.
        tilt = np.radians(np.clip(zen, 0, 90))
        p = cs['dni'].to_numpy() + cs['dhi'].to_numpy() * (1 + np.cos(tilt)) / 2 \
            + 0.2 * cs['ghi'].to_numpy() * (1 - np.cos(tilt)) / 2
        ghi.append(cs['ghi'].to_numpy())
        poa.append(np.where(up, p, 0.0))
    return np.mean(ghi, 0), np.mean(poa, 0)


def smart_from(frame, clear, horizon, min_clear, max_age=24):
    """ratio = power/clear at the latest hour with clear > min_clear (<= max_age h old); times target clear."""
    clear = pd.Series(clear, index=frame.index)
    ratio = (frame.power / clear).where(clear > min_clear)
    src_age = pd.Series(np.where(ratio.notna(), 0, np.nan), index=frame.index)
    ratio = ratio.ffill(limit=max_age)
    target_clear = clear.reindex(frame.index + pd.Timedelta(hours=horizon)).to_numpy()
    pred = ratio.to_numpy() * target_clear
    pred[target_clear <= 20] = 0
    return np.clip(pred, 0, 1.2)


def envelope(frame, horizon):
    """Empirical clear-sky power envelope from 2021 only: 95th pct of power per hour-of-day in a
    +/-15-day day-of-year window. kcs at issue time (envelope >= 0.05) is persisted to the target."""
    dev = frame[frame.index.year == 2021].power.dropna()
    hod, doy = dev.index.hour.to_numpy(), dev.index.dayofyear.to_numpy()
    table = np.full((24, 367), np.nan)
    v = dev.to_numpy()
    for h in range(24):
        mh = hod == h
        for d in range(1, 367):
            dd = np.abs(((doy[mh] - d + 183) % 366) - 183) <= 15
            if dd.sum() >= 10:
                table[h, d] = np.quantile(v[mh][dd], 0.95)
    env = pd.Series(table[frame.index.hour, frame.index.dayofyear], index=frame.index).fillna(0).clip(0, 1.2)
    kcs = (frame.power / env).where(env >= 0.05).ffill(limit=24)
    tgt = env.reindex(frame.index + pd.Timedelta(hours=horizon)).to_numpy()
    pred = np.clip(kcs.to_numpy() * tgt, 0, 1.2)
    return pred


def run(testbed):
    arr = np.load(PREP / testbed / f'h{H}.npz')
    systems = json.loads((PREP / testbed / 'systems.json').read_text())
    issue = pd.to_datetime(arr['issue_ns'])
    test = (issue.year == 2023) & arr['daylight']
    y, refs, sysc = arr['y'][test], arr['references'][test], arr['system'][test]
    issue_t = issue[test]
    out = {name: balanced(np.abs(refs[:, j] - y), sysc)
           for j, name in enumerate(['persistence', 'previous_day', 'climatology', 'smart_v28'])}

    variants = {k: np.full(test.sum(), np.nan) for k in
                ['smart_v28_recomputed', 'smart_ghi_min200', 'smart_poa_min20', 'smart_poa_min200', 'smart_envelope']}
    src_clear = np.full(test.sum(), np.nan)
    for code, name in enumerate(systems):
        fr = pd.read_pickle(PREP / testbed / f'{name}.pkl')
        ghi_cs, poa_cs = clear_components(fr.index, testbed)
        cand = {'smart_v28_recomputed': smart_from(fr, fr.clear_ghi.to_numpy(), H, 20),
                'smart_ghi_min200': smart_from(fr, fr.clear_ghi.to_numpy(), H, 200),
                'smart_poa_min20': smart_from(fr, poa_cs, H, 20),
                'smart_poa_min200': smart_from(fr, poa_cs, H, 200),
                'smart_envelope': envelope(fr, H)}
        # clear GHI at the hour whose ratio v28 actually persisted
        c = fr.clear_ghi.where(fr.clear_ghi > 20)
        src = pd.Series(np.where(c.notna() & fr.power.notna(), fr.clear_ghi, np.nan), index=fr.index).ffill(limit=24)
        pos = fr.index.get_indexer(pd.DatetimeIndex(issue_t[sysc == code]).tz_localize(fr.index.tz, ambiguous='NaT', nonexistent='NaT'))
        ok = pos >= 0
        rows = np.flatnonzero(sysc == code)
        for k, p in cand.items():
            variants[k][rows[ok]] = p[pos[ok]]
        src_clear[rows[ok]] = src.to_numpy()[pos[ok]]
    keep = np.isfinite(np.column_stack(list(variants.values()))).all(1)
    out['keys'] = int(test.sum()); out['keys_all_variants_finite'] = int(keep.sum())
    out['max_abs_diff_v28_vs_recomputed'] = float(np.nanmax(np.abs(variants['smart_v28_recomputed'] - refs[:, 3])))
    for k, p in variants.items():
        out[k + '_on_common_keys'] = balanced(np.abs(p[keep] - y[keep]), sysc[keep])
    out['previous_day_on_common_keys'] = balanced(np.abs(refs[keep, 1] - y[keep]), sysc[keep])
    # Where does the v28 smart reference fail? error by clear GHI of the persisted source hour
    bins = [20, 100, 200, 400, 700, 2000]
    e = np.abs(refs[:, 3] - y)
    cat = pd.cut(src_clear, bins)
    out['v28_error_by_source_clear_ghi'] = {str(b): {'share': float((cat == b).mean()), 'mae': float(e[cat == b].mean())}
                                             for b in cat.categories if (cat == b).any()}
    out['issue_hour_share_night'] = float(np.mean(arr['future'][test][:, 0] > 0) if False else np.mean(
        pd.DatetimeIndex(issue_t).hour.isin(list(range(0, 6)) + list(range(19, 24)))))
    return out


if __name__ == '__main__':
    res = {tb: run(tb) for tb in ('DKASC', 'HKUST')}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'd01_smart_persistence_audit_h{H}.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
