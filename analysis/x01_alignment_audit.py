"""X01 (v44 extension A): timestamp-alignment audit of every power series against solar geometry.

A series is flagged when its power is misaligned with clear-sky irradiance by about a whole hour or more. Two statistics
are computed on the 2021 (development) mean diurnal profile, never on calibration or test data:

  offset_h   the whole-hour shift k in {-2..+2} that maximizes the correlation between the shifted power profile and the
             clear-sky profile; k = -1 means the power must move one hour EARLIER to line up (the series is one hour late);
  lag_h      power-weighted mean hour minus clear-sky-weighted mean hour (a sub-hour diagnostic).

Rule (fixed in protocol/PROTOCOL_v44_extensions.md before any v44 scoring): flag when offset_h != 0 and |lag_h| >= 0.5.
A flagged series is corrected by offset_h hours before windows are built; the shift is logged.

Run on the released 2021 profiles (no private data needed):
    python analysis/x01_alignment_audit.py
Output: results/v44_extensions/x01_alignment_audit.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFSETS = (-2, -1, 0, 1, 2)


def lag_hours(hour, power, clear):
    p = np.clip(np.asarray(power, float), 0, None)
    c = np.clip(np.asarray(clear, float), 0, None)
    return float((hour * p).sum() / p.sum() - (hour * c).sum() / c.sum())


def best_offset(power, clear):
    """power, clear: 24-value mean profiles indexed by completed-hour label 0..23."""
    power = np.asarray(power, float)
    clear = np.asarray(clear, float)
    corr = {k: float(np.corrcoef(np.roll(power, k), clear)[0, 1]) for k in OFFSETS}
    return max(OFFSETS, key=corr.get), corr


def audit_profiles(profile):
    rows = []
    for (bed, system), g in profile.groupby(['testbed', 'system']):
        g = g.sort_values('hour')
        if len(g) != 24:
            raise ValueError(f'{bed}/{system}: expected 24 hourly values, got {len(g)}')
        k, corr = best_offset(g.power, g.clear_ghi)
        lag = lag_hours(g.hour.to_numpy(), g.power, g.clear_ghi)
        rows.append(dict(testbed=bed, system=system, offset_h=k, lag_h=round(lag, 3),
                         irradiance_lag_h=round(lag_hours(g.hour.to_numpy(), g.ghi, g.clear_ghi), 3),
                         **{f'corr_{o:+d}h': round(corr[o], 4) for o in OFFSETS},
                         flagged=bool(k != 0 and abs(lag) >= 0.5)))
    return pd.DataFrame(rows)


def main():
    profile = pd.read_csv(ROOT / 'results/corrected_v28_array1b/development_solar_cycle.csv')
    out = audit_profiles(profile)
    cadence = pd.read_csv(ROOT / 'results/replacement_audit/hkust_audit_files.csv')[['file', 'median_cadence_minutes']]
    systems = pd.read_csv(ROOT / 'results/replacement_audit/hkust_system_map.csv')[['site_file', 'system_id']]
    cadence = systems.merge(cadence, left_on='site_file', right_on='file')[['system_id', 'median_cadence_minutes']]
    out = out.merge(cadence, left_on='system', right_on='system_id', how='left').drop(columns='system_id')
    dest = ROOT / 'results/v44_extensions'
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / 'x01_alignment_audit.csv', index=False, lineterminator='\n')
    pd.set_option('display.width', 200)
    print(out.sort_values('lag_h').to_string(index=False))
    print('\nflagged:', int(out.flagged.sum()), 'of', len(out))
    print(pd.crosstab(out.median_cadence_minutes.fillna(-1), out.flagged))


if __name__ == '__main__':
    main()
