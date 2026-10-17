"""A02: build the hourly frame for PVDAQ system 9068 and audit it WITHOUT scoring any forecast.

Stage 'audit' (run before the protocol is frozen) reads the raw files, builds completed-hour bins with the same
rules as corrected_study.py, checks the local-time convention on 2021 data only, and reports coverage by year.
Stage 'prepare' (run after the protocol is frozen) writes the private frame and h{2,4,6,12}.npz window arrays using
corrected_study.windows() unchanged, including the 2024 confirmation year.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parent  # repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pv_paths import PRIV_PVDAQ  # noqa: E402
RAW = PRIV_PVDAQ / 'raw'
PRIV = PRIV_PVDAQ / 'prepared'
OUT = HERE.parent / 'results/pvdaq9068_evaluation'
PROTO = HERE.parent / 'protocol/protocol_v43_pvdaq9068.json'
LAT, LON, ALT = 40.3864, -104.5512, 1407.0
TZ = 'America/Denver'  # provider timestamps follow local civil time WITH daylight saving (2021 centroid audit)
CAP_KW = None     # set from metadata in load_capacity()


def load_capacity():
    meta = json.loads((RAW / 'pvdaq9068/9068_system_metadata.json').read_text())
    sysm = meta.get('System', {})
    for k in sysm:
        if 'kw dc' in k.lower() or 'dc_capacity' in k.lower() or 'dc capacity' in k.lower():
            return float(sysm[k]), k
    inv = meta.get('Inverters', {})
    ac = sum(float(v.get('AC capacity(kW)', 0)) for v in inv.values())
    return ac, 'sum of inverter AC capacity(kW)'


def hourly(series, expected):
    series = series.sort_index().groupby(level=0).mean()
    grouped = series.resample('1h', closed='left', label='right')
    return grouped.mean().where(grouped.count() >= math.ceil(expected * .75))


def read_series(names, pattern, reduce='mean'):
    parts = []
    for n in names:
        head = pd.read_csv(RAW / 'pvdaq9068' / n, nrows=0).columns
        cols = [c for c in head if pattern(c)]
        df = pd.read_csv(RAW / 'pvdaq9068' / n, usecols=['measured_on'] + cols)
        t = pd.DatetimeIndex(pd.to_datetime(df.measured_on)).tz_localize(TZ, ambiguous='NaT', nonexistent='NaT')
        v = df[cols].apply(pd.to_numeric, errors='coerce')
        s = v.mean(axis=1) if reduce == 'mean' else (v.sum(axis=1, min_count=v.shape[1]) if reduce == 'sum_all' else v.iloc[:, 0])
        parts.append(pd.Series(s.to_numpy(), index=t))
    s = pd.concat(parts)
    return s[s.index.notna()].sort_index(), cols


def isd_humidity():
    parts = []
    for y in range(2020, 2026):
        df = pd.read_csv(RAW / f'isd/72476824051_{y}.csv', usecols=['DATE', 'TMP', 'DEW'], dtype=str)
        t = pd.to_datetime(df.DATE, utc=True)

        def val(col):
            x = df[col].str.split(',', expand=True)
            v = pd.to_numeric(x[0], errors='coerce') / 10
            ok = x[1].isin(['0', '1', '4', '5', '9', 'A', 'C', 'I', 'M', 'P', 'R', 'U']) & (pd.to_numeric(x[0], errors='coerce') != 9999)
            return v.where(ok)
        T, D = val('TMP'), val('DEW')
        rh = 100 * np.exp(17.625 * D / (243.04 + D)) / np.exp(17.625 * T / (243.04 + T))
        parts.append(pd.Series(rh.clip(0, 100).to_numpy(), index=pd.DatetimeIndex(t).tz_convert(TZ)))
    s = pd.concat(parts).dropna().sort_index()
    return hourly(s, 1)  # METAR reports near :53 fall in the completed hour they precede


def solar(index):
    from pvlib.location import Location
    loc = Location(LAT, LON, tz=TZ, altitude=ALT)
    return np.mean([loc.get_clearsky(index - pd.Timedelta(minutes=m))['ghi'].to_numpy() for m in (7.5, 22.5, 37.5, 52.5)], axis=0)


def build_frame():
    cap, cap_source = load_capacity()
    # Revenue meter fails in 2024 (ratio to inverters ~0.5, then zero/negative from August); the sum of the two
    # inverters is consistent in all years (audit a02_power_source.json), so it defines power throughout.
    p, pcols = read_series(['9068_ac_power_data.csv', '9068_ac_power_data_20240101_20250430.csv'],
                           lambda c: c.startswith('inverter_1_ac_power') or c.startswith('inverter_2_ac_power'), reduce='sum_all')
    g, gcols = read_series(['9068_irradiance_data.csv', '9068_irradiance_data_20240101_20250430.csv'],
                           lambda c: 'poa_irradiance_(w/m2)' in c and 'temp_compensated' not in c)
    a, acols = read_series(['9068_environment_data.csv', '9068_environment_data_20240101_20250430.csv'],
                           lambda c: c.startswith('weather_station_ambient_temperature'), reduce='first')
    step = p.index.to_series().diff().dt.total_seconds().median()
    exp = max(1, round(3600 / step))
    f = pd.concat([hourly(p / cap, exp).rename('power'), hourly(g, exp).rename('ghi'),
                   hourly(a, max(1, round(3600 / a.index.to_series().diff().dt.total_seconds().median()))).rename('temperature'),
                   isd_humidity().rename('humidity')], axis=1)
    f = f[(f.index.year >= 2021) & (f.index.year <= 2024)]
    f = f.reindex(pd.date_range(f.index.min().floor('D'), f.index.max().ceil('D'), freq='1h', tz=TZ))
    f['physical_bad'] = f.power.notna() & ~f.power.between(-.05, 1.2)
    f.loc[f.physical_bad, 'power'] = np.nan
    f.loc[~f.ghi.between(-20, 1500), 'ghi'] = np.nan
    f.loc[~f.temperature.between(-40, 60), 'temperature'] = np.nan  # site winters reach below -30 °C
    f.loc[~f.humidity.between(0, 100), 'humidity'] = np.nan
    f['clear_ghi'] = solar(f.index)
    info = dict(capacity_kw=cap, capacity_source=cap_source, power_column=pcols, irradiance_columns=gcols, temperature_column=acols,
                native_step_s=float(step), expected_per_hour=exp)
    return f, info


def audit():
    f, info = build_frame()
    rows = []
    for y in range(2021, 2025):
        q = f[f.index.year == y]
        rows.append(dict(year=y, hours=len(q), power_valid=float(q.power.notna().mean()),
                         all_channels_valid=float(q[['power', 'ghi', 'temperature', 'humidity']].notna().all(axis=1).mean()),
                         days_with_power=int(q.index[q.power.notna()].normalize().nunique()),
                         months_with_power=int(q.index[q.power.notna()].month.nunique()),
                         last_valid=str(q.power.last_valid_index()), physical_bad=int(q.physical_bad.sum()),
                         humidity_valid=float(q.humidity.notna().mean())))
    dev = f[f.index.year == 2021]
    # Local-time check on development data only: mean power and clear-sky by hour; centroid difference in hours.
    prof = dev.groupby(dev.index.hour)[['power', 'clear_ghi']].mean()
    cen = {k: float((prof.index * prof[k]).sum() / prof[k].sum()) for k in ('power', 'clear_ghi')}
    monthly_cen = {}
    for m, q in dev.groupby(dev.index.month):
        pr = q.groupby(q.index.hour).power.mean()
        monthly_cen[int(m)] = round(float((pr.index * pr).sum() / pr.sum()), 2)
    res = dict(info=info, years=rows, timezone=TZ, dev_centroid_hour=cen, dev_monthly_power_centroid=monthly_cen,
               dev_max_power=float(dev.power.max()), dev_p99_power=float(dev.power.quantile(.99)))
    (OUT / 'a02_audit.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def prepare():
    assert PROTO.exists(), 'freeze the protocol first'
    os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(PROTO)
    sys.path.insert(0, str(REL / 'scripts'))
    import corrected_study as cs
    from corrected_study import windows, QUANTILES, save_json, sha
    f, info = build_frame()
    bed = PRIV / 'PVDAQ9068'; bed.mkdir(parents=True, exist_ok=True)
    name = 'pvdaq_9068'
    f.to_pickle(bed / f'{name}.pkl'); save_json(bed / 'systems.json', [name])
    summary = []
    for h in (6, 2, 4, 12):
        w = windows(f, h, 0)
        order = np.lexsort((w['system'], w['issue_ns'])); w = {k: v[order] for k, v in w.items()}
        tt = pd.to_datetime(w['target_ns']); train = tt.year == 2021
        qs = np.zeros((len(w['y']), len(QUANTILES)), dtype='float32')
        for hr in range(24):
            fit = train & (tt.hour == hr); dest = tt.hour == hr
            if fit.sum() < 20:
                fit = train
            qs[dest] = np.clip(w['references'][dest, 3, None] + np.quantile(w['y'][fit] - w['references'][fit, 3], QUANTILES), 0, 1.2)
        w['prob_reference'] = qs
        np.savez_compressed(bed / f'h{h}.npz', **w)
        summary.append(dict(horizon=h, **{str(y): int((tt.year == y).sum()) for y in range(2021, 2025)},
                            **{f'{y}_daylight': int(((tt.year == y) & w['daylight']).sum()) for y in range(2021, 2025)},
                            sha256=sha(bed / f'h{h}.npz')))
    (OUT / 'a02_prepared_windows.json').write_text(json.dumps(dict(info=info, windows=summary), indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    {'audit': audit, 'prepare': prepare}[sys.argv[1]]()
