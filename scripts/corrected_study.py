"""Completed-hour data preparation and exact-time references for revision 27.

Provider observations and training arrays are written only to the explicitly
supplied private directory. Public outputs contain aggregate audit records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_protocol_override = os.environ.get('PV_FORECAST_PROTOCOL_PATH')
PROTOCOL_PATH = Path(_protocol_override).resolve() if _protocol_override else ROOT / 'configs/corrected_study_v28_array1b.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
PUBLIC = ROOT / PROTOCOL.get('public_results_dir', 'results/corrected_v27')
WEATHER = ['ghi', 'temperature', 'humidity']
CHANNELS = PROTOCOL['history_channels']
QUANTILES = np.array(PROTOCOL['quantiles'])


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def private_path(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Observation-containing output must be outside the public release')
    path.mkdir(parents=True, exist_ok=True)
    return path


def hourly(series, expected):
    """The bin [t-1h,t) is labelled by its completion time t."""
    series = series.sort_index().groupby(level=0).mean()
    grouped = series.resample('1h', closed='left', label='right')
    return grouped.mean().where(grouped.count() >= math.ceil(expected * .75))


def solar(index, testbed):
    from pvlib.location import Location
    lat, lon, altitude, zone = ((22.3363, 114.2634, 132, 'Asia/Hong_Kong')
                              if testbed == 'HKUST' else (-23.7626, 133.8748, 580, 'Australia/Darwin'))
    loc = Location(lat, lon, tz=zone, altitude=altitude)
    # Four fixed quadrature points for mean clear-sky GHI over a completed hour.
    values = []
    for minutes in (7.5, 22.5, 37.5, 52.5):
        values.append(loc.get_clearsky(index - pd.Timedelta(minutes=minutes))['ghi'].to_numpy())
    return np.mean(values, axis=0)


def clean_frame(frame, testbed):
    frame = frame.copy()
    frame['physical_bad'] = frame.power.notna() & ~frame.power.between(-.05, 1.2)
    frame.loc[frame.physical_bad, 'power'] = np.nan
    # Small negative meter values are retained as observations, not silently clipped.
    frame.loc[~frame.ghi.between(-20, 1500), 'ghi'] = np.nan
    frame.loc[~frame.temperature.between(-30, 60), 'temperature'] = np.nan
    frame.loc[~frame.humidity.between(0, 100), 'humidity'] = np.nan
    frame['clear_ghi'] = solar(frame.index, testbed)
    return frame


def summarize_source(frame, name, testbed):
    row = {'testbed': testbed, 'system': name}
    for year in (2021, 2022, 2023):
        f = frame[frame.index.year == year]
        valid = f.power.notna()
        row[f'days_{year}'] = int(f.index[valid].normalize().nunique())
        row[f'fraction_{year}'] = float(valid.sum() / 8760)
        row[f'months_{year}'] = int(f.index[valid].month.nunique())
    pre = frame[frame.index.year < 2023]
    row['bad_pretest_fraction'] = float(pre.physical_bad.sum() / max(1, pre.power.notna().sum() + pre.physical_bad.sum()))
    gate = PROTOCOL['coverage']
    row['eligible'] = bool(row['days_2021'] >= gate['development_minimum_days'] and
                           row['days_2021'] + row['days_2022'] >= gate['pretest_minimum_days'] and
                           row['fraction_2022'] >= gate['calibration_minimum_fraction'] and
                           row['bad_pretest_fraction'] <= gate['pretest_physical_bad_fraction_max'])
    row['test_coverage_diagnostic_pass'] = bool(row['days_2023'] >= 300 and row['months_2023'] >= 10 and row['fraction_2023'] >= .9)
    return row


def exact_references(frame, horizon):
    index = frame.index
    target_index = index + pd.Timedelta(hours=horizon)
    y = frame.power.reindex(target_index).to_numpy()
    previous = frame.power.reindex(target_index - pd.Timedelta(days=1)).to_numpy()
    # For pretest diagnostic rows this table is not used for model selection.
    development = frame[frame.index.year == 2021]
    by_hour = development.groupby(development.index.hour).power.mean()
    climate = pd.Series(target_index.hour).map(by_hour).to_numpy()
    target_clear = frame.clear_ghi.reindex(target_index).to_numpy()
    clear_ratio = (frame.power / frame.clear_ghi).where(frame.clear_ghi > 20).ffill(limit=24)
    smart = clear_ratio.to_numpy() * target_clear
    smart[target_clear <= 20] = 0
    refs = np.column_stack([frame.power.to_numpy(), previous, climate, smart])
    return y, np.clip(refs, 0, 1.2), target_clear


def windows(frame, horizon, system_code):
    y, references, target_clear = exact_references(frame, horizon)
    # Exact-time lookup makes missing hours explicit; no row-offset semantics.
    sequence = np.stack([frame[CHANNELS].reindex(frame.index - pd.Timedelta(hours=lag)).to_numpy()
                         for lag in range(23, -1, -1)], axis=1)
    target_time = frame.index + pd.Timedelta(hours=horizon)
    future = np.column_stack([target_clear, np.sin(2*np.pi*target_time.hour/24),
                              np.cos(2*np.pi*target_time.hour/24), np.sin(2*np.pi*target_time.dayofyear/365.25),
                              np.cos(2*np.pi*target_time.dayofyear/365.25)])
    issue_ns = frame.index.tz_localize(None).as_unit('ns').asi8
    target_ns = target_time.tz_localize(None).as_unit('ns').asi8
    # No training label extends into calibration; no calibration label into test.
    valid = np.isfinite(sequence).all(axis=(1,2)) & np.isfinite(future).all(axis=1) & np.isfinite(y) & np.isfinite(references).all(axis=1)
    valid &= frame.index.year == target_time.year
    for boundary in [pd.Timestamp('2021-10-01', tz=frame.index.tz)]:
        valid &= ~((frame.index < boundary) & (target_time >= boundary))
    recent_power = sequence[:, -6:, 0]
    variation = np.full(len(frame), np.nan, dtype='float64')
    recent_supported = np.isfinite(recent_power).any(axis=1)
    variation[recent_supported] = np.nanstd(recent_power[recent_supported], axis=1)
    return {k:v[valid] for k,v in dict(x=sequence.astype('float32'), future=future.astype('float32'),
              y=y.astype('float32'), references=references.astype('float32'),
              issue_ns=issue_ns, target_ns=target_ns, system=np.full(len(frame),system_code,dtype='int32'),
              variation=variation.astype('float32'), daylight=(target_clear>20)).items()}


def prepare(private_root, hkust_root, dkasc_file):
    private_root = private_path(private_root)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    audit_root = ROOT / 'results/replacement_audit'
    mappings = pd.read_csv(audit_root/'hkust_system_map.csv')
    source_audit = json.loads((audit_root/'hkust_audit.json').read_text())
    # Re-read the raw weather files so completed bins meet the source support rule.
    parts = {name:[] for name in WEATHER}
    alias = {'irradiance':'ghi', 'temperature':'temperature', 'humidity':'humidity'}
    sources = {}
    for record in source_audit['files']:
        if record.get('role') != 'weather':
            continue
        for old, field in alias.items():
            column = record.get('columns',{}).get(old)
            if not column:
                continue
            p = Path(hkust_root)/record['file']
            raw = pd.read_csv(p, usecols=[record['columns']['timestamp'],column])
            t = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:,0])).tz_localize('Asia/Hong_Kong')
            parts[field].append(hourly(pd.Series(pd.to_numeric(raw[column],errors='coerce').to_numpy(),index=t),60))
            sources[record['file']] = sha(p)
    weather = pd.concat([pd.concat(series).groupby(level=0).mean().rename(name) for name,series in parts.items()],axis=1)
    all_frames = {'HKUST':{}, 'DKASC':{}}
    rows = []
    for item in mappings.to_dict('records'):
        p = Path(hkust_root)/item['site_file']
        raw = pd.read_csv(p)
        time_col = next(c for c in raw if c.lower() in ('time','timestamp','datetime'))
        power_col = next(c for c in raw if c.lower().startswith('power'))
        t = pd.DatetimeIndex(pd.to_datetime(raw[time_col])).tz_localize('Asia/Hong_Kong')
        step = t.to_series().diff().dt.total_seconds().dropna()
        step = step[step > 0].median()
        s = pd.Series(pd.to_numeric(raw[power_col],errors='coerce').to_numpy()/float(item['capacity_kw'])/1000,index=t)
        f = hourly(s, max(1,round(3600/step))).rename('power').to_frame().join(weather)
        f = clean_frame(f,'HKUST')
        info = summarize_source(f,item['system_id'],'HKUST')
        rows.append(info)
        if info['eligible']:
            all_frames['HKUST'][item['system_id']] = f
        sources[item['site_file']] = sha(p)
        print('AUDIT',item['system_id'],info['eligible'],flush=True)
    dka_source = PROTOCOL.get('sources', {}).get('DKASC', {})
    dka_capacity = float(dka_source.get('capacity_kw', 10.5))
    dka_site_id = str(dka_source.get('site_id', 'dka_m9_bphase'))
    dka_timezone = str(dka_source.get('timezone', PROTOCOL.get('source_timezone', {}).get('DKASC', 'Australia/Darwin')))
    raw = pd.read_csv(dkasc_file)
    t = pd.DatetimeIndex(pd.to_datetime(raw.timestamp)).tz_localize(dka_timezone)
    columns = {'Active_Power':'power', 'Global_Horizontal_Radiation':'ghi',
               'Weather_Temperature_Celsius':'temperature', 'Weather_Relative_Humidity':'humidity'}
    f = pd.concat([hourly(pd.Series(pd.to_numeric(raw[c],errors='coerce').to_numpy()/(dka_capacity if name=='power' else 1),index=t),12).rename(name)
                   for c,name in columns.items()],axis=1)
    f = f[(f.index.year>=2021)&(f.index.year<=2023)]
    f = clean_frame(f,'DKASC')
    info = summarize_source(f,dka_site_id,'DKASC')
    rows.append(info)
    if not info['eligible']:
        raise RuntimeError(f'DKASC pretest coverage gate failed: {info}')
    all_frames['DKASC'][dka_site_id] = f
    sources[f"DKASC/{dka_source.get('provider_filename', dkasc_file.name)}"] = sha(dkasc_file)
    pd.DataFrame(rows).to_csv(PUBLIC/'system_screen.csv',index=False)
    summaries, coverage, cycle = [],[],[]
    for bed,frames in all_frames.items():
        if not frames:
            raise RuntimeError(f'No eligible systems for {bed}')
        bedroot = private_root/bed
        bedroot.mkdir(exist_ok=True)
        names = sorted(frames)
        for name,f in frames.items():
            f.to_pickle(bedroot/(name+'.pkl'))
            for month, part in f.groupby(f.index.strftime('%Y-%m')):
                coverage.append(dict(testbed=bed,system=name,month=month,rows=len(part),valid_power=int(part.power.notna().sum()),
                                     valid_weather=int(part[WEATHER].notna().all(axis=1).sum())))
            pre = f[f.index.year==2021]
            hourly_cycle = pre.groupby(pre.index.hour)[['power','ghi','clear_ghi']].mean()
            for hour,row in hourly_cycle.iterrows():
                cycle.append(dict(testbed=bed,system=name,hour=int(hour),**row.to_dict()))
        for horizon in PROTOCOL['horizons']:
            parts = [windows(frames[name],horizon,i) for i,name in enumerate(names)]
            data = {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
            order = np.lexsort((data['system'],data['issue_ns']))
            data = {k:v[order] for k,v in data.items()}
            target_time = pd.to_datetime(data['target_ns'])
            cal, test = target_time.year==2022, target_time.year==2023
            train = target_time.year==2021
            # Persistence residual distributions are fitted exclusively on development.
            qs = np.zeros((len(data['y']),len(QUANTILES)),dtype='float32')
            for sid in range(len(names)):
                for hr in range(24):
                    fit = train & (data['system']==sid) & (target_time.hour==hr)
                    dest = (data['system']==sid) & (target_time.hour==hr)
                    if fit.sum()<20:
                        fit = train & (data['system']==sid)
                    residual = data['y'][fit]-data['references'][fit,3]
                    qs[dest] = np.clip(data['references'][dest,3,None]+np.quantile(residual,QUANTILES),0,1.2)
            data['prob_reference'] = qs
            np.savez_compressed(bedroot/f'h{horizon}.npz',**data)
            save_json(bedroot/'systems.json',names)
            summary = dict(testbed=bed,horizon=horizon,systems=len(names),development=int(train.sum()),calibration=int(cal.sum()),test=int(test.sum()),
                           test_daylight=int((test & data['daylight']).sum()),sha256=sha(bedroot/f'h{horizon}.npz'))
            summaries.append(summary)
            print('WINDOWS',summary,flush=True)
    pd.DataFrame(coverage).to_csv(PUBLIC/'monthly_coverage.csv',index=False)
    pd.DataFrame(cycle).to_csv(PUBLIC/'development_solar_cycle.csv',index=False)
    versions = {name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','lightgbm','pvlib']}
    save_json(PUBLIC/'preparation.json',dict(protocol_sha256=sha(PROTOCOL_PATH),code_sha256=sha(Path(__file__)),source_hashes=sources,versions=versions,
               populations=summaries,timezone_evidence='Local civil-time interpretation supported by development-period daily cycles; provider timezone text not located',
               observations_private=True,completed_hour_contract=PROTOCOL['timestamp_contract']))


def conformal_correction(scores, level):
    values = np.asarray(scores,dtype=float)
    values = values[np.isfinite(values)]
    rank = math.ceil((len(values)+1)*level)
    if not len(values) or rank>len(values):
        raise ValueError('Calibration group too small for requested level')
    return float(np.partition(values,rank-1)[rank-1])


def balanced_mae(y,pred,systems):
    return float(np.mean([np.abs(y[systems==s]-pred[systems==s]).mean() for s in np.unique(systems)]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--hkust-root',type=Path,required=True)
    parser.add_argument('--dkasc-file',type=Path,required=True)
    a=parser.parse_args()
    prepare(a.private_root,a.hkust_root,a.dkasc_file)
