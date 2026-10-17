"""Targeted, retrospective controls for the v29 manuscript.

Original predictions are immutable. New predictions and fitted models stay in
private workspace storage; only configuration, provenance and aggregates are
written to the release. Run --stage calibration, updating, or temporal.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(ROOT/'configs/corrected_study_v28_array1b.json')
import numpy as np
import pandas as pd
from corrected_study import (PROTOCOL, QUANTILES, hourly, clean_frame, windows,
                             save_json, sha, conformal_correction)
from evaluate_corrected_study import balanced, block_interval, interval_score, holm_adjust

CONFIG = ROOT/'configs/publication_revision_v29.json'
PLAN = json.loads(CONFIG.read_text())
OUT = ROOT/'results/publication_v29'
WORK = ROOT.parents[3]
OLD = WORK/'tmp/corrected_v28_array1b'
PRIVATE = WORK/'tmp/publication_v29'
PARENT = ROOT/'results/corrected_v28_array1b'


def load(path, features=False):
    with np.load(path) as z:
        return {k:z[k] for k in z.files if features or k not in ('x','future')}


def log(event, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    row=dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(),event=event,**details)
    with (OUT/'execution.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row,default=str)+'\n')
    save_json(OUT/'progress.json', row)
    print(json.dumps(row,default=str),flush=True)


def paired(loss, reference_loss, d, mask, **labels):
    if pd.to_datetime(d['target_ns'][mask]).normalize().nunique()<14:
        return dict(**labels,status='insufficient_distinct_days')
    return dict(**labels,status='complete',**block_interval(
        loss[mask],reference_loss[mask],d['system'][mask],d['target_ns'][mask],days=7,resamples=2000))


def residual_quantiles(d, center):
    times=pd.to_datetime(d['target_ns']); fit_year=times.year==2021
    q=np.full((len(times),len(QUANTILES)),np.nan,dtype=np.float32)
    for s in np.unique(d['system']):
        for hr in range(24):
            fit=fit_year & (d['system']==s) & (times.hour==hr)
            if fit.sum()<20: fit=fit_year & (d['system']==s)
            assert fit.any() and np.all(times[fit].year==2021)
            dest=(d['system']==s)&(times.hour==hr)
            q[dest]=np.clip(center[dest,None]+np.quantile(d['y'][fit]-center[fit],QUANTILES),0,1.2)
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
    return q


def correction_maps(d, q, level):
    times=pd.to_datetime(d['target_ns']); cal=times.year==2022
    train=times.year==2021; lo,hi=(1,5) if level==.8 else (0,6)
    scores=np.maximum.reduce([q[:,lo]-d['y'],d['y']-q[:,hi],np.zeros(len(q))])
    a0=conformal_correction(scores[cal],level)
    threshold=np.quantile(d['variation'][train & d['daylight']],[1/3,2/3])
    labels=np.where(~d['daylight'],0,1+np.searchsorted(threshold,d['variation'],side='left'))
    maps={'raw':np.zeros(len(q)),'global_all_hours':np.full(len(q),a0)}
    rows=[]
    for name,groups in [('global_day_night',d['daylight'].astype(int)),('variability_groups',labels)]:
        adjust=np.full(len(q),a0)
        for group in np.unique(groups):
            fit=cal & (groups==group)
            value=conformal_correction(scores[fit],level) if fit.sum()>=100 else a0
            adjust[groups==group]=value
            rows.append(dict(method=name,group=int(group),n=int(fit.sum()),correction=value,fallback=bool(fit.sum()<100)))
        maps[name]=adjust
    return maps,rows,lo,hi


def calibration():
    metrics=[]; inference=[]; distributions=[]; group_rows=[]; checks=[]
    original=pd.read_csv(PARENT/'interval_metrics.csv')
    for bed in ['DKASC','HKUST']:
        path=OLD/bed/'h6.npz'; d=load(path)
        times=pd.to_datetime(d['target_ns']); test=times.year==2023
        models={'probabilistic_persistence':d['prob_reference'],
                'previous_day_residual':residual_quantiles(d,d['references'][:,1])}
        reproduced=residual_quantiles(d,d['references'][:,3])
        assert np.allclose(reproduced,d['prob_reference'],atol=2e-7)
        for model in ['lightgbm','transformer']:
            paths=[OLD/'runs'/f'{bed}_h6_{model}_s{s}_quantile_full/prediction.npz' for s in PROTOCOL['seeds']]
            models[model]=np.mean([load(p)['prediction'] for p in paths],axis=0)
        distribution_losses={}
        for model,q in models.items():
            error=d['y'][:,None]-q
            pinball=np.maximum(QUANTILES*error,(QUANTILES-1)*error)
            crps=2*np.trapezoid(pinball,QUANTILES,axis=1)
            distribution_losses[model]=crps
            for pop,mask in [('all',test),('daylight',test & d['daylight'])]:
                distributions.append(dict(testbed=bed,model=model,population=pop,n=int(mask.sum()),
                    crps7_truncated=balanced(crps[mask],d['system'][mask]),pinball=balanced(pinball[mask].mean(axis=1),d['system'][mask])))
            for level in [.8,.9]:
                maps,groups,lo,hi=correction_maps(d,q,level)
                group_rows += [dict(testbed=bed,model=model,level=level,**r) for r in groups]
                losses={}
                for method,a in maps.items():
                    lower=q[:,lo]-a; upper=q[:,hi]+a
                    losses[method]=interval_score(d['y'],lower,upper,level)
                    for pop,mask in [('all',test),('daylight',test & d['daylight'])]:
                        s=d['system'][mask]
                        row=dict(testbed=bed,model=model,level=level,method=method,population=pop,n=int(mask.sum()),
                            coverage=balanced((d['y'][mask]>=lower[mask]) & (d['y'][mask]<=upper[mask]),s),
                            width=balanced(upper[mask]-lower[mask],s),interval_score=balanced(losses[method][mask],s))
                        metrics.append(row)
                        old_name={'raw':'raw','global_all_hours':'global','variability_groups':'regime'}.get(method)
                        if model!='previous_day_residual' and old_name:
                            old=original[(original.testbed==bed)&(original.model==model)&(original.level==level)&(original.method==old_name)&(original.population==pop)]
                            assert len(old)==1
                            for field in ['coverage','width','interval_score']:
                                assert abs(row[field]-float(old.iloc[0][field]))<1e-7,(row,field)
                            checks.append(dict(testbed=bed,model=model,level=level,method=method,population=pop,status='v28_reproduced'))
                for method,ref in [('global_day_night','global_all_hours'),('variability_groups','global_day_night')]:
                    inference.append(paired(losses[method],losses[ref],d,test & d['daylight'],
                        analysis='calibration_control',year=2023,testbed=bed,model=model,level=level,method=method,
                        reference=ref,population='daylight',metric='interval_score'))
            log('calibration_model_complete',testbed=bed,model=model)
        for model in ['lightgbm','transformer','probabilistic_persistence']:
            inference.append(paired(distribution_losses[model],distribution_losses['previous_day_residual'],d,test & d['daylight'],
                analysis='distribution_reference',year=2023,testbed=bed,model=model,level=0,method='raw',reference='previous_day_residual',population='daylight',metric='crps7_truncated'))
    for name,rows in [('calibration_metrics',metrics),('calibration_groups',group_rows),('distribution_metrics',distributions)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    table=holm_adjust(pd.DataFrame(inference),['year','testbed','analysis','metric'])
    table.to_csv(OUT/'calibration_inference.csv',index=False)
    save_json(OUT/'calibration_verification.json',dict(status='passed',v28_rows_reproduced=len(checks),checks=checks,
        residual_fit_year=2021,calibration_fit_year=2022,test_year=2023,protocol_sha256=sha(CONFIG)))
    log('calibration_complete',comparisons=len(table))


def extended_data():
    cached=PRIVATE/'DKASC_extended/h2.npz'
    if cached.exists():
        gate=json.loads((OUT/'temporal_preparation.json').read_text())
        assert gate['protocol_sha256']==sha(CONFIG)
        return
    source=ROOT/'data/raw/87-Site_DKA-M9_A+C-Phases.csv'
    assert sha(source)==PROTOCOL['sources']['DKASC']['source_sha256']
    columns={'Active_Power':'power','Global_Horizontal_Radiation':'ghi',
             'Weather_Temperature_Celsius':'temperature','Weather_Relative_Humidity':'humidity'}
    raw=pd.read_csv(source,usecols=['timestamp',*columns])
    ts=pd.DatetimeIndex(pd.to_datetime(raw.timestamp)).tz_localize('Australia/Darwin')
    frame=pd.concat([hourly(pd.Series(pd.to_numeric(raw[c],errors='coerce').to_numpy()/(23.4 if name=='power' else 1),index=ts),12).rename(name)
                     for c,name in columns.items()],axis=1)
    frame=clean_frame(frame[(frame.index.year>=2021)&(frame.index.year<=2024)],'DKASC')
    dest=PRIVATE/'DKASC_extended'; dest.mkdir(parents=True,exist_ok=True)
    summaries=[]; coverage=[]
    for h in [2,6]:
        d=windows(frame,h,0)
        target=pd.to_datetime(d['target_ns']); mask=target.year==2024
        original=load(OLD/'DKASC'/f'h{h}.npz',features=True)
        prior=(target.year>=2021)&(target.year<=2023)
        for key in ['target_ns','issue_ns','y','references','x','future']:
            assert np.allclose(d[key][prior],original[key],rtol=0,atol=2e-6),(h,key)
        np.savez_compressed(dest/f'h{h}.npz',**d)
        summaries.append(dict(horizon=h,year=2024,all_windows=int(mask.sum()),daylight_windows=int((mask&d['daylight']).sum()),
            first_target=str(target[mask].min()),last_target=str(target[mask].max()),sha256=sha(dest/f'h{h}.npz')))
        for month in range(1,13):
            m=mask & (target.month==month) & d['daylight']
            calendar=frame[(frame.index.year==2024)&(frame.index.month==month)]
            den=int((calendar.clear_ghi>20).sum())
            coverage.append(dict(year=2024,horizon=h,month=month,n=int(m.sum()),daylight_calendar_hours=den,
                retention=float(m.sum()/den) if den else None,distinct_days=int(target[m].normalize().nunique())))
    pd.DataFrame(coverage).to_csv(OUT/'temporal_coverage.csv',index=False)
    save_json(OUT/'temporal_preparation.json',dict(status='passed',protocol_sha256=sha(CONFIG),source_sha256=sha(source),
        v28_preparation_reproduced=True,populations=summaries,source_previously_quality_audited=True,untouched_holdout=False))
    log('temporal_preparation_complete',populations=summaries)


def fit_regressor(x,y,mask,seed,iterations):
    import lightgbm as lgb
    p=PROTOCOL['lightgbm']
    model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=iterations,learning_rate=p['learning_rate'],
        num_leaves=p['num_leaves'],min_child_samples=p['min_child_samples'],colsample_bytree=p['feature_fraction'],
        random_state=seed,n_jobs=PROTOCOL['compute_threads'],verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x[mask],y[mask])
    return model


def prediction_policies(bed,h,year,d):
    import lightgbm as lgb
    x=np.concatenate([d['x'].reshape(len(d['y']),-1),d['future']],axis=1)
    target=pd.to_datetime(d['target_ns']); issue=pd.to_datetime(d['issue_ns'])
    test=target.year==year; systems=np.unique(d['system'][test])
    all_predictions={p:[] for p in PLAN['point_update']['policies']}
    for seed in PROTOCOL['seeds']:
        original_key=f'{bed}_h{h}_lightgbm_s{seed}_point_full'
        original=json.loads((PARENT/'runs'/(original_key+'.json')).read_text())
        iterations=int(original['settings']['selected_iterations'][0])
        frozen_model=OLD/'runs'/original_key/'model_None.txt'
        key=f'{bed}_{year}_h{h}_s{seed}'
        dest=PRIVATE/'update_runs'/key; dest.mkdir(parents=True,exist_ok=True)
        record=OUT/'update_runs'/(key+'.json')
        data_file=(PRIVATE/'DKASC_extended'/f'h{h}.npz') if year==2024 else OLD/bed/f'h{h}.npz'
        signature=dict(protocol=sha(CONFIG),code=sha(Path(__file__)),data=sha(data_file),parent_model=sha(frozen_model))
        if record.exists():
            previous=json.loads(record.read_text())
            assert previous['signature']==signature,'Changed experiment requires a new output revision'
            pp=load(dest/'predictions.npz')
            for policy in all_predictions: all_predictions[policy].append(pp[policy])
            log('update_resume',key=key)
            continue
        start=time.monotonic(); log('update_start',key=key,selected_trees=iterations)
        frozen=np.full(len(target),np.nan,dtype=np.float32)
        booster=lgb.Booster(model_file=str(frozen_model))
        frozen[test]=np.clip(booster.predict(x[test],num_threads=4),0,1.2)
        if year==2023:
            old_pred=load(OLD/'runs'/original_key/'prediction.npz')['prediction'][:,0]
            assert np.allclose(frozen[test],old_pred[test],atol=1e-6)
        annual=np.full_like(frozen,np.nan); monthly=np.full_like(frozen,np.nan)
        cutoff=pd.Timestamp(year=year,month=1,day=1)
        fit=(target.year==year-1) & (target<cutoff)
        assert all(int((fit & (d['system']==s)).sum())>=100 for s in systems)
        annual_model=fit_regressor(x,d['y'],fit,seed,iterations)
        annual_model.booster_.save_model(str(dest/'annual.txt'))
        annual[test]=np.clip(annual_model.predict(x[test]),0,1.2)
        records=[dict(policy='annual_previous_year',cutoff=str(cutoff),train_n=int(fit.sum()),
                      train_max=str(target[fit].max()),min_issue=str(issue[test].min()),fallback=False)]
        last_model=annual_model
        for month in range(1,13):
            cutoff=pd.Timestamp(year=year,month=month,day=1)
            end=cutoff+pd.offsets.MonthBegin(1)
            predict=test & (issue>=cutoff)&(issue<end)
            if not predict.any(): continue
            fit=(target>=cutoff-pd.Timedelta(days=365)) & (target<cutoff)
            enough=all(int((fit & (d['system']==s)).sum())>=100 for s in systems)
            if enough:
                last_model=fit_regressor(x,d['y'],fit,seed,iterations)
                last_model.booster_.save_model(str(dest/f'month_{month:02}.txt'))
            assert target[fit].max()<issue[predict].min()
            monthly[predict]=np.clip(last_model.predict(x[predict]),0,1.2)
            records.append(dict(policy='monthly_trailing_365_days',cutoff=str(cutoff),train_n=int(fit.sum()),
                train_max=str(target[fit].max()),min_issue=str(issue[predict].min()),predictions=int(predict.sum()),fallback=not enough))
        pp={'frozen_2021':frozen,'annual_previous_year':annual,'monthly_trailing_365_days':monthly}
        assert all(np.isfinite(p[test]).all() for p in pp.values())
        np.savez_compressed(dest/'predictions.npz',**pp)
        save_json(record,dict(status='complete',key=key,signature=signature,iterations=iterations,records=records,
            prediction_sha256=sha(dest/'predictions.npz'),runtime_seconds=time.monotonic()-start))
        for policy in all_predictions: all_predictions[policy].append(pp[policy])
        log('update_complete',key=key,seconds=round(time.monotonic()-start,1))
    return all_predictions


def updating(year=2023):
    beds=['DKASC','HKUST'] if year==2023 else ['DKASC']
    metrics=[]; inference=[]; monthly=[]
    if year==2024: extended_data()
    for bed in beds:
        for horizon in [2,6]:
            d=load((OLD/bed if year==2023 else PRIVATE/'DKASC_extended')/f'h{horizon}.npz',features=True)
            times=pd.to_datetime(d['target_ns']); test=times.year==year; daylight=test & d['daylight']
            if not daylight.any():
                log('no_eligible_temporal_targets',testbed=bed,year=year,horizon=horizon)
                continue
            pred=prediction_policies(bed,horizon,year,d)
            losses={name:np.mean([np.abs(p-d['y']) for p in pp],axis=0) for name,pp in pred.items()}
            for j,name in enumerate(PROTOCOL['references']): losses[name]=np.abs(d['references'][:,j]-d['y'])
            for name,loss in losses.items():
                for pop,mask in [('all',test),('daylight',daylight)]:
                    metrics.append(dict(year=year,testbed=bed,horizon=horizon,model='lightgbm' if name in pred else 'reference',policy=name,
                        population=pop,n=int(mask.sum()),mae=balanced(loss[mask],d['system'][mask]),
                        skill_vs_previous_day=1-balanced(loss[mask],d['system'][mask])/balanced(losses['previous_day'][mask],d['system'][mask])))
                for month in range(1,13):
                    mask=daylight & (times.month==month)
                    if mask.any(): monthly.append(dict(year=year,testbed=bed,horizon=horizon,policy=name,month=month,n=int(mask.sum()),
                        mae=balanced(loss[mask],d['system'][mask]),skill_vs_previous_day=1-balanced(loss[mask],d['system'][mask])/balanced(losses['previous_day'][mask],d['system'][mask])))
            for method in pred:
                for ref in ['previous_day','frozen_2021']:
                    if method==ref: continue
                    inference.append(paired(losses[method],losses[ref],d,daylight,analysis='point_updating',year=year,
                        testbed=bed,horizon=horizon,model='lightgbm',method=method,reference=ref,population='daylight',metric='mae'))
            log('point_population_complete',testbed=bed,horizon=horizon,year=year,daylight_n=int(daylight.sum()))
    pd.DataFrame(metrics).to_csv(OUT/f'updating_metrics_{year}.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/f'updating_monthly_{year}.csv',index=False)
    if inference:
        table=holm_adjust(pd.DataFrame(inference),['year','testbed','analysis','metric'])
        table.to_csv(OUT/f'updating_inference_{year}.csv',index=False)
    log('updating_year_complete',year=year)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['calibration','updating','temporal','all'],default='all')
    a=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    log('stage_started',stage=a.stage,protocol_sha256=sha(CONFIG),script_sha256=sha(Path(__file__)),
        versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','lightgbm','scipy','pvlib']})
    if a.stage in ['calibration','all']: calibration()
    if a.stage in ['updating','all']: updating(2023)
    if a.stage in ['temporal','all']: updating(2024)
    log('stage_complete',stage=a.stage)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        log('failed',error=repr(exc))
        raise
