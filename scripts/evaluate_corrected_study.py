"""Paired evaluation, synchronized block inference and calibration for revision 27."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from corrected_study import ROOT, PROTOCOL, PUBLIC, QUANTILES, sha, save_json, conformal_correction, balanced_mae


def balanced(values,system):
    return float(np.mean([np.mean(values[system==s]) for s in np.unique(system)]))


def holm_adjust(table,families):
    table=table.copy(); table['p_holm']=np.nan
    for _,g in table.groupby(families):
        order=g.p_raw.sort_values().index
        adjusted=np.maximum.accumulate(table.loc[order,'p_raw'].to_numpy()*np.arange(len(order),0,-1))
        table.loc[order,'p_holm']=np.minimum(adjusted,1)
    return table


def interval_score(y,lo,hi,level):
    return hi-lo+2/(1-level)*(np.maximum(lo-y,0)+np.maximum(y-hi,0))


def sparse_quantile_pit(y,qmat,seed=42):
    """Interpolate within the quantile grid; clamp unidentified tails.

    Only equal-valued quantiles are randomized. This is a sparse-grid diagnostic,
    not an identified full predictive CDF or an iid calibration test.
    """
    y=np.asarray(y); qmat=np.asarray(qmat)
    if np.any(np.diff(qmat,axis=1)<0): raise ValueError('Quantiles must be sorted')
    out=np.empty(len(y)); rng=np.random.default_rng(seed)
    for i,(value,q) in enumerate(zip(y,qmat)):
        equal=np.flatnonzero(q==value)
        if len(equal)>1:
            out[i]=rng.uniform(QUANTILES[equal[0]],QUANTILES[equal[-1]])
        elif len(equal)==1: out[i]=QUANTILES[equal[0]]
        else: out[i]=np.interp(value,q,QUANTILES)
    return out


def block_interval(y_loss,ref_loss,system,target_ns,days=7,resamples=2000):
    """Common blocks preserve within-day, between-day and simultaneous site errors."""
    dates=pd.to_datetime(target_ns).normalize()
    calendar=pd.date_range(dates.min(),dates.max(),freq='D')
    if len(calendar)<days: raise ValueError('Block exceeds available calendar span')
    day=calendar.get_indexer(dates)
    sites=np.unique(system)
    site=np.searchsorted(sites,system)
    shape=(len(calendar),len(sites))
    n=np.zeros(shape); e=np.zeros(shape); b=np.zeros(shape)
    np.add.at(n,(day,site),1); np.add.at(e,(day,site),y_loss); np.add.at(b,(day,site),ref_loss)
    rng=np.random.default_rng(PROTOCOL['inference']['seed'])
    draws=[]; deltas=[]
    for _ in range(resamples):
        starts=rng.integers(0,max(1,len(calendar)-days+1),size=math.ceil(len(calendar)/days))
        idx=(starts[:,None]+np.arange(days)[None,:]).ravel()[:len(calendar)]
        nn=n[idx].sum(axis=0)
        if np.any(nn==0): continue
        ee=(e[idx].sum(axis=0)/nn).mean(); bb=(b[idx].sum(axis=0)/nn).mean()
        draws.append(1-ee/bb); deltas.append(bb-ee)
    estimate=1-balanced(y_loss,system)/balanced(ref_loss,system)
    ci=np.quantile(draws,[.025,.975])
    # Fixed site-count weights reproduce the system-balanced difference exactly.
    # Retain absent calendar days as zero contributions rather than compressing
    # gaps and treating non-adjacent dates as adjacent HAC lags.
    daily=len(calendar)*np.mean((b-e)/n.sum(axis=0)[None,:],axis=1)
    z=daily-daily.mean(); lag=min(days-1,len(z)-1)
    var=np.dot(z,z)/len(z)
    for k in range(1,lag+1): var+=2*(1-k/(lag+1))*np.dot(z[k:],z[:-k])/len(z)
    se=math.sqrt(max(var,0)/len(z))
    dm=float(daily.mean()/se) if se>0 else 0.
    return dict(skill=estimate,ci_low=float(ci[0]),ci_high=float(ci[1]),
                loss_reduction=balanced(ref_loss,system)-balanced(y_loss,system),
                reduction_ci_low=float(np.quantile(deltas,.025)),
                reduction_ci_high=float(np.quantile(deltas,.975)),
                hac_stat=dm,p_raw=float(2*stats.norm.sf(abs(dm))),block_days=days,
                resamples=len(draws),calendar_days=len(calendar))


def point_tables(private_root):
    records=[]; inference=[]; sources=[]; matched=[]
    for bed in PROTOCOL['testbeds']:
        all_data={}
        common=None
        for horizon in PROTOCOL['horizons']:
            with np.load(private_root/bed/f'h{horizon}.npz') as z: d={k:z[k] for k in z.files if k not in ['x','future']}
            test=pd.to_datetime(d['target_ns']).year==2023
            keys=set(zip(d['system'][test].tolist(),d['target_ns'][test].tolist()))
            common=keys if common is None else common&keys
            all_data[horizon]=d
        for horizon,d in all_data.items():
            test=pd.to_datetime(d['target_ns']).year==2023
            common_mask=np.array([(int(s),int(t)) in common for s,t in zip(d['system'],d['target_ns'])])
            predictions={name:[(0,d['references'][:,i])] for i,name in enumerate(PROTOCOL['references'])}
            for model in PROTOCOL['models']:
                predictions[model]=[]
                for seed in PROTOCOL['seeds']:
                    key=f'{bed}_h{horizon}_{model}_s{seed}_point_full'
                    p=private_root/'runs'/key/'prediction.npz'
                    if p.exists() and (PUBLIC/'runs'/(key+'.json')).exists():
                        with np.load(p) as z: pred=z['prediction'][:,0]
                        predictions[model].append((seed,pred)); sources.append(key)
            for name,group in predictions.items():
                for seed,pred in group:
                    for population,mask in [('all',test),('daylight',test & d['daylight']),('matched_targets',test & common_mask),('matched_daylight',test & common_mask & d['daylight'])]:
                        y=d['y'][mask]; s=d['system'][mask]; error=pred[mask]-y
                        reference=d['references'][mask,3]
                        row=dict(testbed=bed,horizon=horizon,model=name,seed=seed,population=population,n=int(mask.sum()),systems=len(np.unique(s)),
                                 mae=balanced(np.abs(error),s),rmse=math.sqrt(balanced(error**2,s)),bias=balanced(error,s),
                                 smart_mae=balanced_mae(y,reference,s))
                        row['skill_vs_smart']=1-row['mae']/row['smart_mae']
                        records.append(row)
                if name in PROTOCOL['models'] and len(group)==len(PROTOCOL['seeds']):
                    losses=np.mean([np.abs(pred-d['y']) for _,pred in group],axis=0)
                    for population,mask in [('all',test),('daylight',test & d['daylight']),('matched_daylight',test & common_mask & d['daylight'])]:
                        for days in ([1,7] if horizon==6 and population!='matched_daylight' else [7]):
                            result=block_interval(losses[mask],np.abs(d['references'][mask,3]-d['y'][mask]),d['system'][mask],d['target_ns'][mask],days)
                            inference.append(dict(testbed=bed,horizon=horizon,model=name,population=population,**result))
    pd.DataFrame(records).to_csv(PUBLIC/'point_metrics.csv',index=False)
    if inference:
        table=holm_adjust(pd.DataFrame(inference),['testbed','population','block_days'])
        table.to_csv(PUBLIC/'point_inference.csv',index=False)
    return records


def calibrate_intervals(y,qmat,systems,regime,cal_mask,test_mask,level,kappa=100):
    lo_idx,hi_idx=(1,5) if level==.8 else (0,6)
    scores=np.maximum.reduce([qmat[:,lo_idx]-y,y-qmat[:,hi_idx],np.zeros(len(y))])
    global_q=conformal_correction(scores[cal_mask],level)
    global_map=np.full(len(y),global_q)
    site_map=global_map.copy(); regime_map=global_map.copy(); partial=global_map.copy()
    for label in np.unique(regime):
        fit=cal_mask & (regime==label)
        if fit.sum()>=100: regime_map[regime==label]=conformal_correction(scores[fit],level)
    for s in np.unique(systems):
        fit=cal_mask & (systems==s); target=systems==s
        if fit.sum()>=100:
            site_q=conformal_correction(scores[fit],level)
            site_map[target]=site_q
            weight=fit.sum()/(fit.sum()+kappa)
            # Explicit heuristic shrinkage of corrections; no finite-sample guarantee.
            partial[target]=weight*site_q+(1-weight)*global_q
    maps={'raw':np.zeros(len(y)),'global':global_map,'regime':regime_map,'system':site_map,'partial':partial}
    results={name:(qmat[test_mask,lo_idx]-value[test_mask],qmat[test_mask,hi_idx]+value[test_mask]) for name,value in maps.items()}
    return results,{name:float(value[test_mask].mean()) for name,value in maps.items()}


def probabilistic_tables(private_root):
    intervals=[]; distribution=[]; tuning=[]; budgets=[]; reliability=[]; site_rows=[]; paired=[]
    for bed in PROTOCOL['testbeds']:
        with np.load(private_root/bed/'h6.npz') as z: d={k:z[k] for k in z.files if k not in ['x','future']}
        times=pd.to_datetime(d['target_ns']); train=times.year==2021; cal=times.year==2022; test=times.year==2023
        threshold=np.quantile(d['variation'][train & d['daylight']],[1/3,2/3])
        regime=np.where(~d['daylight'],'target_night',np.where(d['variation']<=threshold[0],'low_variability',np.where(d['variation']<=threshold[1],'medium_variability','high_variability')))
        models={'probabilistic_persistence':d['prob_reference']}
        for model in PROTOCOL['primary_quantile_models']:
            predictions=[]
            for seed in PROTOCOL['seeds']:
                key=f'{bed}_h6_{model}_s{seed}_quantile_full'
                p=private_root/'runs'/key/'prediction.npz'
                if p.exists() and (PUBLIC/'runs'/(key+'.json')).exists():
                    with np.load(p) as z: predictions.append(z['prediction'])
            if len(predictions)==len(PROTOCOL['seeds']): models[model]=np.mean(predictions,axis=0)
        for model,qmat in models.items():
            # Quantile-averaged seed ensemble; explicitly distinct from mean seed loss.
            error=d['y'][:,None]-qmat
            pinball=np.maximum(QUANTILES*error,(QUANTILES-1)*error)
            crps7=2*np.trapezoid(pinball,QUANTILES,axis=1)
            for population,mask in [('all',test),('daylight',test & d['daylight'])]:
                y=d['y'][mask]; q=qmat[mask]; s=d['system'][mask]
                pit=sparse_quantile_pit(y,q)
                for j,tau in enumerate(QUANTILES):
                    reliability.append(dict(testbed=bed,model=model,population=population,
                                            quantile=tau,observed_fraction=balanced(y<=q[:,j],s),
                                            n=len(y),systems=len(np.unique(s))))
                distribution.append(dict(testbed=bed,model=model,population=population,n=int(mask.sum()),
                                         crps7_truncated=balanced(crps7[mask],s),pinball=balanced(pinball[mask].mean(axis=1),s),
                                         pit_cvm_descriptive=float(stats.cramervonmises(pit,'uniform').statistic)))
            for level in [.8,.9]:
                first=cal & (times<pd.Timestamp('2022-07-01')); second=cal & ~first
                candidates=[]
                for kappa in PROTOCOL['calibration']['partial_pool_kappa_candidates']:
                    candidate,_=calibrate_intervals(d['y'],qmat,d['system'],regime,first,second,level,kappa)
                    lo,hi=candidate['partial']
                    loss=balanced(interval_score(d['y'][second],lo,hi,level),d['system'][second])
                    candidates.append((loss,kappa))
                    tuning.append(dict(testbed=bed,model=model,level=level,kappa=kappa,validation_interval_score=loss))
                _,chosen=min(candidates)
                calibrated,corrections=calibrate_intervals(d['y'],qmat,d['system'],regime,cal,test,level,chosen)
                for method,(lo,hi) in calibrated.items():
                    test_regime=regime[test]
                    for population,mask in [('all',np.ones(test.sum(),dtype=bool)),('daylight',d['daylight'][test])]+[(r,test_regime==r) for r in np.unique(test_regime)]:
                        yy=d['y'][test][mask]; ss=d['system'][test][mask]
                        if not len(yy): continue
                        hit=(yy>=lo[mask])&(yy<=hi[mask])
                        intervals.append(dict(testbed=bed,model=model,level=level,method=method,population=population,n=len(yy),
                                              coverage=balanced(hit,ss),width=balanced(hi[mask]-lo[mask],ss),
                                              interval_score=balanced(interval_score(yy,lo[mask],hi[mask],level),ss),selected_kappa=chosen))
                    for site in np.unique(d['system'][test]):
                        mask=d['system'][test]==site
                        yy=d['y'][test][mask]
                        site_rows.append(dict(testbed=bed,model=model,level=level,method=method,
                                              system=int(site),n=int(mask.sum()),
                                              coverage=float(np.mean((yy>=lo[mask])&(yy<=hi[mask]))),
                                              width=float(np.mean(hi[mask]-lo[mask])),
                                              interval_score=float(np.mean(interval_score(yy,lo[mask],hi[mask],level)))))
                yy=d['y'][test]; ss=d['system'][test]; tt=d['target_ns'][test]
                for method,reference in [('global','raw'),('regime','global'),('system','global'),('partial','global')]:
                    lo,hi=calibrated[method]; rlo,rhi=calibrated[reference]
                    score=interval_score(yy,lo,hi,level); refscore=interval_score(yy,rlo,rhi,level)
                    for population,mask in [('all',np.ones(len(yy),bool)),('daylight',d['daylight'][test])]:
                        result=block_interval(score[mask],refscore[mask],ss[mask],tt[mask])
                        paired.append(dict(testbed=bed,model=model,level=level,method=method,reference=reference,
                                           population=population,estimand='interval_score_reduction',**result))
                for days in [30,90,180,365]:
                    fit=cal & (times>=pd.Timestamp('2023-01-01')-pd.Timedelta(days=days))
                    calculated,_=calibrate_intervals(d['y'],qmat,d['system'],regime,fit,test,level,chosen)
                    for method in ['global','system','partial']:
                        lo,hi=calculated[method]; s=d['system'][test]
                        budgets.append(dict(testbed=bed,model=model,level=level,days=days,method=method,
                                            coverage=balanced((d['y'][test]>=lo)&(d['y'][test]<=hi),s),
                                            width=balanced(hi-lo,s),interval_score=balanced(interval_score(d['y'][test],lo,hi,level),s)))
    for name,rows in [('interval_metrics',intervals),('distribution_metrics',distribution),('calibration_tuning',tuning),('calibration_budget',budgets),('quantile_reliability',reliability),('system_interval_metrics',site_rows),('calibration_inference',paired)]:
        table=pd.DataFrame(rows)
        if name=='calibration_inference' and rows:
            table=holm_adjust(table,['testbed','population','block_days'])
        table.to_csv(PUBLIC/(name+'.csv'),index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--stage',choices=['point','probabilistic','all'],default='all')
    parser.add_argument('--allow-partial',action='store_true',help='Explicitly produce a nonfinal snapshot while runs are missing')
    a=parser.parse_args()
    gate=json.loads((PUBLIC/'scientific_verification.json').read_text())
    if gate['science_gate']!='complete' and not a.allow_partial:
        raise RuntimeError('Final evaluation requires all verified runs; use --allow-partial only for a labelled snapshot')
    if a.stage in ['point','all']: point_tables(a.private_root)
    if a.stage in ['probabilistic','all']: probabilistic_tables(a.private_root)
    save_json(PUBLIC/'evaluation_notes.json',dict(status='partial' if a.allow_partial else 'complete',stage=a.stage,
                 script_sha256=sha(Path(__file__)),verification_sha256=sha(PUBLIC/'scientific_verification.json'),
                 bootstrap='Synchronized non-circular consecutive calendar-day blocks across all systems; 1/7-day sensitivity',
                 test_history='Retrospective correction of examined 2023 data',crps='Trapezoidal integral over seven quantiles from .05 to .95; explicitly truncated, not exact CRPS',
                 pit='Piecewise linear interpolation between supplied quantiles; randomized only at repeated quantiles; unidentified tails clamped to .05/.95. CvM descriptive only, not a full-CDF test.',
                 calibration_budget='Retrospective correction-window sensitivity; partial-pooling kappa is fixed using the full 2022 tuning exercise, so this is not a strict total-label-budget experiment.',
                 partial_pool='Heuristic weighted average of site/global corrections, validation-selected kappa; no conditional coverage guarantee'))


if __name__=='__main__': main()
