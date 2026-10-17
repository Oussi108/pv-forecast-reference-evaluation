"""Resume the declared point/quantile experiment matrix for revision 27.

Model selection uses the last quarter of 2021. Final models are refit on all
eligible 2021 examples for the selected training duration. 2022 and 2023 labels
are never used in model fitting or stopping. Results and traces are checkpointed.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from corrected_study import ROOT, PUBLIC, PROTOCOL, PROTOCOL_PATH, QUANTILES, sha, save_json, private_path, balanced_mae


def split_masks(data):
    times=pd.to_datetime(data['target_ns'])
    return dict(select_train=times<pd.Timestamp('2021-10-01'),
                select_val=(times>=pd.Timestamp('2021-10-01'))&(times<pd.Timestamp('2022-01-01')),
                development=times.year==2021,calibration=times.year==2022,test=times.year==2023)


def inputs(data, variant='full'):
    channels=[0,1,2,3,4]
    future=[0,1,2,3,4]
    if variant=='without_power': channels.remove(0)
    if variant=='without_irradiance': channels.remove(1)
    if variant=='without_other_weather': channels=[0,1,4]
    if variant=='without_solar': channels.remove(4); future.remove(0)
    if variant=='without_calendar': future=[0]
    return data['x'][:,:,channels],data['future'][:,future]


def lgb_fit(data, masks, seed, quantile, variant, runroot):
    import lightgbm as lgb
    x,f=inputs(data,variant)
    x=np.concatenate([x.reshape(len(x),-1),f],axis=1)
    p=PROTOCOL['lightgbm']
    outputs=[]; traces=[]; iterations=[]
    levels=QUANTILES if quantile else [None]
    for level in levels:
        objective='regression_l1' if level is None else 'quantile'
        params=dict(objective=objective,n_estimators=p['n_estimators'],learning_rate=p['learning_rate'],
                    num_leaves=p['num_leaves'],min_child_samples=p['min_child_samples'],
                    colsample_bytree=p['feature_fraction'],random_state=seed,n_jobs=PROTOCOL['compute_threads'],
                    verbosity=-1,deterministic=True,force_col_wise=True)
        if level is not None: params['alpha']=float(level)
        search=lgb.LGBMRegressor(**params)
        history={}
        a,b=masks['select_train'],masks['select_val']
        search.fit(x[a],data['y'][a],eval_set=[(x[a],data['y'][a]),(x[b],data['y'][b])],
                   eval_names=['train','validation'],callbacks=[lgb.early_stopping(p['early_stopping_rounds'],verbose=False),lgb.record_evaluation(history)])
        duration=int(search.best_iteration_)
        iterations.append(duration)
        for split,values in history.items():
            for metric,losses in values.items():
                traces.extend(dict(stage='selection',quantile=level,split=split,epoch=i+1,loss=float(loss),metric=metric) for i,loss in enumerate(losses))
        params['n_estimators']=duration
        model=lgb.LGBMRegressor(**params)
        a=masks['development']
        model.fit(x[a],data['y'][a])
        model.booster_.save_model(str(runroot/f'model_{level}.txt'))
        outputs.append(model.predict(x, validate_features=False))
    prediction=np.column_stack(outputs).astype('float32')
    prediction=np.clip(prediction,0,1.2)
    if quantile: prediction.sort(axis=1)
    return prediction,dict(selected_iterations=iterations,objective=objective),traces


def neural_fit(data,masks,seed,model_name,quantile,variant,runroot):
    import torch
    from torch import nn
    torch.set_num_threads(PROTOCOL['compute_threads'])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    p=PROTOCOL['neural']
    x,f=inputs(data,variant)
    dim= len(QUANTILES) if quantile else 1

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            width=p['hidden']
            if model_name=='gru':
                self.encoder=nn.GRU(x.shape[2],width,num_layers=p['layers'],batch_first=True)
            else:
                self.projection=nn.Linear(x.shape[2],width)
                position=torch.zeros(24,width)
                values=torch.arange(24).unsqueeze(1)*torch.exp(torch.arange(0,width,2)*(-np.log(10000.)/width))
                position[:,0::2]=torch.sin(values); position[:,1::2]=torch.cos(values)
                self.register_buffer('position',position)
                layer=nn.TransformerEncoderLayer(width,p['attention_heads'],width*2,p['dropout'],batch_first=True)
                self.encoder=nn.TransformerEncoder(layer,p['layers'],enable_nested_tensor=False)
            self.head=nn.Sequential(nn.Linear(width+f.shape[1],width),nn.ReLU(),nn.Dropout(p['dropout']),nn.Linear(width,dim))

        def forward(self,x,future):
            if model_name=='gru': encoded=self.encoder(x)[0][:,-1]
            else: encoded=self.encoder(self.projection(x)+self.position)[:,-1]
            return self.head(torch.cat([encoded,future],dim=1))

    def tensor_inputs(mask):
        mean=x[mask].mean(axis=(0,1)); scale=x[mask].std(axis=(0,1)); scale[scale<1e-6]=1
        fm=f[mask].mean(axis=0); fs=f[mask].std(axis=0); fs[fs<1e-6]=1
        return torch.tensor((x-mean)/scale),torch.tensor((f-fm)/fs),dict(mean=mean.tolist(),scale=scale.tolist(),future_mean=fm.tolist(),future_scale=fs.tolist())

    y=torch.tensor(data['y'][:,None])
    q=torch.tensor(QUANTILES,dtype=torch.float32)[None,:]
    def loss_fn(pred,actual):
        error=actual-pred
        return torch.maximum(q*error,(q-1)*error).mean() if quantile else error.abs().mean()

    def evaluate(model,tx,tf,indices):
        total=0.
        model.eval()
        with torch.no_grad():
            for start in range(0,len(indices),p['batch_size']):
                idx=indices[start:start+p['batch_size']]
                total+=loss_fn(model(tx[idx],tf[idx]),y[idx]).item()*len(idx)
        return total/len(indices)

    def one_epoch(model,optimizer,tx,tf,indices,rng):
        model.train(); total=0.
        order=rng.permutation(indices)
        for start in range(0,len(order),p['batch_size']):
            idx=order[start:start+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(tx[idx],tf[idx]),y[idx]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); total+=loss.item()*len(idx)
        return total/len(order)

    tx,tf,normalization=tensor_inputs(masks['select_train'])
    model=Net(); optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    rng=np.random.default_rng(seed); traces=[]; best=float('inf'); best_epoch=0
    a,b=np.flatnonzero(masks['select_train']),np.flatnonzero(masks['select_val'])
    for epoch in range(1,p['max_epochs']+1):
        train_loss=one_epoch(model,optimizer,tx,tf,a,rng)
        val_loss=evaluate(model,tx,tf,b)
        save_json(runroot/'live_epoch.json',dict(stage='selection',epoch=epoch,train_loss=train_loss,validation_loss=val_loss))
        if epoch==1 or epoch%5==0: print('EPOCH',runroot.name,epoch,round(val_loss,6),flush=True)
        traces.extend([dict(stage='selection',split='train',epoch=epoch,loss=train_loss),dict(stage='selection',split='validation',epoch=epoch,loss=val_loss)])
        if val_loss<best-1e-6: best=val_loss; best_epoch=epoch
        if epoch-best_epoch>=p['patience']: break
    # Refit from a fresh, deterministic initialization on the whole development year.
    torch.manual_seed(seed)
    model=Net(); optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    tx,tf,normalization=tensor_inputs(masks['development'])
    rng=np.random.default_rng(seed); a=np.flatnonzero(masks['development'])
    for epoch in range(1,best_epoch+1):
        loss=one_epoch(model,optimizer,tx,tf,a,rng)
        save_json(runroot/'live_epoch.json',dict(stage='refit',epoch=epoch,selected_epochs=best_epoch,loss=loss))
        traces.append(dict(stage='refit',split='development',epoch=epoch,loss=loss))
    prediction=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(x),p['batch_size']):
            prediction.append(model(tx[start:start+p['batch_size']],tf[start:start+p['batch_size']]).numpy())
    prediction=np.clip(np.concatenate(prediction),0,1.2)
    if quantile: prediction.sort(axis=1)
    torch.save(model.state_dict(),runroot/'model.pt')
    save_json(runroot/'normalization.json',normalization)
    return prediction,dict(selected_epochs=best_epoch,selection_validation_loss=best,architecture=model_name,parameters=sum(v.numel() for v in model.parameters())),traces


def run_one(private_root,bed,horizon,model,seed,quantile=False,variant='full'):
    key=f'{bed}_h{horizon}_{model}_s{seed}_{"quantile" if quantile else "point"}_{variant}'
    runroot=private_root/'runs'/key
    signature={'protocol':sha(PROTOCOL_PATH),'data':sha(private_root/bed/f'h{horizon}.npz'),
               'training_code':sha(Path(__file__)),'preparation_code':sha(ROOT/'scripts/corrected_study.py')}
    done=PUBLIC/'runs'/(key+'.json')
    if done.exists():
        previous=json.loads(done.read_text())
        if previous.get('signature')==signature and (runroot/'prediction.npz').exists():
            print('RESUME_SKIP',key,flush=True); return
        raise RuntimeError(f'Existing run {key} has a different signature; use a new revision directory')
    runroot.mkdir(parents=True,exist_ok=True)
    start=time.time()
    save_json(PUBLIC/'progress.json',dict(status='running',current=key,started_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('START',key,flush=True)
    with np.load(private_root/bed/f'h{horizon}.npz') as archive: data={k:archive[k] for k in archive.files}
    masks=split_masks(data)
    if min(int(masks[k].sum()) for k in masks)<20: raise RuntimeError('Insufficient split population')
    if model=='lightgbm': pred,settings,trace=lgb_fit(data,masks,seed,quantile,variant,runroot)
    else: pred,settings,trace=neural_fit(data,masks,seed,model,quantile,variant,runroot)
    np.savez_compressed(runroot/'prediction.npz',prediction=pred)
    PUBLIC.joinpath('curves').mkdir(exist_ok=True)
    pd.DataFrame(trace).to_csv(PUBLIC/'curves'/(key+'.csv'),index=False)
    mask=masks['test']; point=pred[:,3] if quantile else pred[:,0]
    mae=balanced_mae(data['y'][mask],point[mask],data['system'][mask])
    report=dict(key=key,testbed=bed,horizon=horizon,model=model,seed=seed,quantile=quantile,variant=variant,
                settings=settings,signature=signature,runtime_seconds=time.time()-start,
                counts={k:int(v.sum()) for k,v in masks.items()},test_system_balanced_mae=mae,
                prediction_sha256=sha(runroot/'prediction.npz'),status='complete')
    save_json(done,report)
    print('DONE',key,'MAE',round(mae,6),'seconds',round(report['runtime_seconds']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--stage',choices=['point','quantile','ablation','all'],default='all')
    p.add_argument('--beds',nargs='+',default=['DKASC','HKUST'])
    p.add_argument('--models',nargs='+',default=PROTOCOL['models'])
    p.add_argument('--horizons',nargs='+',type=int,default=PROTOCOL['horizons'])
    p.add_argument('--seeds',nargs='+',type=int,default=PROTOCOL['seeds'])
    a=p.parse_args(); private_root=private_path(a.private_root)
    if a.stage in ['point','all']:
        for horizon in a.horizons:
            for model in a.models:
                for bed in a.beds:
                    for seed in a.seeds: run_one(private_root,bed,horizon,model,seed)
    if a.stage in ['quantile','all']:
        for model in PROTOCOL['primary_quantile_models']:
            for bed in a.beds:
                for seed in a.seeds: run_one(private_root,bed,6,model,seed,True)
    if a.stage in ['ablation','all']:
        for model in ['lightgbm','gru']:
            for bed in a.beds:
                for variant in ['without_power','without_irradiance','without_other_weather','without_solar','without_calendar']:
                    run_one(private_root,bed,6,model,11,variant=variant)
    save_json(PUBLIC/'progress.json',dict(status='requested_stage_complete',stage=a.stage,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        current=json.loads((PUBLIC/'progress.json').read_text()) if (PUBLIC/'progress.json').exists() else {}
        current = {k:v for k,v in current.items() if k not in ('status','error')}
        current.update(status='failed', error=repr(exc))
        save_json(PUBLIC/'progress.json',current)
        raise
