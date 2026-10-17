"""U01: frozen/annual/monthly update policies at PVDAQ9068 for 2023 (primary) and 2024 (confirmation).

LightGBM logic follows publication_revision_v29.prediction_policies; neural logic follows run_p03_updates.fit_and_predict
(v28 architecture, inherited development-selected epochs, deterministic cold start). Checkpointed; resumable.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
REL = WS / 'WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting'
os.environ['PV_FORECAST_PROTOCOL_PATH'] = str(HERE.parent / 'protocol/protocol_v43_pvdaq9068.json')
sys.path.insert(0, str(REL / 'scripts'))
from corrected_study import PROTOCOL, PUBLIC, sha  # noqa: E402

PRIV = WS / 'tmp/paper1_v43_private/prepared'
RUNS = WS / 'tmp/paper1_v43_private/update_runs'
LOG = HERE.parent / 'logs/u01_events.jsonl'
BED = 'PVDAQ9068'
SEEDS = PROTOCOL['seeds']
POL = ('frozen_2021', 'annual_previous_year', 'monthly_trailing_365_days')


def log(event, **kw):
    row = dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(), event=event, **kw)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print(json.dumps(row, default=str), flush=True)


def load(h):
    with np.load(PRIV / BED / f'h{h}.npz') as z:
        return {k: z[k] for k in z.files}


def lgb_model(x, y, mask, seed, iters):
    import lightgbm as lgb
    p = PROTOCOL['lightgbm']
    m = lgb.LGBMRegressor(objective='regression_l1', n_estimators=iters, learning_rate=p['learning_rate'], num_leaves=p['num_leaves'],
                          min_child_samples=p['min_child_samples'], colsample_bytree=p['feature_fraction'], random_state=seed,
                          n_jobs=PROTOCOL['compute_threads'], verbosity=-1, deterministic=True, force_col_wise=True)
    m.fit(x[mask], y[mask])
    return m


def neural_fit_predict(d, fit, pred_mask, name, seed, epochs):
    import torch
    from torch import nn
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    x = d['x'].astype('float32'); fu = d['future'].astype('float32'); y = d['y'].astype('float32')
    mean = x[fit].mean(axis=(0, 1)); sc = x[fit].std(axis=(0, 1)); sc[sc < 1e-6] = 1
    fm = fu[fit].mean(axis=0); fs = fu[fit].std(axis=0); fs[fs < 1e-6] = 1

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); w = 32; self.n = name
            if name == 'gru':
                self.enc = nn.GRU(5, w, num_layers=1, batch_first=True)
            else:
                self.proj = nn.Linear(5, w)
                pos = torch.zeros(24, w); v = torch.arange(24).unsqueeze(1) * torch.exp(torch.arange(0, w, 2) * (-np.log(10000.) / w))
                pos[:, 0::2] = torch.sin(v); pos[:, 1::2] = torch.cos(v); self.register_buffer('pos', pos)
                self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, w * 2, 0.1, batch_first=True), 1, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(w + 5, w), nn.ReLU(), nn.Dropout(0.1), nn.Linear(w, 1))

        def forward(self, a, b):
            e = self.enc(a)[0][:, -1] if self.n == 'gru' else self.enc(self.proj(a) + self.pos)[:, -1]
            return self.head(torch.cat([e, b], 1))
    tx = torch.tensor((x[fit] - mean) / sc); tf = torch.tensor((fu[fit] - fm) / fs); ty = torch.tensor(y[fit, None])
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(seed); idx = np.arange(len(tx))
    for _ in range(epochs):
        net.train(); order = rng.permutation(idx)
        for s in range(0, len(order), 1024):
            b = order[s:s + 1024]; opt.zero_grad(set_to_none=True)
            loss = (net(tx[b], tf[b]) - ty[b]).abs().mean(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    out = np.full(len(y), np.nan, dtype='float32'); net.eval()
    px = torch.tensor((x[pred_mask] - mean) / sc); pf = torch.tensor((fu[pred_mask] - fm) / fs)
    with torch.no_grad():
        out[pred_mask] = np.clip(np.concatenate([net(px[s:s + 1024], pf[s:s + 1024]).numpy().reshape(-1) for s in range(0, len(px), 1024)]), 0, 1.2)
    return out


def run(model, h, year, seed):
    key = f'{BED}_{year}_h{h}_{model}_s{seed}'
    dest = RUNS / key; done = dest / 'predictions.npz'
    if done.exists():
        return
    dest.mkdir(parents=True, exist_ok=True); t0 = time.time()
    d = load(h); target = pd.to_datetime(d['target_ns']); issue = pd.to_datetime(d['issue_ns'])
    test = target.year == year
    parent = json.loads((PUBLIC / 'runs' / f'{BED}_h{h}_{model}_s{seed}_point_full.json').read_text())
    frozen = np.load(WS / f'tmp/paper1_v43_private/prepared/runs/{BED}_h{h}_{model}_s{seed}_point_full/prediction.npz')['prediction'][:, 0].astype('float32')
    if model == 'lightgbm':
        iters = int(parent['settings']['selected_iterations'][0])
        x = np.concatenate([d['x'].reshape(len(d['y']), -1), d['future']], axis=1)
        fitpred = lambda fit, pm: np.where(pm, np.clip(lgb_model(x, d['y'], fit, seed, iters).predict(x), 0, 1.2), np.nan)
    else:
        ep = int(parent['settings']['selected_epochs'])
        fitpred = lambda fit, pm: neural_fit_predict(d, fit, pm, model, seed, ep)
    annual_fit = (target.year == year - 1) & (target < pd.Timestamp(year=year, month=1, day=1))
    assert annual_fit.sum() >= 100
    annual = fitpred(annual_fit, test)
    monthly = np.full(len(target), np.nan, dtype='float32'); recs = []; last = annual
    for m in range(1, 13):
        cut = pd.Timestamp(year=year, month=m, day=1); end = cut + pd.offsets.MonthBegin(1)
        pm = test & (issue >= cut) & (issue < end)
        if not pm.any():
            continue
        fit = (target >= cut - pd.Timedelta(days=365)) & (target < cut)
        if fit.sum() >= 100:
            assert target[fit].max() < issue[pm].min()
            last = fitpred(fit, pm)
            monthly[pm] = last[pm]; recs.append(dict(month=m, fit_n=int(fit.sum()), fallback=False))
        else:
            monthly[pm] = last[pm]; recs.append(dict(month=m, fit_n=int(fit.sum()), fallback=True))
    pp = dict(frozen_2021=frozen, annual_previous_year=annual, monthly_trailing_365_days=monthly)
    assert all(np.isfinite(v[test]).all() for v in pp.values()), key
    np.savez_compressed(done, **pp)
    (dest / 'record.json').write_text(json.dumps(dict(key=key, months=recs, seconds=time.time() - t0, parent_prediction_sha256=parent['prediction_sha256']), indent=1))
    log('update_done', key=key, seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    for year in (2023, 2024):
        for h in (6, 2, 4, 12):
            for seed in SEEDS:
                run('lightgbm', h, year, seed)
        for model in ('gru', 'transformer'):
            for seed in SEEDS:
                run(model, 6, year, seed)
    log('u01_complete')
