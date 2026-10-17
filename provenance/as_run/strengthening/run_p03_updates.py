"""Sequential fixed/annual/monthly neural update evaluation for P03.

This uses the v28 architecture and inherited development-selected epoch counts,
but fits each updated model from a deterministic cold start in the new private
namespace. It never writes to the release or v29 parent roots.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\master in china\s2\research\solar energy forecasting")
RELEASE = ROOT / "WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting"
EXEC = ROOT / "tmp/paper1_strengthening_20260914"
PARENT = ROOT / "tmp/corrected_v28_array1b"
RUNROOT = EXEC / "private_data/p03_updates"
OUT = EXEC / "aggregates/p03_updates"
LOG = EXEC / "logs/execution.jsonl"
RUNROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["PV_FORECAST_PROTOCOL_PATH"] = str(RELEASE / "configs/corrected_study_v28_array1b.json")
sys.path.insert(0, str(RELEASE / "scripts"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

SEEDS = [11, 23, 37, 51, 73]
MODELS = ["gru", "transformer"]
POLICIES = ["annual_previous_year", "monthly_trailing_365_days"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(event: str, **details) -> None:
    row = {"schema_version": "1.0", "event_id": f"{time.time_ns()}-{event}", "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(), "experiment_id": "PAPER1-STRENGTHEN-20260914", "task_id": "P03", "event": event, "pid": os.getpid(), **details}
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def load(bed: str, horizon: int) -> dict[str, np.ndarray]:
    with np.load(PARENT / bed / f"h{horizon}.npz") as z:
        return {key: z[key] for key in z.files}


def inherited_epochs(bed: str, horizon: int, model: str, seed: int) -> int:
    path = RELEASE / "results/corrected_v28_array1b/runs" / f"{bed}_h{horizon}_{model}_s{seed}_point_full.json"
    record = json.loads(path.read_text())
    return int(record["settings"]["selected_epochs"])


class ForecastNet(nn.Module):
    def __init__(self, model_name: str, input_channels: int = 5, future_dim: int = 5):
        super().__init__()
        width = 32
        self.model_name = model_name
        if model_name == "gru":
            self.encoder = nn.GRU(input_channels, width, num_layers=1, batch_first=True)
        else:
            self.projection = nn.Linear(input_channels, width)
            position = torch.zeros(24, width)
            values = torch.arange(24).unsqueeze(1) * torch.exp(torch.arange(0, width, 2) * (-np.log(10000.0) / width))
            position[:, 0::2] = torch.sin(values)
            position[:, 1::2] = torch.cos(values)
            self.register_buffer("position", position)
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, 0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(width + future_dim, width), nn.ReLU(), nn.Dropout(0.1), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if self.model_name == "gru":
            encoded = self.encoder(x)[0][:, -1]
        else:
            encoded = self.encoder(self.projection(x) + self.position)[:, -1]
        return self.head(torch.cat([encoded, future], dim=1))


def fit_and_predict(data: dict[str, np.ndarray], fit_mask: np.ndarray, predict_mask: np.ndarray, model_name: str, seed: int, epochs: int, run_dir: Path) -> tuple[np.ndarray, dict]:
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    x = data["x"].astype("float32")
    future = data["future"].astype("float32")
    y = data["y"].astype("float32")
    mean = x[fit_mask].mean(axis=(0, 1)); scale = x[fit_mask].std(axis=(0, 1)); scale[scale < 1e-6] = 1.0
    fmean = future[fit_mask].mean(axis=0); fscale = future[fit_mask].std(axis=0); fscale[fscale < 1e-6] = 1.0
    tx = torch.tensor((x[fit_mask] - mean) / scale)
    tf = torch.tensor((future[fit_mask] - fmean) / fscale)
    ty = torch.tensor(y[fit_mask, None])
    model = ForecastNet(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed)
    batch_size = 1024
    losses = []
    indexes = np.arange(len(tx))
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(indexes); total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tx[idx], tf[idx]) - ty[idx]).abs().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * len(idx)
        mean_loss = total / len(order); losses.append(mean_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            emit("update_epoch", run_id=run_dir.name, stage="refit", epoch=epoch, epochs=epochs, training_loss=mean_loss, fit_n=int(fit_mask.sum()))
    prediction = np.full(len(y), np.nan, dtype="float32")
    testx = torch.tensor((x[predict_mask] - mean) / scale)
    testf = torch.tensor((future[predict_mask] - fmean) / fscale)
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(testx), batch_size):
            out.append(model(testx[start:start + batch_size], testf[start:start + batch_size]).numpy().reshape(-1))
    prediction[predict_mask] = np.clip(np.concatenate(out), 0.0, 1.2)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "normalization.json").write_text(json.dumps({"mean": mean.tolist(), "scale": scale.tolist(), "future_mean": fmean.tolist(), "future_scale": fscale.tolist()}), encoding="utf-8")
    (run_dir / "training.json").write_text(json.dumps({"epochs": epochs, "losses": losses, "fit_n": int(fit_mask.sum()), "prediction_n": int(predict_mask.sum()), "model": model_name, "seed": seed}, indent=2), encoding="utf-8")
    return prediction, {"epochs": epochs, "fit_n": int(fit_mask.sum()), "prediction_n": int(predict_mask.sum()), "last_loss": losses[-1], "model_sha256": sha(run_dir / "model.pt")}


def bundle(bed: str, horizon: int, model: str, seed: int, year: int) -> None:
    data = load(bed, horizon)
    target = pd.to_datetime(data["target_ns"])
    issue = pd.to_datetime(data["issue_ns"])
    test = target.year == year
    key = f"{bed}_{year}_h{horizon}_{model}_s{seed}"
    out_dir = RUNROOT / key
    record_path = OUT / f"{key}.json"
    epochs = inherited_epochs(bed, horizon, model, seed)
    predictions = {}
    parent_pred = PARENT / "runs" / f"{bed}_h{horizon}_{model}_s{seed}_point_full" / "prediction.npz"
    if parent_pred.exists():
        with np.load(parent_pred) as z:
            predictions["frozen_2021"] = z["prediction"][:, 0]
    else:
        raise FileNotFoundError(parent_pred)
    fit = (target.year == year - 1) & (target < pd.Timestamp(year=year, month=1, day=1))
    annual_dir = out_dir / "annual"
    annual_pred_path = annual_dir / "prediction.npy"
    if annual_pred_path.exists():
        predictions["annual_previous_year"] = np.load(annual_pred_path)
        emit("run_reused", run_id=key, policy="annual_previous_year", reason="verified local checkpoint")
    else:
        emit("run_started", run_id=key, policy="annual_previous_year", model=model, bed=bed, horizon=horizon, seed=seed, fit_n=int(fit.sum()))
        predictions["annual_previous_year"], details = fit_and_predict(data, fit, test, model, seed, epochs, annual_dir)
        np.save(annual_pred_path, predictions["annual_previous_year"])
        emit("run_completed", run_id=key, policy="annual_previous_year", details=details)
    monthly = np.full(len(target), np.nan, dtype="float32")
    monthly_records = []
    previous_model_pred = predictions["annual_previous_year"]
    for month in range(1, 13):
        cutoff = pd.Timestamp(year=year, month=month, day=1)
        end = cutoff + pd.offsets.MonthBegin(1)
        predict = test & (issue >= cutoff) & (issue < end)
        if not predict.any():
            monthly_records.append({"month": month, "status": "no_prediction_rows"}); continue
        fit = (target >= cutoff - pd.Timedelta(days=365)) & (target < cutoff)
        if any(int((fit & (data["system"] == s)).sum()) < 100 for s in np.unique(data["system"][test])):
            monthly[predict] = previous_model_pred[predict]
            monthly_records.append({"month": month, "status": "fallback_annual", "fit_n": int(fit.sum()), "prediction_n": int(predict.sum()), "cutoff": str(cutoff)})
            emit("fallback_used", run_id=key, policy="monthly_trailing_365_days", month=month, reason="system support below 100")
            continue
        month_dir = out_dir / f"month_{month:02d}"
        pred_path = month_dir / "prediction.npy"
        if pred_path.exists():
            pred = np.load(pred_path)
            monthly[predict] = pred[predict]
            monthly_records.append({"month": month, "status": "reused", "fit_n": int(fit.sum()), "prediction_n": int(predict.sum()), "cutoff": str(cutoff)})
            continue
        assert target[fit].max() < issue[predict].min()
        emit("run_started", run_id=key, policy="monthly_trailing_365_days", month=month, fit_n=int(fit.sum()), prediction_n=int(predict.sum()), cutoff=str(cutoff))
        pred, details = fit_and_predict(data, fit, predict, model, seed, epochs, month_dir)
        np.save(pred_path, pred)
        monthly[predict] = pred[predict]
        monthly_records.append({"month": month, "status": "complete", "fit_n": int(fit.sum()), "prediction_n": int(predict.sum()), "cutoff": str(cutoff), "train_target_max": str(target[fit].max()), **details})
        emit("run_completed", run_id=key, policy="monthly_trailing_365_days", month=month, details=details)
    predictions["monthly_trailing_365_days"] = monthly
    np.savez_compressed(out_dir / "predictions.npz", **predictions)
    write = {"status": "complete", "run_id": key, "bed": bed, "year": year, "horizon": horizon, "model": model, "seed": seed, "selected_epochs": epochs, "parent_prediction_sha256": sha(parent_pred), "data_sha256": sha(PARENT / bed / f"h{horizon}.npz"), "monthly": monthly_records, "prediction_sha256": sha(out_dir / "predictions.npz")}
    record_path.write_text(json.dumps(write, indent=2, default=str), encoding="utf-8")


def summarize(beds: list[str], horizons: list[int], years: list[int]) -> None:
    rows = []
    for bed in beds:
        for horizon in horizons:
            data = load(bed, horizon)
            target = pd.to_datetime(data["target_ns"])
            for year in years:
                if year != 2023:
                    continue
                test = target.year == year; daylight = test & data["daylight"]
                for model in MODELS:
                    pred_sets = {policy: [] for policy in ["frozen_2021", "annual_previous_year", "monthly_trailing_365_days"]}
                    for seed in SEEDS:
                        with np.load(RUNROOT / f"{bed}_{year}_h{horizon}_{model}_s{seed}" / "predictions.npz") as z:
                            for policy in pred_sets: pred_sets[policy].append(z[policy])
                    for policy, arrays in pred_sets.items():
                        pred = np.mean(np.stack(arrays), axis=0)
                        for population, mask in [("all", test), ("daylight", daylight)]:
                            loss = np.abs(data["y"] - pred)
                            base = np.abs(data["y"] - data["references"][:, 1])
                            def bal(v):
                                return float(np.mean([
                                    np.mean(v[mask & (data["system"] == s)])
                                    for s in np.unique(data["system"][mask])
                                    if np.any(mask & (data["system"] == s))
                                ]))
                            rows.append({"testbed": bed, "year": year, "horizon": horizon, "model": model, "policy": policy, "population": population, "n": int(mask.sum()), "mae": bal(loss), "previous_day_mae": bal(base), "skill_vs_previous_day": 1 - bal(loss) / bal(base)})
    frame = pd.DataFrame(rows)
    # Preserve every requested horizon.  The original single-file write used
    # horizons[0], which could silently discard the h=12 rows when h=4 and h=12
    # were executed together.  Keep per-horizon files for compatibility and a
    # combined file for the multi-horizon P03C aggregation.
    for horizon in sorted(set(horizons)):
        frame[frame["horizon"] == horizon].to_csv(
            OUT / f"update_metrics_h{horizon}.csv", index=False
        )
    if len(set(horizons)) > 1:
        frame.to_csv(OUT / "update_metrics_h4_h12_neural.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[6])
    parser.add_argument("--beds", nargs="+", default=["DKASC", "HKUST"])
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    emit("task_started", task_id="P03", stage="neural_updates", horizons=args.horizons, beds=args.beds, year=args.year)
    if args.pilot:
        bundle(args.beds[0], args.horizons[0], "gru", 11, args.year)
        emit("task_complete", task_id="P03", stage="pilot", message="one GRU annual/monthly bundle completed")
        return
    for bed in args.beds:
        for horizon in args.horizons:
            for model in MODELS:
                for seed in SEEDS:
                    bundle(bed, horizon, model, seed, args.year)
    summarize(args.beds, args.horizons, [args.year])
    emit("task_complete", task_id="P03", stage="neural_updates", horizons=args.horizons, beds=args.beds, year=args.year)


if __name__ == "__main__":
    main()
