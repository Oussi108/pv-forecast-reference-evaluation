"""Isolated LightGBM update runner for the P03C h=4/h=12 extension.

The v29 LightGBM update grid covered h=2 and h=6 only.  P03C therefore
requires new h=4 and h=12 annual/monthly fits; this runner keeps them in a
separate namespace and uses the frozen Array 1B protocol, inherited tree
counts, cold-start refits and issue-time monthly assignment.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pv_paths import REPO, RESULTS, PREP_V28, PRIV_UPDATES  # noqa: E402
RELEASE = REPO
EXEC = PRIV_UPDATES
PARENT = PREP_V28
RUNROOT = EXEC / "private_data/p03_lightgbm_updates"
OUT = RESULTS / "updates_2023/p03_lightgbm_updates"
LOG = EXEC / "logs/execution.jsonl"
CONFIG = RELEASE / "configs/corrected_study_v28_array1b.json"
RUNROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["PV_FORECAST_PROTOCOL_PATH"] = str(CONFIG)

SEEDS = [11, 23, 37, 51, 73]
DEFAULT_MODELS = ["lightgbm"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(event: str, **details: object) -> None:
    row = {
        "schema_version": "1.0",
        "event_id": f"{time.time_ns()}-{event}",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "experiment_id": "PAPER1-STRENGTHEN-20260914",
        "task_id": "P03C",
        "event": event,
        "pid": os.getpid(),
        **details,
    }
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def load(bed: str, horizon: int) -> dict[str, np.ndarray]:
    with np.load(PARENT / bed / f"h{horizon}.npz") as values:
        return {key: values[key] for key in values.files}


def inherited_iterations(bed: str, horizon: int, seed: int) -> int:
    record = json.loads(
        (RELEASE / "results/corrected_v28_array1b/runs" /
         f"{bed}_h{horizon}_lightgbm_s{seed}_point_full.json").read_text()
    )
    return int(record["settings"]["selected_iterations"][0])


def fit_regressor(x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                  seed: int, iterations: int):
    import lightgbm as lgb

    protocol = json.loads(CONFIG.read_text())
    settings = protocol["lightgbm"]
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=iterations,
        learning_rate=settings["learning_rate"],
        num_leaves=settings["num_leaves"],
        min_child_samples=settings["min_child_samples"],
        colsample_bytree=settings["feature_fraction"],
        random_state=seed,
        n_jobs=protocol["compute_threads"],
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(x[mask], y[mask])
    return model


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def bundle(bed: str, horizon: int, seed: int, year: int) -> None:
    data = load(bed, horizon)
    target = pd.to_datetime(data["target_ns"])
    issue = pd.to_datetime(data["issue_ns"])
    x = np.concatenate([data["x"].reshape(len(data["y"]), -1), data["future"]], axis=1)
    y = data["y"]
    test = target.year == year
    systems = np.unique(data["system"][test])
    key = f"{bed}_{year}_h{horizon}_lightgbm_s{seed}"
    run_dir = RUNROOT / key
    record_path = OUT / f"{key}.json"
    parent_key = f"{bed}_h{horizon}_lightgbm_s{seed}_point_full"
    parent_prediction = PARENT / "runs" / parent_key / "prediction.npz"
    parent_model = PARENT / "runs" / parent_key / "model_None.txt"
    parent_record = RELEASE / "results/corrected_v28_array1b/runs" / f"{parent_key}.json"
    if not parent_prediction.exists() or not parent_model.exists() or not parent_record.exists():
        raise FileNotFoundError(f"missing verified parent artifacts for {key}")
    iterations = inherited_iterations(bed, horizon, seed)
    signature = {
        "protocol": sha(CONFIG),
        "code": sha(Path(__file__)),
        "data": sha(PARENT / bed / f"h{horizon}.npz"),
        "parent_model": sha(parent_model),
        "parent_prediction": sha(parent_prediction),
    }
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("signature") != signature or previous.get("status") != "complete":
            raise RuntimeError(f"completed record signature mismatch for {key}")
        emit("run_reused", run_id=key, reason="verified local checkpoint")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frozen = np.full(len(y), np.nan, dtype="float32")
    import lightgbm as lgb
    frozen_model = lgb.Booster(model_file=str(parent_model))
    frozen[test] = np.clip(frozen_model.predict(x[test], num_threads=4), 0.0, 1.2)
    with np.load(parent_prediction) as parent_values:
        expected = parent_values["prediction"][:, 0]
    if not np.allclose(frozen[test], expected[test], rtol=0, atol=1e-6):
        raise RuntimeError(f"frozen prediction mismatch for {key}")

    annual = np.full(len(y), np.nan, dtype="float32")
    monthly = np.full(len(y), np.nan, dtype="float32")
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    fit = (target.year == year - 1) & (target < cutoff)
    if any(int((fit & (data["system"] == system)).sum()) < 100 for system in systems):
        raise RuntimeError(f"annual support failure for {key}")
    emit("run_started", run_id=key, policy="annual_previous_year", model="lightgbm",
         bed=bed, horizon=horizon, seed=seed, fit_n=int(fit.sum()), iterations=iterations)
    annual_model = fit_regressor(x, y, fit, seed, iterations)
    annual_model.booster_.save_model(str(run_dir / "annual.txt"))
    annual[test] = np.clip(annual_model.predict(x[test]), 0.0, 1.2)
    records = [{
        "policy": "annual_previous_year", "cutoff": str(cutoff),
        "train_n": int(fit.sum()), "train_max": str(target[fit].max()),
        "min_issue": str(issue[test].min()), "fallback": False,
    }]
    previous_model = annual_model
    for month in range(1, 13):
        cutoff = pd.Timestamp(year=year, month=month, day=1)
        end = cutoff + pd.offsets.MonthBegin(1)
        predict = test & (issue >= cutoff) & (issue < end)
        if not predict.any():
            records.append({"month": month, "status": "no_prediction_rows"})
            continue
        fit = (target >= cutoff - pd.Timedelta(days=365)) & (target < cutoff)
        enough = all(int((fit & (data["system"] == system)).sum()) >= 100 for system in systems)
        if enough:
            if target[fit].max() >= issue[predict].min():
                raise RuntimeError(f"cutoff leakage for {key} month {month}")
            emit("run_started", run_id=key, policy="monthly_trailing_365_days",
                 month=month, fit_n=int(fit.sum()), prediction_n=int(predict.sum()),
                 cutoff=str(cutoff), iterations=iterations)
            previous_model = fit_regressor(x, y, fit, seed, iterations)
            previous_model.booster_.save_model(str(run_dir / f"month_{month:02d}.txt"))
            status = "complete"
            fallback = False
            emit("run_completed", run_id=key, policy="monthly_trailing_365_days",
                 month=month, details={"fit_n": int(fit.sum()), "prediction_n": int(predict.sum()),
                                       "iterations": iterations})
        else:
            status = "fallback_annual"
            fallback = True
            emit("fallback_used", run_id=key, policy="monthly_trailing_365_days",
                 month=month, reason="system support below 100")
        monthly[predict] = np.clip(previous_model.predict(x[predict]), 0.0, 1.2)
        records.append({
            "policy": "monthly_trailing_365_days", "month": month,
            "cutoff": str(cutoff), "train_n": int(fit.sum()),
            "train_max": str(target[fit].max()), "min_issue": str(issue[predict].min()),
            "predictions": int(predict.sum()), "fallback": fallback, "status": status,
        })

    predictions = {"frozen_2021": frozen, "annual_previous_year": annual,
                   "monthly_trailing_365_days": monthly}
    if not all(np.isfinite(values[test]).all() for values in predictions.values()):
        raise RuntimeError(f"non-finite test prediction for {key}")
    prediction_path = run_dir / "predictions.npz"
    atomic_npz(prediction_path, **predictions)
    record = {
        "status": "complete", "key": key, "bed": bed, "year": year,
        "horizon": horizon, "model": "lightgbm", "seed": seed,
        "iterations": iterations, "signature": signature, "records": records,
        "prediction_sha256": sha(prediction_path),
        "runtime_seconds": time.monotonic() - started,
    }
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    emit("bundle_completed", run_id=key, runtime_seconds=record["runtime_seconds"],
         prediction_sha256=record["prediction_sha256"])


def balanced(values: np.ndarray, mask: np.ndarray, systems: np.ndarray) -> float:
    return float(np.mean([np.mean(values[mask & (systems == system)])
                          for system in np.unique(systems[mask])
                          if np.any(mask & (systems == system))]))


def summarize(beds: list[str], horizons: list[int], year: int) -> None:
    rows: list[dict[str, object]] = []
    for bed in beds:
        for horizon in horizons:
            data = load(bed, horizon)
            target = pd.to_datetime(data["target_ns"])
            test = target.year == year
            daylight = test & data["daylight"]
            reference = np.abs(data["y"] - data["references"][:, 1])
            for policy in ["frozen_2021", "annual_previous_year", "monthly_trailing_365_days"]:
                seeds = []
                for seed in SEEDS:
                    with np.load(RUNROOT / f"{bed}_{year}_h{horizon}_lightgbm_s{seed}" / "predictions.npz") as values:
                        seeds.append(np.abs(data["y"] - values[policy]))
                loss = np.mean(np.stack(seeds), axis=0)
                for population, mask in [("all", test), ("daylight", daylight)]:
                    base = balanced(reference, mask, data["system"])
                    candidate = balanced(loss, mask, data["system"])
                    rows.append({"year": year, "testbed": bed, "horizon": horizon,
                                 "model": "lightgbm", "policy": policy,
                                 "population": population, "n": int(mask.sum()),
                                 "mae": candidate, "previous_day_mae": base,
                                 "skill_vs_previous_day": 1.0 - candidate / base})
    path = OUT / "update_metrics_h4_h12_lightgbm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    emit("summary_written", artifact_path=str(path), rows=len(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 12])
    parser.add_argument("--beds", nargs="+", default=["DKASC", "HKUST"])
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()
    emit("task_started", task_id="P03C", stage="lightgbm_updates",
         horizons=args.horizons, beds=args.beds, year=args.year)
    for bed in args.beds:
        for horizon in args.horizons:
            for seed in SEEDS:
                bundle(bed, horizon, seed, args.year)
    summarize(args.beds, args.horizons, args.year)
    emit("task_complete", task_id="P03C", stage="lightgbm_updates",
         horizons=args.horizons, beds=args.beds, year=args.year)


if __name__ == "__main__":
    main()
