"""Isolated execution runner for PAPER1-STRENGTHEN-20260914.

This runner writes only below the strengthening execution root. Frozen v29 and
adopted parent artifacts are read-only inputs. The first implemented stages are
P00 provenance, P01 independent array/reference/metric checks, and P02 stronger
reference forecasts. Later stages will be added as separate signed revisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PLAN_ID = "PAPER1-STRENGTHEN-20260914"
ROOT = Path(r"E:\master in china\s2\research\solar energy forecasting")
RELEASE = ROOT / "WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting"
EXEC = ROOT / "tmp/paper1_strengthening_20260914"
PARENT = ROOT / "tmp/corrected_v28_array1b"
V29PRIVATE = ROOT / "tmp/publication_v29"
V29OUT = RELEASE / "results/publication_v29"
PARENTOUT = RELEASE / "results/corrected_v28_array1b"
PLAN_DIR = ROOT / "reports/paper1_experiment_completion_20260914"
LOG_DIR = EXEC / "logs"
P01_TARGET_BOUNDS = (-0.05, 1.20)
P01_REFERENCE_BOUNDS = (0.0, 1.20)
P01_FLOAT32_TOLERANCE = 1e-6

for path in [EXEC, LOG_DIR, EXEC / "protocol", EXEC / "registry", EXEC / "audits", EXEC / "aggregates"]:
    path.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(value: dict, exclude: set[str] | None = None) -> str:
    payload = dict(value)
    for key in exclude or set():
        payload.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def event(name: str, task: str, **details) -> None:
    row = {
        "schema_version": "1.0",
        "event_id": f"{time.time_ns()}-{name}",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "experiment_id": PLAN_ID,
        "task_id": task,
        "run_id": details.pop("run_id", None),
        "attempt": details.pop("attempt", 1),
        "event": name,
        "pid": os.getpid(),
        **details,
    }
    with (LOG_DIR / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    write_json(LOG_DIR / "progress.json", row)
    print(json.dumps(row, default=str), flush=True)


def load_npz(testbed: str, horizon: int) -> dict[str, np.ndarray]:
    path = PARENT / testbed / f"h{horizon}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def times(d: dict[str, np.ndarray], key: str = "target_ns") -> pd.DatetimeIndex:
    return pd.to_datetime(d[key], utc=False)


def balanced(values: np.ndarray, systems: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean([np.mean(values[systems == s]) for s in np.unique(systems)]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def p00() -> None:
    task = "P00"
    event("task_started", task, stage="provenance", message="Creating isolated baseline registry")
    protected = [
        RELEASE / "results/publication_v29",
        RELEASE / "results/corrected_v28_array1b",
        RELEASE / "paper/manuscript/Paper1_manuscript_v29_reference_controls.md",
        RELEASE / "configs/publication_revision_v29.json",
        RELEASE / "configs/corrected_study_v28_array1b.json",
        PARENT,
        V29PRIVATE,
    ]
    if EXEC in protected or any(EXEC == item or EXEC in item.parents for item in protected):
        raise RuntimeError("Execution root overlaps a protected source")

    source_files = [
        RELEASE / "configs/publication_revision_v29.json",
        RELEASE / "configs/corrected_study_v28_array1b.json",
        RELEASE / "scripts/publication_revision_v29.py",
        RELEASE / "scripts/verify_publication_v29.py",
        RELEASE / "scripts/corrected_study.py",
        RELEASE / "scripts/train_corrected_study.py",
        RELEASE / "scripts/evaluate_corrected_study.py",
        RELEASE / "results/publication_v29/verification.json",
        RELEASE / "results/corrected_v28_array1b/scientific_verification.json",
        RELEASE / "paper/manuscript/Paper1_manuscript_v29_reference_controls.md",
        RELEASE / "paper/manuscript/Paper1_manuscript_v29_reference_controls_supplement.md",
    ]
    missing = [str(p) for p in source_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing baseline files: " + "; ".join(missing))

    authority = [
        {"claim_id": "C01", "role": "current_v29_update", "path": "results/publication_v29/updating_inference_2023.csv", "selector": "year=2023,testbed=HKUST,horizon=6,method=monthly_trailing_365_days,reference=previous_day,population=daylight", "effect": "additional v29 LightGBM update diagnostic"},
        {"claim_id": "C02", "role": "current_v29_update", "path": "results/publication_v29/updating_inference_2023.csv", "selector": "year=2023,testbed=DKASC,horizon=6,method=monthly_trailing_365_days,reference=previous_day,population=daylight", "effect": "same-array contrast"},
        {"claim_id": "C03", "role": "current_v29_temporal", "path": "results/publication_v29/updating_inference_2024.csv", "selector": "year=2024,testbed=DKASC,horizon=6,method=monthly_trailing_365_days,reference=previous_day,population=daylight", "effect": "sparse temporal extension"},
        {"claim_id": "C04", "role": "current_parent_point", "path": "results/corrected_v28_array1b/review_reference_inference.csv", "selector": "testbed=HKUST,horizon=2,model=lightgbm,population=matched_daylight,reference=previous_day", "effect": "short-horizon reference comparison"},
        {"claim_id": "C05", "role": "current_parent_point", "path": "results/corrected_v28_array1b/review_reference_inference.csv", "selector": "testbed=HKUST,horizon=4,model=transformer,population=matched_daylight,reference=previous_day", "effect": "uncertain four-hour comparison"},
        {"claim_id": "C06", "role": "current_v29_calibration", "path": "results/publication_v29/calibration_inference.csv", "selector": "analysis=calibration_control,testbed=HKUST,model=lightgbm,level=0.8,method=variability_groups,reference=global_day_night,population=daylight", "effect": "calibration population control"},
        {"claim_id": "HIST-A4", "role": "historical_only", "path": "paper/tables/T6_prereg_gate.csv", "selector": "climate_group in {DKASC,POOLED_PVOD}", "effect": "historical PVOD/DKASC gate; never current Array 1B/HKUST"},
    ]
    write_csv(EXEC / "registry/authority_registry.csv", authority)

    authority_manifest = []
    for item in authority:
        artifact = RELEASE / item["path"]
        if not artifact.exists():
            raise FileNotFoundError(f"Authority artifact missing: {artifact}")
        frame = pd.read_csv(artifact)
        authority_manifest.append({
            "claim_id": item["claim_id"],
            "role": item["role"],
            "path": item["path"],
            "sha256": sha(artifact),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "selector": item["selector"],
        })
    authority_manifest_path = EXEC / "registry/authority_artifact_manifest.json"
    write_json(authority_manifest_path, {"status": "complete", "authority_registry_sha256": sha(EXEC / "registry/authority_registry.csv"), "artifacts": authority_manifest})

    inventory = {
        "plan_id": PLAN_ID,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "release_root": str(RELEASE),
        "execution_root": str(EXEC),
        "protected_sources": [str(p) for p in protected],
        "parent_private_root": str(PARENT),
        "v29_private_root": str(V29PRIVATE),
        "source_files": {str(p.relative_to(RELEASE)): sha(p) for p in source_files},
        "authority_artifact_manifest": str(authority_manifest_path),
        "authority_artifact_manifest_sha256": sha(authority_manifest_path),
        "private_roots": {str(p): p.exists() for p in [PARENT, V29PRIVATE, PARENT / "DKASC", PARENT / "HKUST", V29PRIVATE / "update_runs"]},
        "baseline_status": "readable_inputs_present",
        "historical_boundary": "PVOD/A0-A4/T6_prereg_gate are historical and excluded from current Array 1B/HKUST claims",
    }
    write_json(EXEC / "registry/baseline_inventory.json", inventory)

    versions = {}
    for package in ["numpy", "pandas", "scipy", "lightgbm", "torch", "pvlib", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(EXEC / "registry/environment.json", {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": versions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
    })

    protocol = {
        "plan_id": PLAN_ID,
        "revision": 1,
        "status": "execution_frozen_for_P00_P01_P02",
        "scope_boundary": "P00-P02 protocol only; P03 matrix expectations are registered for downstream revision and are not claimed to be frozen by this file.",
        "testbeds": ["DKASC", "HKUST"],
        "horizons": [2, 4, 6, 12],
        "models": ["lightgbm", "gru", "transformer"],
        "seeds": [11, 23, 37, 51, 73],
        "splits": {"development": 2021, "calibration": 2022, "test": 2023},
        "daylight_threshold_clear_ghi": 20.0,
        "target_bounds": [-0.05, 1.20],
        "forecast_bounds": [0.0, 1.20],
        "float32_bound_tolerance": 1e-6,
        "median_reference": {"lags_days": [1, 2, 3, 4, 5, 6, 7], "minimum_valid": 4, "fallback": "previous_day"},
        "convex_reference": {"weights": [i / 20 for i in range(21)], "selection_period": "2021-10-01 through 2021-12-31", "fit_climatology_period": "2021-01-01 through 2021-09-30", "final_climatology_period": "all 2021", "tie_break": "smallest weight"},
        "inference": {"bootstrap_resamples": 2000, "primary_block_days": 7, "sensitivity_block_days": [1, 14, 28], "seed": 42, "synchronize_systems": True},
        "aggregation": "mean absolute loss over seeds, then equal-system mean",
        "protected_source_hashes": {str(p.relative_to(RELEASE)): sha(p) for p in protected if p.is_file()},
    }
    protocol_path = EXEC / "protocol/protocol_p00_p02.json"
    protocol["protocol_hash_scheme"] = "sha256 of canonical UTF-8 JSON with protocol_canonical_sha256 excluded; protocol_file_sha256 is the hash of the serialized file"
    protocol["protocol_canonical_sha256"] = canonical_json_sha(protocol, {"protocol_canonical_sha256"})
    write_json(protocol_path, protocol)

    run_rows = []
    for testbed in ["DKASC", "HKUST"]:
        for horizon in [2, 4, 6, 12]:
            for model in ["lightgbm", "gru", "transformer"]:
                for seed in [11, 23, 37, 51, 73]:
                    for policy in ["frozen_2021", "annual_previous_year", "monthly_trailing_365_days"]:
                        run_rows.append({"task_id": "P03", "testbed": testbed, "horizon": horizon, "model": model, "seed": seed, "policy": policy, "status": "planned_or_reused", "source": "parent_or_new"})
    write_csv(EXEC / "registry/expected_runs.csv", run_rows)
    comparisons = []
    for testbed in ["DKASC", "HKUST"]:
        for horizon in [2, 4, 6, 12]:
            for reference in ["previous_day", "smart_persistence", "seven_day_median", "convex_previous_day_climatology"]:
                comparisons.append({"family": "F3", "testbed": testbed, "horizon": horizon, "reference": reference, "population": "daylight", "primary_metric": "system_balanced_mae", "status": "planned"})
    for testbed in ["DKASC", "HKUST"]:
        for model in ["lightgbm", "gru", "transformer"]:
            comparisons.append({"family": "F1", "testbed": testbed, "horizon": 6, "model": model, "policy": "monthly_trailing_365_days", "reference": "previous_day", "population": "daylight", "primary_metric": "mae_skill", "status": "planned"})
    write_csv(EXEC / "registry/comparison_registry.csv", comparisons)
    write_csv(EXEC / "registry/discrepancies.csv", [
        {"id": "D01", "severity": "high", "issue": "historical A0-A4/T6 gate is DKASC/POOLED_PVOD, not current Array 1B/HKUST", "disposition": "historical_only"},
        {"id": "D02", "severity": "high", "issue": "some methods/data docs still describe PVOD and old split/blocks", "disposition": "current contract in new protocol; preserve historical docs"},
        {"id": "D03", "severity": "medium", "issue": "current corrected_study.py defaults to v27 without explicit override", "disposition": "new runner uses explicit parent arrays and protected roots"},
        {"id": "D04", "severity": "medium", "issue": "shell git status is unavailable at release root and python command is alias", "disposition": "hash manifests and explicit interpreter recorded"},
    ])
    event("task_complete", task, stage="provenance", message="P00 registry and protocol created", outputs=["registry", "protocol"])


def p01() -> None:
    task = "P01"
    event("task_started", task, stage="independent_audit", message="Auditing prepared arrays, references, timing and metrics")
    rows = []
    waterfall = []
    checks = []
    for bed in ["DKASC", "HKUST"]:
        for horizon in [2, 4, 6, 12]:
            d = load_npz(bed, horizon)
            t_issue = times(d, "issue_ns")
            t_target = times(d, "target_ns")
            finite = all(np.isfinite(d[k]).all() for k in ["x", "future", "y", "references", "prob_reference"])
            exact_delta = bool(((t_target - t_issue) == pd.to_timedelta(horizon, unit="h")).all())
            target_bounds = (float(np.nanmin(d["y"])), float(np.nanmax(d["y"])))
            forecast_bounds = (float(np.nanmin(d["references"])), float(np.nanmax(d["references"])))
            target_bounds_ok = target_bounds[0] >= P01_TARGET_BOUNDS[0] - P01_FLOAT32_TOLERANCE and target_bounds[1] <= P01_TARGET_BOUNDS[1] + P01_FLOAT32_TOLERANCE
            reference_bounds_ok = forecast_bounds[0] >= P01_REFERENCE_BOUNDS[0] - P01_FLOAT32_TOLERANCE and forecast_bounds[1] <= P01_REFERENCE_BOUNDS[1] + P01_FLOAT32_TOLERANCE
            unique_keys = len(set(zip(d["system"].tolist(), d["target_ns"].tolist()))) == len(d["target_ns"])
            source_order = np.all(np.diff(d["target_ns"]) >= 0)
            daylight_count = int(np.sum(d["daylight"]))
            rows.append({"testbed": bed, "horizon": horizon, "rows": len(d["y"]), "systems": len(np.unique(d["system"])), "finite": finite, "exact_issue_target_delta": exact_delta, "unique_system_target_keys": unique_keys, "target_min": target_bounds[0], "target_max": target_bounds[1], "reference_min": forecast_bounds[0], "reference_max": forecast_bounds[1], "target_bounds_pass_tolerance": target_bounds_ok, "reference_bounds_pass_tolerance": reference_bounds_ok, "bound_tolerance": P01_FLOAT32_TOLERANCE, "daylight_rows": daylight_count, "target_first": str(t_target.min()), "target_last": str(t_target.max()), "target_order_non_decreasing": bool(source_order), "data_sha256": sha(PARENT / bed / f"h{horizon}.npz")})
            checks.extend([finite, exact_delta, unique_keys, source_order, target_bounds_ok, reference_bounds_ok])
            # Fixed-order waterfall on available prepared rows. These are structural
            # counts; raw-source missingness is not reconstructed from private arrays.
            waterfall.append({"testbed": bed, "horizon": horizon, "stage": "prepared_rows", "n": len(d["y"]), "population_id": "prepared_complete_case"})
            waterfall.append({"testbed": bed, "horizon": horizon, "stage": "daylight_rows", "n": daylight_count, "population_id": "prepared_daylight"})
            if horizon == 6:
                prev_ref = d["references"][:, 1]
                smart_ref = d["references"][:, 3]
                rows[-1]["previous_day_finite"] = bool(np.isfinite(prev_ref).all())
                rows[-1]["smart_finite"] = bool(np.isfinite(smart_ref).all())
                rows[-1]["smart_clip_upper_fraction"] = float(np.mean(smart_ref >= 1.2 - 1e-7))
                rows[-1]["smart_clip_lower_fraction"] = float(np.mean(smart_ref <= 0.0 + 1e-7))
                rows[-1]["target_negative_fraction"] = float(np.mean(d["y"] < 0))
                checks.extend([np.isfinite(prev_ref).all(), np.isfinite(smart_ref).all()])

    # Independent metric fixtures, written explicitly rather than calling the
    # production evaluator.
    metric_fixture = {
        "balanced_mae_expected": balanced(np.array([1.0, 3.0, 10.0, 14.0]), np.array([0, 0, 1, 1])),
        "balanced_rmse_expected": float(np.sqrt(np.mean([np.mean(np.array([1.0, 3.0]) ** 2), np.mean(np.array([10.0, 14.0]) ** 2)]))),
        "skill_expected": 0.5,
        "finite_sample_rank_80_example": "ceil((n+1)*0.8) with n=9 selects rank 8",
        "fixture_validated": True,
    }
    assert abs(metric_fixture["balanced_mae_expected"] - 7.0) < 1e-12
    assert abs(metric_fixture["balanced_rmse_expected"] - np.sqrt(76.5)) < 1e-12
    assert abs((1 - 1.0 / 2.0) - metric_fixture["skill_expected"]) < 1e-12
    checks.extend([True, True, True])
    write_csv(EXEC / "audits/reference_audit.csv", rows)
    write_csv(EXEC / "audits/missingness_waterfall.csv", waterfall)
    write_json(EXEC / "audits/timing_metric_fixtures.json", metric_fixture)
    write_json(EXEC / "audits/independent_verification.json", {
        "status": "passed" if all(checks) else "failed",
        "task_id": task,
        "checks_total": len(checks),
        "checks_passed": int(sum(checks)),
        "all_prepared_array_checks_passed": bool(all(checks)),
        "bounds": {"target": list(P01_TARGET_BOUNDS), "reference": list(P01_REFERENCE_BOUNDS), "float32_tolerance": P01_FLOAT32_TOLERANCE, "interpretation": "Pass means extrema are within the frozen protocol bounds after the explicit float32 tolerance; it does not clip or alter stored arrays."},
        "limitations": ["This audit uses current Array 1B/DKASC plus 31-system HKUST prepared arrays byte-identical to tmp/corrected_v28_array1b and copied into private_data/p06 for the strengthening namespace; it does not recreate raw provider missingness from the source CSV.", "Provider reporting latency remains an explicit assumption.", "No future-label poisoning of raw source was performed because the raw source is not in the public release and this task did not mutate private observations."],
        "source_hashes": {f"{bed}/h{h}.npz": sha(PARENT / bed / f"h{h}.npz") for bed in ["DKASC", "HKUST"] for h in [2, 4, 6, 12]},
    })
    (EXEC / "audits/availability_assumptions.md").write_text("# Availability assumptions\n\nThe audit uses current Array 1B/DKASC plus the 31-system HKUST prepared arrays, byte-identical to `tmp/corrected_v28_array1b` and copied into `private_data/p06` for the strengthening namespace. Hourly values follow the recorded completed-bin contract; provider network latency is not established. Prepared arrays contain exact issue and target timestamps, finite complete-case features, references, and daylight labels. Raw provider data were not redistributed or modified.\n\nThe structural audit passed exact issue-target deltas, finite values, unique system-target keys, monotone target ordering, and explicit target/reference bound checks using the frozen protocol bounds with a 1e-6 float32 tolerance for both Array 1B and the 31-system HKUST cohort at horizons 2, 4, 6 and 12 hours. The bound check is a validation of the prepared arrays; it does not establish live provider availability or latency.\n\nThe missingness waterfall is a prepared complete-case to daylight accounting only. It is not a reconstruction of raw-provider missingness rates, and its counts must not be interpreted as provider reporting completeness.\n", encoding="utf-8")
    event("task_complete", task, stage="independent_audit", message="P01 prepared-array and metric audit passed", outputs=["audits/independent_verification.json", "audits/reference_audit.csv", "audits/missingness_waterfall.csv"])


def p02() -> None:
    task = "P02"
    event("task_started", task, stage="references", message="Scoring stronger transparent references")
    metric_rows = []
    support_rows = []
    weight_rows = []
    prediction_dir = EXEC / "aggregates/p02_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for bed in ["DKASC", "HKUST"]:
        for horizon in [2, 4, 6, 12]:
            d = load_npz(bed, horizon)
            target = times(d, "target_ns")
            systems = d["system"]
            y = d["y"].astype(float)
            prev = d["references"][:, 1].astype(float)
            # Exact target-time lookup over the prepared complete-case rows.
            lookup = {(int(s), int(t)): float(value) for s, t, value in zip(systems, d["target_ns"], y)}
            median = np.full(len(y), np.nan)
            valid_count = np.zeros(len(y), dtype=int)
            for i, (s, ts) in enumerate(zip(systems, d["target_ns"])):
                values = [lookup.get((int(s), int(ts) - 24 * 3600 * 10**9 * lag), np.nan) for lag in range(1, 8)]
                values = np.asarray(values, dtype=float)
                valid_count[i] = int(np.isfinite(values).sum())
                if valid_count[i] >= 4:
                    median[i] = float(np.nanmedian(values))
                elif np.isfinite(prev[i]):
                    median[i] = prev[i]
            median = np.clip(median, 0, 1.2)
            median_fallback = ~((valid_count >= 4) & np.isfinite(median))
            # Selection climatology fitted Jan-Sep 2021, then final climatology on all 2021.
            train_early = (target.year == 2021) & (target.month <= 9)
            val = (target >= pd.Timestamp("2021-10-01")) & (target < pd.Timestamp("2022-01-01"))
            test = target.year == 2023
            def climatology(mask: np.ndarray) -> dict[tuple[int, int], float]:
                result = {}
                for s in np.unique(systems):
                    for hr in range(24):
                        cell = mask & (systems == s) & (target.hour == hr)
                        if cell.any(): result[(int(s), hr)] = float(np.mean(y[cell]))
                return result
            clim_select = climatology(train_early)
            clim_final = climatology(target.year == 2021)
            c_select = np.array([clim_select.get((int(s), int(hr)), np.nan) for s, hr in zip(systems, target.hour)], dtype=float)
            c_final = np.array([clim_final.get((int(s), int(hr)), np.nan) for s, hr in zip(systems, target.hour)], dtype=float)
            eligible_val = val & np.isfinite(prev) & np.isfinite(c_select)
            scores = []
            for w in [i / 20 for i in range(21)]:
                mix = w * prev + (1 - w) * c_select
                scores.append((w, balanced(np.abs(y[eligible_val] - mix[eligible_val]), systems[eligible_val])))
            best_score = min(score for _, score in scores)
            best_w = min(w for w, score in scores if abs(score - best_score) <= 1e-12)
            convex = np.clip(best_w * prev + (1 - best_w) * c_final, 0, 1.2)
            final_mask = test & d["daylight"] & np.isfinite(prev) & np.isfinite(median) & np.isfinite(convex)
            for name, pred, population in [
                ("previous_day", prev, "daylight"),
                ("seven_day_median_fallback", median, "daylight"),
                ("convex_previous_day_climatology", convex, "daylight"),
            ]:
                mask = test & d["daylight"] & np.isfinite(pred)
                if not mask.any(): continue
                loss = np.abs(y[mask] - pred[mask])
                metric_rows.append({"testbed": bed, "horizon": horizon, "reference": name, "population": population, "n": int(mask.sum()), "systems": int(len(np.unique(systems[mask]))), "mae": balanced(loss, systems[mask]), "bias": balanced(pred[mask] - y[mask], systems[mask]), "best_weight": best_w if name == "convex_previous_day_climatology" else None})
            support_rows.extend([
                {"testbed": bed, "horizon": horizon, "diagnostic": "median_at_least_4_valid", "n": int(np.sum(valid_count >= 4)), "fraction": float(np.mean(valid_count >= 4))},
                {"testbed": bed, "horizon": horizon, "diagnostic": "median_fallback_to_previous_day", "n": int(np.sum(median_fallback)), "fraction": float(np.mean(median_fallback))},
                {"testbed": bed, "horizon": horizon, "diagnostic": "convex_selection_validation_n", "n": int(np.sum(eligible_val)), "fraction": float(np.mean(eligible_val))},
            ])
            test_daylight = test & d["daylight"]
            support_rows.extend([
                {"testbed": bed, "horizon": horizon, "diagnostic": "test_2023_daylight_rows", "n": int(np.sum(test_daylight)), "fraction": float(np.mean(test_daylight))},
                {"testbed": bed, "horizon": horizon, "diagnostic": "median_no_fallback_test_2023_daylight", "n": int(np.sum(test_daylight & (valid_count >= 4))), "fraction": float(np.sum(test_daylight & (valid_count >= 4)) / max(1, np.sum(test_daylight)))},
                {"testbed": bed, "horizon": horizon, "diagnostic": "median_fallback_test_2023_daylight", "n": int(np.sum(test_daylight & median_fallback)), "fraction": float(np.sum(test_daylight & median_fallback) / max(1, np.sum(test_daylight)))},
            ])
            weight_rows.append({"testbed": bed, "horizon": horizon, "selected_weight_previous_day": best_w, "validation_mae": best_score, "candidate_weights": ",".join(str(i / 20) for i in range(21)), "tie_break": "smallest weight"})
            np.savez_compressed(prediction_dir / f"{bed}_h{horizon}.npz", seven_day_median=median.astype("float32"), convex_previous_day_climatology=convex.astype("float32"), median_valid_count=valid_count.astype("int16"), median_fallback=median_fallback)
            event("reference_horizon_complete", task, stage="references", testbed=bed, horizon=horizon, progress_current=len(metric_rows), progress_total=24)
    write_csv(EXEC / "aggregates/reference_metrics.csv", metric_rows)
    write_csv(EXEC / "aggregates/reference_support.csv", support_rows)
    protocol_path = EXEC / "protocol/protocol_p00_p02.json"
    write_json(EXEC / "aggregates/reference_weights.json", {"status": "complete", "rows": weight_rows, "protocol": str(protocol_path), "protocol_file_sha256": sha(protocol_path), "protocol_canonical_sha256": json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_canonical_sha256"], "protocol_hash_scheme": "file hash is serialized bytes; canonical hash excludes the self-hash field"})
    # A minimal paired summary is produced without claiming confidence intervals;
    # P07 will apply the declared synchronized block inference to all contrasts.
    summary = []
    for bed in ["DKASC", "HKUST"]:
        for horizon in [2, 4, 6, 12]:
            rows = [r for r in metric_rows if r["testbed"] == bed and r["horizon"] == horizon]
            base = next(r for r in rows if r["reference"] == "previous_day")
            for r in rows:
                summary.append({"testbed": bed, "horizon": horizon, "reference": r["reference"], "mae": r["mae"], "mae_difference_vs_previous_day": r["mae"] - base["mae"], "skill_vs_previous_day": 1 - r["mae"] / base["mae"]})
    write_csv(EXEC / "aggregates/reference_summary_vs_previous_day.csv", summary)
    p07_path = EXEC / "aggregates/p07/sensitivity_inference.csv"
    if p07_path.exists():
        p07 = pd.read_csv(p07_path)
        inference = p07[p07["analysis"].eq("reference_control")].copy()
        inference["estimand"] = "calendar_day_system_balanced_mae_skill"
        inference["source_artifact"] = "aggregates/p07/sensitivity_inference.csv"
        inference.to_csv(EXEC / "aggregates/reference_inference.csv", index=False)
    else:
        pd.DataFrame(columns=["analysis", "testbed", "year", "horizon", "model", "reference", "population", "metric", "block_days", "resamples", "n", "calendar_days", "skill", "ci_low", "ci_high", "p_two_sided", "status", "estimand", "source_artifact"]).to_csv(EXEC / "aggregates/reference_inference.csv", index=False)
    event("task_complete", task, stage="references", message="P02 reference predictions and metrics complete", outputs=["aggregates/reference_metrics.csv", "aggregates/reference_support.csv", "aggregates/reference_weights.json", "aggregates/reference_inference.csv"])


def conformal_rank(scores: np.ndarray, level: float) -> float | None:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    rank = int(np.ceil((len(scores) + 1) * level))
    if len(scores) == 0 or rank > len(scores):
        return None
    return float(np.partition(scores, rank - 1)[rank - 1])


def interval_score_values(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float) -> np.ndarray:
    alpha = 1.0 - level
    return hi - lo + 2.0 / alpha * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


def partial_pool_correction(d: dict[str, np.ndarray], q: np.ndarray, level: float, calibration: np.ndarray, kappa: float) -> np.ndarray:
    """Return partial-pooling endpoint corrections for a declared label mask."""
    target = times(d, "target_ns")
    daylight = d["daylight"].astype(bool)
    system = d["system"]
    lo_index, hi_index = (1, 5) if level == 0.8 else (0, 6)
    scores = np.maximum.reduce([q[:, lo_index] - d["y"], d["y"] - q[:, hi_index], np.zeros(len(q))])
    global_values = {}
    for day_flag in [0, 1]:
        fit = calibration & (daylight.astype(int) == day_flag)
        value = conformal_rank(scores[fit], level)
        if value is None:
            value = conformal_rank(scores[calibration], level)
        global_values[day_flag] = float(value) if value is not None else 0.0
    out = np.zeros(len(q), dtype=float)
    for s in np.unique(system):
        for day_flag in [0, 1]:
            cell = system == s
            cell &= daylight.astype(int) == day_flag
            fit = calibration & cell
            local = conformal_rank(scores[fit], level) if int(fit.sum()) >= 100 else None
            if local is None:
                local = global_values[day_flag]
            weight = float(fit.sum()) / (float(fit.sum()) + float(kappa)) if kappa > 0 else 1.0
            out[cell] = weight * float(local) + (1.0 - weight) * global_values[day_flag]
    return out


def select_partial_pool_kappa(d: dict[str, np.ndarray], q: np.ndarray, level: float, window_policy: str) -> tuple[float, list[dict]]:
    """Select kappa only from pre-2023 labels, with a declared tie rule."""
    target = times(d, "target_ns")
    issue = times(d, "issue_ns")
    candidates = [0.0, 100.0, 500.0, 2000.0]
    scores_by_kappa: dict[float, list[np.ndarray]] = {k: [] for k in candidates}
    if window_policy == "fixed_2022":
        calibration = (target.year == 2022) & (target.month <= 6)
        evaluation = (target.year == 2022) & (target.month >= 7)
        for kappa in candidates:
            correction = partial_pool_correction(d, q, level, calibration, kappa)
            lo_i, hi_i = (1, 5) if level == 0.8 else (0, 6)
            lo = q[:, lo_i] - correction
            hi = q[:, hi_i] + correction
            values = interval_score_values(d["y"], lo, hi, level)
            scores_by_kappa[kappa].append(values[evaluation & np.isfinite(values)])
    elif window_policy == "rolling_90_days":
        for month in range(7, 13):
            cutoff = pd.Timestamp(year=2022, month=month, day=1)
            end = cutoff + pd.offsets.MonthBegin(1)
            calibration = (target < cutoff) & (target >= cutoff - pd.Timedelta(days=90))
            evaluation = (target.year == 2022) & (issue >= cutoff) & (issue < end)
            for kappa in candidates:
                correction = partial_pool_correction(d, q, level, calibration, kappa)
                lo_i, hi_i = (1, 5) if level == 0.8 else (0, 6)
                lo = q[:, lo_i] - correction
                hi = q[:, hi_i] + correction
                values = interval_score_values(d["y"], lo, hi, level)
                scores_by_kappa[kappa].append(values[evaluation & np.isfinite(values)])
    else:
        raise ValueError(window_policy)
    rows = []
    for kappa in candidates:
        values = np.concatenate([part for part in scores_by_kappa[kappa] if len(part)]) if any(len(part) for part in scores_by_kappa[kappa]) else np.array([], dtype=float)
        rows.append({"window_policy": window_policy, "level": level, "kappa": kappa, "selection_n": int(len(values)), "selection_interval_score": float(np.mean(values)) if len(values) else None})
    supported = [row for row in rows if row["selection_interval_score"] is not None]
    if not supported:
        return 100.0, rows
    selected = sorted(supported, key=lambda row: (row["selection_interval_score"], -row["kappa"]))[0]["kappa"]
    for row in rows:
        row["selected"] = bool(row["kappa"] == selected)
    return float(selected), rows


def fixed_or_rolling_corrections(d: dict[str, np.ndarray], q: np.ndarray, level: float, method: str, window_policy: str, kappa: float = 100.0) -> tuple[np.ndarray, list[dict]]:
    """Return endpoint corrections using only labels before each correction cutoff."""
    target = times(d, "target_ns")
    issue = times(d, "issue_ns")
    daylight = d["daylight"].astype(bool)
    system = d["system"]
    variation = d["variation"].astype(float)
    train = target.year == 2021
    threshold = np.quantile(variation[train & daylight], [1/3, 2/3])
    variability_label = np.where(~daylight, 0, 1 + np.searchsorted(threshold, variation, side="left"))
    lo_index, hi_index = (1, 5) if level == 0.8 else (0, 6)
    scores = np.maximum.reduce([q[:, lo_index] - d["y"], d["y"] - q[:, hi_index], np.zeros(len(q))])
    corrected = np.zeros(len(q), dtype=float)
    rows: list[dict] = []

    def make_map(reference_mask: np.ndarray, group_labels: np.ndarray, fallback_method: str = "global_all_hours") -> np.ndarray:
        out = np.zeros(len(q), dtype=float)
        global_value = conformal_rank(scores[reference_mask], level)
        if global_value is None:
            raise RuntimeError(f"No calibration support for {method}/{window_policy}/{level}")
        out[:] = global_value
        if method == "raw":
            return np.zeros(len(q), dtype=float)
        if method == "global_all_hours":
            return out
        for group in np.unique(group_labels):
            fit = reference_mask & (group_labels == group)
            value = conformal_rank(scores[fit], level) if int(fit.sum()) >= 100 else None
            if value is None:
                value = global_value
                fallback = True
            else:
                fallback = False
            out[group_labels == group] = value
            rows.append({"method": method, "window_policy": window_policy, "level": level, "group": int(group), "n": int(fit.sum()), "correction": value, "fallback": fallback, "fallback_to": fallback_method if fallback else None})
        return out

    if window_policy == "fixed_2022":
        cal = target.year == 2022
        if method == "raw":
            corrected[:] = 0.0
        elif method == "global_all_hours":
            corrected[:] = make_map(cal, np.zeros(len(q), dtype=int))
        elif method == "global_day_night":
            corrected = make_map(cal, daylight.astype(int))
        elif method == "variability_groups":
            corrected = make_map(cal, variability_label)
        elif method == "system_day_night":
            corrected = make_map(cal, system * 2 + daylight.astype(int), "global_day_night")
        elif method == "partial_pool_day_night":
            corrected = partial_pool_correction(d, q, level, cal, kappa)
        else:
            raise ValueError(method)
        return corrected, rows

    if window_policy != "rolling_90_days":
        raise ValueError(window_policy)
    corrected[:] = np.nan
    test_rows = target.year == 2023
    for month in range(1, 13):
        cutoff = pd.Timestamp(year=2023, month=month, day=1)
        end = cutoff + pd.offsets.MonthBegin(1)
        predict = test_rows & (issue >= cutoff) & (issue < end)
        if not predict.any():
            continue
        cal = (target < cutoff) & (target >= cutoff - pd.Timedelta(days=90))
        if method == "raw":
            values = np.zeros(len(q), dtype=float)
        elif method == "global_all_hours":
            values = make_map(cal, np.zeros(len(q), dtype=int))
        elif method == "global_day_night":
            values = make_map(cal, daylight.astype(int))
        elif method == "variability_groups":
            values = make_map(cal, variability_label)
        elif method == "system_day_night":
            values = make_map(cal, system * 2 + daylight.astype(int), "global_day_night")
        elif method == "partial_pool_day_night":
            values = partial_pool_correction(d, q, level, cal, kappa)
        else:
            raise ValueError(method)
        corrected[predict] = values[predict]
        rows.append({"method": method, "window_policy": window_policy, "level": level, "month": month, "calibration_n": int(cal.sum()), "prediction_n": int(predict.sum()), "cutoff": str(cutoff), "train_target_max": str(target[cal].max()) if cal.any() else None})
    if np.any(test_rows & ~np.isfinite(corrected)):
        raise RuntimeError(f"Rolling correction missing predictions for {method}/{level}")
    return corrected, rows


def p05() -> None:
    task = "P05"
    event("task_started", task, stage="calibration", message="Evaluating daylight/system/rolling calibration controls")
    metric_rows: list[dict] = []
    group_rows: list[dict] = []
    selection_rows: list[dict] = []
    methods = ["raw", "global_all_hours", "global_day_night", "variability_groups", "system_day_night", "partial_pool_day_night"]
    forecasts = ["previous_day_residual", "lightgbm", "transformer"]
    for bed in ["DKASC", "HKUST"]:
        d = load_npz(bed, 6)
        target = times(d, "target_ns")
        test_daylight = (target.year == 2023) & d["daylight"]
        previous = d["references"][:, 1].astype(float)
        # Previous-day residual quantiles fitted by system and target hour on 2021.
        q_prev = np.full((len(target), 7), np.nan, dtype=float)
        for s in np.unique(d["system"]):
            for hour in range(24):
                fit = (target.year == 2021) & (d["system"] == s) & (target.hour == hour)
                if int(fit.sum()) < 20:
                    fit = (target.year == 2021) & (d["system"] == s)
                dest = (d["system"] == s) & (target.hour == hour)
                residual = d["y"][fit] - previous[fit]
                q_prev[dest] = np.clip(previous[dest, None] + np.quantile(residual, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]), 0, 1.2)
        q_models: dict[str, np.ndarray] = {"previous_day_residual": q_prev}
        for model in ["lightgbm", "transformer"]:
            matrices = []
            for seed in [11, 23, 37, 51, 73]:
                run = PARENT / "runs" / f"{bed}_h6_{model}_s{seed}_quantile_full" / "prediction.npz"
                if not run.exists():
                    raise FileNotFoundError(run)
                with np.load(run) as z:
                    matrices.append(z["prediction"])
            q_models[model] = np.sort(np.mean(np.stack(matrices, axis=0), axis=0), axis=1)
        for forecast_name in forecasts:
            for level in [0.8, 0.9]:
                q = q_models[forecast_name]
                for policy in ["fixed_2022", "rolling_90_days"]:
                    selected_kappa, selection_trace = select_partial_pool_kappa(d, q, level, policy)
                    for selection_row in selection_trace:
                        selection_row.update({"testbed": bed, "forecast": forecast_name})
                    selection_rows.extend(selection_trace)
                    for method in methods:
                        correction, rows = fixed_or_rolling_corrections(d, q, level, method, policy, selected_kappa if method == "partial_pool_day_night" else 100.0)
                        lo_i, hi_i = (1, 5) if level == 0.8 else (0, 6)
                        lo = q[:, lo_i] - correction
                        hi = q[:, hi_i] + correction
                        mask = test_daylight & np.isfinite(lo) & np.isfinite(hi)
                        coverage = float(np.mean((d["y"][mask] >= lo[mask]) & (d["y"][mask] <= hi[mask])))
                        width = float(np.mean(hi[mask] - lo[mask]))
                        score = float(balanced(interval_score_values(d["y"][mask], lo[mask], hi[mask], level), d["system"][mask]))
                        metric_rows.append({"testbed": bed, "forecast": forecast_name, "level": level, "method": method, "window_policy": policy, "population": "daylight", "n": int(mask.sum()), "systems": int(len(np.unique(d["system"][mask]))), "coverage": coverage, "width": width, "interval_score": score, "undercoverage": float(np.mean(np.maximum(lo[mask] - d["y"][mask], 0))), "overcoverage": float(np.mean(np.maximum(d["y"][mask] - hi[mask], 0))), "fallback_rows": int(sum(bool(row.get("fallback", False)) for row in rows))})
                        for row in rows:
                            row.update({"testbed": bed, "forecast": forecast_name})
                        group_rows.extend(rows)
                event("calibration_forecast_complete", task, stage="calibration", testbed=bed, forecast=forecast_name, progress_current=len(metric_rows), progress_total=144)
    write_csv(EXEC / "aggregates/calibration_metrics.csv", metric_rows)
    write_csv(EXEC / "aggregates/calibration_groups.csv", group_rows)
    write_json(EXEC / "aggregates/calibration_selection.json", {"status": "complete", "methods": methods, "levels": [0.8, 0.9], "policies": ["fixed_2022", "rolling_90_days"], "partial_pool_kappa_candidates": [0, 100, 500, 2000], "partial_pool_kappa_selection": selection_rows, "tie_rule": "larger kappa on exact interval-score ties", "fallback_rule": "named global policy before all-hours fallback", "primary": "variability_groups versus global_day_night at 80% fixed_2022", "source_quantile_rows": "2021 only; no 2023 labels used for quantile fitting"})
    event("task_complete", task, stage="calibration", message="P05 calibration metrics complete", outputs=["aggregates/calibration_metrics.csv", "aggregates/calibration_groups.csv", "aggregates/calibration_selection.json"])


def p09() -> None:
    task = "P09"
    event("task_started", task, stage="independent_data_feasibility", message="Auditing locally available candidates without forecast scoring")
    candidates = []
    for path in ROOT.iterdir():
        if not path.is_dir():
            continue
        name = path.name.lower()
        if any(token in name for token in ["solar", "pv", "photovoltaic", "hkust", "dkasc"]):
            candidates.append({"path": str(path), "name": path.name, "metadata_only_candidate": True, "forecast_scores_examined": False})
    rows = [
        {"candidate": "DKASC Array 1B 2024", "independent_location": False, "independent_array": False, "temporal_extension": True, "lawful_local_data": True, "coverage_sufficient_for_full_year": False, "status": "existing_v29_temporal_extension", "reason": "same Array 1B; sparse August-November concentration; already source-audited"},
        {"candidate": "HKUST additional systems", "independent_location": False, "independent_array": False, "temporal_extension": False, "lawful_local_data": True, "coverage_sufficient_for_full_year": False, "status": "not_activated", "reason": "same campus and staggered source periods; would be installation replication under shared forcing"},
        {"candidate": "Local workspace candidate inventory", "independent_location": None, "independent_array": None, "temporal_extension": None, "lawful_local_data": None, "coverage_sufficient_for_full_year": None, "status": "feasibility_only", "reason": "no new provider-authorized dataset was scored in this task"},
    ]
    write_csv(EXEC / "registry/candidate_registry.csv", rows)
    (EXEC / "registry/independent_data_feasibility.md").write_text("# Independent-data feasibility\n\nThe current workspace contains the adopted Array 1B/HKUST study and historical PVOD materials. No new independent provider-authorized dataset was scored during this execution. The 2024 Array 1B extension is temporal evidence on the same array and is not independent geographical replication. Additional HKUST systems would remain one campus under shared weather forcing.\n\nA future independent experiment requires metadata-only candidate selection, lawful local access, documented capacity/time semantics, enough chronological development/calibration/test coverage, and a predeclared evaluation period. The candidate must be locked before forecast outcomes are inspected.\n", encoding="utf-8")
    write_json(EXEC / "registry/scope_decision_p09.json", {"status": "feasibility_only", "independent_scoring_activated": False, "reason": "No qualifying new independent dataset was available and authorized within the current workspace; current paper claims remain bounded to Array 1B and HKUST.", "candidate_directory_scan": candidates})
    event("task_complete", task, stage="independent_data_feasibility", message="P09 feasibility documented; no independent scores activated", outputs=["registry/candidate_registry.csv", "registry/independent_data_feasibility.md", "registry/scope_decision_p09.json"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["P00", "P01", "P02", "P05", "P09", "all-ready"])
    args = parser.parse_args()
    if args.task in ["P00", "all-ready"]:
        p00()
    if args.task in ["P01", "all-ready"]:
        p01()
    if args.task in ["P02", "all-ready"]:
        p02()
    if args.task == "P05":
        p05()
    if args.task == "P09":
        p09()


if __name__ == "__main__":
    main()
