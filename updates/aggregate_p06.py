"""Aggregate completed P06 seed-level feature-removal records."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pv_paths import REPO, RESULTS  # noqa: E402
ROOT = REPO
P06 = RESULTS / "updates_2023/p06_runs"
PARENT = REPO / "results/corrected_v28_array1b/runs"


def parent_mae(bed: str, model: str, seed: int) -> float | None:
    path = PARENT / f"{bed}_h6_{model}_s{seed}_point_full.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["test_system_balanced_mae"])


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted((P06 / "runs").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        bed, model, seed, variant = row["testbed"], row["model"], int(row["seed"]), row["variant"]
        base = parent_mae(bed, model, seed)
        row.update({"baseline_full_mae": base, "removal_cost_mae": None if base is None else float(row["test_system_balanced_mae"] - base), "record_path": str(path)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    P06.mkdir(parents=True, exist_ok=True)
    frame.to_csv(P06 / "p06_seed_records.csv", index=False)
    # Stable declared alias for the seed-level evidence table.  The alias is
    # intentionally separate from the historical worker filename so downstream
    # reviewers have an explicit contract.
    frame.to_csv(P06 / "ablation_seed_metrics.csv", index=False)
    if frame.empty:
        return
    summary = frame.groupby(["testbed", "model", "variant"], as_index=False).agg(
        completed_seeds=("seed", "nunique"), seed_mae_mean=("test_system_balanced_mae", "mean"),
        seed_mae_sd=("test_system_balanced_mae", "std"), mean_removal_cost=("removal_cost_mae", "mean"),
        removal_cost_sd=("removal_cost_mae", "std"), runtime_mean_seconds=("runtime_seconds", "mean"),
        all_status=("status", lambda x: ";".join(sorted(set(x)))),
    )
    summary.to_csv(P06 / "p06_summary.csv", index=False)
    stable = frame.groupby(["testbed", "model", "variant"], as_index=False).agg(
        completed_seeds=("seed", "nunique"), positive_cost_fraction=("removal_cost_mae", lambda x: float((x > 0).mean())),
        min_cost=("removal_cost_mae", "min"), max_cost=("removal_cost_mae", "max"),
    )
    stable.to_csv(P06 / "p06_seed_spread.csv", index=False)
    paired = frame.groupby(["testbed", "model", "variant"], as_index=False).agg(
        paired_n=("removal_cost_mae", "count"),
        paired_effect_mean=("removal_cost_mae", "mean"),
        paired_effect_sd=("removal_cost_mae", "std"),
        paired_positive_fraction=("removal_cost_mae", lambda x: float((x > 0).mean())),
        paired_effect_min=("removal_cost_mae", "min"),
        paired_effect_max=("removal_cost_mae", "max"),
    )
    paired["inference_status"] = "descriptive_only"
    paired["inference_eligible"] = False
    paired["interval_method"] = "not_estimated"
    paired["p_value"] = float("nan")
    paired["ci_low"] = float("nan")
    paired["ci_high"] = float("nan")
    paired["reason"] = "Five training seeds support paired descriptive spread only; no inferential interval or multiplicity claim is authorized by P06."
    paired.to_csv(P06 / "ablation_paired_inference.csv", index=False)

    feature_map = {
        "status": "complete",
        "task_id": "P06",
        "scope": "current Array 1B/DKASC plus 31-system HKUST; h6 point forecasts; LightGBM and GRU; five seeds",
        "feature_order": ["power", "ghi", "temperature", "humidity", "clear_ghi"],
        "variants": {
            "without_power": {"removed": ["history.power"], "retained": "all other history and future channels"},
            "without_irradiance": {"removed": ["history.ghi"], "retained": "all other history and future channels"},
            "without_other_weather": {"removed": ["history.temperature", "history.humidity"], "retained": "power, ghi, clear_ghi history plus future inputs"},
            "without_solar": {"removed": ["history.clear_ghi", "future.target_clear_ghi"], "retained": "power, ghi, temperature, humidity history plus calendar inputs"},
            "without_calendar": {"removed": ["future.hour_harmonics", "future.day_of_year_harmonics"], "retained": "future.target_clear_ghi and all history channels"},
        },
        "source_files": [
            {"path": "scripts/train_corrected_study.py", "sha256": sha256(ROOT / "scripts/train_corrected_study.py")},
            {"path": "scripts/corrected_study.py", "sha256": sha256(ROOT / "scripts/corrected_study.py")},
            {"path": "configs/corrected_study_v28_array1b.json", "sha256": sha256(ROOT / "configs/corrected_study_v28_array1b.json")},
        ],
        "interpretation": "Removal costs are conditional prediction diagnostics, not causal physical contributions, additive importance scores, or universal rankings.",
    }
    (P06 / "feature_map.json").write_text(json.dumps(feature_map, indent=2, sort_keys=True), encoding="utf-8")
    (P06 / "progress.json").write_text(json.dumps({"status": "complete", "records": len(frame), "groups": len(summary), "seeds": sorted(frame.seed.unique().tolist()), "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "supersedes": "stale running record from initial launcher"}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(frame), "groups": len(summary), "outputs": ["p06_seed_records.csv", "ablation_seed_metrics.csv", "p06_summary.csv", "p06_seed_spread.csv", "ablation_paired_inference.csv", "feature_map.json", "progress.json"]}))


if __name__ == "__main__":
    main()
