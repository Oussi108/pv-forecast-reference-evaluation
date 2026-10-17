"""Run the remaining h4/h12 neural bundles with canonical isolated workers.

This is execution scheduling only. Every task calls the unchanged canonical
``run_p03_updates.bundle`` function with the declared bed, horizon, model,
seed and year. No data, model, policy, cutoff or epoch setting is changed.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


TASKS = [
    (bed, horizon, model, seed, 2023)
    for bed in ["DKASC", "HKUST"]
    for horizon in [4, 12]
    for model in ["gru", "transformer"]
    for seed in [11, 23, 37, 51, 73]
]


def run_task(task: tuple[str, int, str, int, int]) -> dict:
    bed, horizon, model, seed, year = task
    runner = importlib.import_module("run_p03_updates")
    started = time.time()
    runner.bundle(bed, horizon, model, seed, year)
    return {
        "bed": bed,
        "horizon": horizon,
        "model": model,
        "seed": seed,
        "year": year,
        "pid": os.getpid(),
        "elapsed_s": round(time.time() - started, 3),
        "status": "complete",
    }


def main() -> int:
    max_workers = 2
    print(json.dumps({
        "event": "parallel_stage_started",
        "stage": "P03C_H4_H12_NEURAL",
        "task_count": len(TASKS),
        "max_workers": max_workers,
        "protocol": "canonical_bundle_with_atomic_checkpoint_reuse",
    }), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, task): task for task in TASKS}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(json.dumps({"event": "task_failed", "task": task, "error": repr(exc)}), flush=True)
                raise
            results.append(result)
            print(json.dumps({"event": "task_completed", **result}), flush=True)

    runner = importlib.import_module("run_p03_updates")
    runner.summarize(["DKASC", "HKUST"], [4, 12], [2023])
    print(json.dumps({
        "event": "parallel_stage_completed",
        "stage": "P03C_H4_H12_NEURAL",
        "results": sorted(results, key=lambda row: (row["bed"], row["horizon"], row["model"], row["seed"])),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
