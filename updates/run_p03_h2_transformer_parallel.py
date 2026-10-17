"""Run the fixed HKUST h2 Transformer bundles in isolated seed processes.

This is an execution-scheduling helper only. It calls the same ``bundle``
function, with the same data, seeds, policies, cutoffs, and inherited epoch
settings as the canonical sequential runner. Existing atomic checkpoints are
reused. The helper exists to avoid a multi-hour sequential bottleneck; it does
not change the scientific protocol.
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


def run_seed(seed: int) -> dict:
    runner = importlib.import_module("run_p03_updates")
    started = time.time()
    runner.bundle("HKUST", 2, "transformer", seed, 2023)
    return {"seed": seed, "pid": os.getpid(), "elapsed_s": round(time.time() - started, 3), "status": "complete"}


def main() -> int:
    seeds = [11, 23, 37, 51, 73]
    max_workers = 2
    print(json.dumps({"event": "parallel_stage_started", "stage": "P03B_HKUST_H2_TRANSFORMER", "seeds": seeds, "max_workers": max_workers, "protocol": "canonical_bundle_with_atomic_checkpoint_reuse"}), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # fail closed; the verifier will also reject missing bundles
                print(json.dumps({"event": "seed_failed", "seed": seed, "error": repr(exc)}), flush=True)
                raise
            results.append(result)
            print(json.dumps({"event": "seed_completed", **result}), flush=True)

    # Rebuild the same h2 neural summary that the canonical runner emits.
    runner = importlib.import_module("run_p03_updates")
    runner.summarize(["DKASC", "HKUST"], [2], [2023])
    print(json.dumps({"event": "parallel_stage_completed", "stage": "P03B_HKUST_H2_TRANSFORMER", "results": sorted(results, key=lambda row: row["seed"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
