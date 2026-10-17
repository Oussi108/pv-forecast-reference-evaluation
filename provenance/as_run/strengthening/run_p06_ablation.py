"""Run the declared five-seed feature-removal extension in an isolated root."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"E:\master in china\s2\research\solar energy forecasting")
RELEASE = ROOT / "WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting"
EXEC = ROOT / "tmp/paper1_strengthening_20260914"
PRIVATE = EXEC / "private_data/p06"
PUBLIC = EXEC / "aggregates/p06_runs"
PRIVATE.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)
os.environ["PV_FORECAST_PROTOCOL_PATH"] = str(RELEASE / "configs/corrected_study_v28_array1b.json")
sys.path.insert(0, str(RELEASE / "scripts"))
import train_corrected_study as trainer  # noqa: E402

trainer.PUBLIC = PUBLIC
trainer.PROTOCOL = json.loads((RELEASE / "configs/corrected_study_v28_array1b.json").read_text())
trainer.PROTOCOL_PATH = RELEASE / "configs/corrected_study_v28_array1b.json"

def main() -> None:
    beds = ["DKASC", "HKUST"]
    variants = ["without_power", "without_irradiance", "without_other_weather", "without_solar", "without_calendar"]
    seeds = [11, 23, 37, 51, 73]
    for bed in beds:
        for horizon in [6]:
            for seed in seeds:
                for variant in variants:
                    trainer.run_one(PRIVATE, bed, horizon, "lightgbm", seed, variant=variant)
                    trainer.run_one(PRIVATE, bed, horizon, "gru", seed, variant=variant)
    (EXEC / "logs/p06_complete.json").write_text(json.dumps({"status":"complete","beds":beds,"horizon":6,"seeds":seeds,"variants":variants},indent=2),encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (EXEC / "logs/p06_failed.json").write_text(json.dumps({"status":"failed","error":repr(exc)},indent=2),encoding="utf-8")
        raise
