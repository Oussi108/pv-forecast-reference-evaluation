"""Write protocol/FREEZE_v44.json: SHA-256 of the v44 protocol and of the scripts that score it. Run once, then commit."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['protocol/PROTOCOL_v44_extensions.md', 'analysis/x02_optimal_reference.py', 'analysis/x03_drift_updates.py',
         'analysis/x04_seed_bootstrap.py', 'analysis/x05_conditional_calibration.py', 'analysis/x06_run_9068.py']
out = ROOT / 'protocol/FREEZE_v44.json'
if out.exists():
    raise SystemExit('FREEZE_v44.json already exists; a freeze is never rewritten')
record = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
record['_frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
record['_statement'] = 'Frozen before any v44 extension result on 2023 or 2024 data was computed.'
out.write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record, indent=1))
