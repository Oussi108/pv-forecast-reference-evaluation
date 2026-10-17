"""Build a conservative site-file to Brick-system mapping for HKUST.

Only normalized exact matches and one documented UK/US spelling alias are accepted.
Unmatched files remain ``review``; no fuzzy matching is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_map(data_root: Path, metadata_csv: Path, audit_json: Path) -> list[dict[str, Any]]:
    metadata = list(csv.DictReader(metadata_csv.open(encoding="utf-8", newline="")))
    by_norm = {normalise(row["system_id"]): row for row in metadata}
    # The dataset uses American ``Center`` in Brick IDs and British ``Centre`` in
    # the site CSV filename. This is the only non-exact alias permitted here.
    aliases = {normalise("Indoor Sports Centre"): "Indoor_Sport_Center"}
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    audit_by_file = {row["file"]: row for row in audit.get("files", [])}
    site_files = sorted(
        path for path in data_root.rglob("*.csv")
        if path.is_file() and "site level dataset" in path.as_posix().lower()
    )
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for path in site_files:
        relative = path.relative_to(data_root).as_posix()
        stem = path.stem
        candidate_id = aliases.get(normalise(stem))
        method = "explicit_alias" if candidate_id else "normalized_exact"
        if candidate_id is None:
            metadata_row = by_norm.get(normalise(stem))
        else:
            metadata_row = next((row for row in metadata if row["system_id"] == candidate_id), None)
        status = "mapped" if metadata_row is not None else "review"
        system_id = metadata_row["system_id"] if metadata_row else None
        if system_id and system_id in used:
            status = "review"
            method = "duplicate_system_id"
        if system_id:
            used.add(system_id)
        audit_row = audit_by_file.get(relative, {})
        rows.append({
            "site_file": relative,
            "site_name": stem,
            "system_id": system_id,
            "mapping_status": status,
            "mapping_method": method if metadata_row else "unmatched",
            "audit_status": audit_row.get("status"),
            "capacity_kw": metadata_row.get("capacity_kw") if metadata_row else None,
            "connection_date": metadata_row.get("connection_date") if metadata_row else None,
            "location": metadata_row.get("location") if metadata_row else None,
            "azimuth": metadata_row.get("azimuth") if metadata_row else None,
            "tilt": metadata_row.get("tilt") if metadata_row else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_map(args.data_root.resolve(), args.metadata_csv.resolve(), args.audit_json.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["site_file", "mapping_status"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mapped = sum(row["mapping_status"] == "mapped" for row in rows)
    audit_pass = sum(row["audit_status"] == "pass" for row in rows)
    print(json.dumps({"site_files": len(rows), "mapped": mapped,
                      "unmatched_or_duplicate": len(rows) - mapped,
                      "audit_pass": audit_pass}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

