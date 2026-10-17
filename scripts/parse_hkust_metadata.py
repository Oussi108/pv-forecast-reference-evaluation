"""Extract a reviewable system table from the HKUST Brick metadata file.

The parser is intentionally conservative: it extracts identifiers and metadata but
does not invent a filename-to-system mapping.  That mapping is reviewed against the
site-level CSV names before ingestion, preventing accidental inverter
pseudoreplication or capacity mismatches.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "results" / "replacement_audit" / "hkust_metadata.csv"


def _value(pattern: str, block: str) -> str | None:
    match = re.search(pattern, block, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def parse_metadata(path: Path) -> list[dict[str, Any]]:
    """Parse PV generation-system blocks from a Brick/Turtle metadata file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.finditer(
        r"pvsystem:(\S+)\s+a\s+brick:PV_Generation_System\s*;(.*?)(?=\n\s*\npvsystem:|\Z)",
        text,
        flags=re.DOTALL,
    )
    rows: list[dict[str, Any]] = []
    for match in blocks:
        system_id, block = match.group(1), match.group(2)
        capacity = _value(r"ext:ratedPowerOutput\s+\[.*?brick:value\s+([0-9.]+)", block)
        rows.append({
            "system_id": system_id,
            "location": _value(r"brick:hasLocation\s+\[\s*brick:value\s+\"([^\"]+)\"", block),
            "latitude": _value(r"brick:latitude\s+([0-9.+-]+)", block),
            "longitude": _value(r"brick:longitude\s+([0-9.+-]+)", block),
            "altitude_m": _value(r"ext:altitude\s+\[.*?brick:value\s+([0-9.]+)", block),
            "connection_date": _value(r"ext:connectionDate\s+\[.*?brick:value\s+\"([^\"]+)\"", block),
            "capacity_kw": capacity,
            "azimuth": _value(r"ext:azimuth\s+\[\s*brick:value\s+\"([^\"]+)\"", block),
            "tilt": _value(r"ext:tiltAngle\s+\[\s*brick:value\s+\"([^\"]+)\"", block),
            "inverter_resource": _value(r"brick:hasPart\s+([^;]+)", block),
        })
    rows.sort(key=lambda row: row["system_id"])
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system_id", "location", "latitude", "longitude", "altitude_m",
              "connection_date", "capacity_kw", "azimuth", "tilt", "inverter_resource"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None,
                        help="optional JSON summary path")
    args = parser.parse_args(argv)
    source = args.metadata_file.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Metadata file not found: {source}")
    rows = parse_metadata(source)
    output = args.output.expanduser().resolve()
    write_csv(rows, output)
    summary = {
        "schema_version": "hkust-metadata-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"dataset_doi": "10.5061/dryad.m37pvmd99",
                   "metadata_file": "PV generation system metadata.ttl"},
        "systems_parsed": len(rows),
        "missing_capacity": sum(row["capacity_kw"] is None for row in rows),
        "missing_connection_date": sum(row["connection_date"] is None for row in rows),
        "duplicate_locations": sorted({location for location in
            (row["location"] for row in rows if row["location"])
            if sum(other["location"] == location for other in rows) > 1}),
        "note": "Review against site-level CSV names before ingestion; no automatic mapping was inferred.",
    }
    summary_output = args.summary_output or output.with_name(output.stem + "_summary.json")
    summary_output = summary_output.expanduser().resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

