"""Audit an extracted HKUST PV archive before it enters the forecasting study.

The script is intentionally an audit, not an ingest or downloader.  It scans CSV
files in a user-supplied directory in chunks and writes a machine-readable report
with file-level coverage, timestamp cadence, candidate weather columns and missing
values.  Raw third-party observations stay outside the repository.

Example
-------
    python scripts/audit_hkust_dataset.py --data-root D:/data/hkust

The report is a gate: a file marked ``pass`` is only a *candidate* until the
authors review units, inverter-to-system mapping, timezone conventions and the
chronological split.  The script does not fit a model or inspect future labels
outside the supplied archive.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "results" / "replacement_audit" / "hkust_audit.json"
DEFAULT_CONFIG = REPO / "configs" / "hkust_dataset.yaml"

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date_time", "date", "DateTime"),
    "power": ("power", "ac_power", "active_power", "pac", "pv_power", "power_kw", "power_w"),
    "irradiance": ("ghi", "irradiance", "poa", "poa_irradiance", "solar_irradiance", "global_irradiance"),
    "temperature": ("temp", "temperature", "air_temperature", "temp_air", "ambient_temperature"),
    "humidity": ("humidity", "relative_humidity", "rh"),
    "wind": ("wind", "wind_speed", "windspeed"),
    "pressure": ("pressure", "sea_level_pressure", "slp", "barometric_pressure"),
    "visibility": ("visibility", "visibility_km", "visibility_m", "vis"),
    "rainfall": ("rainfall", "rain", "precipitation", "precip"),
}

WEATHER_KEYS = ("irradiance", "temperature", "humidity", "wind", "pressure", "visibility", "rainfall")


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_normalised = {_normalise(c): c for c in columns}
    for candidate in candidates:
        found = by_normalised.get(_normalise(candidate))
        if found is not None:
            return found
    # HKUST appends units to most weather headers, e.g. ``Irradiance (W/m2)``.
    # Treat a candidate as matched when it is an unambiguous prefix after unit
    # punctuation is removed, while retaining exact-match precedence above.
    normalised_columns = list(by_normalised.items())
    for candidate in candidates:
        wanted = _normalise(candidate)
        matches = [column for normalised, column in normalised_columns
                   if normalised.startswith(wanted)]
        if len(matches) == 1:
            return matches[0]
    return None


def _classify_files(root: Path, include_inverter: bool = False) -> tuple[list[Path], list[Path], list[Path]]:
    """Return target site files, weather files, and intentionally ignored files.

    HKUST ships both site-level and inverter-level exports.  Site-level exports are
    the default target because counting every inverter as an independent site would
    create pseudoreplication.  The weather files are audited separately because they
    do not contain a power column.
    """
    csv_files = sorted(p for p in root.rglob("*.csv") if p.is_file())
    targets, weather, ignored = [], [], []
    for path in csv_files:
        lower = path.as_posix().lower()
        if "meteorological dataset" in lower:
            weather.append(path)
        elif "site level dataset" in lower or (include_inverter and "inverter level dataset" in lower):
            targets.append(path)
        else:
            ignored.append(path)
    return targets, weather, ignored


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit_file(path: Path, root: Path, test_year: int, chunk_rows: int,
                max_rows: int | None, maximum_missing_fraction: float,
                minimum_operating_days: int, minimum_pretest_days: int,
                minimum_test_days: int, minimum_test_months: int,
                role: str = "target") -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "file": path.relative_to(root).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "status": "fail",
        "error": None,
        "columns": {},
        "rows_scanned": 0,
        "valid_timestamp_rows": 0,
        "power_non_null_rows": 0,
        "power_missing_fraction": None,
        "weather_non_null_fraction": {},
        "start_utc": None,
        "end_utc": None,
        "span_days": None,
        "pretest_days_observed": 0,
        "test_year_days_observed": 0,
        "test_year_months_observed": 0,
        "median_cadence_minutes": None,
        "candidate_reasons": [],
        "audit_started_utc": started,
    }
    try:
        header = pd.read_csv(path, nrows=0)
        columns = [str(c) for c in header.columns]
        selected = {key: _find_column(columns, aliases) for key, aliases in ALIASES.items()}
        result["columns"] = selected
        required = ["timestamp", "power"] if role == "target" else ["timestamp"]
        missing_weather = role == "weather" and all(selected[key] is None for key in WEATHER_KEYS)
        if missing_weather:
            required.append("weather field")
        if any(selected[key] is None for key in required if key in selected) or missing_weather:
            missing = [key for key in required if key in selected and selected[key] is None]
            if missing_weather:
                missing.append("weather field")
            result["candidate_reasons"].append("missing required column(s): " + ", ".join(missing))
            return result

        usecols = [column for column in selected.values() if column is not None]
        # Avoid duplicate usecols when an alias maps two logical fields to one column.
        usecols = list(dict.fromkeys(usecols))
        first_time: pd.Timestamp | None = None
        last_time: pd.Timestamp | None = None
        cadence_values: list[float] = []
        test_days: set[str] = set()
        pretest_days: set[str] = set()
        test_months: set[str] = set()
        weather_counts = {key: 0 for key in WEATHER_KEYS
                          if selected[key] is not None}
        row_limit = max_rows if max_rows is not None and max_rows > 0 else None

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows,
                                 low_memory=False):
            if row_limit is not None:
                remaining = row_limit - result["rows_scanned"]
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]
            n = len(chunk)
            result["rows_scanned"] += n
            timestamp_col = selected["timestamp"]
            power_col = selected["power"]
            assert timestamp_col is not None
            timestamps = pd.to_datetime(chunk[timestamp_col], errors="coerce")
            # The HKUST README gives local Hong Kong civil timestamps but no
            # explicit timezone field. Record the declared source timezone here,
            # then convert to UTC for all coverage calculations.
            if getattr(timestamps.dt, "tz", None) is None:
                timestamps = timestamps.dt.tz_localize("Asia/Hong_Kong")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")
            valid = timestamps.notna()
            valid_timestamps = timestamps[valid]
            result["valid_timestamp_rows"] += int(valid.sum())
            if role == "target":
                assert power_col is not None
                result["power_non_null_rows"] += int(chunk[power_col].notna().sum())
            for key, column in selected.items():
                if key in weather_counts and column is not None:
                    weather_counts[key] += int(chunk[column].notna().sum())
            if not valid_timestamps.empty:
                current_first, current_last = valid_timestamps.iloc[0], valid_timestamps.iloc[-1]
                first_time = current_first if first_time is None else min(first_time, current_first)
                last_time = current_last if last_time is None else max(last_time, current_last)
                # Keep cadence computation bounded while still catching obvious source frequency.
                sample = valid_timestamps.iloc[: min(2000, len(valid_timestamps))].sort_values()
                if len(sample) > 1:
                    diffs = sample.diff().dropna().dt.total_seconds().div(60)
                    cadence_values.extend(diffs[(diffs > 0) & (diffs < 24 * 60)].tolist())
                test_timestamps = valid_timestamps[valid_timestamps.dt.year == test_year]
                test_days.update(test_timestamps.dt.strftime("%Y-%m-%d").tolist())
                test_months.update(test_timestamps.dt.strftime("%Y-%m").tolist())
                pretest_days.update(valid_timestamps[valid_timestamps.dt.year < test_year]
                                    .dt.strftime("%Y-%m-%d").tolist())
            if row_limit is not None and result["rows_scanned"] >= row_limit:
                break

        rows = result["rows_scanned"]
        valid_rows = result["valid_timestamp_rows"]
        if role == "target":
            result["power_missing_fraction"] = round(
                1.0 - (result["power_non_null_rows"] / rows), 6) if rows else None
        result["weather_non_null_fraction"] = {
            key: round(value / rows, 6) if rows else None
            for key, value in weather_counts.items()
        }
        result["start_utc"] = first_time.isoformat() if first_time is not None else None
        result["end_utc"] = last_time.isoformat() if last_time is not None else None
        if first_time is not None and last_time is not None:
            result["span_days"] = round((last_time - first_time).total_seconds() / 86400, 3)
        result["test_year_days_observed"] = len(test_days)
        result["pretest_days_observed"] = len(pretest_days)
        result["test_year_months_observed"] = len(test_months)
        if cadence_values:
            result["median_cadence_minutes"] = round(float(pd.Series(cadence_values).median()), 3)

        reasons: list[str] = []
        if valid_rows == 0:
            reasons.append("no parseable timestamps")
        if result["span_days"] is None or result["span_days"] < minimum_operating_days:
            reasons.append(f"span below {minimum_operating_days} operating days")
        if result["pretest_days_observed"] < minimum_pretest_days:
            reasons.append(f"fewer than {minimum_pretest_days} pre-{test_year} observed days")
        if result["test_year_days_observed"] < minimum_test_days:
            reasons.append(f"fewer than {minimum_test_days} observed days in {test_year}")
        if result["test_year_months_observed"] < minimum_test_months:
            reasons.append(f"fewer than {minimum_test_months} observed months in {test_year}")
        missing_fraction = result["power_missing_fraction"]
        if role == "target" and (missing_fraction is None or missing_fraction > maximum_missing_fraction):
            reasons.append(f"power missingness exceeds {maximum_missing_fraction:.0%}")
        result["candidate_reasons"] = reasons
        result["status"] = "pass" if not reasons else "review"
        return result
    except Exception as exc:  # report a file-level failure and continue the audit
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["candidate_reasons"].append("file could not be scanned")
        return result


def audit(root: Path, *, test_year: int = 2023, chunk_rows: int = 100_000,
          max_rows: int | None = None, maximum_missing_fraction: float = 0.10,
          minimum_operating_days: int = 700, minimum_pretest_days: int = 500,
          minimum_test_days: int = 300, minimum_test_months: int = 10,
          include_inverter: bool = False) -> dict[str, Any]:
    """Return a deterministic audit report for an extracted archive directory."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HKUST data root is not a directory: {root}")
    target_files, weather_files, ignored_files = _classify_files(root, include_inverter)
    target_records = [_audit_file(path, root, test_year, chunk_rows, max_rows,
                           maximum_missing_fraction, minimum_operating_days,
                           minimum_pretest_days, minimum_test_days,
                           minimum_test_months, role="target") for path in target_files]
    weather_records = [_audit_file(path, root, test_year, chunk_rows, max_rows,
                             maximum_missing_fraction, minimum_operating_days,
                             minimum_pretest_days, minimum_test_days,
                             minimum_test_months, role="weather") for path in weather_files]
    records = target_records + weather_records
    passed = sum(record["status"] == "pass" for record in target_records)
    review = sum(record["status"] == "review" for record in target_records)
    failed = sum(record["status"] == "fail" for record in target_records)
    return {
        "schema_version": "hkust-audit-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "HKUST rooftop PV dataset",
            "doi": "10.5061/dryad.m37pvmd99",
            "license": "CC0-1.0",
            "paper_doi": "10.1038/s41597-025-04397-y",
        },
        "input": {
            "root": str(root),
            "csv_files_scanned": len(records),
            "target_files_scanned": len(target_files),
            "weather_files_scanned": len(weather_files),
            "ignored_csv_files": len(ignored_files),
            "xlsx_files": len(list(root.rglob("*.xlsx"))),
            "inverter_files_included": include_inverter,
            "test_year": test_year,
            "chunk_rows": chunk_rows,
            "max_rows": max_rows,
        },
        "thresholds": {
            "minimum_operating_days": minimum_operating_days,
            "minimum_pretest_days": minimum_pretest_days,
            "minimum_test_days": minimum_test_days,
            "minimum_test_months": minimum_test_months,
            "maximum_power_missing_fraction": maximum_missing_fraction,
        },
        "summary": {
            "pass_files": passed,
            "review_files": review,
            "failed_files": failed,
            "eligible_target_files": passed,
            # A green gate requires every scanned CSV to pass.  A mixture of
            # pass/review files remains a review state so it cannot be mistaken
            # for an accepted fleet during a later automated run.
            "gate": "pass" if target_records and passed == len(target_records) and failed == 0 else "review",
            "interpretation": "Target files are screened separately from shared weather files; system mapping and protocol review remain required.",
        },
        "files": records,
    }


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = report.get("files", [])
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["file", "status", "bytes", "rows_scanned", "start_utc", "end_utc",
            "span_days", "test_year_days_observed", "median_cadence_minutes",
            "power_missing_fraction", "candidate_reasons"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in keys}
            out["candidate_reasons"] = "; ".join(row.get("candidate_reasons", []))
            writer.writerow(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory containing the extracted HKUST CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"JSON report path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="optional scan cap per file for a quick smoke audit")
    parser.add_argument("--test-year", type=int, default=2023)
    parser.add_argument("--minimum-operating-days", type=int, default=700)
    parser.add_argument("--minimum-pretest-days", type=int, default=500)
    parser.add_argument("--minimum-test-days", type=int, default=300)
    parser.add_argument("--minimum-test-months", type=int, default=10)
    parser.add_argument("--include-inverter", action="store_true",
                        help="also scan inverter-level exports (not counted as independent systems)")
    parser.add_argument("--redact-root", action="store_true",
                        help="replace the local input path with <external-data-root> in the JSON report")
    args = parser.parse_args(argv)
    report = audit(args.data_root, test_year=args.test_year, chunk_rows=args.chunk_rows,
                   max_rows=args.max_rows,
                   minimum_operating_days=args.minimum_operating_days,
                   minimum_pretest_days=args.minimum_pretest_days,
                   minimum_test_days=args.minimum_test_days,
                   minimum_test_months=args.minimum_test_months,
                   include_inverter=args.include_inverter)
    if args.redact_root:
        report["input"]["root"] = "<external-data-root>"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(report, output.with_name(output.stem + "_files.csv"))
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
