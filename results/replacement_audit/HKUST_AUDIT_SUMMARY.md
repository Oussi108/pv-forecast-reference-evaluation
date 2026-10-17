# HKUST archive audit summary

Audit run: 2026-09-07 against the supplied Dryad archive (September 2024 version).
The raw archive was extracted outside the release tree. This summary and the JSON
report contain no observations and redact the local input path.

## What was found

The archive contains 124 CSV files and one rainfall workbook:

- 60 site-level PV exports (used for target screening);
- 44 inverter-level exports (excluded from the site count to avoid
  pseudoreplication; they can be used later to validate aggregation);
- 20 shared meteorological files (2021–2023);
- one rainfall workbook requiring an explicit parser decision.

The site-level files use `Time,generation(kWh),power(W)`. Weather is stored in
separate files with unit-bearing headers such as `Irradiance (W/m2)`, `Temp (Degree
Celsius)`, `RH (%)`, `SLP (hPa)` and `Wind Speed (m/s)`.

The provider README reports five-minute PV sampling, but the site-level exports
audited here have mixed observed cadences: 37 files are 15-minute and 23 files are
60-minute by the median positive timestamp difference. This is a material ingest
finding, not an error to hide. The planned analysis therefore harmonises retained
site targets to an hourly grid, with interpolation restricted to each chronological
split. The README gives Hong Kong civil timestamps but no explicit timezone column;
the audit provisionally localises naive timestamps as `Asia/Hong_Kong`, pending
confirmation in the final data-access record.

## Provisional coverage screen

The audit thresholds were:

- at least 700 calendar days spanning the file;
- at least 500 observed days before 2023;
- at least 300 observed days and 10 observed months in 2023;
- no more than 10% missing site-level power values.

Results:

| Category | Count |
|---|---:|
| Site-level files scanned | 60 |
| Site-level files passing the provisional screen | 34 |
| Site-level files requiring review | 26 |
| Site-level files with scan errors | 0 |
| Weather files scanned | 20 |

The 34 passing files are the initial candidate pool. They include both optimizer
and non-optimizer campus systems and are not yet a final analysis fleet.

The conservative filename-to-Brick-system mapping then matched all 60 site-level
files to unique metadata system IDs (59 normalized exact matches plus one explicit
`Centre`/`Center` spelling alias). The mapping is recorded in
`hkust_system_map.csv`; it does not decide which systems enter the experiment.

## Interpretation and remaining gate

The result is **promising but not a final pass**. The 34 candidates provide more
than enough systems for a campus-fleet replication, but the following must still be
completed before model fitting:

1. review the completed file-to-system mapping and verify capacities from the metadata;
2. join the shared weather files without future-information leakage;
3. decide whether the rainfall workbook is included in the common contract;
4. verify timestamps, units, daylight conventions and five-minute aggregation;
5. document the retained systems without selecting them by forecasting performance;
6. run the causal preprocessing and target-key audit for the matched 2021/2022/2023
   development–calibration–test protocol.

The complete machine-readable report is `hkust_audit.json`; the file-level table is
`hkust_audit_files.csv`. Read `docs/HKUST_REPLACEMENT_PLAN.md` before promoting the
candidate to a primary testbed.
