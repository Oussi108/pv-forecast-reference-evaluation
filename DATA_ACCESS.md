# Data access

No observations are distributed with this repository. All inputs are public, but each comes under its provider's terms.
Put them in a private folder **outside** the repository and point the code to it:

```bash
export PV_PRIVATE_ROOT=/path/to/pv_private        # Windows PowerShell: $env:PV_PRIVATE_ROOT = "D:\pv_private"
```

Expected layout (the code creates all other sub-folders itself):

```
pv_private/
  raw/DKASC/87-Site_DKA-M9_A+C-Phases.csv
  raw/HKUST/Dataset/...                    (extracted Dryad archive)
  paper1_v43_private/raw/...               (filled by analysis/a01_download.py)
```

## 1. DKASC Array 1B (Alice Springs, Australia)

- **Source:** DKA Solar Centre, site "DKA-M9 A+C-Phases" (Array 1B, 23.4 kW, Trina mono-Si, dual-axis tracker, installed 2009),
  <https://dkasolarcentre.com.au/source/alice-springs/dka-m9-a-c-phases>.
- **File:** `87-Site_DKA-M9_A+C-Phases.csv` (5-minute data). The study used the download with SHA-256
  `adebe983a30f58c5900247308790ed0659de9fdff2a4fd59f6e464147e3c2d14`. A later download may contain more recent records;
  the code uses 2021–2023 for the main study and 2024 for the sparse extension.
- **Terms:** free use with mandatory acknowledgement of the DKA Solar Centre as the source. Redistribution of unaltered
  data is limited, so the file is not included here.
- **Place at:** `$PV_PRIVATE_ROOT/raw/DKASC/87-Site_DKA-M9_A+C-Phases.csv`.

## 2. HKUST rooftop photovoltaic dataset (Hong Kong)

- **Source:** Lin et al. (2025), *Scientific Data*, <https://doi.org/10.1038/s41597-025-04397-y>. The data are on Dryad,
  <https://doi.org/10.5061/dryad.m37pvmd99>; the study used the version of 8 September 2024.
- **Licence:** CC0 1.0.
- **Place at:** extract the archive into `$PV_PRIVATE_ROOT/raw/HKUST/`, so that the folder `Dataset/` sits directly under it.
- **Screening:** the systems used (31 of 60) and the column mapping are fixed in `results/replacement_audit/`
  (`hkust_audit.json`, `hkust_system_map.csv`). They were produced by `scripts/audit_hkust_dataset.py`,
  `scripts/parse_hkust_metadata.py` and `scripts/build_hkust_system_map.py`.

## 3. NREL PVDAQ system 9068 (Kersey, Colorado, USA)

- **Source:** PVDAQ public datasets on the Open Energy Data Initiative (OEDI),
  <https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/9068_OEDI/>; dataset DOI
  <https://doi.org/10.25984/1846021>.
- **Licence:** CC BY 4.0.
- **Download:** `python analysis/a01_download.py` downloads the seven data files and the system metadata into
  `$PV_PRIVATE_ROOT/paper1_v43_private/raw/pvdaq9068/`. It also records each file's size and SHA-256 in
  `results/pvdaq9068_evaluation/a01_download_manifest.json`, which you can compare with the released copy.

## 4. NOAA Integrated Surface Database, station 724768-24051 (Greeley–Weld County Airport)

- **Use:** relative humidity for PVDAQ 9068, which has no humidity sensor; the airport is 8.8 km from the site.
- **Source:** NOAA NCEI global-hourly files, `https://www.ncei.noaa.gov/data/global-hourly/access/<year>/72476824051.csv`
  for 2020–2025.
- **Licence:** U.S. Government work, public domain.
- **Download:** `analysis/a01_download.py` downloads these files into `$PV_PRIVATE_ROOT/paper1_v43_private/raw/isd/`.

## Acknowledgement text

When you use these data, acknowledge the DKA Solar Centre, cite Lin et al. (2025) and the Dryad dataset for HKUST, cite
the PVDAQ dataset (DOI 10.25984/1846021) and the Integrated Surface Database (Smith, Lott and Vose, 2011, *Bulletin of the
American Meteorological Society* 92:704–708).
