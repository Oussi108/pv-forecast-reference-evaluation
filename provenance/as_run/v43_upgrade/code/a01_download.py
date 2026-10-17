"""A01: download PVDAQ system 9068 (CC BY 4.0, OEDI) and NOAA ISD Greeley-Weld (public domain) raw files; record hashes.
Raw observations stay in tmp/paper1_v43_private (never inside the public release)."""
import hashlib, json, time, urllib.request
from pathlib import Path
WS = Path(__file__).resolve().parents[3]
RAW = WS / 'tmp/paper1_v43_private/raw'
OUT = Path(__file__).resolve().parents[1] / 'results'
B = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/9068_OEDI'
files = {f'pvdaq9068/{n}': f'{B}/data/{n}' for n in [
    '9068_ac_power_data.csv', '9068_ac_power_data_20240101_20250430.csv', '9068_irradiance_data.csv',
    '9068_irradiance_data_20240101_20250430.csv', '9068_environment_data.csv', '9068_environment_data_20240101_20250430.csv',
    '9068_meter_data_20240101_20250430.csv']}
files['pvdaq9068/9068_system_metadata.json'] = f'{B}/metadata/9068_system_metadata.json'
for y in range(2020, 2026):
    files[f'isd/72476824051_{y}.csv'] = f'https://www.ncei.noaa.gov/data/global-hourly/access/{y}/72476824051.csv'
manifest = {}
for rel, url in files.items():
    dest = RAW / rel
    if not dest.exists():
        t0 = time.time()
        urllib.request.urlretrieve(url, dest)
        print('downloaded', rel, round(dest.stat().st_size / 1e6, 1), 'MB', round(time.time() - t0), 's', flush=True)
    manifest[rel] = dict(url=url, bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
manifest['_licences'] = {'pvdaq9068': 'PVDAQ public datasets, OEDI submission 4568, Creative Commons Attribution 4.0 (DOI 10.25984/1846021)',
                         'isd': 'NOAA NCEI Integrated Surface Dataset, U.S. Government public domain'}
manifest['_accessed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
(OUT / 'a01_download_manifest.json').write_text(json.dumps(manifest, indent=1))
print('total MB', round(sum(v['bytes'] for k, v in manifest.items() if not k.startswith('_')) / 1e6, 1))
