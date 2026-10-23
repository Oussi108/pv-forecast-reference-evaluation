"""P01: derive protocol_v43_pvdaq9068.json from the frozen v28 protocol, changing only the testbed-specific fields."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
REL = HERE.parents[2] / 'WorkLoad/solar_pipeline_scaffold/release/six-hour-pv-forecasting'
p = json.loads((REL / 'configs/corrected_study_v28_array1b.json').read_text())
audit = json.loads((HERE.parent / 'results/a02_audit.json').read_text())
p.update({
    'protocol_id': 'v43-pvdaq9068-2026-09-26',
    'parent_protocol_id': p['protocol_id'],
    'status': 'frozen_before_any_forecast_scoring_at_PVDAQ9068',
    'revision_type': 'independent_third_testbed_confirmatory',
    'test_history': 'No forecast for PVDAQ9068 had been produced or scored when this protocol was frozen; 2023 and 2024 labels were examined only as coverage counts.',
    'public_results_dir': 'results/v43_pvdaq9068',
    'testbeds': ['PVDAQ9068'],
    'confirmation_year': 2024,
})
p['sources'] = {'PVDAQ9068': {
    'provider': 'NREL PVDAQ 2023 Solar Data Prize, system 9068', 'licence': 'CC BY 4.0 (OEDI 4568, DOI 10.25984/1846021)',
    'latitude': 40.3864, 'longitude': -104.5512, 'altitude_m': 1407, 'koppen': 'BSk', 'structure': 'single-axis tracker',
    'timezone': 'America/Denver', 'capacity_kw': audit['info']['capacity_kw'], 'capacity_source': audit['info']['capacity_source'],
    'power_definition': 'inverter_1 + inverter_2 AC power, both required; revenue meter excluded (2024 fault)',
    'irradiance_channel': 'mean measured plane-of-array irradiance (two pyranometers); no GHI sensor at the site',
    'humidity_source': 'NOAA ISD 724768-24051 Greeley-Weld County Airport (8.8 km), RH from temperature and dew point',
    'temperature_range_c': [-40, 60]}}
p['source_timezone'] = {'PVDAQ9068': 'America/Denver'}
p['preregistration'] = 'protocol/PREREGISTRATION_v43.md'
(HERE.parent / 'protocol/protocol_v43_pvdaq9068.json').write_text(json.dumps(p, indent=2))
print('written')
