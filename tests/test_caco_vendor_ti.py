"""CACO z01 vendor turbulence intensity from the workbook wind speed dispersion.

The WindCube V2-96 workbook's wind speed dispersion is the scan-to-scan
sigma(WS) at each range gate, so it yields a TI commensurable with the
instruments processed from high-rate series. The workbook's separate Z-wind
Dispersion is the vertical beam's radial-velocity standard deviation, not the
VAD-fit sigma_w, so ingesting it would break the TKE definition shared by
the other instruments.
"""

import os

import numpy as np
import pandas as pd
import pytest

from windprof.config import get_qc_params, round_profile_values
from windprof.discovery_and_export import TI_METHODS
from windprof.lidar_parsers import parse_caco_lidar_file
from windprof.lidars import process_caco_z01_lidar

MIN_TI_AVAILABILITY = get_qc_params('cape_cod', 'z01')['min_ti_availability']

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKBOOK = 'caco.lidar.z01.00.20240516.000000.xlsx'

# The workbook is not shipped: the Wind Data Hub copy is authoritative and a
# duplicate here would be a second source of truth. See tests/data/README.md.
SAMPLE = next(
    (c for c in (os.path.join(_REPO, 'tests', 'data', _WORKBOOK),
                 os.path.join(_REPO, 'caco_z01_lidar_sample', _WORKBOOK))
     if os.path.exists(c)),
    os.path.join(_REPO, 'tests', 'data', _WORKBOOK))

needs_workbook = pytest.mark.skipif(
    not os.path.exists(SAMPLE),
    reason='vendor workbook not present; see tests/data/README.md')

# The vendor's range gates for this instrument, in meters above the lidar.
CACO_Z01_GATES = [40, 60, 80, 100, 120, 137, 160, 180, 200, 220, 248, 280]


def _synthetic(n=1, ws=5.0, dispersion=0.5, wd=100.0, availability=100.0):
    """A stand-in for parse_caco_lidar_file output at a single gate."""
    times = pd.Series(pd.date_range('2024-06-12 00:00', periods=n, freq='2min'))
    return {
        'time': times, 'heights': [40], 'latitude': 42.03, 'longitude': -70.05,
        'measurements': {40: {
            'wind_speed': np.full(n, ws, dtype=float),
            'wind_direction': np.full(n, wd, dtype=float),
            'w': np.zeros(n),
            'ws_dispersion': np.full(n, dispersion, dtype=float),
            'cnr': np.full(n, 0.0),
            'availability': np.full(n, availability, dtype=float)}},
    }


@needs_workbook
def test_parser_reads_wind_speed_dispersion_and_not_z_wind():
    parsed = parse_caco_lidar_file(SAMPLE, use_cache=False)
    assert parsed['heights'] == CACO_Z01_GATES

    for height in parsed['heights']:
        keys = parsed['measurements'][height]
        assert 'ws_dispersion' in keys
        # Z-wind Dispersion is present in the workbook but deliberately dropped.
        assert not any('z_wind_dispersion' in k or 'w_dispersion' in k for k in keys)

    disp = parsed['measurements'][40]['ws_dispersion']
    assert np.isfinite(disp).any(), 'no finite dispersion read at the lowest gate'
    finite = disp[np.isfinite(disp)]
    assert (finite >= 0).all(), 'wind speed dispersion cannot be negative'


@needs_workbook
def test_ti_equals_dispersion_over_wind_speed_on_a_real_day():
    """Every exported TI is the vendor ratio at the pipeline's output precision."""
    parsed = parse_caco_lidar_file(SAMPLE, use_cache=False)
    times = pd.to_datetime(parsed['time'])
    start = times.min().floor('10min')

    checked = 0
    for _ in range(144):  # one day of 10-minute windows
        res = process_caco_z01_lidar(parsed, start, location='cape_cod', time_window=10)
        start = start + pd.Timedelta(minutes=10)
        if res is None:
            continue
        window_times = (times >= res['time']) & (times < res['time'] + pd.Timedelta(minutes=10))
        for height, turb in res['turbulence_profiles'].items():
            src = parsed['measurements'][int(height)]
            ws = src['wind_speed'][window_times]
            disp = src['ws_dispersion'][window_times]
            avail = src['availability'][window_times]
            valid = (np.isfinite(ws) & np.isfinite(disp) & (ws > 0)
                     & (avail >= MIN_TI_AVAILABILITY))
            expected = round_profile_values({'ti': float(np.nanmean(disp[valid] / ws[valid]))})
            assert turb['ti'] == expected['ti']
            assert avail[valid].min() >= MIN_TI_AVAILABILITY
            checked += 1

    assert checked > 100, f'only {checked} TI values exercised'


def test_ti_is_the_only_turbulence_quantity_reported():
    """No TKE or component standard deviations: the workbook cannot form them."""
    res = process_caco_z01_lidar(_synthetic(), pd.Timestamp('2024-06-12 00:00:00'),
                                 location='cape_cod', time_window=10)
    assert set(res['turbulence_profiles'][40.0]) == {'ti'}
    assert res['turbulence_profiles'][40.0]['ti'] == pytest.approx(0.1, abs=5e-4)


def test_missing_dispersion_leaves_wind_untouched():
    with_disp = _synthetic(dispersion=0.5)
    without = _synthetic(dispersion=np.nan)

    a = process_caco_z01_lidar(with_disp, pd.Timestamp('2024-06-12 00:00:00'),
                               location='cape_cod', time_window=10)
    b = process_caco_z01_lidar(without, pd.Timestamp('2024-06-12 00:00:00'),
                               location='cape_cod', time_window=10)

    assert a['wind_profiles'] == b['wind_profiles']
    assert a['availability'] == b['availability']
    assert b['turbulence_profiles'] == {}


def test_calm_records_do_not_produce_a_ratio():
    """TI is undefined at zero wind speed, so a calm record contributes nothing."""
    res = process_caco_z01_lidar(_synthetic(ws=0.0, dispersion=0.5),
                                 pd.Timestamp('2024-06-12 00:00:00'),
                                 location='cape_cod', time_window=10)
    assert res['turbulence_profiles'] == {}
    assert 40.0 in res['wind_profiles']


def test_implausible_ti_is_rejected_by_the_shared_physics_gate():
    """The IEC 61400-1 (2019) bound applied to every instrument applies here too."""
    res = process_caco_z01_lidar(_synthetic(ws=0.1, dispersion=1.0),
                                 pd.Timestamp('2024-06-12 00:00:00'),
                                 location='cape_cod', time_window=10)
    assert res['turbulence_profiles'] == {}
    assert 40.0 in res['wind_profiles']


def test_low_vendor_availability_gates_turbulence_but_not_wind():
    """No TI below the minimum: the dispersion is degenerate. The vendor's wind
    speed there is still delivered, so the gate must not reach it.
    """
    args = dict(location='cape_cod', time_window=10)
    t0 = pd.Timestamp('2024-06-12 00:00:00')

    below = process_caco_z01_lidar(_synthetic(availability=MIN_TI_AVAILABILITY - 1), t0, **args)
    at = process_caco_z01_lidar(_synthetic(availability=MIN_TI_AVAILABILITY), t0, **args)
    full = process_caco_z01_lidar(_synthetic(availability=100.0), t0, **args)

    assert below['turbulence_profiles'] == {}
    assert at['turbulence_profiles'][40.0]['ti'] == pytest.approx(0.1, abs=5e-4)
    assert below['wind_profiles'] == full['wind_profiles']
    assert below['availability'] == full['availability']

def test_published_method_string_names_the_availability_gate():
    """The threshold is discoverable from the file, not only from the source."""
    method = TI_METHODS[('cape_cod', 'lidar_z01')]
    assert f'{MIN_TI_AVAILABILITY}%' in method
    assert 'dispersion' in method
    assert 'not reported' not in method
