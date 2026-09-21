"""Pins for the consolidated per-instrument QC (config.QC_CONFIG), the single
source of truth shared by the beam filter, the availability calculation and the
instrument processors. Each threshold is pinned on the scale of the signal it
screens: an intensity threshold placed on the CNR scale would pass every beam.
The 4-beam VAD minimum leaves dof >= 1, without which sigma_ws is undefined.
"""

import numpy as np
import pandas as pd
import pytest

from windprof.config import QC_CONFIG, MIN_VAD_BEAMS, get_qc_params
from windprof.quality_control import (filter_data_by_qc_criteria,
                                             calculate_data_availability)
from windprof.wind_analysis import fit_chi_winds
from windprof.lidars import process_caco_z01_lidar, _process_profiling_csv


def test_qc_config_values_pinned():
    assert QC_CONFIG['nantucket']['z01'] == {'qc_type': 'intensity', 'threshold': 1.008, 'min_beams': 4}
    assert QC_CONFIG['nantucket']['z02']['threshold'] == 1.008
    assert QC_CONFIG['block_island']['z01']['threshold'] == 1.008
    assert QC_CONFIG['nantucket']['z03'] == {'qc_type': 'cnr', 'threshold': -23}
    assert QC_CONFIG['block_island']['z03'] == {'qc_type': 'cnr', 'threshold': -22}
    assert QC_CONFIG['rhode_island']['z03'] == {'qc_type': 'samples_rain', 'min_samples': 20}
    assert QC_CONFIG['cape_cod']['z01'] == {'qc_type': 'prefiltered',
                                            'min_ti_availability': 10}
    assert MIN_VAD_BEAMS == 4
    with pytest.raises(ValueError):
        get_qc_params('nantucket', 'z99')


def _beam_data(values, key):
    n = len(values)
    return {
        'heights': [100.0],
        'time': pd.date_range('2024-06-12', periods=n, freq='4s'),
        'position': np.arange(n) * 60.0 % 360,
        'measurements': {100.0: {key: np.array(values, dtype=float),
                                 'vr': np.zeros(n)}},
    }


def test_intensity_filter_uses_config_threshold():
    # The gate is inclusive: 1.008 passes, 1.007 fails.
    vals = [1.05, 1.2, 1.009, 1.008, 1.05, 1.0, 0.99, 1.007]
    out = filter_data_by_qc_criteria(_beam_data(vals, 'intensity'),
                                     instrument='z01', location='nantucket', verbose=False)
    assert out is not None
    assert len(out['measurements'][100.0]['vr']) == 5

    # Only 3 passing beams < min_beams=4 -> the height is dropped entirely.
    out2 = filter_data_by_qc_criteria(_beam_data([1.05, 1.05, 1.05, 1.0, 1.0], 'intensity'),
                                      instrument='z01', location='nantucket', verbose=False)
    assert out2 is None


def test_cnr_filter_distinguishes_site_thresholds():
    # -22.5 dB passes NANT (-23) but fails BLOC (-22): behavioral pin of both.
    vals = [-22.5, -21, -20, -19, -18]
    nant = filter_data_by_qc_criteria(_beam_data(vals, 'cnr'),
                                      instrument='z03', location='nantucket', verbose=False)
    bloc = filter_data_by_qc_criteria(_beam_data(vals, 'cnr'),
                                      instrument='z03', location='block_island', verbose=False)
    assert len(nant['measurements'][100.0]['vr']) == 5
    assert len(bloc['measurements'][100.0]['vr']) == 4


def test_prefiltered_short_circuit_passes_data_through():
    data = _beam_data([-99.0, -99.0], 'cnr')  # values that would fail any threshold
    out = filter_data_by_qc_criteria(data, instrument='z01', location='cape_cod', verbose=False)
    assert out is data  # untouched: QC applied upstream by the instrument


def test_availability_requires_and_matches_filter_rule():
    data = _beam_data([1.05, 1.05, 1.0, 1.0], 'intensity')
    # qc_type/threshold are mandatory: the rule is passed in, never inferred.
    with pytest.raises(ValueError):
        calculate_data_availability(data, data, 0.5)
    av = calculate_data_availability(data, data, 0.5,
                                     qc_type='intensity', threshold=1.008)
    assert av['height_availability'][100.0]['availability'] == pytest.approx(0.5)


def _vad_rv(az, el, u, v):
    az, el = np.deg2rad(np.asarray(az, float)), np.deg2rad(np.asarray(el, float))
    return u * np.sin(az) * np.cos(el) + v * np.cos(az) * np.cos(el)


def test_three_beam_fit_rejected_with_insufficient_dof():
    az3, el3 = [0.0, 120.0, 240.0], [60.0] * 3
    out3 = fit_chi_winds(az3, el3, _vad_rv(az3, el3, 8.0, 3.0))
    assert np.isnan(out3['ws'])
    assert out3['diagnostics']['failure_reason'] == 'insufficient_dof'

    az4, el4 = [0.0, 90.0, 180.0, 270.0], [60.0] * 4
    out4 = fit_chi_winds(az4, el4, _vad_rv(az4, el4, 8.0, 3.0))
    assert out4['ws'] == pytest.approx(float(np.hypot(8.0, 3.0)), abs=0.02)


def test_caco_prefiltered_keeps_low_cnr_rows():
    """Cape Cod is 'prefiltered': the vendor blanks wind wherever its own
    screening fails, so low CNR rows must pass through untouched."""
    n = 3
    times = pd.Series(pd.date_range('2024-06-12 00:00', periods=n, freq='2min'))
    parsed = {
        'time': times, 'heights': [40], 'latitude': 42.03, 'longitude': -70.05,
        'measurements': {40: {
            'wind_speed': np.full(n, 5.0), 'wind_direction': np.full(n, 100.0),
            'w': np.zeros(n), 'ws_dispersion': np.full(n, 0.5),
            'cnr': np.full(n, -30.0), 'availability': np.full(n, 100.0)}},
    }
    res = process_caco_z01_lidar(parsed, pd.Timestamp('2024-06-12 00:00:00'),
                                 location='cape_cod', time_window=10)
    assert res is not None and 40.0 in res['wind_profiles']
    assert res['wind_profiles'][40.0]['ws'] == pytest.approx(5.0)


def test_zephir_min_samples_gate_is_config_driven():
    n = 4
    times = pd.Series(pd.date_range('2024-06-12 00:00', periods=n, freq='2min'))
    packets = np.array([25.0, 25.0, 19.0, 25.0])
    df = pd.DataFrame({
        'Proportion Of Packets With Rain (%)': np.zeros(n),
        'Packets in Average at 40m': packets,
    })
    csv_data = {
        'heights': [40.0], 'time': times, 'file_type': 'csv',
        'measurements': {40.0: {
            'time': times, 'wind_speed': np.full(n, 6.0),
            'wind_direction': np.full(n, 210.0), 'w': np.zeros(n),
            'TI': np.full(n, 0.1), 'std_dev': np.full(n, 0.5), 'packets': packets}},
        'metadata': {'original_df': df},
    }
    common = dict(availability_threshold=0.5, ground_elevation=3.37,
                  wind_dir_correction=0, latitude=41.45, longitude=-71.43,
                  location='rhode_island', verbose=False)

    keep = _process_profiling_csv(csv_data, pd.Timestamp('2024-06-12 00:00:00'), 10,
                                  min_samples=20, **common)
    assert keep is not None and 40.0 in keep['wind_profiles']

    drop = _process_profiling_csv(csv_data, pd.Timestamp('2024-06-12 00:00:00'), 10,
                                  min_samples=26, **common)
    assert drop is None  # every record below the sample minimum
