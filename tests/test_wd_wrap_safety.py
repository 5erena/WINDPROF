"""Wrap-safety for vendor wind-direction averaging.

CACO z01 and RHOD z03 deliver one vendor row per 10-min window today (both
verified against real files), so their per-window averaging is currently a
no-op. These tests force the multi-row case that a future finer-cadence
delivery could produce, and pin that the circular mean is used: two rows
straddling 0/360 must average toward 0 rather than collapse to 180, which
is what an arithmetic mean produces.
"""

import numpy as np
import pandas as pd
import pytest

from windprof.lidars import process_caco_z01_lidar, _process_profiling_csv

T0 = pd.Timestamp("2024-06-12 00:00:00")


def _near_zero(deg):
    return min(deg % 360.0, 360.0 - (deg % 360.0))


def test_caco_wd_uses_circular_mean_across_the_wrap():
    n = 2
    times = pd.Series(pd.to_datetime(["2024-06-12 00:02:00", "2024-06-12 00:07:00"]))
    parsed = {
        "time": times, "heights": [40], "latitude": 42.03, "longitude": -70.05,
        "measurements": {40: {
            "wind_speed": np.full(n, 5.0),
            "wind_direction": np.array([350.0, 10.0]),
            "w": np.zeros(n), "ws_dispersion": np.full(n, 0.5),
            "cnr": np.full(n, 0.0),
            "availability": np.full(n, 100.0)}},
    }
    res = process_caco_z01_lidar(parsed, T0, location="cape_cod", time_window=10)
    assert res is not None
    wd = res["wind_profiles"][40.0]["wd"]
    assert _near_zero(wd) < 1.0, f"expected ~0 (circular), got {wd}"


def test_rhod_wd_uses_circular_mean_across_the_wrap():
    n = 2
    times = pd.Series(pd.to_datetime(["2024-06-12 00:02:00", "2024-06-12 00:07:00"]))
    packets = np.full(n, 30.0)
    df = pd.DataFrame({
        "Proportion Of Packets With Rain (%)": np.zeros(n),
        "Packets in Average at 40m": packets,
    })
    csv_data = {
        "heights": [40.0], "time": times, "file_type": "csv",
        "measurements": {40.0: {
            "time": times, "wind_speed": np.full(n, 6.0),
            "wind_direction": np.array([355.0, 5.0]), "w": np.zeros(n),
            "TI": np.full(n, 0.1), "std_dev": np.full(n, 0.5), "packets": packets}},
        "metadata": {"original_df": df},
    }
    res = _process_profiling_csv(csv_data, T0, 10, availability_threshold=0.5,
                                 ground_elevation=3.37, wind_dir_correction=0,
                                 latitude=41.45, longitude=-71.43,
                                 location="rhode_island", verbose=False,
                                 min_samples=20)
    assert res is not None
    wd = res["wind_profiles"][40.0]["wd"]
    assert _near_zero(wd) < 1.0, f"expected ~0 (circular), got {wd}"


def test_single_row_window_is_unchanged_no_op():
    times = pd.Series(pd.to_datetime(["2024-06-12 00:03:00"]))
    parsed = {
        "time": times, "heights": [40], "latitude": 42.03, "longitude": -70.05,
        "measurements": {40: {
            "wind_speed": np.array([5.0]), "wind_direction": np.array([137.0]),
            "w": np.zeros(1), "ws_dispersion": np.array([0.5]),
            "cnr": np.array([0.0]), "availability": np.array([100.0])}},
    }
    res = process_caco_z01_lidar(parsed, T0, location="cape_cod", time_window=10)
    assert res["wind_profiles"][40.0]["wd"] == pytest.approx(137.0, abs=0.05)
