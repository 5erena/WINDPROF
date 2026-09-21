"""Tests for the near-surface grid levels and sonic placement.

Grid levels at 4/5/10 m (derived from config anemometer heights) place
every single-height sonic exactly where it measured; 0 m carries no
instrument and is not a grid level. Sonics stay isolated points,
nothing may be interpolated between a sonic level and the lowest lidar
gate, and the lidar may not extrapolate below its lowest gate.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof.anemometers import extract_wind_from_surface_met
from windprof.config import get_near_surface_levels
from windprof.discovery_and_export import save_results_to_netcdf
from windprof.merging import create_combined_results, process_surface_met_for_date

T0 = pd.Timestamp("2024-06-12 00:00:00")


def sonic(height, ws=5.0):
    return {"time": T0, "wind_profiles": {height: {"ws": ws, "wd": 200.0, "w": 0.1}},
            "turbulence_profiles": {}}


def test_grid_levels_derived_from_config():
    assert get_near_surface_levels() == (4.0, 5.0, 10.0)


@pytest.mark.parametrize("location,instrument,height", [
    ("rhode_island", "met_z01", 4.0),
    ("cape_cod", "met_z01", 4.0),
    ("nantucket", "met_z02", 5.0),
    ("block_island", "met_z01", 10.0),
    ("cape_cod", "met_z02", 10.0),
])
def test_each_sonic_lands_at_its_config_height(location, instrument, height):
    out = create_combined_results(T0, {instrument: sonic(height)}, False, location=location)
    merged = out["wind_profiles"]["merged"]
    assert height in merged and merged[height]["ws"] == pytest.approx(5.0)
    assert 0.0 not in merged
    assert height in out["wind_profiles"]["individual"][instrument]


def test_caco_sonics_occupy_separate_levels_not_averaged():
    data = {"met_z01": sonic(4.0, ws=3.0), "met_z02": sonic(10.0, ws=5.0)}
    out = create_combined_results(T0, data, False, location="cape_cod")
    merged = out["wind_profiles"]["merged"]
    assert merged[4.0]["ws"] == pytest.approx(3.0)
    assert merged[10.0]["ws"] == pytest.approx(5.0)
    assert 0.0 not in merged


def test_no_bridge_between_sonic_and_lowest_lidar_gate():
    lidar = {"time": T0, "turbulence_profiles": {},
             "wind_profiles": {40.0: {"ws": 6.0, "wd": 200.0, "w": 0.0},
                               60.0: {"ws": 7.0, "wd": 202.0, "w": 0.0},
                               80.0: {"ws": 8.0, "wd": 204.0, "w": 0.0},
                               100.0: {"ws": 9.0, "wd": 206.0, "w": 0.0}}}
    data = {"met_z01": sonic(10.0, ws=5.0), "lidar_z03": lidar}
    out = create_combined_results(T0, data, False, location="block_island")

    merged = out["wind_profiles"]["merged"]
    assert 10.0 in merged and merged[10.0]["ws"] == pytest.approx(5.0)
    assert 20.0 not in merged, "20 m must not be fabricated between sonic and lidar"
    assert 4.0 not in merged and 5.0 not in merged

    lidar_interp = out["wind_profiles"]["individual"]["lidar_z03"]
    assert min(lidar_interp) == pytest.approx(40.0), "lidar must not extrapolate below its lowest gate"
    sonic_interp = out["wind_profiles"]["individual"]["met_z01"]
    assert list(sonic_interp) == [10.0]


def test_bloc_surface_met_wind_lands_at_10m(tmp_path):
    """End to end: the builder keys surface-met wind at the configured 10 m height."""
    times = pd.date_range("2024-06-12 00:00:00", periods=6, freq="100s")
    ds = xr.Dataset({
        "wind_speed": ("time", np.full(6, 7.0)),
        "wind_direction": ("time", np.full(6, 200.0)),
        "pressure": ("time", np.full(6, 1013.0)),
        "temperature": ("time", np.full(6, 15.0)),
        "relative_humidity": ("time", np.full(6, 80.0)),
    }, coords={"time": times})
    path = tmp_path / "bloc.met.z01.c1.20240612.000000.nc"
    ds.to_netcdf(path)

    windows = process_surface_met_for_date("2024-06-12", file_path=str(path),
                                           location="block_island")
    first = windows[int(T0.timestamp())]
    assert list(first["wind_profile"]) == [10.0]

    results = extract_wind_from_surface_met(windows, "block_island")
    assert results, "expected extracted surface-met wind intervals"
    for res in results:
        assert list(res["wind_profiles"]) == [10.0]


def test_measurement_height_attr_agrees_with_grid_level(tmp_path):
    t0 = int(T0.timestamp())
    interval = {
        "time": t0,
        "wind_profiles": {
            "merged": {10.0: {"ws": 5.0, "wd": 200.0}},
            "individual": {"met_z01": {10.0: {"ws": 5.0, "wd": 200.0}}},
            "quality_flags": {10.0: {"ws": 0, "wd": 0, "w": 0}},
        },
        "turbulence_profiles": {"merged": {}, "individual": {}, "quality_flags": {}},
    }
    path = save_results_to_netcdf({"time_intervals": [interval]},
                                  str(tmp_path / "bloc.windprof.z01.c1.20240612.000000.nc"),
                                  location="block_island")
    out = xr.open_dataset(path)
    try:
        var = out["wind_speed_met_z01"]
        assert var.attrs["measurement_height_agl"] == 10.0
        populated = out["height"].values[np.isfinite(var.values).any(axis=0)]
        assert list(populated) == [10.0]
    finally:
        out.close()
