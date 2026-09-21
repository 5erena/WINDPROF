"""Tests for NetCDF TI metadata. TI definitions differ by instrument by
design, so each file must record the method each instrument used.
"""

import pandas as pd
import pytest
import xarray as xr

from windprof.discovery_and_export import save_results_to_netcdf, TI_METHODS


def _minimal_results():
    t0 = int(pd.Timestamp("2024-06-12 00:00:00").timestamp())
    interval = {
        "time": t0,
        "wind_profiles": {
            "merged": {100.0: {"ws": 5.0, "wd": 200.0}},
            "individual": {"lidar_z01": {100.0: {"ws": 5.0, "wd": 200.0}}},
            "quality_flags": {100.0: {"ws": 0, "wd": 0, "w": 0}},
        },
        "turbulence_profiles": {
            "merged": {100.0: {"ti": 0.1}},
            "individual": {
                "lidar_z01": {100.0: {"ti": 0.1}},
                "lidar_z03": {100.0: {"ti": 0.12}},
            },
            "quality_flags": {100.0: {"ti": 0}},
        },
    }
    return {"time_intervals": [interval]}


def test_ti_method_attributes_written(tmp_path):
    path = save_results_to_netcdf(_minimal_results(), str(tmp_path / "out.nc"),
                                  location="block_island")
    ds = xr.open_dataset(path)
    try:
        comment = ds["turbulence_intensity_merged"].attrs.get("comment", "")
        assert "different TI definitions" in comment

        z01 = ds["turbulence_intensity_lidar_z01"].attrs.get("method", "")
        assert "Rosenbusch" in z01 and "0.55" in z01

        z03 = ds["turbulence_intensity_lidar_z03"].attrs.get("method", "")
        assert "TKE" in z03  # BLOC z03 = sqrt(TKE)/WS from STA statistics

        met = ds["turbulence_intensity_met_z01"].attrs.get("method", "")
        assert "not reported" in met  # BLOC met_z01 has no turbulence data

        ws_met = ds["wind_speed_met_z01"].attrs
        assert ws_met.get("measurement_height_agl") == 10.0  # BLOC z01 sonic
        assert "not interpolated" in ws_met.get("comment", "")
    finally:
        ds.close()


def test_ti_methods_cover_all_turbulence_instruments():
    """Every (site, turbulence instrument) pair needs a documented TI method,
    or its variable falls back to the generic placeholder."""
    from windprof.config import SITE_INSTRUMENT_MAPPINGS, TURBULENCE_INSTRUMENTS

    missing = []
    for location, site in SITE_INSTRUMENT_MAPPINGS.items():
        for instrument in site["instruments"]:
            if instrument in TURBULENCE_INSTRUMENTS and (location, instrument) not in TI_METHODS:
                missing.append((location, instrument))
    assert not missing, f"TI method undocumented for: {missing}"


# The definition each instrument's turbulence intensity actually follows. A
# label naming a different one publishes a method the values do not use.
EXPECTED_TI_METHOD = {
    ('nantucket', 'lidar_z01'): 'Rosenbusch',
    ('nantucket', 'lidar_z02'): 'Rosenbusch',
    ('nantucket', 'lidar_z03'): 'Rosenbusch',
    ('nantucket', 'met_z02'): 'Rosenbusch',
    ('block_island', 'lidar_z01'): 'Rosenbusch',
    ('block_island', 'lidar_z03'): 'TKE',
    ('block_island', 'met_z01'): 'not reported',
    ('rhode_island', 'lidar_z01'): 'Rosenbusch',
    ('rhode_island', 'lidar_z03'): 'vendor-reported',
    ('rhode_island', 'met_z01'): 'EddyPro',
    ('cape_cod', 'lidar_z01'): 'dispersion',
    ('cape_cod', 'lidar_z02'): 'Rosenbusch',
    ('cape_cod', 'met_z01'): 'EddyPro',
    ('cape_cod', 'met_z02'): 'EddyPro',
}


def test_ti_method_strings_name_the_definition_used():
    """Presence is not enough: a label naming the wrong definition passes a
    coverage check while telling users the values were formed another way."""
    wrong = [(key, token, TI_METHODS.get(key, ''))
             for key, token in EXPECTED_TI_METHOD.items()
             if token not in TI_METHODS.get(key, '')]
    assert not wrong, f"TI method does not name its definition: {wrong}"

    unreported = [key for key, method in TI_METHODS.items()
                  if 'not reported' in method and EXPECTED_TI_METHOD.get(key) != 'not reported']
    assert not unreported, f"labelled 'not reported' but carries turbulence: {unreported}"
