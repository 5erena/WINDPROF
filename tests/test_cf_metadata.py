"""CF-1.10 compliance and value-neutrality of the NetCDF export.

The export stage adds CF metadata and structure only, so stored values
round-trip bit-identically.
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof.discovery_and_export import save_results_to_netcdf

CANONICAL = "bloc.windprof.z01.c1.20240612.000000.nc"


def _results():
    t = [int(pd.Timestamp("2024-06-12 00:00:00").timestamp()) + i * 600 for i in range(4)]
    intervals = []
    for ti in t:
        intervals.append({
            "time": ti,
            "ground_elevation": 35.1,
            "wind_profiles": {
                "merged": {100.0: {"ws": 5.23, "wd": 200.7, "w": 0.11},
                           130.0: {"ws": 6.44, "wd": 205.1, "w": 0.02}},
                "individual": {
                    "lidar_z01": {100.0: {"ws": 5.23, "wd": 200.7, "w": 0.11,
                                          "wserr": 0.31, "wderr": 2.0, "uerr": 0.2,
                                          "verr": 0.2, "werr": 0.1}},
                    "radar": {130.0: {"ws": 6.5, "wd": 206.0, "w": 0.0}},
                    "met_z01": {10.0: {"ws": 4.0, "wd": 190.0, "w": 0.05}},
                },
                "quality_flags": {100.0: {"ws": 0, "wd": 0, "w": 0},
                                  130.0: {"ws": 1, "wd": 0, "w": 0}},
            },
            "turbulence_profiles": {
                "merged": {100.0: {"ti": 0.1, "tke": 0.5, "std_u": 0.3,
                                   "std_v": 0.3, "std_w": 0.2}},
                "individual": {"lidar_z01": {100.0: {"ti": 0.1, "tke": 0.5,
                                                     "std_u": 0.3, "std_v": 0.3,
                                                     "std_w": 0.2}}},
                "quality_flags": {100.0: {"ti": 0}},
            },
            "surface_met": {"pressure": 1013.0, "temperature": 15.0,
                            "relative_humidity": 80.0, "precipitation": 0.0},
        })
    return {"time_intervals": intervals}


@pytest.fixture
def product(tmp_path):
    path = save_results_to_netcdf(_results(), str(tmp_path / CANONICAL),
                                  location="block_island")
    return path


def test_data_arrays_bit_identical_value_neutral(product):
    """The metadata pass must not perturb any stored value."""
    raw = xr.open_dataset(product, decode_cf=False, mask_and_scale=False)
    try:
        ws = raw["wind_speed_merged"].values
        assert ws[0, list(raw["height"].values).index(100.0)] == 5.23
        # Radar reaches only 130 m, so its lower levels hold raw -9999 sentinels.
        assert (raw["wind_speed_radar"].values == -9999.0).any()
        assert raw["wind_speed_merged"].dtype == np.float64
        assert raw["qc_wind_speed_merged"].dtype == np.int32
    finally:
        raw.close()


def test_time_decodes_with_calendar(product):
    ds = xr.open_dataset(product, decode_cf=True)
    try:
        assert ds["time"].dtype.kind == "M"  # decoded to datetime64
        assert np.datetime_as_string(ds["time"].values[0]).startswith("2024-06-12T00:00")
        assert ds.attrs["Conventions"] == "CF-1.10, ACDD-1.3"
        text = " ".join(str(v) for v in ds.attrs.values()).lower()
        assert "co-authored" not in text and "generated with" not in text
    finally:
        ds.close()


def test_availability_uses_flag_masks_not_flag_values(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        av = raw["instrument_availability"].attrs
        assert "flag_masks" in av and "flag_values" not in av
        masks = list(np.asarray(av["flag_masks"]))
        assert masks == [1 << i for i in range(len(masks))]
        assert len(av["flag_meanings"].split()) == len(masks)
        assert raw["instrument_availability"].dtype == np.int32
    finally:
        raw.close()


def test_qc_vars_carry_flag_values_matching_dtype(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        for name in ("qc_wind_speed_merged", "qc_wind_speed_lidar_z01"):
            attrs = raw[name].attrs
            assert "flag_values" in attrs and "flag_masks" not in attrs
            assert np.asarray(attrs["flag_values"]).dtype == raw[name].dtype
        assert list(np.asarray(raw["qc_wind_speed_merged"].attrs["flag_values"])) == [0, 1, 2, 3]
    finally:
        raw.close()


def test_ancillary_variables_resolve(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        found = False
        for name, var in raw.data_vars.items():
            anc = var.attrs.get("ancillary_variables")
            if anc:
                found = True
                for target in anc.split():
                    assert target in raw.variables, f"{name} -> missing {target}"
        assert found, "expected at least one ancillary_variables link"
        anc = raw["wind_speed_lidar_z01"].attrs["ancillary_variables"].split()
        assert "qc_wind_speed_lidar_z01" in anc
        assert "wind_speed_error_lidar_z01" in anc
    finally:
        raw.close()


def test_canonical_units_and_standard_names(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        assert raw["wind_speed_merged"].attrs["units"] == "m s-1"
        assert raw["wind_speed_merged"].attrs["standard_name"] == "wind_speed"
        assert raw["wind_direction_merged"].attrs["standard_name"] == "wind_from_direction"
        assert raw["turbulent_kinetic_energy_merged"].attrs["units"] == "m2 s-2"
        # CF-undefined quantities carry no standard_name.
        assert "standard_name" not in raw["turbulence_intensity_merged"].attrs
        assert "standard_name" not in raw["u_std_merged"].attrs
        # The TKE standard name postdates table v79, hence the v93 check below.
        assert (raw["turbulent_kinetic_energy_merged"].attrs["standard_name"]
                == "specific_turbulent_kinetic_energy_of_air")
        assert (raw["turbulent_kinetic_energy_lidar_z01"].attrs["standard_name"]
                == "specific_turbulent_kinetic_energy_of_air")
        assert (raw["qc_turbulent_kinetic_energy_lidar_z01"].attrs["standard_name"]
                == "status_flag")
        assert raw["qc_wind_speed_merged"].attrs["standard_name"] == "status_flag"
        assert "v93" in raw.attrs["standard_name_vocabulary"]
        # Merged TKE carries its own flag: the merge judges TKE and TI
        # separately, so reusing TI's flag would assert a judgement never made.
        assert "qc_turbulent_kinetic_energy_merged" in raw.variables
        assert (raw["turbulent_kinetic_energy_merged"].attrs["ancillary_variables"]
                == "qc_turbulent_kinetic_energy_merged")
        assert (list(np.asarray(raw["qc_turbulent_kinetic_energy_merged"]
                                .attrs["flag_values"])) == [0, 1, 2, 3])
        assert (raw["qc_turbulent_kinetic_energy_merged"].attrs["flag_meanings"]
                == raw["qc_turbulence_intensity_merged"].attrs["flag_meanings"])
        for comp in ("u", "v", "w"):
            assert f"qc_{comp}_std_merged" in raw.variables
            assert (raw[f"{comp}_std_merged"].attrs["ancillary_variables"]
                    == f"qc_{comp}_std_merged")
        assert raw["ground_elevation"].attrs["standard_name"] == "surface_altitude"
        assert raw["height"].attrs["positive"] == "up"
    finally:
        raw.close()


def test_acdd_attribution_and_identification(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        a = raw.attrs
        assert a["creator_name"] == "Serena Lipari"
        assert a["creator_email"] == "serena.lipari@pnnl.gov"
        assert a["creator_institution"] == "Pacific Northwest National Laboratory"
        assert a["publisher_name"] == "Wind Data Hub"
        assert a["publisher_email"] == "wdhteam@pnnl.gov"
        assert a["publisher_url"] == "https://wdh.energy.gov/"
        assert a["license"] == "CC0-1.0"
        assert a["id"].startswith("bloc.windprof.z01.c1.")
        assert a["time_coverage_resolution"] == "PT10M"
        assert "creation_date" not in a
        # The product is not versioned, so history must not imply otherwise;
        # date_created distinguishes generations.
        assert a["history"].startswith(a["date_created"])
        assert "WINDPROF processing pipeline" in a["history"]
        assert "user" in a["history"] and "machine" in a["history"]
        for versioned in ("v1.", "v2."):
            assert versioned not in a["history"]
        # The contributing datastreams are listed once, under the ACDD name.
        assert "instrument" in a and "instruments" not in a
    finally:
        raw.close()


def test_time_bounds_match_start_of_window_convention(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        assert raw["time"].attrs["bounds"] == "time_bounds"
        assert raw["time"].attrs["calendar"] == "proleptic_gregorian"
        t0 = int(raw["time"].values[0])
        b0 = raw["time_bounds"].values[0]
        # Start-stamped windows: bounds are [t, t + 10 min], not center +/- 5.
        assert list(b0) == [t0, t0 + 600]
    finally:
        raw.close()


def test_status_flag_and_coverage_content_types(product):
    raw = xr.open_dataset(product, decode_cf=False)
    try:
        assert raw["qc_wind_speed_merged"].attrs["standard_name"] == "status_flag"
        assert (raw["wind_speed_error_lidar_z01"].attrs["coverage_content_type"]
                == "qualityInformation")
        assert raw["u_std_merged"].attrs["coverage_content_type"] == "physicalMeasurement"
        assert raw["wind_speed_merged"].attrs["coverage_content_type"] == "physicalMeasurement"
        meanings = raw["instrument_availability"].attrs["flag_meanings"].split()
        assert all(m.endswith("_available") for m in meanings)
        # met provenance: BLOC met_z01 is surface-met-derived, not a sonic.
        assert "surface meteorological station" in raw["wind_speed_met_z01"].attrs["source"]
    finally:
        raw.close()


def test_ioos_cf_1_10_compliance(product):
    checker = pytest.importorskip("compliance_checker.runner")
    cs = checker.CheckSuite()
    cs.load_all_available_checkers()
    return_value, had_errors = checker.ComplianceChecker.run_checker(
        ds_loc=product, checker_names=["cf:1.10"], verbose=0,
        criteria="normal", output_filename=os.devnull, output_format="text")
    assert not had_errors
    assert return_value
