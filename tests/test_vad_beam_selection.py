"""Tests for VAD beam selection in the scanning-lidar path.

Only beams at the cone elevation (|el - 60| < 2 deg) may enter the retrieval.
Composite scans also contain 90-degree vertical stares and ~1-degree sector
sweeps, whose near-horizontal beams carry twice a VAD beam's u/v leverage at a
far lower altitude than the height they key to. min_beams counts VAD beams only.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof.lidars import process_scanning_lidar

LOCATION = "block_island"  # config: z01 wd correction 180, w sign +1
START = pd.Timestamp("2024-06-12 00:00:00")
U, V = 8.0, 3.0  # synthetic wind
WS_TRUE = float(np.hypot(U, V))


def _vr(az_deg, el_deg, u=U, v=V, w=0.0):
    az = np.deg2rad(np.asarray(az_deg, float))
    el = np.deg2rad(np.asarray(el_deg, float))
    return u * np.sin(az) * np.cos(el) + v * np.cos(az) * np.cos(el) + w * np.sin(el)


def make_ds(azimuths, elevations, vr_override=None, slant=(120.0, 180.0)):
    """Synthetic scanning dataset. vr_override entries replace the physical
    radial velocity for the matching beam index (to simulate contamination)."""
    azimuths = np.asarray(azimuths, float)
    elevations = np.asarray(elevations, float)
    n = len(azimuths)
    vr = _vr(azimuths, elevations)
    if vr_override is not None:
        for idx, value in vr_override.items():
            vr[idx] = value
    slant = np.asarray(slant, float)
    times = pd.date_range(START, periods=n, freq="4s")
    return xr.Dataset(
        {
            "radial_wind_speed": (("time", "range_gate"), np.tile(vr[:, None], (1, len(slant)))),
            "intensity": (("time", "range_gate"), np.full((n, len(slant)), 1.05)),
            "azimuth": ("time", azimuths),
            "elevation": ("time", elevations),
            "distance": ("range_gate", slant),
            "lat": ("index", np.array([41.167])),
            "lon": ("index", np.array([-71.58])),
        },
        coords={"time": times},
    )


def vad_cycle(n_cycles):
    az = np.tile([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], n_cycles)
    el = np.full(az.shape, 60.0)
    return az, el


def test_stares_do_not_corrupt_w():
    """90-degree stares with a fake +5 m/s signal must not move w (or ws)."""
    az, el = vad_cycle(3)
    az = np.concatenate([az, [0.0, 0.0, 0.0]])
    el = np.concatenate([el, [90.0, 90.0, 90.0]])
    ds = make_ds(az, el, vr_override={18: 5.0, 19: 5.0, 20: 5.0})

    result = process_scanning_lidar(ds, START, "z01", LOCATION, time_window=10)
    assert result["wind_profiles"], "expected wind profiles from clean VAD beams"
    for prof in result["wind_profiles"].values():
        assert prof["ws"] == pytest.approx(WS_TRUE, abs=0.02)
        assert prof["w"] == pytest.approx(0.0, abs=0.05)


def test_low_elevation_sweeps_do_not_corrupt_uv():
    """~1-degree sector-sweep beams with garbage vr must not move ws."""
    az, el = vad_cycle(3)
    sweep_az = np.array([86.0, 112.0, 138.0, 164.0])
    az = np.concatenate([az, sweep_az])
    el = np.concatenate([el, np.full(4, 1.0)])
    overrides = {18 + i: 15.0 for i in range(4)}
    ds = make_ds(az, el, vr_override=overrides)

    result = process_scanning_lidar(ds, START, "z01", LOCATION, time_window=10)
    assert result["wind_profiles"]
    for prof in result["wind_profiles"].values():
        assert prof["ws"] == pytest.approx(WS_TRUE, abs=0.02)


def test_min_beams_counts_only_vad_beams():
    """3 VAD beams + stares + sweeps must fail min_beams=4; 4 VAD beams pass."""
    az3 = [0.0, 120.0, 240.0, 10.0, 20.0, 100.0, 140.0]
    el3 = [60.0, 60.0, 60.0, 90.0, 90.0, 1.0, 1.0]
    ds3 = make_ds(az3, el3)
    result3 = process_scanning_lidar(ds3, START, "z01", LOCATION, time_window=10,
                                     min_beams=4)
    assert result3 is None or not result3["wind_profiles"], (
        "3 VAD beams padded with stares/sweeps must not satisfy min_beams=4")

    az4 = [0.0, 90.0, 180.0, 270.0, 10.0, 100.0]
    el4 = [60.0, 60.0, 60.0, 60.0, 90.0, 1.0]
    ds4 = make_ds(az4, el4)
    result4 = process_scanning_lidar(ds4, START, "z01", LOCATION, time_window=10,
                                     min_beams=4)
    assert result4["wind_profiles"], "4 VAD beams must satisfy min_beams=4"


def test_window_with_only_stares_returns_none():
    az = [0.0, 0.0, 0.0, 0.0]
    el = [90.0, 90.0, 90.0, 90.0]
    ds = make_ds(az, el)
    assert process_scanning_lidar(ds, START, "z01", LOCATION, time_window=10) is None
