"""Tests for the lidar height datum.

Profile heights in the merged product are height above ground level at the
instrument. Parser and scan-geometry heights are already instrument-relative,
so the site's ASL ground elevation must never be subtracted from them. Both
tests run at Block Island, whose ~34-35 m ground elevation makes an accidental
datum shift larger than one height grid cell, so it fails loudly.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof.config import LOCATION_CONFIG
from windprof.lidars import _process_profiling_rtd, process_scanning_lidar

LOCATION = "block_island"
ELEVATION_ANGLE_PROFILING = 62.0  # WindCube cone elevation used by the RTD path
START = pd.Timestamp("2024-06-12 00:00:00")


def _radial_velocity(az_deg, el_deg, u, v, w):
    """Forward VAD model used to synthesize radial velocities."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    return u * np.sin(az) * np.cos(el) + v * np.cos(az) * np.cos(el) + w * np.sin(el)


def test_profiling_heights_pass_through_unchanged():
    """Native WindCube altitudes must be the output profile heights, exactly."""
    heights = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0]
    n = 120  # 10 minutes at ~5 s per beam
    times = pd.date_range(START, periods=n, freq="5s")
    azimuths = np.tile([0.0, 90.0, 180.0, 270.0], n // 4)
    vr = _radial_velocity(azimuths, ELEVATION_ANGLE_PROFILING, u=6.0, v=2.0, w=0.0)

    rtd_data = {
        "heights": heights,
        "time": times,
        "position": pd.Series([f"{a:.0f}" for a in azimuths]),
        "measurements": {
            h: {"cnr": np.zeros(n), "vr": vr.copy()} for h in heights
        },
    }

    ground_elevation = LOCATION_CONFIG[LOCATION]["elevations"]["z03"]
    result = _process_profiling_rtd(
        rtd_data, START, time_window=10, cnr_threshold=-23, min_beams=4,
        wind_dir_correction=0, min_beams_per_scan=3,
        availability_threshold=0.5, ground_elevation=ground_elevation,
        latitude=41.17, longitude=-71.58, location=LOCATION, verbose=False,
    )

    assert result is not None, "profiling processor returned no result"
    got = sorted(result["wind_profiles"])
    assert got == pytest.approx(heights), (
        f"profile heights {got} differ from native instrument altitudes "
        f"{heights}: heights must not be shifted by any elevation offset"
    )
    # 40 m is the instrument's minimum range: below that is physically impossible.
    assert min(got) >= 40.0


def test_scanning_heights_follow_geometry_and_100m_floor():
    """Scanning output heights = slant * sin(el); nothing below the 100 m rule."""
    elevation = 60.0
    # One gate below the 100 m floor (52 m vertical), two safely above it.
    slant = np.array([60.0, 120.0, 180.0])
    expected = sorted(slant[1:] * np.sin(np.deg2rad(elevation)))  # ~103.9, ~155.9 m

    n = 120
    times = pd.date_range(START, periods=n, freq="5s")
    azimuths = np.tile([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], n // 6)
    vr = _radial_velocity(azimuths, elevation, u=8.0, v=3.0, w=0.0)

    ds = xr.Dataset(
        {
            "radial_wind_speed": (("time", "range_gate"), np.tile(vr[:, None], (1, len(slant)))),
            "intensity": (("time", "range_gate"), np.full((n, len(slant)), 1.05)),
            "azimuth": ("time", azimuths),
            "elevation": ("time", np.full(n, elevation)),
            "distance": ("range_gate", slant),
            "lat": ("index", np.array([41.167])),
            "lon": ("index", np.array([-71.58])),
        },
        coords={"time": times},
    )

    result = process_scanning_lidar(ds, START, "z01", LOCATION, time_window=10)

    assert result is not None, "scanning processor returned no result"
    got = sorted(result["wind_profiles"])
    assert got == pytest.approx(expected, abs=0.01), (
        f"profile heights {got} differ from slant*sin(el) {expected}: "
        f"heights must not be shifted by any elevation offset"
    )
    # Documented scanning-lidar floor: no retrievals below 100 m AGL.
    assert min(got) >= 100.0
