"""Smoke tests for the WINDPROF pipeline.

Two tiers:

  - Import-only tests (always run): every module in ``windprof``
    imports cleanly, and a couple of core numerical helpers (``fit_chi_winds``)
    return sensible results on synthetic data.
  - Integration test (skipped unless ``tests/data/`` is populated): runs the
    full pipeline on one day of real BLOC campaign data and checks that the
    output NetCDF is structurally valid. See ``tests/data/README.md`` for
    instructions on populating the sample inputs from the DOE Wind Data Hub.

Deeper validation lives in the Lipari et al. (2026) companion paper and in
notebook-driven comparisons against WINDoe retrievals.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


MODULES = [
    "windprof.config",
    "windprof.wind_analysis",
    "windprof.quality_control",
    "windprof.anemometers",
    "windprof.lidar_parsers",
    "windprof.lidars",
    "windprof.radars",
    "windprof.merging",
    "windprof.discovery_and_export",
    "windprof.pipeline",
    "windprof.process_parallel",
    "windprof.plotting",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_vad_fit_on_synthetic_scan():
    """Inject a noise-free uniform flow and confirm VAD recovers it to <0.1 m/s."""
    from windprof.wind_analysis import fit_chi_winds

    true_u, true_v, true_w = 6.0, -3.0, 0.2
    az_deg = np.linspace(0, 350, 36)
    el_deg = np.full_like(az_deg, 75.0)

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    # Radial velocity for an idealized uniform wind field sampled by a VAD scan.
    vr = true_u * np.sin(az) * np.cos(el) + true_v * np.cos(az) * np.cos(el) + true_w * np.sin(el)

    result = fit_chi_winds(az_deg, el_deg, vr)
    assert np.isfinite(result["ws"])
    recovered_ws = np.sqrt(result["u"] ** 2 + result["v"] ** 2)
    true_ws = np.sqrt(true_u ** 2 + true_v ** 2)
    assert abs(recovered_ws - true_ws) < 0.1


# ---------------------------------------------------------------------------
# Integration test (real data; skipped if tests/data/ is empty)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "tests" / "data"
SAMPLE_DATA = SAMPLE_ROOT / "data"
SAMPLE_ARCHIVE = SAMPLE_ROOT / "archive"
SAMPLE_RESULTS = SAMPLE_ROOT / "results"

# Reference test day: BLOC 2024-09-15 (smallest instrument inventory of the
# four sites; first verified during refactor validation).
REF_SITE = "block_island"
REF_SITE_CODE = "bloc"
REF_DATE = "2024-09-15"


def _sample_data_present():
    """True if at least one input file exists under tests/data/data/."""
    return SAMPLE_DATA.exists() and any(SAMPLE_DATA.rglob("*.nc"))


@pytest.mark.skipif(
    not _sample_data_present(),
    reason="tests/data/ not populated; see tests/data/README.md to enable.",
)
def test_integration_bloc_one_day():
    """Run the pipeline end-to-end on one day of BLOC data and validate the output.

    Checks structural sanity (variables present, dimensions populated, no
    all-NaN heights for primary winds) rather than byte equivalence against a
    reference file, so the test stays robust under harmless code changes.
    """
    import xarray as xr

    SAMPLE_RESULTS.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["WINDPROF_DATA_PATH"] = str(SAMPLE_DATA)
    env["WINDPROF_ARCHIVE_PATH"] = str(SAMPLE_ARCHIVE)
    env["WINDPROF_RESULTS_PATH"] = str(SAMPLE_RESULTS)

    # Run as a subprocess so config.py picks up the test env vars at import time.
    result = subprocess.run(
        [sys.executable, "-m", "windprof.process_parallel",
         REF_SITE, "--start-date", REF_DATE, "--end-date", REF_DATE, "--no-skip"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"Pipeline failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    yyyymm = REF_DATE[:4] + REF_DATE[5:7]
    yyyymmdd = yyyymm + REF_DATE[8:10]
    out_path = (SAMPLE_RESULTS / REF_SITE_CODE / yyyymm / yyyymmdd /
                f"{REF_SITE_CODE}.windprof.z01.c1.{yyyymmdd}.000000.nc")
    assert out_path.exists(), f"Expected output NetCDF not produced: {out_path}"

    with xr.open_dataset(out_path) as ds:
        # Dimensions populated.
        assert ds.sizes.get("time", 0) > 0, "time dimension is empty"
        assert ds.sizes.get("height", 0) > 0, "height dimension is empty"

        # Required merged variables present.
        for var in ["wind_speed", "wind_direction", "height"]:
            assert var in ds, f"Missing expected variable: {var}"

        # Heights are monotone increasing.
        h = ds["height"].values
        assert np.all(np.diff(h) > 0), "height grid is not monotone increasing"

        # Some valid data for the dominant variable. We don't assert a specific
        # count because instrument availability varies day-to-day, but at least
        # one finite value should exist for a normal day.
        ws = ds["wind_speed"].values
        assert np.isfinite(ws).any(), "wind_speed is entirely NaN"

        # Plausible wind-speed range.
        finite_ws = ws[np.isfinite(ws)]
        assert finite_ws.min() >= 0.0, "negative wind_speed detected"
        assert finite_ws.max() < 100.0, "implausibly large wind_speed detected"

        # Wind direction is bounded [0, 360).
        wd = ds["wind_direction"].values
        finite_wd = wd[np.isfinite(wd)]
        if finite_wd.size:
            assert finite_wd.min() >= 0.0 and finite_wd.max() < 360.0, \
                "wind_direction outside [0, 360)"
