"""Smoke tests for the WINDPROF pipeline.

The integration test runs the full pipeline on one day of real BLOC data and
is skipped unless ``tests/data/`` is populated; ``tests/data/README.md`` says
how to fetch the sample inputs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "tests" / "data"
SAMPLE_DATA = SAMPLE_ROOT / "data"
SAMPLE_ARCHIVE = SAMPLE_ROOT / "archive"
SAMPLE_RESULTS = SAMPLE_ROOT / "results"

# Reference test day: BLOC has the smallest instrument inventory of the four sites.
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
    """Run the pipeline end-to-end on one day of BLOC data.

    Checks structural sanity, not byte equivalence against a reference file,
    so the test survives harmless code changes.
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
        assert ds.sizes.get("time", 0) > 0, "time dimension is empty"
        assert ds.sizes.get("height", 0) > 0, "height dimension is empty"

        for var in ["wind_speed", "wind_direction", "height"]:
            assert var in ds, f"Missing expected variable: {var}"

        h = ds["height"].values
        assert np.all(np.diff(h) > 0), "height grid is not monotone increasing"

        # No specific count: instrument availability varies day to day.
        ws = ds["wind_speed"].values
        assert np.isfinite(ws).any(), "wind_speed is entirely NaN"

        finite_ws = ws[np.isfinite(ws)]
        assert finite_ws.min() >= 0.0, "negative wind_speed detected"
        assert finite_ws.max() < 100.0, "implausibly large wind_speed detected"

        wd = ds["wind_direction"].values
        finite_wd = wd[np.isfinite(wd)]
        if finite_wd.size:
            assert finite_wd.min() >= 0.0 and finite_wd.max() < 360.0, \
                "wind_direction outside [0, 360)"
