"""Tests for output filenames and success/failure bookkeeping.

The canonical daily name is built from the requested date at 000000, never
from the first data interval.
"""

import os

import pandas as pd
import pytest

import windprof.pipeline as pipeline
import windprof.process_parallel as process_parallel
from windprof.discovery_and_export import save_results_to_netcdf


def _results(first_interval="2024-06-12 03:10:00"):
    t0 = int(pd.Timestamp(first_interval).timestamp())
    interval = {
        "time": t0,
        "wind_profiles": {
            "merged": {100.0: {"ws": 5.0, "wd": 200.0}},
            "individual": {"lidar_z01": {100.0: {"ws": 5.0, "wd": 200.0}}},
            "quality_flags": {100.0: {"ws": 0, "wd": 0, "w": 0}},
        },
        "turbulence_profiles": {"merged": {}, "individual": {}, "quality_flags": {}},
    }
    return {"time_intervals": [interval]}


CANONICAL = "bloc.windprof.z01.c1.20240612.000000.nc"

def test_reprocess_clears_stale_twin(tmp_path):
    stale = tmp_path / "bloc.windprof.z01.c1.20240612.031000.nc"
    stale.write_bytes(b"stale twin from a partial-day run")
    written = save_results_to_netcdf(_results(), str(tmp_path / CANONICAL),
                                     location="block_island")
    remaining = sorted(p.name for p in tmp_path.glob("*.windprof.*.nc"))
    assert remaining == [CANONICAL]
    assert os.path.basename(written) == CANONICAL


def _run_pipeline_day(monkeypatch, tmp_path, save_behavior, plot_behavior):
    """Drive one day through the pipeline with save and plot stubbed,
    returning its output dict."""
    fake_configs = {"2024-06-12": {"lidar_z01": {"filename": "x.nc", "params": {}}}}
    monkeypatch.setattr(pipeline, "discover_instrument_files_for_date_range",
                        lambda *a, **k: fake_configs)
    monkeypatch.setattr(pipeline, "process_combined_profiles",
                        lambda *a, **k: _results())
    monkeypatch.setattr(pipeline, "save_results_to_netcdf", save_behavior)
    monkeypatch.setattr(pipeline, "_create_plots_for_results", plot_behavior)
    return pipeline.process_wind_profiles_for_date(
        "2024-06-12", base_dir=str(tmp_path), location="block_island",
        output_dir=str(tmp_path), create_hovmoller=False, save_netcdf=True)


def test_save_failure_marks_day_failed(monkeypatch, tmp_path):
    def failing_save(*a, **k):
        raise OSError("disk full")

    out = _run_pipeline_day(monkeypatch, tmp_path, failing_save, lambda *a, **k: [])
    assert out["netcdf_file"] is None
    assert out["processing_summary"]["status"] == "failed"
    assert "disk full" in out["processing_summary"]["error"]


def test_plot_crash_keeps_saved_day_successful(monkeypatch, tmp_path):
    saved = {}

    def good_save(results, path, **k):
        saved["path"] = path
        return path

    def crashing_plots(*a, **k):
        raise RuntimeError("matplotlib exploded")

    out = _run_pipeline_day(monkeypatch, tmp_path, good_save, crashing_plots)
    assert out["netcdf_file"] == saved["path"]
    assert os.path.basename(saved["path"]) == CANONICAL
    assert out["processing_summary"]["status"] == "success"


def test_check_if_processed_ignores_noncanonical_stale_file(monkeypatch, tmp_path):
    day_dir = tmp_path / "bloc" / "202406" / "20240612"
    day_dir.mkdir(parents=True)
    monkeypatch.setattr(process_parallel, "RESULTS_BASE_PATH", str(tmp_path) + "/")

    (day_dir / "bloc.windprof.z01.c1.20240612.031000.nc").write_bytes(b"stale")
    assert process_parallel.check_if_processed("2024-06-12", "block_island") is False

    (day_dir / CANONICAL).write_bytes(b"canonical")
    assert process_parallel.check_if_processed("2024-06-12", "block_island") is True


def test_worker_success_requires_written_file(monkeypatch):
    def fake_day(*a, **k):
        return {"results": {"time_intervals": [1]}, "netcdf_file": None,
                "processing_summary": {"status": "failed",
                                       "error": "NetCDF save failed: disk full"}}

    monkeypatch.setattr(process_parallel, "process_wind_profiles_for_date", fake_day)
    date_str, ok, reason = process_parallel.process_single_date(("2024-06-12", 2024, 6))
    assert ok is False and "disk full" in reason

    monkeypatch.setattr(process_parallel, "process_wind_profiles_for_date",
                        lambda *a, **k: {"results": {"time_intervals": [1]},
                                         "netcdf_file": "/x/" + CANONICAL,
                                         "processing_summary": {"status": "success"}})
    date_str, ok, reason = process_parallel.process_single_date(("2024-06-12", 2024, 6))
    assert ok is True and reason is None
