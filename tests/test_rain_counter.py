"""Cape Cod surface meteorology from a WXT536 whose telegrams are occasionally corrupted.

Rain is reported as a running total, not per interval: these tests pin how that
counter becomes 10 minute totals across outages and restarts, how a dry window is
told from a silent station, and that other sites are untouched.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof.merging import (process_surface_met_for_date, rain_counter_increments,
                                     screen_single_readings)

DAY = "2024-08-15"
T0 = pd.Timestamp(DAY)


def window(result, hh, mm):
    key = int((T0 + pd.Timedelta(hours=hh, minutes=mm)).timestamp())
    return result[key]


def write_caco_day(tmp_path, counter, ptu_mask=None):
    """One day on a 5 s grid in the Cape Cod file layout; ``counter=None``
    writes a file with no rain counter at all, as some delivered days are."""
    t = pd.date_range(T0, periods=17280, freq="5s")
    ptu = np.full(t.size, 1012.0)
    if ptu_mask is not None:
        ptu = np.where(ptu_mask, ptu, np.nan)
    data = {"air_pressure": ("time", ptu),
            "air_temperature": ("time", np.where(np.isfinite(ptu), 20.0, np.nan)),
            "relative_humidity": ("time", np.where(np.isfinite(ptu), 80.0, np.nan))}
    if counter is not None:
        data["rain_accumulation"] = ("time", counter)
    ds = xr.Dataset(data, coords={"time": t})
    d = tmp_path / "2024" / "08" / "15"
    d.mkdir(parents=True)
    ds.to_netcdf(d / "caco.met.z01.00.20240815.000000.nc")
    return t


def test_increments_rise_restart_and_gap():
    t = pd.date_range(T0, periods=7, freq="5s").append(pd.DatetimeIndex([T0 + pd.Timedelta(minutes=30)]))
    counter = np.array([5.0, 5.0, 5.2, 5.5, 0.1, 0.1, np.nan, 3.0])
    out = rain_counter_increments(t, counter)
    assert np.isnan(out[0])                          # no predecessor
    assert out[1:4] == pytest.approx([0.0, 0.2, 0.3])
    assert out[4] == pytest.approx(0.1)              # restart: rain since the restart
    assert out[5] == pytest.approx(0.0)
    assert np.isnan(out[6])
    assert np.isnan(out[7])                          # rise across a 30 min gap is not placed


def test_reading_after_an_outage_is_not_a_glitch():
    t = pd.DatetimeIndex([T0, T0 + pd.Timedelta(seconds=5), T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=5, seconds=5)])
    out = rain_counter_increments(t, np.array([106.15, 106.15, 0.01, 0.02]))
    assert np.nansum(out) == pytest.approx(0.01)     # only the rise after the restart, not the outage


def test_corrupted_reading_is_not_rain():
    t = pd.date_range(T0, periods=7, freq="5s")
    for glitch in (5.80, 99.0):                      # digits lost, or a spurious high reading
        counter = np.array([57.83, 57.83, 57.83, glitch, 57.83, 57.83, 57.84])
        out = rain_counter_increments(t, counter)
        assert np.nansum(out) == pytest.approx(0.01)
        assert np.isnan(out[3])


def test_small_dip_and_recovery_is_not_rain():
    t = pd.date_range(T0, periods=6, freq="5s")
    out = rain_counter_increments(t, np.array([66.06, 66.06, 66.00, 66.06, 66.06, 66.06]))
    assert np.nansum(out) == pytest.approx(0.0)


def test_persistent_drop_is_a_restart():
    t = pd.date_range(T0, periods=6, freq="5s")
    counter = np.array([481.77, 481.77, 0.0, 0.0, 0.0, 0.02])
    out = rain_counter_increments(t, counter)
    assert np.nansum(out) == pytest.approx(0.02)


def test_ten_minute_totals_dry_and_silent_windows(tmp_path):
    t = pd.date_range(T0, periods=17280, freq="5s")
    counter = np.full(t.size, np.nan)
    rain = (t >= T0 + pd.Timedelta(hours=1)) & (t < T0 + pd.Timedelta(hours=1, minutes=10))
    # rain telegram only while raining: 0.01 mm per 10 s report for ten minutes
    reports = np.flatnonzero(rain)[::2]
    counter[reports] = 20.0 + 0.01 * np.arange(1, reports.size + 1)
    silent = (t >= T0 + pd.Timedelta(hours=3)) & (t < T0 + pd.Timedelta(hours=3, minutes=10))
    write_caco_day(tmp_path, counter, ptu_mask=~silent)
    res = process_surface_met_for_date(DAY, data_dir=str(tmp_path), location="cape_cod")
    # the first report of the event follows a long gap, so its rise is not counted
    assert window(res, 1, 0)["precipitation"] == pytest.approx(0.01 * (reports.size - 1), abs=1e-9)
    assert window(res, 2, 0)["precipitation"] == 0.0     # no rain report, station reporting
    assert np.isnan(window(res, 3, 0)["precipitation"])
    assert window(res, 2, 0)["pressure"] == pytest.approx(1012.0)


def test_day_without_counter_variable(tmp_path):
    write_caco_day(tmp_path, None)
    res = process_surface_met_for_date(DAY, data_dir=str(tmp_path), location="cape_cod")
    assert window(res, 12, 0)["precipitation"] == 0.0


def test_lone_corrupted_ptu_readings_do_not_reach_the_means(tmp_path):
    t = pd.date_range(T0, periods=17280, freq="5s")
    i = int(np.flatnonzero(t == T0 + pd.Timedelta(hours=13, seconds=5))[0])
    p = np.full(t.size, 1016.3); p[i] = 16.3                 # leading digits lost
    rh = np.full(t.size, 73.3); rh[i + 1] = 733.0            # decimal point lost
    ta = np.full(t.size, 14.8); ta[i + 2] = 48.0             # digit lost
    ds = xr.Dataset({"air_pressure": ("time", p), "air_temperature": ("time", ta),
                     "relative_humidity": ("time", rh)}, coords={"time": t})
    d = tmp_path / "2024" / "08" / "15"; d.mkdir(parents=True)
    ds.to_netcdf(d / "caco.met.z01.00.20240815.000000.nc")
    w = window(process_surface_met_for_date(DAY, data_dir=str(tmp_path), location="cape_cod"), 13, 0)
    assert w["pressure"] == pytest.approx(1016.3)
    assert w["relative_humidity"] == pytest.approx(73.3)
    assert w["temperature"] == pytest.approx(14.8)


def test_screen_keeps_real_change_and_drops_out_of_range():
    t = pd.date_range(T0, periods=6, freq="5s")
    rising = np.array([15.0, 17.5, 20.0, 22.5, 25.0, 27.5])  # the screen drops only a reading both neighbors contradict
    assert np.all(np.isfinite(screen_single_readings(t, rising, 2.0, (-52.0, 60.0))))
    pair = np.array([87.0, 87.0, 870.0, 870.0, 87.0, 87.0])
    out = screen_single_readings(t, pair, 5.0, (0.0, 100.0))
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.nansum(np.isfinite(out)) == 4


def test_other_sites_keep_per_sample_precipitation(tmp_path):
    t = pd.date_range(T0, periods=1440, freq="1min")
    precip = np.zeros(t.size); precip[60:70] = 0.2
    pres = np.full(t.size, 1010.0); pres[125] = 10.0
    ds = xr.Dataset({"atmos_pressure": ("time", pres),
                     "precipitation": ("time", precip)}, coords={"time": t})
    d = tmp_path / "nant"; d.mkdir()
    ds.to_netcdf(d / "nant.met.z05.c1.20240815.000000.nc")
    res = process_surface_met_for_date(DAY, data_dir=str(tmp_path), location="nantucket")
    assert window(res, 1, 0)["precipitation"] == pytest.approx(2.0)
    assert window(res, 2, 0)["precipitation"] == 0.0
    assert window(res, 2, 0)["pressure"] == pytest.approx(1010.0 - 100.0)   # no screen outside Cape Cod
