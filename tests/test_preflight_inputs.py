"""A campaign refuses to start when a configured datastream resolves nothing.

Unreadable instruments are not errors to the per-day processors, just fewer
contributors, so a run against an unreachable archive reports success while
omitting whole instruments and overwriting good products. Absent everywhere is a
broken path, absent for part of the range is a deployment window.
"""

import pandas as pd
import pytest

from windprof import discovery_and_export as dex
from windprof import process_parallel as pp

SITE = 'nantucket'
START, END = '2024-02-01', '2024-03-01'


def _fake_discovery(present_on):
    """Stand in for per-date discovery. present_on maps instrument -> predicate
    on the date string, so a stream can be made to appear for part of a range."""
    def fake(target_date, location='nantucket', verbose=False):
        return {inst: {'filename': f'/x/{inst}.nc', 'params': {}}
                for inst, pred in present_on.items() if pred(str(target_date))}
    return fake


ALL_STREAMS = ['lidar_z01', 'lidar_z02', 'lidar_z03', 'met_z02', 'surface_met']


def test_all_streams_present_passes(monkeypatch):
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date',
                        _fake_discovery({i: (lambda d: True) for i in ALL_STREAMS}))
    ok, report = dex.preflight_check_inputs(START, END, SITE, verbose=False)
    assert ok
    assert set(report) == set(ALL_STREAMS)


def test_one_stream_absent_everywhere_fails_and_is_named(monkeypatch):
    present = {i: (lambda d: True) for i in ALL_STREAMS}
    present['met_z02'] = lambda d: False
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date', _fake_discovery(present))
    ok, report = dex.preflight_check_inputs(START, END, SITE, verbose=False)
    assert not ok
    assert report['met_z02'] == 0
    assert all(report[i] > 0 for i in ALL_STREAMS if i != 'met_z02')


def test_every_stream_absent_is_the_unmounted_archive_case(monkeypatch):
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date',
                        _fake_discovery({}))
    ok, report = dex.preflight_check_inputs(START, END, SITE, verbose=False)
    assert not ok
    assert set(report.values()) == {0}


def test_a_deployment_window_is_not_a_failure(monkeypatch):
    present = {i: (lambda d: True) for i in ALL_STREAMS}
    present['met_z02'] = lambda d: d >= '2024-02-20'      # installed mid-range
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date', _fake_discovery(present))
    ok, report = dex.preflight_check_inputs(START, END, SITE, verbose=False)
    assert ok, 'an instrument with a real deployment window must not block the run'
    assert 0 < report['met_z02'] < report['lidar_z01']


def test_probing_stops_early_when_everything_is_found(monkeypatch):
    calls = []

    def counting(target_date, location='nantucket', verbose=False):
        calls.append(target_date)
        return {i: {'filename': 'x', 'params': {}} for i in ALL_STREAMS}

    monkeypatch.setattr(dex, 'discover_instrument_files_for_date', counting)
    ok, _ = dex.preflight_check_inputs('2024-02-01', '2025-09-09', SITE, verbose=False)
    assert ok
    assert len(calls) == 1, f'probed {len(calls)} dates when one sufficed'


def test_probes_are_spread_across_the_range(monkeypatch):
    """Probes must span the range, or a late-installed instrument reads as missing."""
    calls = []

    def counting(target_date, location='nantucket', verbose=False):
        calls.append(pd.to_datetime(target_date))
        return {}          # nothing found, so probing runs to the cap

    monkeypatch.setattr(dex, 'discover_instrument_files_for_date', counting)
    dex.preflight_check_inputs('2024-02-01', '2025-09-09', SITE, verbose=False,
                               max_probe_dates=10)
    assert len(calls) == 10
    assert calls[0] == pd.Timestamp('2024-02-01')
    assert calls[-1] == pd.Timestamp('2025-09-09')


def test_process_site_refuses_by_default_and_writes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(pp, 'preflight_check_inputs',
                        lambda *a, **k: (False, {'met_z02': 0}))
    called = []
    monkeypatch.setattr(pp, 'get_dates_to_process',
                        lambda *a, **k: called.append(1) or [])
    pp.process_site(SITE, START, END)
    assert not called, 'must refuse before enumerating dates, let alone writing'
    assert 'Refusing to start' in capsys.readouterr().out


def test_the_override_exists_and_proceeds(monkeypatch, capsys):
    monkeypatch.setattr(pp, 'preflight_check_inputs',
                        lambda *a, **k: (False, {'met_z02': 0}))
    monkeypatch.setattr(pp, 'get_dates_to_process', lambda *a, **k: [])
    pp.process_site(SITE, START, END, allow_missing_inputs=True)
    out = capsys.readouterr().out
    assert 'Proceeding without a missing datastream' in out
    assert 'Refusing to start' not in out


def _bloc_mapping(monkeypatch):
    """Block Island as configured: met_z01 and surface_met name the same file."""
    monkeypatch.setitem(dex.SITE_INSTRUMENT_MAPPINGS, 'block_island',
                        {'prefix': 'bloc',
                         'instruments': {'lidar_z01': 'bloc.lidar.z01.a0',
                                         'met_z01': 'bloc.met.z01.c1',
                                         'surface_met': 'bloc.met.z01.c1'}})


def test_a_datastream_named_by_two_keys_is_satisfied_by_either(monkeypatch):
    """Only surface_met is discoverable: the met_z01 wind is lifted from it
    downstream. Reachability belongs to the datastream, not the key."""
    _bloc_mapping(monkeypatch)
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date',
                        _fake_discovery({'lidar_z01': (lambda d: True),
                                         'surface_met': (lambda d: True)}))
    ok, report = dex.preflight_check_inputs(START, END, 'block_island', verbose=False)
    assert ok, 'met_z01 must not block the run: its datastream is reachable via surface_met'
    assert report['met_z01'] == 0 and report['surface_met'] > 0


def test_a_genuinely_unreachable_stream_still_fails_when_aliased(monkeypatch):
    _bloc_mapping(monkeypatch)
    monkeypatch.setattr(dex, 'discover_instrument_files_for_date',
                        _fake_discovery({'lidar_z01': (lambda d: True)}))
    ok, _ = dex.preflight_check_inputs(START, END, 'block_island', verbose=False)
    assert not ok, 'neither key resolved the met datastream, so it is unreachable'
