"""Tests for PSL WINDS radar ingestion: parsing, selection-based mapping,
mode combination, QC gating, and wrap-safe direction handling.

The synthetic builder mirrors the WFIP3 sample block layout: 3 beams
(vertical plus two obliques), two modes per timestamp, 29-min consensus
at 15-min cadence. Real-file tests skip when the samples are absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windprof.radars import (parse_psl_winds_file, select_consensus_for_window,
                                    combine_mode_profiles, process_radar_time_series,
                                    average_netcdf_bin, MET_QC_KEEP_DEFAULT)

REPO = Path(__file__).resolve().parents[1]


def _sample_path(name):
    """Radar samples are gitignored: sample_data/ is populated locally."""
    for candidate in (REPO / "sample_data" / name, REPO / name):
        if candidate.exists():
            return candidate
    return REPO / "sample_data" / name


SAMPLES = [_sample_path("bloc.radar.20241117.txt"),
           _sample_path("nant.radar.20241126.txt")]

BEAMS = " 30 90.0  30 66.4  300 66.4"
HEADER_COLS = ("    HT      SPD      DIR   MET_QC      RAD      RAD      RAD"
               "      CNT      CNT      CNT      SNR      SNR      SNR       QC       QC       QC")


def winds_block(begin, gates, dur_min=29, beams_line=BEAMS, ngates=None):
    """One '$'-terminated WINDS consensus block.

    gates: list of (ht_km, spd, dir, met_qc, rad3) tuples; CNT/SNR/QC-beam
    columns are filled with passing values (SNR 10, beam QC 0).
    """
    ngates = ngates if ngates is not None else len(gates)
    lines = [
        " BID",
        " WINDS    rev 5.1",
        "  41.17  -71.58     35",
        f"  {begin.year - 2000:02d} {begin.month:02d} {begin.day:02d}"
        f" {begin.hour:02d} {begin.minute:02d} {begin.second:02d}   0",
        f"  {dur_min}  3  {ngates}",
        " 00:11 (0.0) 01:12 (0.0) 01:12 (0.0)",
        "  50 50 50 50 417 417 75 75",
        "  21.8  21.8  0  4083 4083 50 50 417 417",
        beams_line,
        HEADER_COLS,
    ]
    for ht, spd, wdir, qc, rad in gates:
        rad_s = "".join(f" {r:8}" for r in rad)
        lines.append(f" {ht:.3f} {spd:8} {wdir:8} {qc:8}{rad_s}"
                     "       11       11       11       10       10       10"
                     "      0.0      0.0      0.0")
    return "\n".join(lines) + "\n$\n"


def low_gates(spds=(10.0, 12.0, 14.0), dirs=(327, 330, 333), qcs=(0, 0, 0)):
    heights = (0.121, 0.179, 0.237)
    return [(h, s, d, q, (-0.3, 2.1, 4.0))
            for h, s, d, q in zip(heights, spds, dirs, qcs)]


def high_gates(spds=(11.0, 13.0, 15.0), dirs=(328, 331, 334), qcs=(0, 0, 0)):
    heights = (0.164, 0.261, 0.358)
    return [(h, s, d, q, (-0.2, 2.5, 4.6))
            for h, s, d, q in zip(heights, spds, dirs, qcs)]


def write_winds(tmp_path, blocks):
    path = tmp_path / "test.radar.20241117.txt"
    path.write_text("".join(blocks))
    return str(path)


T0 = pd.Timestamp("2024-11-17 00:00:23")
T1 = pd.Timestamp("2024-11-17 00:15:06")


def test_parser_reads_begin_time_duration_beams_and_sentinels(tmp_path):
    gates = low_gates(spds=(10.0, 999999, 14.0))  # middle gate SPD missing
    path = write_winds(tmp_path, [winds_block(T0, gates)])
    parsed = parse_psl_winds_file(path)

    assert parsed['site']['elev_m'] == 35.0
    assert len(parsed['blocks']) == 1
    b = parsed['blocks'][0]
    # Line 4 is the consensus BEGIN time, not its center or end.
    assert b['begin'] == T0
    assert b['duration_min'] == 29
    assert b['nbeams'] == 3
    assert b['beam_el'] == [90.0, 66.4, 66.4]
    assert b['gates'][0]['ht_m'] == pytest.approx(121.0)
    assert np.isnan(b['gates'][1]['spd'])
    assert b['gate_spacing_m'] == pytest.approx(58.0, abs=0.1)


def test_selection_overlap_and_center_tiebreak(tmp_path):
    path = write_winds(tmp_path, [
        winds_block(pd.Timestamp("2024-11-17 00:00:00"), low_gates()),
        winds_block(pd.Timestamp("2024-11-17 00:15:00"), low_gates()),
    ])
    parsed = parse_psl_winds_file(path)
    by_begin = {}
    for b in parsed['blocks']:
        by_begin.setdefault(b['begin'], []).append(b)
    begins = sorted(by_begin)

    def pick(win_start_str):
        ws = pd.Timestamp(win_start_str)
        return select_consensus_for_window(begins, by_begin, ws,
                                           ws + pd.Timedelta(minutes=10))

    # Window fully inside the first consensus only.
    assert pick("2024-11-17 00:00:00") == pd.Timestamp("2024-11-17 00:00:00")
    # Both consensus windows cover 00:15-00:25 fully (overlap tie);
    # the first block's center (00:14:30) is nearer the window center.
    assert pick("2024-11-17 00:15:00") == pd.Timestamp("2024-11-17 00:00:00")
    # At 00:20 the second block overlaps more (10 min vs 9 min).
    assert pick("2024-11-17 00:20:00") == pd.Timestamp("2024-11-17 00:15:00")
    assert pick("2024-11-17 03:00:00") is None


def test_profiles_are_selected_not_averaged(tmp_path):
    b0 = winds_block(pd.Timestamp("2024-11-17 00:00:00"),
                     low_gates(spds=(10.0, 12.0, 14.0), dirs=(350, 350, 350)))
    b1 = winds_block(pd.Timestamp("2024-11-17 00:15:00"),
                     low_gates(spds=(20.0, 22.0, 24.0), dirs=(10, 10, 10)))
    path = write_winds(tmp_path, [b0, b1])

    results = process_radar_time_series(
        path, pd.Timestamp("2024-11-17 00:00:00"),
        pd.Timestamp("2024-11-17 00:10:00"), 'block_island')

    assert len(results) == 1
    prof = results[0]['profiles']
    # Values equal the selected block's gates exactly: an average with the
    # second block (dir 10 deg) would have collapsed wd toward 180.
    assert prof[121.0]['ws'] == pytest.approx(10.0)
    assert prof[121.0]['wd'] == pytest.approx(350.0)
    assert prof[237.0]['ws'] == pytest.approx(14.0)
    # w: vertical beam RAD (-0.3, +toward radar) x block_island w_sign (-1).
    assert prof[121.0]['w'] == pytest.approx(0.3)


def test_mode_combination_low_priority_high_above(tmp_path):
    path = write_winds(tmp_path, [
        winds_block(T0, low_gates()),
        winds_block(T0, high_gates()),
    ])
    results = process_radar_time_series(
        path, pd.Timestamp("2024-11-17 00:00:00"),
        pd.Timestamp("2024-11-17 00:10:00"), 'block_island')

    heights = sorted(results[0]['profiles'])
    # High mode is kept only above the low ceiling (237 m); 164 m is dropped.
    assert heights == pytest.approx([121.0, 179.0, 237.0, 261.0, 358.0])


def test_mode_combination_high_alone_when_low_invalid(tmp_path):
    path = write_winds(tmp_path, [
        winds_block(T0, low_gates(qcs=(9, 9, 9))),
        winds_block(T0, high_gates()),
    ])
    results = process_radar_time_series(
        path, pd.Timestamp("2024-11-17 00:00:00"),
        pd.Timestamp("2024-11-17 00:10:00"), 'block_island')
    assert sorted(results[0]['profiles']) == pytest.approx([164.0, 261.0, 358.0])


def test_met_qc_gating_default_and_opt_in(tmp_path):
    gates = [(0.121, 10.0, 300, 0, (-0.3, 2.1, 4.0)),
             (0.179, 11.0, 301, 2, (-0.3, 2.2, 4.1)),
             (0.237, 12.0, 302, 7, (-0.3, 2.3, 4.2)),
             (0.295, 13.0, 303, 8, (-0.3, 2.4, 4.3)),
             (0.353, 14.0, 304, 9, (-0.3, 2.5, 4.4))]
    path = write_winds(tmp_path, [winds_block(T0, gates)])
    win = (pd.Timestamp("2024-11-17 00:00:00"), pd.Timestamp("2024-11-17 00:10:00"))

    default = process_radar_time_series(path, *win, 'block_island')
    assert sorted(default[0]['profiles']) == [121.0]
    assert MET_QC_KEEP_DEFAULT == frozenset({0})

    widened = process_radar_time_series(path, *win, 'block_island',
                                        met_qc_keep=frozenset({0, 2}))
    assert sorted(widened[0]['profiles']) == [121.0, 179.0]


def test_missing_vertical_beam_gives_nan_w_but_keeps_gate(tmp_path):
    gates = [(0.121, 10.0, 300, 0, (999999, 2.1, 4.0))]
    path = write_winds(tmp_path, [winds_block(T0, gates)])
    results = process_radar_time_series(
        path, pd.Timestamp("2024-11-17 00:00:00"),
        pd.Timestamp("2024-11-17 00:10:00"), 'block_island')
    entry = results[0]['profiles'][121.0]
    assert entry['ws'] == pytest.approx(10.0)
    assert np.isnan(entry['w'])


def test_netcdf_bin_average_is_circular_and_nan_safe():
    def m(wd, w):
        return {'ws': 5.0, 'wd': wd, 'w': w, 'u': 1.0, 'v': 1.0,
                'ws_err': np.nan, 'wd_err': np.nan, 'w_err': np.nan}

    out = average_netcdf_bin([m(350.0, 2.0), m(10.0, np.nan)])
    # Circular mean of 350/10 is 0 (not 180); NaN w must not poison the bin.
    assert min(out['wd'], 360 - out['wd']) == pytest.approx(0.0, abs=1e-6)
    assert out['w'] == pytest.approx(2.0)
    assert np.isnan(out['ws_err'])


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_real_sample_files(sample):
    if not sample.exists():
        pytest.skip(f"sample file not present: {sample.name}")
    parsed = parse_psl_winds_file(str(sample))
    blocks = parsed['blocks']
    assert len(blocks) == 192  # 96 timestamps x 2 modes
    assert {b['duration_min'] for b in blocks} == {29}
    assert {b['nbeams'] for b in blocks} == {3}
    begins = {b['begin'] for b in blocks}
    assert len(begins) == 96

    location = 'block_island' if sample.name.startswith('bloc') else 'nantucket'
    day = min(begins).normalize()
    results = process_radar_time_series(str(sample), day, day + pd.Timedelta(hours=24),
                                        location)
    # Nearly every 10-min window is served by one consensus; a window whose
    # selected consensus has no QC-passing gates yields no interval (correct).
    assert 140 <= len(results) <= 144
    for res in results[:12]:
        for ht, prof in res['profiles'].items():
            assert ht >= 121.0  # first PSL gate; heights are native AGL
            assert 0.0 <= prof['wd'] < 360.0
            assert np.isfinite(prof['ws'])
