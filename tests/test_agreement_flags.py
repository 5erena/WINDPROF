"""Tests for inter-instrument agreement flags (merged-value QC).

Flags cover only the instruments that formed the merged value: one excluded
from the merge (radar below the lidar ceiling, or a MALFUNCTION_PERIODS
window) must not mark cells it did not contribute to as suspect.
"""

import pandas as pd
import pytest

from windprof.merging import average_multi_instrument_profiles

WS = ['ws', 'wd', 'w']


def profile(height, ws, wd=180.0, w=0.1):
    return {height: {'ws': ws, 'wd': wd, 'w': w}}


def test_excluded_instrument_cannot_flag_suspect():
    """Below 1000 m the radar is excluded wherever lidar has data; a large
    lidar-radar disagreement there must not flag the lidar-only value."""
    height = 500.0
    data = {
        'lidar_z01': profile(height, ws=10.0),
        'radar': profile(height, ws=20.0, w=9.0),  # ws spread 10 m/s; |w| > 7
    }
    out = average_multi_instrument_profiles(data, [height], WS)

    assert out['profiles'][height]['ws'] == pytest.approx(10.0)
    assert out['flags'][height]['ws'] == 0
    assert out['flags'][height]['w'] == 0


def test_used_instruments_disagreeing_still_flag_suspect():
    height = 500.0
    data = {
        'lidar_z01': profile(height, ws=10.0),
        'lidar_z02': profile(height, ws=20.0),
    }
    out = average_multi_instrument_profiles(data, [height], WS)

    assert out['profiles'][height]['ws'] == pytest.approx(15.0)
    assert out['flags'][height]['ws'] == 1  # spread 10 > max(1.5, 0.5*median)


def test_malfunction_period_instrument_excluded_from_flags():
    """NANT radar malfunction window 2025-05-10 to 16: inside it the radar
    is excluded from the merged value and from the flags."""
    height = 1500.0
    data = {
        'lidar_z01': profile(height, ws=10.0),
        'radar': profile(height, ws=20.0),
    }

    inside = average_multi_instrument_profiles(
        data, [height], WS, location='nantucket',
        time_interval=pd.Timestamp('2025-05-12 06:00:00'))
    assert inside['profiles'][height]['ws'] == pytest.approx(10.0)
    assert inside['flags'][height]['ws'] == 0

    outside = average_multi_instrument_profiles(
        data, [height], WS, location='nantucket',
        time_interval=pd.Timestamp('2025-06-01 06:00:00'))
    assert outside['profiles'][height]['ws'] == pytest.approx(15.0)
    assert outside['flags'][height]['ws'] == 1