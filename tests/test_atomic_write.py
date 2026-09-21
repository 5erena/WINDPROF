"""The daily product is replaced atomically, so a failed write costs nothing.

Written to a temp file in the same directory and renamed into place: the file
on disk is only ever the complete old one or the complete new one.
"""

import errno
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from windprof import discovery_and_export as dex

CANONICAL = 'nant.windprof.z01.c1.20240811.000000.nc'
TWIN = 'nant.windprof.z01.c1.20240811.013000.nc'


def _interval(ts):
    """One merged interval, the shape save_results_to_netcdf consumes."""
    return {
        'time': int(ts.timestamp()),
        'ground_elevation': 3.0,
        'wind_profiles': {
            'merged': {100.0: {'ws': 8.0, 'wd': 210.0, 'w': 0.05}},
            'individual': {'lidar_z01': {100.0: {'ws': 8.0, 'wd': 210.0, 'w': 0.05}}},
            'quality_flags': {100.0: {'ws': 0, 'wd': 0, 'w': 0}},
            'contributing_instruments': ['lidar_z01'],
        },
        'turbulence_profiles': {
            'merged': {100.0: {'ti': 0.08}},
            'individual': {'lidar_z01': {100.0: {'ti': 0.08}}},
            'quality_flags': {100.0: {'ti': 0}},
            'contributing_instruments': ['lidar_z01'],
        },
    }


@pytest.fixture
def results():
    t0 = pd.Timestamp('2024-08-11 00:00:00')
    return {'time_intervals':
            [_interval(t0 + pd.Timedelta(minutes=10 * i)) for i in range(6)]}


def _write(tmp_path, results, name=CANONICAL):
    return dex.save_results_to_netcdf(results, str(tmp_path / name),
                                      location='nantucket')

def test_failed_write_leaves_the_delivered_file_untouched(tmp_path, results, monkeypatch):
    target = tmp_path / CANONICAL
    _write(tmp_path, results)
    before = target.read_bytes()

    def boom(self, path, *a, **kw):
        # Fail the way a full disk does, after the temp file exists, so the
        # rename is what protects the delivered product.
        with open(path, 'wb') as fh:
            fh.write(b'partial')
        raise OSError(errno.ENOSPC, 'No space left on device')

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', boom)
    with pytest.raises(OSError):
        _write(tmp_path, results)

    assert target.read_bytes() == before, 'delivered product was modified'
    with xr.open_dataset(target) as ds:
        assert 'wind_speed_merged' in ds, 'delivered product no longer opens'


def test_failed_write_leaves_no_temp_file_behind(tmp_path, results, monkeypatch):
    _write(tmp_path, results)

    def boom(self, path, *a, **kw):
        # Fail the way a full disk does, after the temp file exists, so the
        # rename is what protects the delivered product.
        with open(path, 'wb') as fh:
            fh.write(b'partial')
        raise OSError(errno.ENOSPC, 'No space left on device')

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', boom)
    with pytest.raises(OSError):
        _write(tmp_path, results)

    assert not list(tmp_path.glob('.*.tmp')), 'temp file leaked'
    # A leaked temp must never look like a product to the reprocess checkpoint.
    assert sorted(p.name for p in tmp_path.glob('*.nc')) == [CANONICAL]


def test_twins_are_cleared_only_after_the_new_product_lands(tmp_path, results, monkeypatch):
    """A failed write must leave the day's other copies alone, not orphan it."""
    _write(tmp_path, results)
    (tmp_path / TWIN).write_bytes((tmp_path / CANONICAL).read_bytes())

    def boom(self, path, *a, **kw):
        # Fail the way a full disk does, after the temp file exists, so the
        # rename is what protects the delivered product.
        with open(path, 'wb') as fh:
            fh.write(b'partial')
        raise OSError(errno.ENOSPC, 'No space left on device')

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', boom)
    with pytest.raises(OSError):
        _write(tmp_path, results)
    assert (tmp_path / TWIN).exists(), 'twin removed despite the write failing'

    monkeypatch.undo()
    _write(tmp_path, results)
    assert not (tmp_path / TWIN).exists()
    assert sorted(p.name for p in tmp_path.glob('*.nc')) == [CANONICAL]


def test_precheck_refuses_when_the_filesystem_is_too_full(tmp_path, results, monkeypatch):
    """A doomed campaign should stop on day one with a legible message."""
    monkeypatch.setattr(dex, 'MIN_FREE_BYTES_FLOOR', 1 << 60)
    with pytest.raises(OSError) as e:
        _write(tmp_path, results)
    assert e.value.errno == errno.ENOSPC
    assert 'refusing to write' in str(e.value)
    assert not list(tmp_path.glob('*.nc')), 'nothing should be written'


def test_precheck_does_not_consult_root_reserved_blocks(tmp_path, results, monkeypatch):
    """f_bavail, not f_bfree: an unprivileged writer cannot use the reserve."""
    seen = {}
    real = os.statvfs

    class FakeStat:
        f_frsize = 4096
        f_bavail = 0
        f_bfree = 1 << 40

    def fake(path):
        seen['path'] = path
        return FakeStat()

    monkeypatch.setattr(os, 'statvfs', fake)
    with pytest.raises(OSError) as e:
        _write(tmp_path, results)
    assert e.value.errno == errno.ENOSPC
    monkeypatch.setattr(os, 'statvfs', real)


def test_product_permissions_match_a_plain_write(tmp_path, results):
    """The temp file must not carry 0600 through the rename."""
    out = _write(tmp_path, results)
    plain = tmp_path / 'plain.nc'
    xr.Dataset({'x': ('t', np.arange(3.0))}).to_netcdf(plain)
    assert (os.stat(out).st_mode & 0o777) == (os.stat(plain).st_mode & 0o777)
