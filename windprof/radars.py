"""NOAA PSL 915 MHz radar wind profiler processing

Reads consensus-averaged winds from the PSL "WINDS" sub-hourly text format
(Weber-Wuertz) and maps them onto the WINDPROF 10-minute analysis grid by
*selection*: each 10-minute profile receives the single most-overlapping
consensus record, and consensus blocks are never averaged together. The
WFIP3 radars ran 29-minute consensus at ~15-minute cadence, so successive
records overlap in time.

Format notes (verified against WFIP3 sample files):

  - Each "$"-terminated block is one consensus record; every timestamp has
    two blocks (low mode: finer gate spacing and lower ceiling; high mode:
    coarser spacing, higher ceiling).
  - Per-gate columns: HT (km AGL), SPD, DIR, MET_QC, then per-beam RAD /
    CNT / SNR / QC column groups sized by the header beam count. 999999
    marks missing values. Radar heights are AGL.

Gates are kept when MET_QC (resultant-wind quality) is in ``met_qc_keep``,
which defaults to {0}: valid retrievals only. Widening it to {0, 2} to
admit estimated winds is a science decision.

Entry points:

  - ``process_radar_time_series``: PSL text format (Nantucket, Block Island)
  - ``process_radar_netcdf_time_series``: pre-ingested NetCDF (Rhode Island)
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from .config import (get_wind_correction, get_w_sign_correction,
                     get_ground_elevation_from_config, get_instrument_coordinates,
                     round_profile_values, wind_direction_average, TIME_WINDOW_MINUTES)

# Resultant-wind QC values retained by default (0 = valid). PSL also emits
# 2 = estimated, 7 = suspect, 8 = invalid, 9 = missing.
MET_QC_KEEP_DEFAULT = frozenset({0})

# PSL missing-value sentinel for SPD/DIR/RAD columns.
_MISSING = 999999.0

# Vertical velocities beyond this are implausible for a wind profiler.
_W_SANITY_MAX = 50.0


### PSL "WINDS" text format


def _to_float(token):
    """Float conversion mapping the PSL 999999 sentinel (and parse errors) to NaN."""
    try:
        value = float(token)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if value == _MISSING else value


def parse_psl_winds_file(filename):
    """
    Parse one PSL WINDS text file into a list of consensus blocks.

    Returns
    -------
    dict with keys ``site`` and ``blocks``:
        site : {lat, lon, elev_m} from the first block header (NaN if absent)
        blocks : one dict per consensus block, with keys
            begin        : pd.Timestamp (UTC, start of the consensus window)
            duration_min : consensus averaging time in minutes
            nbeams       : beam count
            ngates       : gate count declared in the header
            beam_az/beam_el : per-beam pointing in degrees
            gate_spacing_m  : vertical spacing derived from the first two gates
            gates        : list of per-gate dicts with keys
                           ht_m, spd, dir, met_qc, rad, snr, qc_beam
    """
    with open(filename, 'r') as f:
        content = f.read()

    site = {'lat': np.nan, 'lon': np.nan, 'elev_m': np.nan}
    blocks = []

    for section in content.split('$'):
        lines = [line for line in section.splitlines() if line.strip()]
        if len(lines) < 11 or 'WINDS' not in lines[1]:
            continue

        try:
            lat, lon, elev = (float(x) for x in lines[2].split()[:3])
            yy, mo, dd, hh, mi, ss, utc_off = (int(x) for x in lines[3].split()[:7])
            dur_min, nbeams, ngates = (int(float(x)) for x in lines[4].split()[:3])

            # Column-label line anchors the layout; beam line is directly above it.
            hdr_i = next(i for i, l in enumerate(lines) if l.split()[:2] == ['HT', 'SPD'])
            beam_tokens = [float(x) for x in lines[hdr_i - 1].split()]
        except (StopIteration, ValueError, IndexError):
            continue

        beam_az, beam_el = [], []
        for i in range(0, 2 * nbeams, 2):
            az, el = beam_tokens[i], beam_tokens[i + 1]
            # Some file revisions write elevation with an implicit decimal
            # (9000 = 90.00 deg).
            if el > 90:
                el = el / 100.0
            beam_az.append(az)
            beam_el.append(el)

        # Header line 4 is the BEGIN time of the consensus window. The
        # utc-offset field is hours from UTC (0 for all WFIP3 files).
        begin = pd.Timestamp(year=2000 + yy, month=mo, day=dd,
                             hour=hh, minute=mi, second=ss)
        if utc_off != 0:
            begin = begin - pd.Timedelta(hours=utc_off)

        gates = []
        for line in lines[hdr_i + 1:]:
            parts = line.split()
            if len(parts) < 4 + 4 * nbeams:
                continue
            try:
                met_qc = int(float(parts[3]))
            except ValueError:
                continue
            gates.append({
                'ht_m': float(parts[0]) * 1000.0,  # km AGL -> m AGL
                'spd': _to_float(parts[1]),
                'dir': _to_float(parts[2]),
                'met_qc': met_qc,
                'rad': [_to_float(x) for x in parts[4:4 + nbeams]],
                'snr': [_to_float(x) for x in parts[4 + 2 * nbeams:4 + 3 * nbeams]],
                'qc_beam': [_to_float(x) for x in parts[4 + 3 * nbeams:4 + 4 * nbeams]],
            })

        if not gates:
            continue

        spacing = gates[1]['ht_m'] - gates[0]['ht_m'] if len(gates) > 1 else np.nan
        blocks.append({
            'begin': begin, 'duration_min': dur_min, 'nbeams': nbeams,
            'ngates': ngates, 'beam_az': beam_az, 'beam_el': beam_el,
            'gate_spacing_m': spacing, 'gates': gates,
        })
        if np.isnan(site['lat']):
            site.update({'lat': lat, 'lon': lon, 'elev_m': elev})

    return {'site': site, 'blocks': blocks}


def select_consensus_for_window(begins, blocks_by_begin, window_start, window_end):
    """
    Pick the consensus begin-time whose averaging window most overlaps
    [window_start, window_end). Ties (common with overlapping consensus
    windows) go to the block whose center is nearest the window center;
    any remaining tie goes to the earlier begin for determinism.
    Returns the chosen begin timestamp, or None if nothing overlaps.
    """
    window_center = window_start + (window_end - window_start) / 2
    best_key, best_begin = None, None

    for begin in begins:
        block = blocks_by_begin[begin][0]
        cons_end = begin + pd.Timedelta(minutes=block['duration_min'])
        overlap = (min(cons_end, window_end) - max(begin, window_start)).total_seconds()
        if overlap <= 0:
            continue
        center_dist = abs(((begin + (cons_end - begin) / 2) - window_center).total_seconds())
        key = (overlap, -center_dist, -begin.value)
        if best_key is None or key > best_key:
            best_key, best_begin = key, begin

    return best_begin


def _beam_snr_ok(gate, snr_threshold, min_good_beams=2):
    """
    Require at least ``min_good_beams`` beams with SNR >= threshold and
    per-beam QC <= 2. The -23 dB default is the PM-specified per-beam SNR
    floor for the WFIP3 profilers, applied alongside the MET_QC gate.
    """
    good = 0
    for snr, qc in zip(gate['snr'], gate['qc_beam']):
        if np.isfinite(snr) and snr >= snr_threshold and np.isfinite(qc) and qc <= 2:
            good += 1
    return good >= min_good_beams


def _vertical_beam_index(beam_el):
    """Index of the vertical beam (elevation ~90 deg), or None."""
    for i, el in enumerate(beam_el):
        if np.isclose(el, 90.0, atol=1.0):
            return i
    return None


def _block_profile_entries(block, met_qc_keep, snr_threshold,
                           wind_dir_correction, w_sign_correction):
    """
    Extract QC-passing gates from one consensus block as
    [(ht_m, {'ws','wd','w'}), ...]. No averaging: values map through from
    the consensus record, with the site azimuth correction and w sign
    convention applied.
    """
    vert_i = _vertical_beam_index(block['beam_el'])
    entries = []
    for gate in block['gates']:
        if gate['met_qc'] not in met_qc_keep:
            continue
        if np.isnan(gate['spd']) or np.isnan(gate['dir']):
            continue
        if not _beam_snr_ok(gate, snr_threshold):
            continue

        # RAD is positive toward the radar; the site w_sign_corrections entry
        # converts to the up-positive product convention.
        w = np.nan
        if vert_i is not None:
            w_raw = gate['rad'][vert_i]
            if np.isfinite(w_raw) and abs(w_raw) <= _W_SANITY_MAX:
                w = w_raw * w_sign_correction

        entries.append((gate['ht_m'], {
            'ws': gate['spd'],
            'wd': (gate['dir'] + wind_dir_correction) % 360,
            'w': w,
        }))
    return entries


def combine_mode_profiles(low_entries, high_entries):
    """
    Merge low- and high-mode gate lists into one profile.

    Low mode is kept wherever it has valid gates (finer vertical
    resolution); high mode contributes only strictly above the highest
    valid low-mode gate, so the two pulse volumes are never mixed at one
    height and no transition constant is needed.
    """
    if not low_entries:
        return sorted(high_entries)
    cutoff = max(ht for ht, _ in low_entries)
    combined = list(low_entries) + [e for e in high_entries if e[0] > cutoff]
    return sorted(combined)


def process_radar_time_series(filename, start_time, end_time, location,
                              met_qc_keep=MET_QC_KEEP_DEFAULT,
                              snr_threshold=-23, wind_dir_correction=None,
                              verbose=False, **kwargs):
    """
    Map PSL consensus records onto the 10-minute analysis grid by selection:
    each window takes the single most-overlapping consensus and combines its
    two mode blocks by gate. One 29-minute consensus can serve several
    adjacent windows; records are never averaged together.

    Returns a list of interval dicts shaped for
    ``merging.convert_radar_results_to_combined_format``.
    """
    if not os.path.exists(filename):
        print(f"Error: Radar file not found: {filename}")
        return []

    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'radar')
    w_sign = get_w_sign_correction('radar', location)
    ground_elevation = get_ground_elevation_from_config(location, 'radar')

    parsed = parse_psl_winds_file(filename)
    blocks = parsed['blocks']
    if not blocks:
        print(f"No consensus blocks parsed from {os.path.basename(filename)}")
        return []

    lat, lon = parsed['site']['lat'], parsed['site']['lon']
    if np.isnan(lat) or np.isnan(lon):
        fallback = get_instrument_coordinates(location, 'radar')
        if fallback:
            lat, lon = fallback

    blocks_by_begin = defaultdict(list)
    for block in blocks:
        blocks_by_begin[block['begin']].append(block)
    begins = sorted(blocks_by_begin)

    window_start = start_time.replace(
        minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES,
        second=0, microsecond=0)
    window_len = pd.Timedelta(minutes=TIME_WINDOW_MINUTES)

    results = []
    n_windows = 0
    while window_start < end_time:
        n_windows += 1
        window_end = window_start + window_len
        begin = select_consensus_for_window(begins, blocks_by_begin,
                                            window_start, window_end)
        if begin is not None:
            # Low mode = finer gate spacing; high mode fills in above it.
            ts_blocks = sorted(blocks_by_begin[begin],
                               key=lambda b: b['gate_spacing_m'])
            mode_entries = [
                _block_profile_entries(b, met_qc_keep, snr_threshold,
                                       wind_dir_correction, w_sign)
                for b in ts_blocks
            ]
            low = mode_entries[0] if mode_entries else []
            high = mode_entries[1] if len(mode_entries) > 1 else []
            profiles = {ht: round_profile_values(values)
                        for ht, values in combine_mode_profiles(low, high)}

            if profiles:
                results.append({
                    'time': window_start,
                    'instrument': 'radar',
                    'instrument_code': 'radar',
                    'profiles': profiles,
                    'availability': {'meets_threshold': True},
                    'latitude': lat,
                    'longitude': lon,
                    'ground_elevation': ground_elevation,
                    'consensus_begin': str(begin),
                })
        window_start = window_end

    print(f"Mapped radar data to {len(results)} out of {n_windows} 10-minute intervals")
    return results


### Rhode Island NetCDF format


def process_radar_netcdf_data(filename, start_time=None, end_time=None,
                             qc_threshold=10, snr_threshold=None,
                             wind_dir_correction=0, w_sign_correction=1,
                             verbose=False):
    """
    Read Rhode Island radar NetCDF file (pre-computed winds with QC flags).
    Unlike the PSL text format, this file already contains wind_speed, wind_direction,
    and u/v/w arrays, so processing is just QC filtering + correction application.

    Parameters
    ----------
    filename : str
    start_time, end_time : datetime, optional
        None disables filtering.
    qc_threshold : int
        Maximum acceptable value of the wind QC flag (default 10 keeps
        everything except hard-missing).
    snr_threshold : float, optional
        Optional secondary SNR filter; the file's built-in QC should
        suffice, so this is None by default.
    """
    try:
        ds = xr.open_dataset(filename, engine='netcdf4')
    except Exception as e:
        if verbose:
            print(f"Error reading NetCDF file: {e}")
        return []

    # File timestamps are stored as Unix seconds, not the xarray default.
    times = pd.to_datetime(ds.time.values, unit='s')

    if start_time is not None or end_time is not None:
        time_mask = np.ones(len(times), dtype=bool)
        if start_time is not None:
            time_mask &= times >= start_time
        if end_time is not None:
            time_mask &= times < end_time

        if not np.any(time_mask):
            if verbose:
                print("No data in specified time range")
            ds.close()
            return []
    else:
        time_mask = np.ones(len(times), dtype=bool)

    heights = ds.height.values[time_mask]

    wind_speed_data = ds.wind_speed.values[time_mask]
    wind_direction_data = ds.wind_direction.values[time_mask]
    u_wind_data = ds.u_wind.values[time_mask]
    v_wind_data = ds.v_wind.values[time_mask]
    w_wind_data = ds.w_wind.values[time_mask]

    wind_qc_data = ds.wind_speed_and_direction_data_quality.values[time_mask]

    lat = float(ds.lat.values)
    lon = float(ds.lon.values)
    alt = float(ds.alt.values)

    measurements = []
    valid_times = times[time_mask]

    if verbose:
        print(f"Processing {len(valid_times)} timesteps from {valid_times[0]} to {valid_times[-1]}")

    for t_idx in range(len(valid_times)):
        timestamp = valid_times[t_idx]

        # The height grid varies per timestep in this file (the radar
        # mode can change), so re-derive the valid range gates each pass.
        height_profile = heights[t_idx, :]
        valid_height_mask = ~np.isnan(height_profile)

        if not np.any(valid_height_mask):
            continue

        for r_idx in range(len(height_profile)):
            if not valid_height_mask[r_idx]:
                continue

            height_agl = height_profile[r_idx]

            # Some file revisions report height in km, not m.
            if height_agl < 10:
                height_agl = height_agl * 1000

            ws = wind_speed_data[t_idx, r_idx]
            wd = wind_direction_data[t_idx, r_idx]
            u = u_wind_data[t_idx, r_idx]
            v = v_wind_data[t_idx, r_idx]
            w = w_wind_data[t_idx, r_idx]

            qc_flag = wind_qc_data[t_idx, r_idx]

            if qc_flag >= qc_threshold or np.isnan(ws) or np.isnan(wd) or np.isnan(u) or np.isnan(v):
                continue

            if snr_threshold is not None:
                # SNR is stored per component (u/v/w) in this file.
                has_snr = any(f'{var}_signal_to_noise_ratio' in ds.variables
                            for var in ['u_wind', 'v_wind', 'w_wind'])
                if has_snr:
                    snr_ok = True
                    for var in ['u_wind', 'v_wind', 'w_wind']:
                        if f'{var}_signal_to_noise_ratio' in ds.variables:
                            snr_val = ds[f'{var}_signal_to_noise_ratio'].values[time_mask][t_idx, r_idx]
                            if not np.isnan(snr_val) and snr_val < snr_threshold:
                                snr_ok = False
                                break
                    if not snr_ok:
                        continue

            wd_corrected = (wd + wind_dir_correction) % 360
            w_corrected = w * w_sign_correction if not np.isnan(w) else w

            measurements.append({
                'time': timestamp,
                'height': height_agl,
                'u': u, 'v': v, 'w': w_corrected,
                # No per-component error estimates in the NetCDF input; NaN
                # keeps the schema uniform.
                'u_err': np.nan, 'v_err': np.nan, 'w_err': np.nan,
                'ws': ws, 'wd': wd_corrected,
                'ws_err': np.nan, 'wd_err': np.nan,
                'lat': lat, 'lon': lon
            })

    ds.close()

    if verbose:
        print(f"Successfully processed {len(measurements)} height-time measurements")

    return measurements


def _nanmean_or_nan(values):
    """Mean of the finite entries, NaN when none are finite (no warnings)."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def average_netcdf_bin(measurements):
    """
    Average the measurements that fell into one (interval, height) bin.

    Wind direction uses the circular mean so bins straddling 0/360 deg do
    not collapse toward 180 deg; w and the error fields skip NaN entries
    instead of letting one missing sample null the bin.
    """
    return {
        'ws': _nanmean_or_nan([m['ws'] for m in measurements]),
        'wd': wind_direction_average([m['wd'] for m in measurements
                                      if np.isfinite(m['wd'])]) % 360,
        'w': _nanmean_or_nan([m['w'] for m in measurements]),
        'u': _nanmean_or_nan([m['u'] for m in measurements]),
        'v': _nanmean_or_nan([m['v'] for m in measurements]),
        'ws_err': _nanmean_or_nan([m['ws_err'] for m in measurements]),
        'wd_err': _nanmean_or_nan([m['wd_err'] for m in measurements]),
        'w_err': _nanmean_or_nan([m['w_err'] for m in measurements]),
    }


def process_radar_netcdf_time_series(filenames, start_time, end_time, location='rhode_island', **kwargs):
    """
    Process a list of NetCDF radar files into 10-minute interval profiles
    aligned with the analysis grid. Measurements are grouped by their
    interval-truncated timestamp and averaged within each (interval, height) bin.
    """
    wind_dir_correction = kwargs.get('wind_dir_correction', get_wind_correction(location, 'radar'))
    w_sign_correction = kwargs.get('w_sign_correction', get_w_sign_correction('radar', location))

    all_measurements = []
    for filename in filenames:
        try:
            measurements = process_radar_netcdf_data(
                filename, start_time, end_time,
                wind_dir_correction=wind_dir_correction,
                w_sign_correction=w_sign_correction,
                **kwargs
            )
            all_measurements.extend(measurements)
        except Exception as e:
            if kwargs.get('verbose', False):
                print(f"Error processing {filename}: {e}")
            continue

    if not all_measurements:
        return []

    # interval_data[interval_start_time][height_agl] = [measurement, ...]
    interval_data = defaultdict(lambda: defaultdict(list))

    for measurement in all_measurements:
        time_10min = measurement['time'].replace(
            minute=(measurement['time'].minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES,
            second=0, microsecond=0
        )

        height = measurement['height']
        interval_data[time_10min][height].append(measurement)

    results = []
    for interval_time in sorted(interval_data.keys()):
        if interval_time < start_time or interval_time >= end_time:
            continue

        profiles = {}
        coords = None

        for height, measurements in interval_data[interval_time].items():
            if not measurements:
                continue

            profiles[height] = round_profile_values(average_netcdf_bin(measurements))

            if coords is None:
                coords = (measurements[0]['lat'], measurements[0]['lon'])

        if profiles:
            result = {
                'time': interval_time,
                'instrument': 'radar',
                'profiles': profiles,
                'availability': {'meets_threshold': True},
                'latitude': coords[0] if coords else np.nan,
                'longitude': coords[1] if coords else np.nan
            }
            results.append(result)

    if kwargs.get('verbose', False):
        print(f"Mapped radar data to {len(results)} 10-minute intervals")

    return results
