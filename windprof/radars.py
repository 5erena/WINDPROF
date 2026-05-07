"""NOAA PSL 915 MHz radar wind profiler processing

Reads consensus-averaged horizontal and vertical winds from the PSL
text file format (one section per beam-cycle). Used at Nantucket and
Block Island; the Narragansett radar is excluded from the current
release pending instrument validation, and Cape Cod has no radar.

Entry points:

  - ``process_radar_time_series``: native PSL text format
  - ``process_radar_netcdf_time_series``: pre-ingested NetCDF version

Beam geometry (typically one vertical + four oblique at 75° zenith) and
range-gate spacing (60 m low-mode / 200 m high-mode) are parsed from the
file header. Adapters with a different radar format would need a new
parser; this module is otherwise format-agnostic at the merging
interface.

A documented Nantucket beam-steering malfunction (May 10-16, 2025)
restricts the radar to vertical-only operation during that window; the
exclusion is encoded in ``config.MALFUNCTION_PERIODS``.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import re
from collections import defaultdict
from .config import (get_wind_correction, get_w_sign_correction,
                     get_ground_elevation_from_config, get_instrument_coordinates,
                     round_profile_values, TIME_WINDOW_MINUTES)

### Radar Processing

def _parse_radar_site_info(section):
    """
    Pull beam geometry and site coordinates out of the first section
    header of a NOAA PSL radar text file. Field positions are fixed by
    the file format and are referenced by line number below.
    """
    lines = section.strip().split('\n')

    # Line 5: "BEAMS  N" — number of beams in the scan pattern.
    num_beams = int(lines[4].strip().split()[1])

    # Line 3: "<lat> <lon> <alt>" — radar site location.
    try:
        location_info = lines[2].strip().split()
        lat = float(location_info[0])
        lon = float(location_info[1])
    except Exception as e:
        print(f"Warning: Could not extract coordinates from radar file: {e}, using fallback")
        lat, lon = np.nan, np.nan

    # Line 9: alternating azimuth/elevation pairs for each beam.
    beam_line = lines[8].strip().split()
    beam_azimuths = []
    beam_elevations = []

    for i in range(0, len(beam_line), 2):
        azimuth = float(beam_line[i])
        elevation = float(beam_line[i+1])
        # Some files write the elevation with an implicit decimal
        # (e.g., 9000 -> 90.00°); rescale defensively.
        if elevation > 90:
            elevation = elevation / 100
        beam_azimuths.append(azimuth)
        beam_elevations.append(elevation)

    return {
        'num_beams': num_beams,
        'lat': lat,
        'lon': lon,
        'beam_azimuths': beam_azimuths,
        'beam_elevations': beam_elevations
    }

def _compute_vertical_velocity(radial_velocities, beam_elevations, w_sign_correction=1, threshold=50):
    """
    Use the beam pointed straight up (elevation ≈ 90°) for vertical
    velocity directly, rather than fitting w from the off-zenith beams.
    Reject |w| > 50 m/s as physically implausible and apply the 
    site-specific w sign convention.
    """
    vertical_indices = [i for i, elev in enumerate(beam_elevations) if np.isclose(elev, 90, atol=1)]

    if vertical_indices:
        w_direct = radial_velocities[vertical_indices[0]]
        if abs(w_direct) > threshold:
            return np.nan
        else:
            return w_direct * w_sign_correction

    return np.nan

def _process_radar_section(section, site_info, snr_threshold, wind_dir_correction=0, w_sign_correction=1):
    """
    Process one time-stamped section of the radar text file into a
    {height: {ws, wd, w}} profile, applying the standard PSL QC plus an
    SNR threshold. The PSL format encodes one row per range gate; the
    section ends at a "$" delimiter.
    """
    lines = section.strip().split('\n')
    data_start_idx = next(i for i, line in enumerate(lines) if "HT" in line)

    profiles = {}
    total_points = 0
    points_passed = 0
    qc_failures = {'beam_count': [], 'quality_flag': []}

    data_lines = lines[data_start_idx+1:]
    for line in data_lines:
        if line.strip() == '$':
            break

        values = line.split()
        if len(values) < 13:
            continue

        try:
            height = float(values[0]) * 1000  # km -> m
            total_points += 1

            # 999999 is the PSL sentinel for missing/no-retrieval.
            if values[1] == '999999':
                qc_failures['quality_flag'].append({'height': height, 'reason': 'Missing data'})
                continue

            quality_flag = int(values[3])
            if quality_flag != 0:
                qc_failures['quality_flag'].append({'height': height, 'flag': quality_flag})
                continue

            wind_speed = float(values[1])
            wind_direction = float(values[2])

            wind_direction = (wind_direction + wind_dir_correction) % 360

            # PSL stores per-beam SNR in cols 10-12 and per-beam radar QC
            # flags in the last 3 cols. A beam must clear *both* checks.
            snr_values = np.array([float(values[10]), float(values[11]), float(values[12])])
            radar_qc_values = np.array([float(values[-3]), float(values[-2]), float(values[-1])])

            good_beams = (radar_qc_values <= 2) & (snr_values >= snr_threshold)
            beams_passed = np.sum(good_beams)

            # Need 2 of 3 beams to constrain horizontal wind
            if beams_passed < 2:
                qc_failures['beam_count'].append({
                    'height': height, 'passed': beams_passed, 'total': len(snr_values)
                })
                continue

            # Radial velocities are columns 4 .. 4+num_beams-1
            radial_velocities = [float(val) for val in values[4:4+site_info['num_beams']]]
            vertical_velocity = _compute_vertical_velocity(
                radial_velocities, site_info['beam_elevations'], w_sign_correction
            )

            # Radar heights are reported in AGL already; no offset applied
            agl_height = height
            profiles[agl_height] = {
                'ws': wind_speed,
                'wd': wind_direction,
                'w': vertical_velocity
            }

            points_passed += 1

        except (ValueError, IndexError) as e:
            continue

    return profiles, {
        'total_points': total_points,
        'points_passed': points_passed,
        'qc_failures': qc_failures
    }

def _print_radar_qc_summary(results, qc_failures):
    """One-line per-metric QC summary printed when ``verbose=True``"""
    qc_meta = results['qc_metadata']
    print(f"\nRadar QC Summary:")
    print(f"Total points: {qc_meta['total_points']}")
    print(f"Points passed: {qc_meta['points_passed']} ({qc_meta['points_passed']/qc_meta['total_points']*100:.1f}%)")
    print(f"Beam count failures: {len(qc_failures['beam_count'])}")
    print(f"Quality flag failures: {len(qc_failures['quality_flag'])}")
    print(f"Final profiles: {len(results['profiles'])}")

def process_radar_data(filename, location, start_time=None, end_time=None, target_date=None,
                       snr_threshold=-23, wind_dir_correction=None, verbose=False):
    """
    Read one NOAA PSL radar text file and produce time-averaged profiles
    over the requested window, with combined SNR + per-beam QC filtering.

    Parameters
    ----------
    filename : str
        Path to the radar text file. The date is extracted from the
        filename (8-digit YYYYMMDD).
    location : str
        Site identifier ('nantucket' or 'block_island').
    start_time, end_time : str or datetime, optional
        Time-of-day filter in 'HH:MM' or datetime form. None disables.
    snr_threshold : float
        Minimum per-beam SNR (dB).
    wind_dir_correction : float, optional
        Override the location's default WD correction.
    """
    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'radar')

    date_match = re.search(r'\d{8}', os.path.basename(filename))
    if date_match:
        target_date = pd.to_datetime(date_match.group(0)).date()
    else:
        raise ValueError(f"Could not extract date from filename: {filename}")

    def parse_time_param(time_param):
        # Accept 'HH:MM' strings, ISO datetimes, or datetime objects
        if time_param is None:
            return None, None
        if isinstance(time_param, str):
            if ':' in time_param:
                return map(int, time_param.split(':'))
            else:
                dt = pd.to_datetime(time_param)
                return dt.hour, dt.minute
        else:
            return time_param.hour, time_param.minute

    start_hour, start_minute = parse_time_param(start_time)
    end_hour, end_minute = parse_time_param(end_time)

    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading radar file: {e}")
        return []

    # PSL files are a sequence of "$"-delimited sections, one per scan
    sections = content.split('$')

    # Pre-parse the section header line to bin sections by timestamp,
    # then apply the date+time-of-day filter before doing any column work
    valid_sections = {}
    for section in sections:
        if 'HT' not in section:
            continue

        try:
            lines = section.strip().split('\n')
            # The timestamp line is the first one with at least 6 fields
            # whose first 5 tokens are all integers (YY MM DD HH MM).
            time_line = next(line for line in lines if len(line.split()) >= 6 and
                           all(part.isdigit() for part in line.split()[:5]))
            time_info = time_line.strip().split()

            year = 2000 + int(time_info[0])
            month = int(time_info[1])
            day = int(time_info[2])
            hour = int(time_info[3])
            minute = int(time_info[4])

            section_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

            if target_date and section_time.date() != target_date:
                continue
            if start_hour is not None and (section_time.hour < start_hour or
                                         (section_time.hour == start_hour and
                                          start_minute is not None and section_time.minute < start_minute)):
                continue
            if end_hour is not None and (section_time.hour > end_hour or
                                       (section_time.hour == end_hour and
                                        end_minute is not None and section_time.minute > end_minute)):
                continue

            if section_time not in valid_sections:
                valid_sections[section_time] = []
            valid_sections[section_time].append(section)

        except (StopIteration, ValueError, IndexError):
            continue

    if not valid_sections:
        if verbose:
            print(f"No matching radar data found for the specified time range: {start_time} to {end_time}")
        return []

    time_periods = sorted(valid_sections.keys())

    first_section = valid_sections[time_periods[0]][0]
    site_info = _parse_radar_site_info(first_section)

    # Some radar file revisions omit GPS in the header; fall back to site config
    if np.isnan(site_info['lat']) or np.isnan(site_info['lon']):
        fallback_coords = get_instrument_coordinates(location, 'radar')
        if fallback_coords:
            site_info['lat'], site_info['lon'] = fallback_coords[0], fallback_coords[1]
        else:
            if verbose:
                print(f"Warning: No coordinates available for radar at {location}")

    ground_elevation = get_ground_elevation_from_config(location, 'radar')

    if verbose:
        print(f"Processing radar data from {time_periods[0]} to {time_periods[-1]}")
        print(f"Number of 15-minute periods: {len(time_periods)}")

    w_correction = get_w_sign_correction('radar', location)

    all_measurements = []
    total_qc_stats = {
        'total_points': 0,
        'points_passed': 0,
        'qc_failures': {'beam_count': [], 'quality_flag': []}
    }

    for current_time in time_periods:
        matching_sections = valid_sections[current_time]
        for section in matching_sections:
            section_profiles, section_qc = _process_radar_section(
                section, site_info, snr_threshold, wind_dir_correction, w_correction
            )

            for height, profile in section_profiles.items():
                all_measurements.append((height, profile['ws'], profile['wd'], profile['w']))

            total_qc_stats['total_points'] += section_qc['total_points']
            total_qc_stats['points_passed'] += section_qc['points_passed']
            total_qc_stats['qc_failures']['beam_count'].extend(section_qc['qc_failures']['beam_count'])
            total_qc_stats['qc_failures']['quality_flag'].extend(section_qc['qc_failures']['quality_flag'])

    # Average measurements per height across all sections in the window
    height_measurements = defaultdict(list)
    for height, ws, wd, w in all_measurements:
        height_measurements[height].append({'ws': ws, 'wd': wd, 'w': w})

    final_profiles = {}
    for height, measurements in height_measurements.items():
        final_profiles[height] = round_profile_values({
            'ws': sum(m['ws'] for m in measurements) / len(measurements),
            'wd': sum(m['wd'] for m in measurements) / len(measurements),
            'w': sum(m['w'] for m in measurements) / len(measurements)
        })

    results = {
        'time': time_periods[0],
        'instrument': 'RADAR',
        'instrument_code': 'radar',
        'profiles': final_profiles,
        'ground_elevation': ground_elevation,
        'latitude': site_info['lat'],
        'longitude': site_info['lon'],
        'beam_azimuths': site_info['beam_azimuths'],
        'beam_elevations': site_info['beam_elevations'],
        'qc_metadata': {
            'total_points': total_qc_stats['total_points'],
            'points_passed': total_qc_stats['points_passed'],
            'snr_threshold': snr_threshold
        }
    }

    if verbose:
        _print_radar_qc_summary(results, total_qc_stats['qc_failures'])

    return results

def match_radar_to_target_time(radar_results, target_time, max_time_diff_minutes=10):
    """
    Pick the radar result closest in time to ``target_time``, returning
    None if nothing is within ``max_time_diff_minutes``. Used to align
    radar's native 15-minute cadence to the 10-minute analysis grid.
    """
    closest_result = None
    min_diff = float('inf')

    for result in radar_results:
        radar_time = result['time']
        time_diff = abs((radar_time - target_time).total_seconds()) / 60

        if time_diff < min_diff and time_diff <= max_time_diff_minutes:
            min_diff = time_diff
            closest_result = result

    return closest_result

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

            # Some file revisions report height in km; investigate and convert.
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
                # SNR is split per-component; require all available
                # components to clear the threshold.
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
                # NetCDF input does not provide per-component error
                # estimates — record NaN so downstream code can still
                # query the field uniformly.
                'u_err': np.nan, 'v_err': np.nan, 'w_err': np.nan,
                'ws': ws, 'wd': wd_corrected,
                'ws_err': np.nan, 'wd_err': np.nan,
                'lat': lat, 'lon': lon
            })

    ds.close()

    if verbose:
        print(f"Successfully processed {len(measurements)} height-time measurements")

    return measurements

### Time-Series Handling

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
        # Snap each measurement's timestamp to the start of its
        # containing analysis window (e.g., 12:07 -> 12:00 for a 10-min
        # window) so we can group by window.
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

            avg_measurement = {
                'ws': np.mean([m['ws'] for m in measurements]),
                'wd': np.mean([m['wd'] for m in measurements]),
                'w': np.mean([m['w'] for m in measurements]),
                'u': np.mean([m['u'] for m in measurements]),
                'v': np.mean([m['v'] for m in measurements]),
                'ws_err': np.mean([m['ws_err'] for m in measurements if not np.isnan(m['ws_err'])]),
                'wd_err': np.mean([m['wd_err'] for m in measurements if not np.isnan(m['wd_err'])]),
                'w_err': np.mean([m['w_err'] for m in measurements if not np.isnan(m['w_err'])])
            }

            profiles[height] = round_profile_values(avg_measurement)

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

def process_radar_time_series(filename, start_time, end_time, location, **kwargs):
    """
    Process a NOAA PSL radar text file across a date range and align its
    native 15-minute scans to the 10-minute analysis grid via
    nearest-neighbor matching with a 10-minute tolerance.
    """
    # Pad the radar query window by ±15 minutes so a scan
    # near the analysis boundary still gets considered.
    radar_start = start_time - pd.Timedelta(minutes=15)
    radar_end = end_time + pd.Timedelta(minutes=15)

    all_radar_results = []
    # Snap to the nearest 15-minute boundary at or before radar_start.
    current_radar_time = radar_start.replace(minute=(radar_start.minute // 15) * 15, second=0, microsecond=0)

    while current_radar_time < radar_end:
        try:
            radar_result = process_radar_data(
                filename,
                location,
                start_time=current_radar_time,
                end_time=current_radar_time + pd.Timedelta(minutes=15),
                **kwargs
            )

            if radar_result and radar_result.get('profiles'):
                standardized_result = {
                    'time': current_radar_time,
                    'instrument': 'radar',
                    'profiles': radar_result['profiles'],
                    'availability': {'meets_threshold': True},
                    'latitude': radar_result.get('latitude'),
                    'longitude': radar_result.get('longitude'),
                    'ground_elevation': radar_result.get('ground_elevation'),
                }
                all_radar_results.append(standardized_result)
        except Exception as e:
            if kwargs.get('verbose', False):
                print(f"No radar data for period {current_radar_time}: {e}")

        current_radar_time += pd.Timedelta(minutes=15)

    if not all_radar_results:
        return []

    # Build the analysis-grid intervals and pull the closest radar scan
    # within tolerance for each one.
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    time_intervals = []
    while current_time < end_time:
        time_intervals.append(current_time)
        current_time += pd.Timedelta(minutes=TIME_WINDOW_MINUTES)

    results = []
    for interval_time in time_intervals:
        matched_radar = match_radar_to_target_time(all_radar_results, interval_time, max_time_diff_minutes=10)
        if matched_radar:
            interval_result = {
                'time': interval_time,
                'instrument': 'radar',
                'profiles': matched_radar['profiles'],
                'availability': matched_radar['availability'],
                'latitude': matched_radar['latitude'],
                'longitude': matched_radar['longitude'],
                'ground_elevation': matched_radar.get('ground_elevation'),
            }
            results.append(interval_result)

    print(f"Mapped radar data to {len(results)} out of {len(time_intervals)} 10-minute intervals")
    return results
