"""Scanning and profiling lidar processing

Entry points:

  - ``process_scanning_lidar`` / ``process_lidar_time_series``: VAD-based
    wind and turbulence retrievals from raw scanning lidar NetCDF files.
    Used for Halo XR, XR+, and Vaisala Galion instruments across all
    four WFIP3 sites.
  - ``process_profiling_lidar``: 10-minute averaged winds from profiling
    lidars; reads via ``lidar_parsers.parse_profiling_lidar_file`` and applies QC.
  - ``process_caco_z01_lidar``: separate entry for the CACO profiling
    lidar (WindCube V2-96), which uses an Excel-workbook delivery
    format and the inverted convention where z01 is profiling.

Per-instrument scan geometries (azimuth reset thresholds, elevation
angles, beam counts), QC metric choices (SNR vs. intensity), and QC
thresholds are configured in the location-keyed dispatch dict near the
end of the module. Adapters with different scan geometries or QC
signals should add a new entry to that dict.

Lidar-specific azimuth corrections (true-north reference) and
vertical-velocity sign corrections are read from ``config.py`` per
site/instrument; they are not embedded here.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from .config import (get_wind_correction, get_w_sign_correction,
                    get_ground_elevation_from_config, get_instrument_coordinates,
                    round_profile_values, wind_direction_average, TIME_WINDOW_MINUTES,
                    get_qc_params, MIN_VAD_BEAMS)
from .wind_analysis import fit_chi_winds, calculate_turbulence_metrics, apply_physics_based_qc
from .quality_control import filter_data_by_qc_criteria, calculate_data_availability
from .lidar_parsers import parse_profiling_lidar_file, parse_profiling_csv_file, parse_caco_lidar_file

# All WFIP3 scanning lidars run their VAD as a 6-position cone at 60° elevation.
# Only beams in this band enter the retrieval: composite scans (NANT z01) also hold
# 90° vertical stares and ~0.5-5° sector sweeps, whose horizontal leverage and
# sampled altitude do not match the height they would be keyed to. They are excluded
# from fitting, beam counting, QC/availability and turbulence segmentation, so
# min_beams counts only fit-entering VAD beams.
VAD_ELEVATION_DEG = 60.0
VAD_ELEVATION_TOL_DEG = 2.0

### Scan Segment Functions

def identify_scanning_lidar_segments(timestamps, az_angles, instrument='z01', min_points=3, debug=False, agl_height=None, reset_threshold=-200):
    """
    Identify scan segments for z01/z02 lidar, which use 6 fixed azimuth angles
    Parameters:
    -----------
    timestamps : array-like
        Array of measurement timestamps
    az_angles : array-like
        Array of azimuth angles
    instrument : str
        'z01' or 'z02' for instrument-specific debug messages
    min_points : int
        Minimum points needed for a valid scan
    debug : bool
        Whether to print debug information
    agl_height : float, optional
        Height value for debug messages
    reset_threshold : float, optional
        Azimuth difference threshold for detecting scan resets (default -200 for VAD
        360->0 wraps; use -50 for sector PPI scans that reset over a shorter span)
    Returns:
    --------
    list
        List of segment indices for each scan
    """
    if len(az_angles) < min_points:
        if debug and agl_height is not None:
            print(f"  Height {agl_height:.1f}m: Not enough points ({len(az_angles)})")
        return []

    az_diffs = np.diff(az_angles)
    # Look for large negative differences (e.g., 300°/360° -> 0°/60° for VAD,
    # or ~164° -> 86° for PPI sector scans with reset_threshold=-50)
    reset_indices = np.where(az_diffs < reset_threshold)[0]
    
    if len(reset_indices) < 2:
        if debug and agl_height is not None:
            print(f"  Height {agl_height:.1f}m: No clear scan pattern transitions found")
        if instrument.lower() == 'z02':
            az_close_to_zero = np.where((az_angles < 10) | (az_angles > 350))[0]
        else:  # z01
            az_close_to_zero = np.where(az_angles < 10)[0]
        
        if len(az_close_to_zero) < 2:
            if debug and agl_height is not None:
                print(f"  Height {agl_height:.1f}m: Unable to identify scans using azimuth values")
            point_count = len(az_angles)
            points_per_scan = 6  # Both instruments have 6 beam positions per scan
            num_scans = point_count // points_per_scan
            segments = []
            for i in range(num_scans):
                start_idx = i * points_per_scan
                end_idx = min((i+1) * points_per_scan, point_count)
                if end_idx - start_idx >= min_points:
                    segments.append(list(range(start_idx, end_idx)))
            
            if debug and agl_height is not None:
                print(f"  Height {agl_height:.1f}m: Created {len(segments)} segments based on {instrument.upper()} scan pattern")
            return segments
        
        segments = []
        for i in range(len(az_close_to_zero) - 1):
            start_idx = az_close_to_zero[i]
            end_idx = az_close_to_zero[i+1]
            
            # For z01, handle repeated 300° measurements
            if instrument.lower() == 'z01':
                segment_az = az_angles[start_idx:end_idx]
                unique_angles = np.unique(np.round(segment_az))
                if len(unique_angles) >= min_points:
                    segments.append(list(range(start_idx, end_idx)))
            else:  # z02
                if end_idx - start_idx >= min_points:
                    segments.append(list(range(start_idx, end_idx)))
        
        if debug and agl_height is not None:
            zero_marker = "0/360°" if instrument.lower() == 'z02' else "0°"
            print(f"  Height {agl_height:.1f}m: Created {len(segments)} segments using {zero_marker} azimuth markers")
        return segments
    
    segments = []
    
    def process_segment(start_idx, end_idx):
        if instrument.lower() == 'z01':
            segment_az = az_angles[start_idx:end_idx]
            # Count unique angles (rounded to handle small variations)
            unique_angles = np.unique(np.round(segment_az))
            if len(unique_angles) >= min_points:
                return list(range(start_idx, end_idx))
            return None
        else:  # z02
            if end_idx - start_idx >= min_points:
                return list(range(start_idx, end_idx))
            return None
    
    first_segment = process_segment(0, reset_indices[0] + 1)
    if first_segment:
        segments.append(first_segment)
    
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i] + 1
        end_idx = reset_indices[i+1] + 1
        segment = process_segment(start_idx, end_idx)
        if segment:
            segments.append(segment)
    
    last_segment = process_segment(reset_indices[-1] + 1, len(az_angles))
    if last_segment:
        segments.append(last_segment)
    
    if debug and agl_height is not None:
        transition_type = "azimuth transitions" if len(reset_indices) >= 2 else "fallback method"
        print(f"  Height {agl_height:.1f}m: Identified {len(segments)} scan segments using {transition_type}")
        if segments:
            first_scan_az = az_angles[segments[0]]
            if instrument.lower() == 'z01':
                unique_angles = len(np.unique(np.round(first_scan_az)))
                print(f"    First scan: {len(segments[0])} points, {unique_angles} unique angles, "
                      f"azimuth range: {np.min(first_scan_az):.1f}° to {np.max(first_scan_az):.1f}°")
            else:  # z02
                print(f"    First scan: {len(segments[0])} points, azimuth range: "
                      f"{np.min(first_scan_az):.1f}° to {np.max(first_scan_az):.1f}°")
    
    return segments

def identify_profiling_lidar_segments(az_angles, min_points=3, debug=False, agl_height=None):
    """
    Identify scan segments based on repeating azimuth patterns
    """
    if len(az_angles) < min_points:
        if debug:
            print(f"  Height {agl_height:.1f}m: Not enough points ({len(az_angles)})")
        return []
    
    try:
        az_angles = np.array(az_angles, dtype=float)
    except (ValueError, TypeError):
        if debug:
            print(f"  Height {agl_height:.1f}m: Could not convert azimuth angles to float")
        return []
    
    # Find where azimuth decreases significantly (likely a reset from 270° to 0°)
    az_diffs = np.diff(az_angles)
    reset_indices = np.where(az_diffs < -180)[0]
    
    if len(reset_indices) < 2:
        if debug:
            print(f"  Height {agl_height:.1f}m: No clear azimuth pattern resets found")
        points_per_scan = 4  # Typical profiling lidar has 4 beam directions per scan
        num_scans = len(az_angles) // points_per_scan
        
        if num_scans == 0:
            return []
        
        segments = []
        for i in range(num_scans):
            start_idx = i * points_per_scan
            end_idx = min((i+1) * points_per_scan, len(az_angles))
            if end_idx - start_idx >= min_points:
                segments.append(list(range(start_idx, end_idx)))
        
        if debug:
            print(f"  Height {agl_height:.1f}m: Created {len(segments)} segments based on expected pattern")
        return segments
    
    segments = []
    
    if reset_indices[0] + 1 >= min_points:
        segments.append(list(range(0, reset_indices[0] + 1)))
    
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i] + 1
        end_idx = reset_indices[i+1] + 1
        if end_idx - start_idx >= min_points:
            segments.append(list(range(start_idx, end_idx)))
    
    if len(az_angles) - (reset_indices[-1] + 1) >= min_points:
        segments.append(list(range(reset_indices[-1] + 1, len(az_angles))))
    
    if debug:
        print(f"  Height {agl_height:.1f}m: Identified {len(segments)} scan cycles based on azimuth patterns")
        if segments:
            print(f"    First scan: {len(segments[0])} points, azimuth range: {np.min(az_angles[segments[0]]):.1f}° to {np.max(az_angles[segments[0]]):.1f}°")
    
    return segments

def extract_profiling_lidar_height_measurements(filtered_results, height, time_mask, times, min_beams):
    """
    Extract valid profiling lidar measurements for turbulence processing at specific height
    Filters out NaN radial velocities and vertical beam positions ('V'), converts
    string positions to float azimuth angles, and validates minimum beam count.
    Parameters:
    -----------
    filtered_results : dict
        QC-filtered profiling lidar data structure
    height : float
        Specific height level to extract
    time_mask : array
        Boolean mask for time window filtering
    times : array
        Timestamp array
    min_beams : int
        Minimum required valid measurements
    Returns:
    --------
    tuple
        (timestamps, az_angles, el_angles, radial_vel) arrays or (None, None, None, None)
    """    
    timestamps = []
    az_angles = []
    el_angles = []
    radial_vel = []
    valid_count = 0
    
    for idx in np.where(time_mask)[0]:
        position = filtered_results['position'].iloc[idx]
        vr_value = filtered_results['measurements'][height]['vr'].iloc[idx]
        
        if np.isnan(vr_value) or position == 'V':
            continue
        
        try:
            az_angle = float(position)
        except (ValueError, TypeError):
            continue  # Skip if can't convert to float
        
        timestamps.append(times.iloc[idx])
        az_angles.append(az_angle)
        el_angles.append(62)  # Fixed elevation for these lidar
        radial_vel.append(vr_value)
        valid_count += 1
    
    if valid_count < min_beams:
        return None, None, None, None
    
    return (np.array(timestamps), np.array(az_angles),
            np.array(el_angles), np.array(radial_vel))

def extract_qc_valid_measurements(qc_data, vr_data, az_data, el_data,
                                        times, threshold, min_beams):
    """Extract valid measurements for turbulence processing"""
   
    valid_mask = (qc_data >= threshold) & ~np.isnan(vr_data)
    
    if np.sum(valid_mask) < min_beams:
        return None, None, None, None
    
    return (np.array(times)[valid_mask], az_data[valid_mask],
            el_data[valid_mask], vr_data[valid_mask])

def identify_complete_scans(timestamps, az_angles, el_angles, radial_vel, instrument='z02', min_beams_per_scan=3):
    """
    Identify complete scans for turbulence analysis
    Returns list of complete scans, each containing all measurements for that scan cycle
    """
    if len(az_angles) < min_beams_per_scan:
        return []
    
    if instrument.lower() in ['z01', 'z02']:
        segments = identify_scanning_lidar_segments(timestamps, az_angles, instrument, min_beams_per_scan)
    else:
        segments = identify_profiling_lidar_segments(az_angles, min_beams_per_scan)
    
    complete_scans = []
    for segment in segments:
        if len(segment) >= min_beams_per_scan:
            scan_data = {
                'timestamps': timestamps[segment],
                'az': az_angles[segment], 
                'el': el_angles[segment],
                'vr': radial_vel[segment],
                'time': timestamps[segment][0]  # Representative time for this scan
            }
            complete_scans.append(scan_data)
    
    return complete_scans

def process_turbulence_from_scans(timestamps, az_angles, el_angles, radial_vel, 
                                 instrument='z02', wind_dir_correction=0, w_sign_correction=1,
                                 min_scans=3, min_beams_per_scan=3):
    """
    Extract turbulence by analyzing multiple complete scans over time
    """
    complete_scans = identify_complete_scans(timestamps, az_angles, el_angles, radial_vel, 
                                           instrument, min_beams_per_scan)
    
    if len(complete_scans) < min_scans:
        return None
        
    scan_results = []
    for scan in complete_scans:
        try:
            wind_result = fit_chi_winds(scan['az'], scan['el'], scan['vr'])
            if not np.isnan(wind_result['ws']):
                corrected_wd = (wind_result['wd'] + wind_dir_correction) % 360
                corrected_w = wind_result['w'] * w_sign_correction
                
                scan_results.append({
                    'time': scan['time'],
                    'u': wind_result['u'], 
                    'v': wind_result['v'],
                    'w': corrected_w,
                    'ws': wind_result['ws']
                })
        except Exception:
            continue
    
    if len(scan_results) < min_scans:
        return None
        
    u_series = np.array([r['u'] for r in scan_results])
    v_series = np.array([r['v'] for r in scan_results]) 
    w_series = np.array([r['w'] for r in scan_results])
    ws_series = np.array([r['ws'] for r in scan_results])
    
    return calculate_turbulence_metrics(u_series, v_series, w_series, ws_series, 'empirical')

### Instrument Processors

def process_scanning_lidar(ds, start_time, instrument_code, location, time_window=TIME_WINDOW_MINUTES,
                          min_beams=None, wind_dir_correction=None,
                          min_beams_per_scan=3,
                          availability_threshold=0.5, ground_elevation=None,
                          verbose=False):
    """
    Process scanning lidar data for combined wind and turbulence profiles
    Parameters:
    -----------
    ds : xarray.Dataset
        lidar datasets containing e.g., time, distance, SNR, radial_wind_speed, azimuth, elevation
    instrument_code : str
        'z01' or 'z02' to specify instrument-specific processing
    location : str
        Location identifier ('nantucket', 'rhode_island', 'cape_cod', or 'block_island')
    start_time : datetime
        Start time for processing window
    time_window : int, optional
        Processing window duration in minutes (default: 10)
    wind_dir_correction : float, optional
        Wind direction correction. If None, uses location defaults
    min_beams : int, optional
        Minimum number of fit-entering VAD beams. Defaults to the
        QC_CONFIG value (4): a 3-beam fit is exactly determined, so its
        wind-speed error is undefined and the wserr gate cannot apply
    min_beams_per_scan : int, optional
        Minimum beams required per scan segment for turbulence analysis (default: 3)
    availability_threshold : float, optional
        Minimum data availability fraction (0-1) for height inclusion (default: 0.5)
    ground_elevation : float, optional
        Ignored: the value is read from the site config.
    verbose : bool, optional
        Enable detailed processing output (default: False)
    Returns:
    --------
    dict or None
        Dictionary containing wind and turbulence profiles or None if insufficient data
    """
    try:
        latitude = float(ds.lat[0].values)
        longitude = float(ds.lon[0].values)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not extract coordinates from {instrument_code} dataset: {e}, using fallback")
        fallback_coords = get_instrument_coordinates(location, f'{instrument_code}_lidar')
        if fallback_coords:
            latitude, longitude = fallback_coords[0], fallback_coords[1]
        else:
            print(f"Error: No coordinates found for {instrument_code} at {location}")
            return None
    
    # Per-(site, instrument) scanning lidar configuration. ``segment_func`` closes
    # over a function in this module, so this dict cannot live in config.py without
    # a circular import. Fields: wind_dir_correction (azimuth offset in degrees,
    # true-north reference), w_sign_correction (+-1 on raw vertical velocity),
    # segment_func (callable that splits the time series into scans).

    config = {
        ('nantucket', 'z01'): {
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            # -50 rather than the -200 default; the VAD wrap (~-300°) still trips it.
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts, reset_threshold=-50)
        },
        ('nantucket', 'z02'): {
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z02'),
            'w_sign_correction': get_w_sign_correction('z02', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z02', min_pts)
        },
        ('block_island', 'z01'): {
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts)
        },
        ('rhode_island', 'z01'): {
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts)
        },
        ('cape_cod', 'z02'): {
            'wind_dir_correction': wind_dir_correction or get_wind_correction('cape_cod', 'z02'),
            'w_sign_correction': get_w_sign_correction('z02', 'cape_cod'),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z02', min_pts)
        }
    }

    instrument_key = instrument_code.lower()
    config_key = (location, instrument_key)
    if config_key not in config:
        raise ValueError(f"Unsupported combination: {instrument_code} at {location}. "
                        f"Supported: {list(config.keys())}")
    cfg = config[config_key]

    # QC signal, threshold, and beam minimum come from config.QC_CONFIG:
    # the single source of truth shared with the availability calculation.
    qc_cfg = get_qc_params(location, instrument_key)
    qc_metric = qc_cfg['qc_type']
    qc_threshold = qc_cfg['threshold']
    if min_beams is None:
        min_beams = qc_cfg.get('min_beams', MIN_VAD_BEAMS)

    if verbose:
        print(f"Using {location} {instrument_key} QC: {qc_metric} threshold {qc_threshold}, min_beams {min_beams}")

    times = pd.to_datetime(ds.time.values)
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    if not time_mask.any():
        return None

    if verbose:
        print(f"Processing {instrument_key} combined data from {start_time} to {end_time}")

    heights = ds.distance.values
    if heights.ndim == 2:
        # For 2D arrays, take heights from first time point (assuming consistent across time)
        if 'range_gate' in ds.distance.dims and ds.distance.dims[0] == 'time':
            heights = heights[0, :]
        elif heights.shape[0] > heights.shape[1]:
            # Shape is (time, height) - take first time point
            heights = heights[0, :]
        else:
            # Shape is (height, time) - take first column  
            heights = heights[:, 0]
    elif heights.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected distance array dimensions: {heights.shape}")
    heights = np.asarray(heights).flatten()

    vr_data = ds.radial_wind_speed[time_mask, :].values
    azimuth_data = ds.azimuth[time_mask].values
    elevation_data = ds.elevation[time_mask].values

    # Keep only beams at the VAD cone elevation (see VAD_ELEVATION_DEG above).
    vad_mask = np.abs(elevation_data - VAD_ELEVATION_DEG) < VAD_ELEVATION_TOL_DEG
    if not vad_mask.any():
        return None
    vr_data = vr_data[vad_mask, :]
    azimuth_data = azimuth_data[vad_mask]
    elevation_data = elevation_data[vad_mask]
    window_times = times[time_mask][vad_mask]

    # Slant range to geometric height: height = distance * sin(elevation).
    median_elevation = np.nanmedian(elevation_data)
    heights = heights * np.sin(np.radians(median_elevation))

    if qc_metric == 'snr':
        qc_data = ds.SNR[time_mask, :].values[vad_mask, :]
    elif qc_metric == 'intensity':
        qc_data = ds.intensity[time_mask, :].values[vad_mask, :]
    else:
        raise ValueError(f"Unsupported QC metric: {qc_metric}")

    ground_elevation = get_ground_elevation_from_config(location, instrument_code)

    filtered_data = {
        'heights': heights,
        'time': window_times,
        'position': azimuth_data,
        'measurements': {}
    }

    for i, height in enumerate(heights):
        if height < 100:  # Skip heights <100m for scanning lidars
            continue
        filtered_data['measurements'][height] = {
            qc_metric: qc_data[:, i],
            'vr': vr_data[:, i],
            'azimuth': azimuth_data,
            'elevation': elevation_data,
            'timestamps': window_times
        }

    qc_params = {
        'qc_type': qc_metric,
        'threshold': qc_threshold,
        'min_beams': min_beams
    }

    filtered_results = filter_data_by_qc_criteria(
        filtered_data,
        instrument=instrument_key,
        qc_params=qc_params,
        verbose=verbose
    )

    if filtered_results is None:
        return None

    # Calculate availability using the same QC rule as the beam filter
    availability_metrics = calculate_data_availability(
        filtered_results, filtered_data, availability_threshold,
        qc_type=qc_metric, threshold=qc_threshold)

    results = {
        'time': start_time,
        'instrument_code': f'lidar_{instrument_key}',
        'wind_profiles': {},
        'turbulence_profiles': {},
        'ground_elevation': ground_elevation,
        'latitude': latitude,
        'longitude': longitude,
        'availability': availability_metrics
    }

    for height, data in filtered_results['measurements'].items():
        availability = availability_metrics['height_availability'].get(height, {}).get('availability', 0)
        if availability < availability_threshold:
            if verbose:
                print(f"Skipping height {height}m - availability {availability:.1%} < {availability_threshold:.1%}")
            continue

        az = data['azimuth']
        el = data['elevation']
        vr = data['vr']
        timestamps_qc = data['timestamps']

        valid_mask = ~np.isnan(vr)
        az_valid = az[valid_mask]
        el_valid = el[valid_mask]
        vr_valid = vr[valid_mask]
        timestamps_valid = timestamps_qc[valid_mask]

        if len(vr_valid) < min_beams:
            if verbose:
                print(f"Height {height}m: Insufficient valid data after NaN removal: {len(vr_valid)} < {min_beams}")
            continue

        # Heights are already AGL; ground_elevation is ASL, carried for metadata.
        true_agl_height = height

        wind_profile = fit_chi_winds(az_valid, el_valid, vr_valid)

        if not np.isnan(wind_profile['ws']):
            wind_profile['wd'] = (wind_profile['wd'] + cfg['wind_dir_correction']) % 360
            wind_profile['w'] *= cfg['w_sign_correction']
            wind_profile['availability'] = availability
            results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        
        turbulence_metrics = process_turbulence_from_scans(
            timestamps_valid, az_valid, el_valid, vr_valid,
            instrument=instrument_key,
            wind_dir_correction=cfg['wind_dir_correction'],
            w_sign_correction=cfg['w_sign_correction'],
            min_scans=3,  # Need at least 3 complete scans for turbulence
            min_beams_per_scan=min_beams_per_scan
        )
        if turbulence_metrics:
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_metrics)

    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final results: {wind_count} Wind Profiles, {turb_count} turbulence profiles")
    
    return results

def process_profiling_lidar(parsed_data, start_time, location, time_window=TIME_WINDOW_MINUTES,
                     wind_dir_correction=None, min_beams_per_scan=3,
                     availability_threshold=0.5, ground_elevation=None,
                     verbose=False):
    """
    Unified z03 lidar processor for both RTD, STA, and CSV formats.

    QC rules (CNR threshold for RTD/STA, sample/rain gating for the ZephIR
    CSV, and the VAD beam minimum) come from ``config.QC_CONFIG`` per site.

    Parameters:
    -----------
    parsed_data : dict
        Parsed z03 data structure from parse_profiling_lidar_file()
    start_time : datetime
        Start time for processing window
    location : str
        Location identifier ('nantucket', 'block island', etc.)
    time_window : int, optional
        Processing window duration in minutes (default: 10)
    wind_dir_correction : float, optional
        Wind direction correction. If None, uses location defaults
    min_beams_per_scan : int, optional
        Minimum beams required per scan segment for turbulence analysis (default: 3)
    availability_threshold : float, optional
        Minimum data availability fraction (0-1) for height inclusion (default: 0.5)
    ground_elevation : float, optional
        Ignored: the value is read from the site config.
    verbose : bool, optional
        Enable detailed processing output (default: False)
    Returns:
    --------
    dict or None
        Dictionary containing wind and turbulence profiles or None if insufficient data
    """
    if parsed_data is None:
        return None

    qc_cfg = get_qc_params(location, 'z03')
    
    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'z03')
    
    if 'latitude' in parsed_data and 'longitude' in parsed_data:
        latitude = parsed_data['latitude']
        longitude = parsed_data['longitude']
    elif 'metadata' in parsed_data and parsed_data['metadata'].get('gps_coords'):
        latitude, longitude = parsed_data['metadata']['gps_coords']
    else:
        fallback_coords = get_instrument_coordinates(location, 'z03_lidar')
        if fallback_coords:
            latitude, longitude = fallback_coords[0], fallback_coords[1]
        else:
            if verbose:
                print(f"Warning: No coordinates found for profiling lidar at {location}")
            latitude, longitude = np.nan, np.nan
    
    # Ground elevation comes from the site config (USGS-derived values cached there)
    ground_elevation = get_ground_elevation_from_config(location, 'z03')

    file_type = parsed_data.get('file_type', 'rtd')  # Default to RTD if not specified

    if file_type == 'rtd':
        return _process_profiling_rtd(parsed_data, start_time, time_window,
                               qc_cfg['threshold'],
                               qc_cfg.get('min_beams', MIN_VAD_BEAMS),
                               wind_dir_correction, min_beams_per_scan,
                               availability_threshold,
                               ground_elevation, latitude, longitude, location, verbose)
    elif file_type == 'sta':
        return _process_profiling_sta(parsed_data, start_time, time_window,
                               qc_cfg['threshold'],
                               availability_threshold, ground_elevation,
                               wind_dir_correction, latitude, longitude, location, verbose)
    elif file_type == 'csv':
        return _process_profiling_csv(parsed_data, start_time, time_window,
                               availability_threshold, ground_elevation,
                               wind_dir_correction, latitude, longitude, location,
                               verbose, min_samples=qc_cfg.get('min_samples', 20))
    else:
        if verbose:
            print(f"Unknown profiling lidar file type: {file_type}")
        return None

def _process_profiling_rtd(rtd_data, start_time, time_window, cnr_threshold, min_beams,
                    wind_dir_correction, min_beams_per_scan,
                    availability_threshold, ground_elevation, latitude, longitude,
                    location, verbose):
    """Process WindCube RTD format (scanning pattern). Used at Nantucket in WFIP3"""

    w_correction = get_w_sign_correction('z03', location)
    times = pd.to_datetime(rtd_data['time'])
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        return None
    
    if verbose:
        print(f"Processing z03 RTD data from {start_time} to {end_time}")
    
    filtered_data = {
        'heights': rtd_data['heights'],
        'time': times[time_mask],
        'position': rtd_data['position'][time_mask],
        'measurements': {}
    }
    
    for height in rtd_data['heights']:
        if height not in rtd_data['measurements']:
            continue
        
        height_data = rtd_data['measurements'][height]
        filtered_data['measurements'][height] = {
            'cnr': height_data['cnr'][time_mask],
            'vr': height_data['vr'][time_mask],
            'timestamps': times[time_mask],
            'position': rtd_data['position'][time_mask]  # Include position for each height
        }
    
    qc_params = {
        'qc_type': 'cnr',
        'threshold': cnr_threshold,
        'min_beams': min_beams
    }
    
    filtered_results = filter_data_by_qc_criteria(filtered_data, instrument='z03', qc_params=qc_params, verbose=verbose)
    if filtered_results is None:
        return None
    
    availability_metrics = calculate_data_availability(
        filtered_results, filtered_data, availability_threshold,
        qc_type='cnr', threshold=cnr_threshold)
    
    results = {
        'time': start_time,
        'instrument_code': 'lidar_z03',
        'wind_profiles': {},
        'turbulence_profiles': {},
        'ground_elevation': ground_elevation,
        'latitude': latitude,
        'longitude': longitude,
        'availability': availability_metrics
    }
    
    for height, data in filtered_results['measurements'].items():
        availability = availability_metrics['height_availability'].get(height, {}).get('availability', 0)
        
        if availability < availability_threshold:
            if verbose:
                print(f"Skipping height {height}m - availability {availability:.1%} < {availability_threshold:.1%}")
            continue
        
        # Heights are already AGL; ground_elevation is ASL, carried for metadata.
        true_agl_height = height
        
        vr = data['vr']
        positions = data['position']
        timestamps_qc = data['timestamps']
        
        az_angles = []
        el_angles = []
        vr_valid = []
        timestamps_valid = []
        
        for i, (pos, vr_val, ts) in enumerate(zip(positions, vr, timestamps_qc)):
            if np.isnan(vr_val) or pos == 'V':
                continue
            
            try:
                az_angle = float(pos)
            except (ValueError, TypeError):
                continue  # Skip if can't convert to float
            
            az_angles.append(az_angle)
            el_angles.append(62)  # Fixed elevation for z03
            vr_valid.append(vr_val)
            timestamps_valid.append(ts)
        
        az_angles = np.array(az_angles)
        el_angles = np.array(el_angles)
        vr_valid = np.array(vr_valid)
        timestamps_valid = np.array(timestamps_valid)
        
        if len(vr_valid) < min_beams:
            if verbose:
                print(f"Height {height}m: Insufficient valid data: {len(vr_valid)} < {min_beams}")
            continue
        
        wind_profile = fit_chi_winds(az_angles, el_angles, vr_valid)
        if not np.isnan(wind_profile['ws']):
            wind_profile['wd'] = (wind_profile['wd'] + wind_dir_correction) % 360
            wind_profile['w'] *= w_correction 
            wind_profile['availability'] = availability
            results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        
        turbulence_metrics = process_turbulence_from_scans(
            timestamps_valid, az_angles, el_angles, vr_valid,
            instrument='z03',
            wind_dir_correction=wind_dir_correction,
            w_sign_correction=w_correction,
            min_scans=3,
            min_beams_per_scan=min_beams_per_scan
        )
        
        if turbulence_metrics:
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_metrics)
    
    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final results: {wind_count} Wind Profiles, {turb_count} turbulence profiles")
    
    return results

def _process_profiling_sta(sta_data, start_time, time_window, cnr_threshold,
                    availability_threshold, ground_elevation, wind_dir_correction,
                    latitude, longitude, location, verbose):
    """Process WindCube STA format (pre-processed wind data). Used at Block Island in WFIP3"""
    w_correction = get_w_sign_correction('z03', location)
    times = pd.to_datetime(sta_data['time'])
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        return None
    
    if verbose:
        print(f"Processing z03 STA data from {start_time} to {end_time}")
    
    results = {
        'time': start_time,
        'instrument_code': 'lidar_z03',
        'wind_profiles': {},
        'turbulence_profiles': {},
        'ground_elevation': ground_elevation,
        'latitude': latitude,
        'longitude': longitude
    }
    
    for height in sta_data['heights']:
        height_data = sta_data['measurements'][height]
        
        if len(height_data['u_native']) == 0:
            if verbose:
                print(f"Height {height}m: Skipping - no data")
            continue
        
        cnr_vals = np.array(height_data['cnr'])
        mask = time_mask & (cnr_vals >= cnr_threshold)
        
        if not mask.any():
            if verbose:
                print(f"Height {height}m: No data above CNR threshold {cnr_threshold}")
            continue
        
        u_native_vals = np.array(height_data['u_native'])[mask]
        v_native_vals = np.array(height_data['v_native'])[mask]
        w_native_vals = np.array(height_data['w_native'])[mask]
        std_u_native_vals = np.array(height_data['std_u_native'])[mask]
        std_v_native_vals = np.array(height_data['std_v_native'])[mask]
        std_w_native_vals = np.array(height_data['std_w_native'])[mask]
        vhm_vals = np.array(height_data['vhm'])[mask]
        std_vhm_vals = np.array(height_data['std_vhm'])[mask]
        avail_vals = np.array(height_data['availability'])[mask]
        
        valid_mask = ~(np.isnan(u_native_vals) | np.isnan(v_native_vals))
        if not valid_mask.any():
            continue
        
        u_native_final = u_native_vals[valid_mask]
        v_native_final = v_native_vals[valid_mask]
        w_native_final = w_native_vals[valid_mask]
        std_u_native_final = std_u_native_vals[valid_mask]
        std_v_native_final = std_v_native_vals[valid_mask]
        std_w_native_final = std_w_native_vals[valid_mask]
        vhm_final = vhm_vals[valid_mask]
        std_vhm_final = std_vhm_vals[valid_mask]
        avail_final = avail_vals[valid_mask]
        
        avg_availability = np.nanmean(avail_final) / 100.0  # Convert percent to fraction
        if avg_availability < availability_threshold:
            if verbose:
                print(f"Height {height}m: Availability {avg_availability:.1%} < {availability_threshold:.1%}")
            continue
        
        # Step 1: Convert to meteorological coordinates (coordinate conversion only)
        u_met_mean = -np.nanmean(v_native_final)  # u_met = -v_native
        v_met_mean = -np.nanmean(u_native_final)  # v_met = -u_native  
        w_met_mean = -np.nanmean(w_native_final)  # w_met = -w_native
        
        # Step 2: Apply sign correction separately (like RTD format)
        w_met_mean *= w_correction
        
        # Convert standard deviations to meteorological coordinates 
        std_u_met_mean = np.nanmean(std_v_native_final)  # std_u_met = std_v_native
        std_v_met_mean = np.nanmean(std_u_native_final)  # std_v_met = std_u_native
        std_w_met_mean = np.nanmean(std_w_native_final)  # std_w_met = std_w_native 
        
        # Wind speed magnitude and meteorological direction (direction *from* which wind blows)
        ws = np.sqrt(u_met_mean**2 + v_met_mean**2)
        wd = (np.rad2deg(np.arctan2(-u_met_mean, -v_met_mean)) + wind_dir_correction) % 360
        
        # Turbulent kinetic energy and turbulence intensity (Stull 1988)
        tke = 0.5 * (std_u_met_mean**2 + std_v_met_mean**2 + std_w_met_mean**2)
        ti = np.sqrt(tke) / ws if ws > 0 else np.nan
        
        scalar_ws_mean = np.nanmean(vhm_final)
        scalar_ws_std = np.nanmean(std_vhm_final)
        
        # Heights are already AGL; ground_elevation is ASL, carried for metadata.
        true_agl_height = height
        
        wind_profile = {
            'u': u_met_mean,
            'v': v_met_mean,
            'w': w_met_mean,
            'ws': ws,
            'wd': wd,
            'availability': avg_availability,
            'scalar_ws': scalar_ws_mean  # Additional scalar wind speed
        }
        
        turbulence_profile = {
            'std_u': std_u_met_mean,
            'std_v': std_v_met_mean,
            'std_w': std_w_met_mean,
            'ti': ti,
            'tke': tke,
            'scalar_ws_std': scalar_ws_std
        }
        
        if turbulence_profile and apply_physics_based_qc(turbulence_profile):
            turbulence_profile = {}  # Clear if rejected

        if turbulence_profile:  # Only store if not empty
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_profile)

        results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_profile)
        
        if verbose:
            print(f"Height {height}m ({true_agl_height}m AGL): WS={ws:.2f} m/s, WD={wd:.1f}°, TI={ti:.3f}")
    
    if verbose:
        print(f"Height {height}m: u_native={len(u_native_vals)}, v_native={len(v_native_vals)}, mask_sum={mask.sum()}, valid_mask_sum={valid_mask.sum() if 'valid_mask' in locals() else 'not_created'}")

    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final results: {wind_count} wind profiles, {turb_count} turbulence profiles")
    
    # Create availability structure for plotting
    availability = {
        'total_availability': 0,
        'height_availability': {},
        'meets_threshold': False
    }
    
    total_availability = 0
    count = 0
    
    for height_agl, wind_profile in results['wind_profiles'].items():
        if 'availability' in wind_profile:
            availability['height_availability'][height_agl] = {
                'availability': wind_profile['availability'],
                'meets_threshold': wind_profile['availability'] >= availability_threshold
            }
            total_availability += wind_profile['availability']
            count += 1
    
    if count > 0:
        availability['total_availability'] = round(total_availability / count, 3)
        availability['meets_threshold'] = any(
            info['meets_threshold'] for info in availability['height_availability'].values()
        )
    
    results['availability'] = availability
    
    return results

def clean_missing_values(data_array):
    """Fast vectorized missing value cleaning"""
    data_clean = data_array.copy()
    
    missing_values = [9999, 9999.0, 9999.9, 10000, -9999, -9999.0, 99999]
    missing_mask = np.isin(data_clean, missing_values)
    data_clean[missing_mask] = np.nan
    
    return data_clean

def _process_profiling_csv(csv_data, start_time, time_window, availability_threshold,
                    ground_elevation, wind_dir_correction, latitude, longitude,
                    location, verbose, min_samples=20):
    """
    Process profiling lidar CSV format (pre-computed wind statistics with coordinate transform).
    Used at Narragansett/Rhode Island (ZephIR-300) in WFIP3.
    
    Parameters:
    -----------
    csv_data : dict
        Parsed Z03 CSV data from parse_profiling_csv_file()
    start_time : datetime
        Start time for processing window
    time_window : int
        Processing window duration in minutes
    availability_threshold : float
        Minimum data availability fraction (0-1) for height inclusion
    ground_elevation : float
        Ground elevation in meters
    wind_dir_correction : float
        Wind direction correction in degrees
    latitude, longitude : float
        GPS coordinates
    verbose : bool
        Print processing details
    
    Returns:
    --------
    dict
        Standardized results with wind and turbulence profiles
    """
    if verbose:
        print(f"Processing Z03 CSV data from {start_time} to {start_time + pd.Timedelta(minutes=time_window)}")
    
    times = csv_data['time']
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        if verbose:
            print("No Z03 CSV data in time window")
        return None
    
    df = csv_data['metadata']['original_df']
    
    results = {
        'time': start_time,
        'instrument_code': 'lidar_z03',
        'wind_profiles': {},
        'turbulence_profiles': {},
        'ground_elevation': ground_elevation,
        'latitude': latitude,
        'longitude': longitude,
        'availability': {'total_availability': 0, 'height_availability': {}, 'meets_threshold': False}
    }
    
    total_availability = 0
    processed_heights = 0
    
    for height in csv_data['heights']:
        height_data = csv_data['measurements'][height]
        
        time_filtered_data = {}
        for key, values in height_data.items():
            time_filtered_data[key] = values[time_mask]
        
        if len(time_filtered_data['wind_speed']) == 0:
            continue
        
        # **RHODE ISLAND Z03 SPECIFIC QC FILTERS (per instrument manager)**
        
        # 1. Rain filter: Remove data with any rain
        rain_mask = np.ones(len(time_filtered_data['wind_speed']), dtype=bool)  # Default: keep all
        if 'Proportion Of Packets With Rain (%)' in df.columns:
            rain_vals = df['Proportion Of Packets With Rain (%)'][time_mask].values
            rain_mask = rain_vals == 0  # Keep only 0% rain periods
            if verbose and np.sum(~rain_mask) > 0:
                print(f"Height {height}m: Filtered out {np.sum(~rain_mask)} data points due to rain")
        
        # 2. Sample count filter: drop records averaged from too few packets
        # (QC_CONFIG 'samples_rain': min_samples)
        packets_mask = np.ones(len(time_filtered_data['wind_speed']), dtype=bool)  # Default: keep all
        packets_col = f'Packets in Average at {int(height)}m'
        if packets_col in df.columns:
            packets_vals = df[packets_col][time_mask].values
            packets_mask = packets_vals >= min_samples
            if verbose and np.sum(~packets_mask) > 0:
                print(f"Height {height}m: Filtered out {np.sum(~packets_mask)} data points due to insufficient samples (< {min_samples})")
        
        ri_qc_mask = rain_mask & packets_mask
        
        if not ri_qc_mask.any():
            if verbose:
                print(f"Height {height}m: No data passed Rhode Island Z03 QC filters")
            continue

        if not ri_qc_mask.any():
            continue
            
        qc_data = {
            'wind_speed': time_filtered_data['wind_speed'][ri_qc_mask],
            'wind_direction': time_filtered_data['wind_direction'][ri_qc_mask], 
            'w': time_filtered_data['w'][ri_qc_mask],
            'TI': time_filtered_data['TI'][ri_qc_mask]
        }

        wind_speed_vals = clean_missing_values(qc_data['wind_speed'])
        wind_dir_vals = clean_missing_values(qc_data['wind_direction'])
        w_vals = clean_missing_values(qc_data['w'])
        ti_vals = clean_missing_values(qc_data['TI'])
        
        wind_speed_vals = clean_missing_values(wind_speed_vals)
        wind_dir_vals = clean_missing_values(wind_dir_vals)
        w_vals = clean_missing_values(w_vals)
        ti_vals = clean_missing_values(ti_vals)
        
        # Calculate data availability (after all QC filters)
        total_points = len(time_filtered_data['wind_speed'])  # Original time-filtered data
        valid_points = np.sum(~np.isnan(wind_speed_vals))     # After all QC + NaN filtering
        availability = valid_points / total_points if total_points > 0 else 0
        
        if availability < availability_threshold:
            if verbose:
                print(f"Height {height}m: Availability {availability:.1%} < {availability_threshold:.1%} after Rhode Island QC")
            continue
        
        # Wind direction uses the circular mean so rows straddling 0/360°
        # cannot collapse toward 180°.
        mean_wind_speed = np.nanmean(wind_speed_vals)
        wd_finite = wind_dir_vals[np.isfinite(wind_dir_vals)]
        mean_wind_dir = float(wind_direction_average(wd_finite)) if len(wd_finite) else np.nan
        mean_w = np.nanmean(w_vals)
        mean_ti = np.nanmean(ti_vals)
        
        if np.isnan(mean_wind_speed) or np.isnan(mean_wind_dir):
            if verbose:
                print(f"Height {height}m: Insufficient valid wind data after all QC")
            continue
        
        corrected_wind_dir = (mean_wind_dir + wind_dir_correction) % 360
        
        w_correction = get_w_sign_correction('z03', location)
        corrected_w = mean_w * w_correction
        
        # Convert to u, v components for consistency with other processors
        wind_dir_rad = np.deg2rad(corrected_wind_dir)
        u_component = -mean_wind_speed * np.sin(wind_dir_rad)  # Meteorological convention
        v_component = -mean_wind_speed * np.cos(wind_dir_rad)
        
        # Heights are already AGL; ground_elevation is ASL, carried for metadata.
        true_agl_height = height
        
        wind_profile = {
            'u': u_component,
            'v': v_component,
            'w': corrected_w,
            'ws': mean_wind_speed,
            'wd': corrected_wind_dir,
            'availability': availability
        }
        
        turbulence_profile = {}
        if not np.isnan(mean_ti):
            turbulence_profile = {
                'ti': mean_ti,
            }
        
        results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        if turbulence_profile:
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_profile)
        
        results['availability']['height_availability'][true_agl_height] = {
            'availability': round(availability, 3),
            'meets_threshold': availability >= availability_threshold
        }
        total_availability += availability
        processed_heights += 1
        
        if verbose:
            rain_filtered = np.sum(~rain_mask) if 'Proportion Of Packets With Rain (%)' in df.columns else 0
            packets_filtered = np.sum(~packets_mask) if packets_col in df.columns else 0
            print(f"Height {height}m ({true_agl_height}m AGL): WS={mean_wind_speed:.2f} m/s, "
                  f"WD={corrected_wind_dir:.1f}°, TI={mean_ti:.3f}, Avail={availability:.1%}")
            print(f"  QC summary - Rain filtered: {rain_filtered}, Low packets filtered: {packets_filtered}")
    
    if processed_heights > 0:
        results['availability']['total_availability'] = round(total_availability / processed_heights, 3)
        results['availability']['meets_threshold'] = any(
            info['meets_threshold'] for info in results['availability']['height_availability'].values()
        )
    
    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final Z03 CSV results: {wind_count} wind profiles, {turb_count} turbulence profiles")
    
    if not results['wind_profiles'] and not results['turbulence_profiles']:
        return None
        
    return results

def process_caco_z01_lidar(parsed_data, start_time, location='cape_cod', time_window=TIME_WINDOW_MINUTES,
                          availability_threshold=0.5, ground_elevation=None,
                          wind_dir_correction=None, verbose=False):
    """
    Process the CACO z01 profiling lidar (WindCube V2-96) for one 10-minute window.

    CACO's profiling lidar is labeled z01 (not z03 like other sites' profiling lidars)
    and delivered as pre-aggregated 10-minute profiles. QC is applied upstream by the
    instrument (QC_CONFIG 'prefiltered'; the vendor blanks wind wherever its own
    screening fails), so this function applies only azimuth and
    vertical-velocity-sign corrections plus NaN screening.

    Turbulence is limited to intensity: the workbook reports a within-window wind
    speed dispersion but no component variances, so TKE and the component standard
    deviations would need an isotropy assumption and are left unreported. Intensity
    is gated on the vendor's per-height availability (QC_CONFIG 'min_ti_availability').

    Parameters
    ----------
    parsed_data : dict
        Output of ``lidar_parsers.parse_caco_lidar_file``, containing
        ``time``, ``heights``, ``wind_speed``, ``wind_direction``,
        ``w``, and ``ws_dispersion`` arrays plus latitude/longitude metadata.
    start_time : datetime
        Beginning of the 10-minute window to extract.
    location : str
        Site key (defaults to 'cape_cod').
    wind_dir_correction : float, optional
        Override the per-site azimuth correction from ``config.py``.

    Returns
    -------
    dict or None
        Profile dict with ``wind_profiles`` (u, v, w, ws, wd, availability),
        ``turbulence_profiles`` (ti), ``latitude``, ``longitude``, and
        metadata; ``None`` if no data falls in the window.
    """
    if parsed_data is None:
        return None
    
    latitude = parsed_data['latitude']
    longitude = parsed_data['longitude']
    
    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'z01')

    ground_elevation = get_ground_elevation_from_config(location, 'z01')
    
    w_correction = get_w_sign_correction('z01', location)

    # Vendor availability minimum, consulted for turbulence only.
    min_ti_availability = get_qc_params(location, 'z01')['min_ti_availability']

    times = parsed_data['time']
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        if verbose:
            print("No CACO Z01 data in time window")
        return None
    
    if verbose:
        print(f"Processing CACO Z01 profiling data from {start_time} to {end_time}")
        print("QC: prefiltered upstream by the instrument (no pipeline CNR gate)")
    
    results = {
        'time': start_time,
        'instrument_code': 'lidar_caco_z01',
        'wind_profiles': {},
        'turbulence_profiles': {},
        'ground_elevation': ground_elevation,
        'latitude': latitude,
        'longitude': longitude,
        'availability': {'total_availability': 0, 'height_availability': {}, 'meets_threshold': False}
    }
    
    total_availability = 0
    processed_heights = 0
    
    for height in parsed_data['heights']:
        height_data = parsed_data['measurements'][height]
        
        wind_speed_vals = height_data['wind_speed'][time_mask]
        wind_dir_vals = height_data['wind_direction'][time_mask]
        w_vals = height_data['w'][time_mask]
        ws_dispersion_vals = height_data['ws_dispersion'][time_mask]
        vendor_availability_vals = height_data['availability'][time_mask]

        if len(wind_speed_vals) == 0:
            continue

        # Wind is prefiltered upstream by the vendor, so only NaN screening remains.
        valid_mask = ~(np.isnan(wind_speed_vals) | np.isnan(wind_dir_vals))
        if not valid_mask.any():
            if verbose:
                print(f"Height {height}m: No valid wind data in window")
            continue

        # Calculate data availability (valid data / total data)
        total_points = len(wind_speed_vals)
        valid_points = np.sum(valid_mask)
        availability = valid_points / total_points if total_points > 0 else 0

        if availability < availability_threshold:
            if verbose:
                print(f"Height {height}m: Availability {availability:.1%} < {availability_threshold:.1%}")
            continue

        # Wind direction uses the circular mean (0/360°-safe).
        mean_wind_speed = np.nanmean(wind_speed_vals[valid_mask])
        mean_wind_dir = float(wind_direction_average(wind_dir_vals[valid_mask]))
        mean_w = np.nanmean(w_vals[valid_mask])

        # TI from the vendor's within-window wind speed dispersion, the same
        # scan-to-scan sigma(WS) the high-rate processors compute. Denominator is
        # the scalar mean, not the Rosenbusch (2021) hybrid used elsewhere, which
        # needs both scalar and vector means the workbook does not deliver. Ratios
        # are formed per record then averaged; calm records (WS = 0) drop out.
        # The vendor's per-height availability gates the dispersion (QC_CONFIG
        # 'min_ti_availability'), a different quantity from the `availability`
        # computed below, which is the fraction of non-NaN records in the window.
        ti_mask = (valid_mask
                   & np.isfinite(ws_dispersion_vals)
                   & (wind_speed_vals > 0)
                   & (vendor_availability_vals >= min_ti_availability))
        if ti_mask.any():
            mean_ti = float(np.nanmean(ws_dispersion_vals[ti_mask] / wind_speed_vals[ti_mask]))
        else:
            mean_ti = np.nan

        if np.isnan(mean_wind_speed) or np.isnan(mean_wind_dir):
            if verbose:
                print(f"Height {height}m: Insufficient valid wind data")
            continue
        
        corrected_wind_dir = (mean_wind_dir + wind_dir_correction) % 360
        corrected_w = mean_w * w_correction
        
        # Convert to u, v components (meteorological convention)
        wind_dir_rad = np.deg2rad(corrected_wind_dir)
        u_component = -mean_wind_speed * np.sin(wind_dir_rad)
        v_component = -mean_wind_speed * np.cos(wind_dir_rad)
        
        # Heights are already AGL; ground_elevation is ASL, carried for metadata.
        true_agl_height = height
        
        wind_profile = {
            'u': u_component,
            'v': v_component,
            'w': corrected_w,
            'ws': mean_wind_speed,
            'wd': corrected_wind_dir,
            'availability': availability
        }

        # Only intensity exists here, so the shared gate acts on the TI bound alone.
        turbulence_profile = {}
        if not np.isnan(mean_ti):
            turbulence_profile = {'ti': mean_ti}
            if apply_physics_based_qc(turbulence_profile):
                turbulence_profile = {}

        results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        if turbulence_profile:
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_profile)

        results['availability']['height_availability'][true_agl_height] = {
            'availability': round(availability, 3),
            'meets_threshold': availability >= availability_threshold
        }
        
        processed_heights += 1
        total_availability += availability
        
        if verbose:
            ti_text = f"{mean_ti:.3f}" if not np.isnan(mean_ti) else "n/a"
            print(f"Height {height}m ({true_agl_height}m AGL): WS={mean_wind_speed:.2f} m/s, "
                  f"WD={corrected_wind_dir:.1f}°, TI={ti_text}, Avail={availability:.1%}")

    if processed_heights > 0:
        results['availability']['total_availability'] = round(total_availability / processed_heights, 3)
        results['availability']['meets_threshold'] = any(
            info['meets_threshold'] for info in results['availability']['height_availability'].values()
        )

    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final CACO Z01 results: {wind_count} wind profiles, "
              f"{turb_count} turbulence profiles (prefiltered upstream)")
    
    if not results['wind_profiles']:
        return None
        
    return results

### Time-Series Handling

def process_lidar_time_series(instrument_code, filename, start_time, end_time, location,
                              time_window=TIME_WINDOW_MINUTES, verbose=False, **kwargs):
    """
    Process a single lidar instrument across multiple 10-minute windows.

    Top-level lidar dispatch: opens the input file once, then iterates the
    appropriate per-window processor (``process_scanning_lidar``,
    ``process_profiling_lidar``, or ``process_caco_z01_lidar``) over the requested
    time range. The choice of processor is based on ``(location, instrument_code)``:

      - All scanning lidars (Halo XR/XR+, Galion) -> ``process_scanning_lidar``
      - z03 profiling lidars (WindCube V1/V2.1, ZephIR-300) -> ``process_profiling_lidar``
      - CACO z01 (WindCube V2-96; pre-aggregated workbook delivery) -> ``process_caco_z01_lidar``

    Parameters
    ----------
    instrument_code : str
        Site-specific instrument label ('z01', 'z02', or 'z03').
    filename : str
        Path to the input file (NetCDF, .rtd, .sta, .csv, or .xlsx
        depending on instrument).
    start_time, end_time : datetime
        Inclusive/exclusive time range to process.
    location : str
        Site key as defined in ``config.LOCATION_CONFIG``.
    time_window : int
        Averaging window in minutes (default 10).
    **kwargs
        Forwarded to the selected per-window processor.

    Returns
    -------
    list of dict
        One profile dict per successfully processed window, in
        chronological order. Empty list on file-open failure.
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return []
    
    if start_time >= end_time:
        print(f"Error: Invalid time range: {start_time} to {end_time}")
        return []
    
    try:
        if instrument_code in ['z01', 'z02']:
            if location.lower() == 'cape_cod' and instrument_code == 'z01':
                ds = parse_caco_lidar_file(filename)
                if ds is None or not ds.get('measurements'):
                    print(f"Error: Failed to read CACO Z01 file or no measurements")
                    return []
            else:
                ds = xr.open_dataset(filename)
                if 'time' not in ds or len(ds.time) == 0:
                    print(f"Error: No time data in {instrument_code} file")
                    return []
        elif instrument_code == 'z03':
            ds = parse_profiling_lidar_file(filename, verbose=False)
            if ds is None or not ds.get('measurements'):
                print(f"Error: Failed to read z03 file or no measurements")
                return []
        elif instrument_code.lower() == 'caco':
            # The site code is accepted as an alias for the Cape Cod z01 workbook
            ds = parse_caco_lidar_file(filename)
            if ds is None or not ds.get('measurements'):
                print(f"Error: Failed to read CACO file or no measurements")
                return []
        else:
            raise ValueError(f"Unknown instrument code: {instrument_code}")
    except Exception as e:
        print(f"Error loading {instrument_code} file {filename}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return []

    ground_elevation = get_ground_elevation_from_config(location, instrument_code)
    
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    time_intervals = []
    while current_time < end_time:
        time_intervals.append(current_time)
        current_time += pd.Timedelta(minutes=time_window)
    
    print(f"Processing {instrument_code} combined data: {len(time_intervals)} time intervals")
    
    results = []
    for interval_time in time_intervals:
        try:
            if instrument_code in ['z01', 'z02']:
                if location.lower() == 'cape_cod' and instrument_code == 'z01':
                    interval_result = process_caco_z01_lidar(ds, interval_time, location, time_window,
                                                ground_elevation=ground_elevation, verbose=verbose, **kwargs)
                else:
                    # Standard scanning lidar processing (includes CACO Z02)
                    interval_result = process_scanning_lidar(ds, interval_time, instrument_code, location, time_window,
                                                    ground_elevation=ground_elevation, **kwargs)
            elif instrument_code == 'z03':
                interval_result = process_profiling_lidar(ds, interval_time, location, time_window,
                                            ground_elevation=ground_elevation, **kwargs)
            elif instrument_code.lower() == 'caco':
                interval_result = process_caco_z01_lidar(ds, interval_time, location, time_window,
                                            ground_elevation=ground_elevation, verbose=verbose, **kwargs)
            else:
                continue
            
            if interval_result and (interval_result.get('wind_profiles') or interval_result.get('turbulence_profiles')):
                results.append(interval_result)
        except Exception as e:
            if verbose:
                print(f"Error processing {instrument_code} interval at {interval_time}: {e}")
            continue
    
    try:
        if hasattr(ds, 'close'):
            ds.close()
    except Exception as e:
        if verbose:
            print(f"Warning: Error closing dataset: {e}")
    
    return results