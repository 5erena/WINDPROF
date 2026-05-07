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
                    round_profile_values, TIME_WINDOW_MINUTES)
from .wind_analysis import fit_chi_winds, calculate_turbulence_metrics, apply_physics_based_qc
from .quality_control import filter_data_by_qc_criteria, calculate_data_availability
from .lidar_parsers import parse_profiling_lidar_file, parse_profiling_csv_file, parse_caco_lidar_file

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
    Notes:
    ------
    - All scanning lidars in WFIP3 use a 6-beam VAD pattern at 60° elevation
      (azimuths 0/360°, 60°, 120°, 180°, 240°, 300°). NANT z01 additionally
      includes a vertical (90° elevation) stare every 7 beams.
    - Resets are detected as large negative azimuth differences (300° -> 0°
      gives ~ -300°). The ``reset_threshold`` defaults to -200; NANT z01 uses
      -50 as a more permissive threshold (the scan still wraps from 300°,
      but the value is set conservatively).
    """
    if len(az_angles) < min_points:
        if debug and agl_height is not None:
            print(f"  Height {agl_height:.1f}m: Not enough points ({len(az_angles)})")
        return []

    # Find where azimuth resets from larger angles back to smaller angles
    az_diffs = np.diff(az_angles)
    # Look for large negative differences (e.g., 300°/360° -> 0°/60° for VAD,
    # or ~164° -> 86° for PPI sector scans with reset_threshold=-50)
    reset_indices = np.where(az_diffs < reset_threshold)[0]
    
    if len(reset_indices) < 2:
        if debug and agl_height is not None:
            print(f"  Height {agl_height:.1f}m: No clear scan pattern transitions found")
        # Look for points where azimuth is close to 0° (or 360° for z02)
        if instrument.lower() == 'z02':
            az_close_to_zero = np.where((az_angles < 10) | (az_angles > 350))[0]
        else:  # z01
            az_close_to_zero = np.where(az_angles < 10)[0]
        
        if len(az_close_to_zero) < 2:
            if debug and agl_height is not None:
                print(f"  Height {agl_height:.1f}m: Unable to identify scans using azimuth values")
            # Fall back to time-based segmentation
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
        
        # Use points where azimuth is close to 0°/360° as scan boundaries
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
    
    # Use identified reset points to create segments
    segments = []
    
    # Process segments with handling for repeated measurements (z01 specific)
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
    
    # First segment
    first_segment = process_segment(0, reset_indices[0] + 1)
    if first_segment:
        segments.append(first_segment)
    
    # Middle segments
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i] + 1
        end_idx = reset_indices[i+1] + 1
        segment = process_segment(start_idx, end_idx)
        if segment:
            segments.append(segment)
    
    # Last segment
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
    
    # Convert to float array if not already
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
        # Alternative: divide into equal segments based on expected pattern
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
    
    # Use reset indices to create segments
    segments = []
    
    # First segment (if enough points)
    if reset_indices[0] + 1 >= min_points:
        segments.append(list(range(0, reset_indices[0] + 1)))
    
    # Middle segments
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i] + 1
        end_idx = reset_indices[i+1] + 1
        if end_idx - start_idx >= min_points:
            segments.append(list(range(start_idx, end_idx)))
    
    # Last segment (if enough points)
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
        
        # Skip NaN values and vertical beams
        if np.isnan(vr_value) or position == 'V':
            continue
        
        # Convert position to float
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
   
    # Single vectorized mask
    valid_mask = (qc_data >= threshold) & ~np.isnan(vr_data)
    
    if np.sum(valid_mask) < min_beams:
        return None, None, None, None
    
    # Apply mask to all arrays at once
    return (np.array(times)[valid_mask], az_data[valid_mask],
            el_data[valid_mask], vr_data[valid_mask])

def identify_complete_scans(timestamps, az_angles, el_angles, radial_vel, instrument='z02', min_beams_per_scan=3):
    """
    Identify complete scans for turbulence analysis
    Returns list of complete scans, each containing all measurements for that scan cycle
    """
    if len(az_angles) < min_beams_per_scan:
        return []
    
    # Use existing segmentation logic to find scan boundaries
    if instrument.lower() in ['z01', 'z02']:
        segments = identify_scanning_lidar_segments(timestamps, az_angles, instrument, min_beams_per_scan)
    else:
        segments = identify_profiling_lidar_segments(az_angles, min_beams_per_scan)
    
    # Convert segments into complete scan data structures
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
    # Identify complete scans
    complete_scans = identify_complete_scans(timestamps, az_angles, el_angles, radial_vel, 
                                           instrument, min_beams_per_scan)
    
    if len(complete_scans) < min_scans:
        return None
        
    # Fit VAD to each complete scan to get wind components  
    scan_results = []
    for scan in complete_scans:
        try:
            wind_result = fit_chi_winds(scan['az'], scan['el'], scan['vr'])
            if not np.isnan(wind_result['ws']):
                # Apply corrections 
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
    
    # Check if we have enough successful fits
    if len(scan_results) < min_scans:
        return None
        
    # Create time series from scan results
    u_series = np.array([r['u'] for r in scan_results])
    v_series = np.array([r['v'] for r in scan_results]) 
    w_series = np.array([r['w'] for r in scan_results])
    ws_series = np.array([r['ws'] for r in scan_results])
    
    # Use existing turbulence calculation function 
    return calculate_turbulence_metrics(u_series, v_series, w_series, ws_series, 'empirical')

### Instrument Processors

def process_scanning_lidar(ds, start_time, instrument_code, location, time_window=TIME_WINDOW_MINUTES,
                          qc_threshold=None, min_beams=3, wind_dir_correction=None,
                          min_beams_per_scan=3, az_range_threshold=60, 
                          availability_threshold=0.5, ground_elevation=None,
                          verbose=False):
    """
    Process scanning lidar data for combined wind and turbulence profiles
    Now supports location-specific configurations for different site deployments
    Parameters:
    -----------
    ds : xarray.Dataset
        lidar datasets containing e.g., time, distance, SNR, radial_wind_speed, azimuth, elevation
    instrument_code : str
        'z01' or 'z02' to specify instrument-specific processing
    location : str
        Location identifier ('nantucket', 'rhode_island', 'cape_cod', or 'block_island')
    qc_threshold : float, optional
        Quality control threshold. If None, uses location+instrument defaults
    start_time : datetime
        Start time for processing window
    time_window : int, optional
        Processing window duration in minutes (default: 10)
    wind_dir_correction : float, optional
        Wind direction correction. If None, uses location defaults
    min_beams : int, optional
        Minimum number of valid beams required for VAD fitting (default: 3)
    min_beams_per_scan : int, optional
        Minimum beams required per scan segment for turbulence analysis (default: 3)
    az_range_threshold : float, optional
        Minimum azimuth range in degrees for valid scans (default: 60)
    availability_threshold : float, optional
        Minimum data availability fraction (0-1) for height inclusion (default: 0.5)
    ground_elevation : float, optional
        Ground elevation in meters. If None, retrieved from USGS service
    verbose : bool, optional
        Enable detailed processing output (default: False)
    Returns:
    --------
    dict or None
        Dictionary containing wind and turbulence profiles or None if insufficient data
    """
    # Extract coordinates from dataset
    try:
        latitude = float(ds.lat[0].values)
        longitude = float(ds.lon[0].values)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not extract coordinates from {instrument_code} dataset: {e}, using fallback")
        # Fallback to location-based coordinates
        fallback_coords = get_instrument_coordinates(location, f'{instrument_code}_lidar')
        if fallback_coords:
            latitude, longitude = fallback_coords[0], fallback_coords[1]
        else:
            print(f"Error: No coordinates found for {instrument_code} at {location}")
            return None
    
    # Per-(site, instrument) scanning lidar configuration. Static fields could
    # live in config.py, but ``segment_func`` is a closure over a function in
    # this module — moving it there would create a circular import. Corrections
    # (wind, w-sign) are read from config.py via the lookup helpers.
    #
    # Fields:
    #   qc_metric         : QC signal name ('intensity', 'snr', 'cnr')
    #   qc_threshold      : numeric threshold; values below are dropped
    #   wind_dir_correction : azimuth offset in degrees (true-north reference)
    #   w_sign_correction : ±1 multiplier on raw vertical velocity
    #   segment_func      : callable that splits the time series into scans

    config = {
        # Nantucket configurations
        ('nantucket', 'z01'): {
            'qc_metric': 'intensity',
            'qc_threshold': qc_threshold or 1.008,
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            # reset_threshold=-50: more permissive than default (-200), verified
            # harmless for the standard VAD wrap (300° -> 0° gives ~−300°).
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts, reset_threshold=-50)
        },
        ('nantucket', 'z02'): {
            'qc_metric': 'intensity',
            'qc_threshold': qc_threshold or 1.008,  
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z02'),
            'w_sign_correction': get_w_sign_correction('z02', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z02', min_pts)
        },
        # Block Island configurations
        ('block_island', 'z01'): {
            'qc_metric': 'intensity',
            'qc_threshold': qc_threshold or 1.008, 
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts)
        },
        # Rhode Island configurations  
        ('rhode_island', 'z01'): {
            'qc_metric': 'intensity',
            'qc_threshold': qc_threshold or 1.008,
            'wind_dir_correction': wind_dir_correction or get_wind_correction(location, 'z01'),
            'w_sign_correction': get_w_sign_correction('z01', location),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z01', min_pts)
        },
        # Cape Cod configurations
        ('cape_cod', 'z02'): {
            'qc_metric': 'intensity', 
            'qc_threshold': qc_threshold or 1.008,  
            'wind_dir_correction': wind_dir_correction or get_wind_correction('cape_cod', 'z02'),
            'w_sign_correction': get_w_sign_correction('z02', 'cape_cod'),
            'segment_func': lambda ts, az, min_pts: identify_scanning_lidar_segments(ts, az, 'z02', min_pts)
        }
    }
    
    # Get configuration for this location + instrument combination
    instrument_key = instrument_code.lower()
    config_key = (location, instrument_key)
    if config_key not in config:
        raise ValueError(f"Unsupported combination: {instrument_code} at {location}. "
                        f"Supported: {list(config.keys())}")
    cfg = config[config_key]
    
    if verbose:
        print(f"Using {location} {instrument_key} config: {cfg['qc_metric']} threshold {cfg['qc_threshold']}")

    # Time filtering
    times = pd.to_datetime(ds.time.values)
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    if not time_mask.any():
        return None

    if verbose:
        print(f"Processing {instrument_key} combined data from {start_time} to {end_time}")

    # Load and prepare data
    heights = ds.distance.values
    if heights.ndim == 2:
        # For 2D arrays, take heights from first time point (assuming consistent across time)
        # Use dimension names for more reliable extraction
        if 'range_gate' in ds.distance.dims and ds.distance.dims[0] == 'time':
            # Shape is (time, range_gate) - take first time point
            heights = heights[0, :]
        elif heights.shape[0] > heights.shape[1]:
            # Shape is (time, height) - take first time point
            heights = heights[0, :]
        else:
            # Shape is (height, time) - take first column  
            heights = heights[:, 0]
    elif heights.ndim == 1:
        # Already 1D, use as-is
        pass
    else:
        raise ValueError(f"Unexpected distance array dimensions: {heights.shape}")
    # Ensure it's a 1D array of scalars
    heights = np.asarray(heights).flatten()

    vr_data = ds.radial_wind_speed[time_mask, :].values
    azimuth_data = ds.azimuth[time_mask].values
    elevation_data = ds.elevation[time_mask].values

    # Convert slant range to geometric height above instrument
    # For scanning lidars at ~60° elevation, height = distance * sin(elevation)
    median_elevation = np.nanmedian(elevation_data)
    heights = heights * np.sin(np.radians(median_elevation))

    # Quality control data - now location and instrument specific
    if cfg['qc_metric'] == 'snr':
        qc_data = ds.SNR[time_mask, :].values
    elif cfg['qc_metric'] == 'intensity':
        qc_data = ds.intensity[time_mask, :].values
    else:
        raise ValueError(f"Unsupported QC metric: {cfg['qc_metric']}")

    ground_elevation = get_ground_elevation_from_config(location, instrument_code)

    # Prepare data structure
    filtered_data = {
        'heights': heights,
        'time': times[time_mask],
        'position': azimuth_data,
        'measurements': {}
    }

    for i, height in enumerate(heights):
        if height < 100:  # Skip heights <100m for scanning lidars
            continue
        filtered_data['measurements'][height] = {
            cfg['qc_metric']: qc_data[:, i],
            'vr': vr_data[:, i],
            'azimuth': azimuth_data,
            'elevation': elevation_data,
            'timestamps': times[time_mask]
        }

    # Apply QC filtering
    qc_params = {
        'qc_type': cfg['qc_metric'],
        'threshold': cfg['qc_threshold'],
        'min_beams': min_beams,
        'az_range_threshold': az_range_threshold,
        'min_height': 100
    }

    filtered_results = filter_data_by_qc_criteria(
        filtered_data,
        instrument=instrument_key,
        qc_params=qc_params,
        verbose=verbose
    )

    if filtered_results is None:
        return None

    # Calculate availability using original data
    availability_metrics = calculate_data_availability(filtered_results, filtered_data, availability_threshold)

    # Initialize results
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

    # Process each height for wind and turbulence
    for height, data in filtered_results['measurements'].items():
        availability = availability_metrics['height_availability'].get(height, {}).get('availability', 0)
        if availability < availability_threshold:
            if verbose:
                print(f"Skipping height {height}m - availability {availability:.1%} < {availability_threshold:.1%}")
            continue

        # Prepare QC-filtered data
        az = data['azimuth']
        el = data['elevation']
        vr = data['vr']
        timestamps_qc = data['timestamps']

        # Remove remaining NaNs
        valid_mask = ~np.isnan(vr)
        az_valid = az[valid_mask]
        el_valid = el[valid_mask]
        vr_valid = vr[valid_mask]
        timestamps_valid = timestamps_qc[valid_mask]

        if len(vr_valid) < min_beams:
            if verbose:
                print(f"Height {height}m: Insufficient valid data after NaN removal: {len(vr_valid)} < {min_beams}")
            continue

        true_agl_height = height - ground_elevation

        # Wind Profile: VAD fitting
        wind_profile = fit_chi_winds(az_valid, el_valid, vr_valid)

        if not np.isnan(wind_profile['ws']):
            wind_profile['wd'] = (wind_profile['wd'] + cfg['wind_dir_correction']) % 360
            wind_profile['w'] *= cfg['w_sign_correction']
            wind_profile['availability'] = availability
            results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        
        # Turbulence Profile: scan-based analysis
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
                     cnr_threshold=-23, min_beams=5, wind_dir_correction=None,
                     min_beams_per_scan=3, az_range_threshold=180,
                     availability_threshold=0.5, ground_elevation=None,
                     verbose=False):
    """
    Unified z03 lidar processor for both RTD, STA, and CSV formats
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
    cnr_threshold : float, optional
        Minimum acceptable Carrier-to-Noise Ratio in dB (default: -23)
    min_beams : int, optional
        Minimum number of valid beams required for VAD fitting (default: 5)
    wind_dir_correction : float, optional
        Wind direction correction. If None, uses location defaults
    min_beams_per_scan : int, optional
        Minimum beams required per scan segment for turbulence analysis (default: 3)
    az_range_threshold : float, optional
        Minimum azimuth range in degrees for valid scans (default: 180)
    availability_threshold : float, optional
        Minimum data availability fraction (0-1) for height inclusion (default: 0.5)
    ground_elevation : float, optional
        Ground elevation in meters. If None, retrieved from USGS service
    verbose : bool, optional
        Enable detailed processing output (default: False)
    Returns:
    --------
    dict or None
        Dictionary containing wind and turbulence profiles or None if insufficient data
    """
    if parsed_data is None:
        return None
    
    # Get wind direction correction
    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'z03')
    
    # Extract coordinates from parsed data
    if 'latitude' in parsed_data and 'longitude' in parsed_data:
        latitude = parsed_data['latitude']
        longitude = parsed_data['longitude']
    elif 'metadata' in parsed_data and parsed_data['metadata'].get('gps_coords'):
        latitude, longitude = parsed_data['metadata']['gps_coords']
    else:
        # Fallback to location-based coordinates
        fallback_coords = get_instrument_coordinates(location, 'z03_lidar')
        if fallback_coords:
            latitude, longitude = fallback_coords[0], fallback_coords[1]
        else:
            if verbose:
                print(f"Warning: No coordinates found for profiling lidar at {location}")
            latitude, longitude = np.nan, np.nan
    
    # Ground elevation comes from the site config (USGS-derived values cached there)
    ground_elevation = get_ground_elevation_from_config(location, 'z03')

    # Determine file type and route to appropriate processor
    file_type = parsed_data.get('file_type', 'rtd')  # Default to RTD if not specified

    if file_type == 'rtd':
        return _process_profiling_rtd(parsed_data, start_time, time_window, cnr_threshold,
                               min_beams, wind_dir_correction, min_beams_per_scan,
                               az_range_threshold, availability_threshold,
                               ground_elevation, latitude, longitude, location, verbose)
    elif file_type == 'sta':
        return _process_profiling_sta(parsed_data, start_time, time_window, cnr_threshold,
                               availability_threshold, ground_elevation,
                               wind_dir_correction, latitude, longitude, location, verbose)
    elif file_type == 'csv':
        return _process_profiling_csv(parsed_data, start_time, time_window,
                               availability_threshold, ground_elevation,
                               wind_dir_correction, latitude, longitude, location, verbose)
    else:
        if verbose:
            print(f"Unknown profiling lidar file type: {file_type}")
        return None

def _process_profiling_rtd(rtd_data, start_time, time_window, cnr_threshold, min_beams,
                    wind_dir_correction, min_beams_per_scan, az_range_threshold,
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
    
    # Prepare data structure for QC - need to restructure RTD data
    filtered_data = {
        'heights': rtd_data['heights'],
        'time': times[time_mask],
        'position': rtd_data['position'][time_mask],
        'measurements': {}
    }
    
    # Structure measurements for QC
    for height in rtd_data['heights']:
        if height not in rtd_data['measurements']:
            continue
        
        height_data = rtd_data['measurements'][height]
        # Apply time mask to all measurements for this height
        filtered_data['measurements'][height] = {
            'cnr': height_data['cnr'][time_mask],
            'vr': height_data['vr'][time_mask],
            'timestamps': times[time_mask],
            'position': rtd_data['position'][time_mask]  # Include position for each height
        }
    
    # Apply QC (shared for both wind and turbulence)
    qc_params = {
        'qc_type': 'cnr',
        'threshold': cnr_threshold,
        'min_beams': min_beams,
        'az_range_threshold': az_range_threshold
    }
    
    filtered_results = filter_data_by_qc_criteria(filtered_data, instrument='z03', qc_params=qc_params, verbose=verbose)
    if filtered_results is None:
        return None
    
    availability_metrics = calculate_data_availability(filtered_results, filtered_data, availability_threshold)
    
    # Initialize combined results
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
    
    # Process each height for both wind and turbulence
    for height, data in filtered_results['measurements'].items():
        availability = availability_metrics['height_availability'].get(height, {}).get('availability', 0)
        
        if availability < availability_threshold:
            if verbose:
                print(f"Skipping height {height}m - availability {availability:.1%} < {availability_threshold:.1%}")
            continue
        
        true_agl_height = height - ground_elevation
        
        # Extract measurements - data is already QC filtered
        vr = data['vr']
        positions = data['position']
        timestamps_qc = data['timestamps']
        
        # Convert positions to azimuth angles, filter out vertical beams and NaN VR
        az_angles = []
        el_angles = []
        vr_valid = []
        timestamps_valid = []
        
        for i, (pos, vr_val, ts) in enumerate(zip(positions, vr, timestamps_qc)):
            # Skip NaN values and vertical beams
            if np.isnan(vr_val) or pos == 'V':
                continue
            
            # Convert position to float
            try:
                az_angle = float(pos)
            except (ValueError, TypeError):
                continue  # Skip if can't convert to float
            
            az_angles.append(az_angle)
            el_angles.append(62)  # Fixed elevation for z03
            vr_valid.append(vr_val)
            timestamps_valid.append(ts)
        
        # Convert to numpy arrays
        az_angles = np.array(az_angles)
        el_angles = np.array(el_angles)
        vr_valid = np.array(vr_valid)
        timestamps_valid = np.array(timestamps_valid)
        
        if len(vr_valid) < min_beams:
            if verbose:
                print(f"Height {height}m: Insufficient valid data: {len(vr_valid)} < {min_beams}")
            continue
        
        # Wind Profile: Single VAD fit using all valid data
        wind_profile = fit_chi_winds(az_angles, el_angles, vr_valid)
        if not np.isnan(wind_profile['ws']):
            wind_profile['wd'] = (wind_profile['wd'] + wind_dir_correction) % 360
            wind_profile['w'] *= w_correction 
            wind_profile['availability'] = availability
            results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        
        # Turbulence Profile: scan-based analysis
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
        
        # Skip heights with no data
        if len(height_data['u_native']) == 0:
            if verbose:
                print(f"Height {height}m: Skipping - no data")
            continue
        
        # Apply time and CNR filters
        cnr_vals = np.array(height_data['cnr'])
        mask = time_mask & (cnr_vals >= cnr_threshold)
        
        if not mask.any():
            if verbose:
                print(f"Height {height}m: No data above CNR threshold {cnr_threshold}")
            continue
        
        # Extract filtered data (native coordinates)
        u_native_vals = np.array(height_data['u_native'])[mask]
        v_native_vals = np.array(height_data['v_native'])[mask]
        w_native_vals = np.array(height_data['w_native'])[mask]
        std_u_native_vals = np.array(height_data['std_u_native'])[mask]
        std_v_native_vals = np.array(height_data['std_v_native'])[mask]
        std_w_native_vals = np.array(height_data['std_w_native'])[mask]
        vhm_vals = np.array(height_data['vhm'])[mask]
        std_vhm_vals = np.array(height_data['std_vhm'])[mask]
        avail_vals = np.array(height_data['availability'])[mask]
        
        # Remove NaN values
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
        
        # Check availability threshold
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
        
        # Also use scalar wind speed statistics
        scalar_ws_mean = np.nanmean(vhm_final)
        scalar_ws_std = np.nanmean(std_vhm_final)
        
        true_agl_height = height - ground_elevation
        
        # Wind profile
        wind_profile = {
            'u': u_met_mean,
            'v': v_met_mean,
            'w': w_met_mean,
            'ws': ws,
            'wd': wd,
            'availability': avg_availability,
            'scalar_ws': scalar_ws_mean  # Additional scalar wind speed
        }
        
        # Turbulence profile
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
    
    # All missing indicators in one operation
    missing_values = [9999, 9999.0, 9999.9, 10000, -9999, -9999.0, 99999]
    missing_mask = np.isin(data_clean, missing_values)
    data_clean[missing_mask] = np.nan
    
    return data_clean

def _process_profiling_csv(csv_data, start_time, time_window, availability_threshold,
                    ground_elevation, wind_dir_correction, latitude, longitude,
                    location, verbose):
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
    
    # Time filtering
    times = csv_data['time']
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        if verbose:
            print("No Z03 CSV data in time window")
        return None
    
    # Get the original DataFrame for QC filtering
    df = csv_data['metadata']['original_df']
    
    # Initialize results
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
    
    # Process each height
    for height in csv_data['heights']:
        height_data = csv_data['measurements'][height]
        
        # Apply time mask to height data
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
        
        # 2. Sample count filter: Remove data with insufficient samples (Nlidar < 20)
        packets_mask = np.ones(len(time_filtered_data['wind_speed']), dtype=bool)  # Default: keep all
        packets_col = f'Packets in Average at {int(height)}m'
        if packets_col in df.columns:
            packets_vals = df[packets_col][time_mask].values
            packets_mask = packets_vals >= 20  # Keep packets >= 20
            if verbose and np.sum(~packets_mask) > 0:
                print(f"Height {height}m: Filtered out {np.sum(~packets_mask)} data points due to insufficient samples (< 20)")
        
        # 3. Combine Rhode Island QC masks
        ri_qc_mask = rain_mask & packets_mask
        
        if not ri_qc_mask.any():
            if verbose:
                print(f"Height {height}m: No data passed Rhode Island Z03 QC filters")
            continue

        # Apply Rhode Island QC mask
        if not ri_qc_mask.any():
            continue
            
        # Extract all QC-filtered data at once
        qc_data = {
            'wind_speed': time_filtered_data['wind_speed'][ri_qc_mask],
            'wind_direction': time_filtered_data['wind_direction'][ri_qc_mask], 
            'w': time_filtered_data['w'][ri_qc_mask],
            'TI': time_filtered_data['TI'][ri_qc_mask]
        }

        # Clean all at once
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
        
        # Check availability threshold
        if availability < availability_threshold:
            if verbose:
                print(f"Height {height}m: Availability {availability:.1%} < {availability_threshold:.1%} after Rhode Island QC")
            continue
        
        # Calculate mean values (removing NaNs)
        mean_wind_speed = np.nanmean(wind_speed_vals)
        mean_wind_dir = np.nanmean(wind_dir_vals)
        mean_w = np.nanmean(w_vals)
        mean_ti = np.nanmean(ti_vals)
        
        # Skip if insufficient valid data
        if np.isnan(mean_wind_speed) or np.isnan(mean_wind_dir):
            if verbose:
                print(f"Height {height}m: Insufficient valid wind data after all QC")
            continue
        
        # Apply wind direction correction
        corrected_wind_dir = (mean_wind_dir + wind_dir_correction) % 360
        
        # Apply w sign correction
        w_correction = get_w_sign_correction('z03', location)
        corrected_w = mean_w * w_correction
        
        # Convert to u, v components for consistency with other processors
        wind_dir_rad = np.deg2rad(corrected_wind_dir)
        u_component = -mean_wind_speed * np.sin(wind_dir_rad)  # Meteorological convention
        v_component = -mean_wind_speed * np.cos(wind_dir_rad)
        
        # Calculate AGL height
        true_agl_height = height - ground_elevation
        
        # Store wind profile
        wind_profile = {
            'u': u_component,
            'v': v_component,
            'w': corrected_w,
            'ws': mean_wind_speed,
            'wd': corrected_wind_dir,
            'availability': availability
        }
        
        # Store turbulence profile
        turbulence_profile = {}
        if not np.isnan(mean_ti):
            turbulence_profile = {
                'ti': mean_ti,
            }
        
        # Round and store results
        results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        if turbulence_profile:
            results['turbulence_profiles'][true_agl_height] = round_profile_values(turbulence_profile)
        
        # Track availability
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
    
    # Calculate overall availability
    if processed_heights > 0:
        results['availability']['total_availability'] = round(total_availability / processed_heights, 3)
        results['availability']['meets_threshold'] = any(
            info['meets_threshold'] for info in results['availability']['height_availability'].values()
        )
    
    if verbose:
        wind_count = len(results['wind_profiles'])
        turb_count = len(results['turbulence_profiles'])
        print(f"Final Z03 CSV results: {wind_count} wind profiles, {turb_count} turbulence profiles")
    
    # Return None if no successful profiles
    if not results['wind_profiles'] and not results['turbulence_profiles']:
        return None
        
    return results

def process_caco_z01_lidar(parsed_data, start_time, location='cape_cod', time_window=TIME_WINDOW_MINUTES,
                          cnr_threshold=-23, availability_threshold=0.5, ground_elevation=None,
                          wind_dir_correction=None, verbose=False):
    """
    Process the CACO z01 profiling lidar (WindCube V2-96) for one 10-minute window.

    CACO's profiling lidar is labeled z01 (not z03 like other sites' profiling lidars) 
    and delivered as pre-aggregated 10-minute profiles. No raw beams available; this 
    function applies only azimuth and vertical-velocity-sign corrections.

    Parameters
    ----------
    parsed_data : dict
        Output of ``lidar_parsers.parse_caco_lidar_file``, containing
        ``time``, ``heights``, ``wind_speed``, ``wind_direction``, and
        ``vertical_velocity`` arrays plus latitude/longitude metadata.
    start_time : datetime
        Beginning of the 10-minute window to extract.
    location : str
        Site key (defaults to 'cape_cod').
    cnr_threshold : float
        Retained for API symmetry with other lidar processors; not used by
        the CACO pipeline because raw CNR is not available.
    wind_dir_correction : float, optional
        Override the per-site azimuth correction from ``config.py``.

    Returns
    -------
    dict or None
        Profile dict with ``wind_speed``, ``wind_direction``, ``w``,
        ``heights``, ``latitude``, ``longitude``, and metadata; ``None`` if
        no data falls in the window.
    """
    if parsed_data is None:
        return None
    
    # Get coordinates
    latitude = parsed_data['latitude']
    longitude = parsed_data['longitude']
    
    # Get wind direction correction
    if wind_dir_correction is None:
        wind_dir_correction = get_wind_correction(location, 'z01')

    ground_elevation = get_ground_elevation_from_config(location, 'z01')
    
    # Get w sign correction
    w_correction = get_w_sign_correction('z01', location)
    
    # Time filtering
    times = parsed_data['time']
    end_time = start_time + pd.Timedelta(minutes=time_window)
    time_mask = (times >= start_time) & (times < end_time)
    
    if not time_mask.any():
        if verbose:
            print("No CACO Z01 data in time window")
        return None
    
    if verbose:
        print(f"Processing CACO Z01 profiling data from {start_time} to {end_time}")
        print(f"Using CNR threshold: {cnr_threshold} dB")
    
    # Initialize results
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
    
    # Process each height
    for height in parsed_data['heights']:
        height_data = parsed_data['measurements'][height]
        
        # Apply time mask
        wind_speed_vals = height_data['wind_speed'][time_mask]
        wind_dir_vals = height_data['wind_direction'][time_mask] 
        w_vals = height_data['w'][time_mask]
        cnr_vals = height_data['cnr'][time_mask]
        availability_vals = height_data['availability'][time_mask]
        
        if len(wind_speed_vals) == 0:
            continue
        
        # Apply CNR filtering
        cnr_mask = cnr_vals >= cnr_threshold
        if not cnr_mask.any():
            if verbose:
                print(f"Height {height}m: No data above CNR threshold {cnr_threshold} dB")
            continue
        
        # Apply CNR mask to all variables
        wind_speed_filtered = wind_speed_vals[cnr_mask]
        wind_dir_filtered = wind_dir_vals[cnr_mask]
        w_filtered = w_vals[cnr_mask]
        availability_filtered = availability_vals[cnr_mask]
        
        # Additional NaN filtering
        valid_mask = ~(np.isnan(wind_speed_filtered) | np.isnan(wind_dir_filtered))
        if not valid_mask.any():
            if verbose:
                print(f"Height {height}m: No valid wind data after CNR and NaN filtering")
            continue
        
        # Calculate data availability (CNR-filtered data / total data)
        total_points = len(wind_speed_vals)
        valid_points = np.sum(valid_mask)
        availability = valid_points / total_points if total_points > 0 else 0
        
        # Check availability threshold
        if availability < availability_threshold:
            if verbose:
                print(f"Height {height}m: Availability {availability:.1%} < {availability_threshold:.1%}")
            continue
        
        # Calculate mean values from valid, CNR-filtered data
        mean_wind_speed = np.nanmean(wind_speed_filtered[valid_mask])
        mean_wind_dir = np.nanmean(wind_dir_filtered[valid_mask])
        mean_w = np.nanmean(w_filtered[valid_mask])
        
        # Skip if insufficient valid data
        if np.isnan(mean_wind_speed) or np.isnan(mean_wind_dir):
            if verbose:
                print(f"Height {height}m: Insufficient valid wind data")
            continue
        
        # Apply corrections
        corrected_wind_dir = (mean_wind_dir + wind_dir_correction) % 360
        corrected_w = mean_w * w_correction
        
        # Convert to u, v components (meteorological convention)
        wind_dir_rad = np.deg2rad(corrected_wind_dir)
        u_component = -mean_wind_speed * np.sin(wind_dir_rad)
        v_component = -mean_wind_speed * np.cos(wind_dir_rad)
        
        # Calculate AGL height
        true_agl_height = height - ground_elevation
        
        # Store wind profile
        wind_profile = {
            'u': u_component,
            'v': v_component,
            'w': corrected_w,
            'ws': mean_wind_speed,
            'wd': corrected_wind_dir,
            'availability': availability
        }
        
        # Store results
        results['wind_profiles'][true_agl_height] = round_profile_values(wind_profile)
        
        # Track availability
        results['availability']['height_availability'][true_agl_height] = {
            'availability': round(availability, 3),
            'meets_threshold': availability >= availability_threshold
        }
        
        processed_heights += 1
        total_availability += availability
        
        if verbose:
            print(f"Height {height}m ({true_agl_height}m AGL): WS={mean_wind_speed:.2f} m/s, "
                  f"WD={corrected_wind_dir:.1f}°, CNR>={cnr_threshold}dB, Avail={availability:.1%}")
    
    # Calculate overall availability
    if processed_heights > 0:
        results['availability']['total_availability'] = round(total_availability / processed_heights, 3)
        results['availability']['meets_threshold'] = any(
            info['meets_threshold'] for info in results['availability']['height_availability'].values()
        )
    
    if verbose:
        wind_count = len(results['wind_profiles'])
        print(f"Final CACO Z01 results: {wind_count} wind profiles (CNR >= {cnr_threshold} dB)")
    
    # Return None if no successful profiles
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
    # Validate inputs
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return []
    
    if start_time >= end_time:
        print(f"Error: Invalid time range: {start_time} to {end_time}")
        return []
    
    # Load data with error context
    try:
        if instrument_code in ['z01', 'z02']:
            # Special handling for CACO Z01 (Excel format)
            if location.lower() == 'cape_cod' and instrument_code == 'z01':
                ds = parse_caco_lidar_file(filename)
                if ds is None or not ds.get('measurements'):
                    print(f"Error: Failed to read CACO Z01 file or no measurements")
                    return []
            else:
                # Standard NetCDF scanning lidars
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
            # Legacy CACO handling (should be same as CACO Z01)
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

    # Get ground elevation from config
    ground_elevation = get_ground_elevation_from_config(location, instrument_code)
    
    # Generate time intervals
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    time_intervals = []
    while current_time < end_time:
        time_intervals.append(current_time)
        current_time += pd.Timedelta(minutes=time_window)
    
    print(f"Processing {instrument_code} combined data: {len(time_intervals)} time intervals")
    
    # Process each interval
    results = []
    for interval_time in time_intervals:
        try:
            if instrument_code in ['z01', 'z02']:
                # Handle CACO Z01 as profiling lidar (like z03)
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
                # Legacy CACO handling
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
    
    # Ensure cleanup
    try:
        if hasattr(ds, 'close'):
            ds.close()
    except Exception as e:
        if verbose:
            print(f"Warning: Error closing dataset: {e}")
    
    return results