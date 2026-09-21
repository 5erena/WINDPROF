"""Hierarchical cross-instrument merging into best-estimate profiles

Each 10-minute window's instrument profiles are interpolated to a common
height grid and then combined by height:

  - Below the lowest lidar gate (~5-20 m AGL): sonic anemometers only.
  - Below 1000 m AGL: lidars take priority, radar fills only heights with
    no lidar measurement.
  - Above 1000 m AGL: radar and lidar are averaged.

This module embeds the WFIP3-specific instrument hierarchy directly in
the merge logic; campaigns with different instrument inventories may
need to revise the merge priorities here.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
from datetime import timedelta
from pathlib import Path
from .config import (round_profile_values, wind_direction_average, normalize_location,
                     MALFUNCTION_PERIODS, get_surface_met_patterns, LOCATION_CONFIG,
                     TIME_WINDOW_MINUTES, get_near_surface_levels)
from .quality_control import calculate_quality_flag_for_height
from .lidars import process_lidar_time_series
from .radars import process_radar_time_series, process_radar_netcdf_time_series
from .anemometers import (process_anemometer_time_series, process_sonic_c1_time_series,
                          process_rhod_sonic_time_series, extract_wind_from_surface_met)

def interpolate_profiles_to_grid(profiles, target_heights, profile_type, max_gap=200):
    """
    Interpolate instrument profiles to target height grid with gap handling
    Parameters:
    -----------
    profiles : dict
        {height: {param: value}} from single instrument
    target_heights : array
        Target height grid
    profile_type : str
        'wind' or 'turbulence'
    max_gap : float
        Maximum height gap (in meters) to allow interpolation across

    Returns:
    --------
    dict
        Interpolated profiles {height: {param: value}}
    """    
    if not profiles:
        return {}

    # A non-finite height cannot be placed on any grid.
    profiles = {h: v for h, v in profiles.items() if np.isfinite(h)}
    if not profiles:
        return {}

    # Single-height instruments (sonics): snap to the nearest grid level within 10 m.
    if len(profiles) == 1:
        single_height = list(profiles.keys())[0]
        distances = np.abs(target_heights - single_height)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= 10:
            return {target_heights[closest_idx]: profiles[single_height]}
        return {}
    
    original_heights = np.array(sorted(profiles.keys()))
    interpolated_profile = {}
    
    if profile_type == 'wind':
        params = ['ws', 'w', 'uerr', 'verr', 'werr', 'wserr', 'wderr']
    else:
        params = ['ti', 'tke', 'std_u', 'std_v', 'std_w']
    
    # Split height array at gaps > max_gap so we don't interpolate across
    # large data voids (e.g. lidar blind zone between 0 and lowest gate).
    height_gaps = np.diff(original_heights)
    large_gap_indices = np.where(height_gaps > max_gap)[0]
    segment_starts = np.concatenate([[0], large_gap_indices + 1])
    segment_ends = np.concatenate([large_gap_indices + 1, [len(original_heights)]])

    for start_idx, end_idx in zip(segment_starts, segment_ends):
        segment_heights = original_heights[start_idx:end_idx]
        if len(segment_heights) < 2:
            continue
        
        segment_min, segment_max = segment_heights[0], segment_heights[-1]
        
        segment_mask = (target_heights >= segment_min) & (target_heights <= segment_max)
        segment_targets = target_heights[segment_mask]
        
        if len(segment_targets) == 0:
            continue
        
        for param in params:
            segment_values = np.array([profiles[h].get(param, np.nan) for h in segment_heights])
            valid_mask = ~np.isnan(segment_values)
            
            if np.sum(valid_mask) >= 2:
                valid_heights = segment_heights[valid_mask]
                valid_values = segment_values[valid_mask]
                
                try:
                    interp_values = np.interp(segment_targets, valid_heights, valid_values)
                    for height, value in zip(segment_targets, interp_values):
                        if height not in interpolated_profile:
                            interpolated_profile[height] = {}
                        interpolated_profile[height][param] = float(value)
                except (ValueError, TypeError):
                    continue

    # Wind direction: handle separately using circular interpolation (sin/cos
    # decomposition) to avoid the 0/360° wraparound discontinuity.
    if profile_type == 'wind':
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            segment_heights = original_heights[start_idx:end_idx]
            if len(segment_heights) < 2:
                continue
            
            segment_min, segment_max = segment_heights[0], segment_heights[-1]
            segment_mask = (target_heights >= segment_min) & (target_heights <= segment_max)
            segment_targets = target_heights[segment_mask]
            
            if len(segment_targets) == 0:
                continue
            
            segment_wd = np.array([profiles[h].get('wd', np.nan) for h in segment_heights])
            valid_mask = ~np.isnan(segment_wd)
            
            if np.sum(valid_mask) >= 2:
                valid_heights = segment_heights[valid_mask]
                valid_wd = segment_wd[valid_mask]

                try:
                    sin_values = np.sin(np.deg2rad(valid_wd))
                    cos_values = np.cos(np.deg2rad(valid_wd))
                    
                    interp_sin = np.interp(segment_targets, valid_heights, sin_values)
                    interp_cos = np.interp(segment_targets, valid_heights, cos_values)
                    
                    interp_wd = np.rad2deg(np.arctan2(interp_sin, interp_cos)) % 360
                    
                    for height, wd_value in zip(segment_targets, interp_wd):
                        if height not in interpolated_profile:
                            interpolated_profile[height] = {}
                        interpolated_profile[height]['wd'] = float(wd_value)
                except (ValueError, TypeError):
                    pass
    
    for height in interpolated_profile:
        interpolated_profile[height] = round_profile_values(interpolated_profile[height])
    
    return interpolated_profile

def extract_availability_data(instruments_data):
    """Collect per-instrument availability dicts from the results, if present"""
    availability_data = {}
    for instrument_name, result in instruments_data.items():
        if 'availability' in result:
            availability_data[instrument_name] = result['availability']
    return availability_data

def average_multi_instrument_profiles(interpolated_data, target_heights, parameters,
                                      location=None, time_interval=None): 
    """
    Merge interpolated profiles from multiple instruments with a hierarchical approach.

    Below 1000 m AGL: lidar takes priority and radar is only used to fill gaps where
    no lidar data exist. Above 1000 m: linear averaging across all available
    instruments. Radar is additionally excluded during known malfunction periods
    listed in MALFUNCTION_PERIODS.

    Parameters:
    -----------
    interpolated_data : dict
        {instrument: {height: {param: value}}}
    target_heights : array
        Target height grid
    parameters : list
        List of parameters to merge

    Returns:
    --------
    dict
        {'profiles': merged_profiles, 'flags': quality_flags}
    """    
    
    in_malfunction = False
    if location in MALFUNCTION_PERIODS and time_interval is not None:
        if isinstance(time_interval, int):
            measurement_time = pd.to_datetime(time_interval, unit='s', utc=True)
        else:
            measurement_time = pd.to_datetime(time_interval, utc=True)
        
        for start_str, end_str in MALFUNCTION_PERIODS[location]:
            start = pd.to_datetime(start_str, utc=True)
            end = pd.to_datetime(end_str, utc=True)
            if start <= measurement_time <= end:
                in_malfunction = True
                break
    
    merged_profiles = {}
    quality_flags = {}
    all_instruments = list(interpolated_data.keys())
    lidar_instruments = [inst for inst in all_instruments if inst.startswith('lidar_')]
    
    for height in target_heights:
        height_values = {}
        
        if height < 1000:
            lidar_has_data = any(
                height in interpolated_data[inst] and 
                any(param in interpolated_data[inst][height] and 
                    interpolated_data[inst][height][param] is not None and 
                    not np.isnan(interpolated_data[inst][height][param])
                    for param in parameters)
                for inst in lidar_instruments
                if inst in interpolated_data
            )
            
            if lidar_has_data:
                instruments_to_use = [inst for inst in all_instruments if inst != 'radar']
            else:
                instruments_to_use = all_instruments
        else:
            instruments_to_use = all_instruments
        
        if in_malfunction and 'radar' in instruments_to_use:
            instruments_to_use = [inst for inst in instruments_to_use if inst != 'radar']
        
        for param in parameters:
            values_list = []
            for instrument in instruments_to_use:
                profile = interpolated_data[instrument]
                if height in profile and param in profile[height]:
                    val = profile[height][param]
                    if val is not None and not np.isnan(val):
                        values_list.append(val)
            
            if values_list:
                height_values[param] = np.array(values_list)
        
        if height_values:
            merged_profiles[height] = {}
            for param, values in height_values.items():
                if param == 'wd':
                    merged_value = wind_direction_average(values)
                    merged_profiles[height][param] = round(merged_value, 1)
                else:
                    merged_value = np.mean(values)
                    decimals = 3 if param in ['ti', 'tke', 'std_u', 'std_v', 'std_w'] else 2
                    merged_profiles[height][param] = round(merged_value, decimals)
            
            merged_profiles[height] = round_profile_values(merged_profiles[height])
            # Flags cover only the instruments that formed the merged value, so an
            # excluded instrument cannot mark a cell it did not contribute to.
            used_data = {inst: interpolated_data[inst] for inst in instruments_to_use}
            height_flags = calculate_quality_flag_for_height(height, used_data, parameters)
            quality_flags[height] = height_flags
    
    return {'profiles': merged_profiles, 'flags': quality_flags}

def create_combined_results(time_interval, instruments_data, include_availability, 
                            location='nantucket', max_gap=200): 
    """
    Create combined time interval with merged wind and turbulence profiles

    Parameters:
    -----------
    time_interval : datetime
        Time for this interval
    instruments_data : dict
        {instrument_name: instrument_result} for this time interval
    include_availability : bool
        Whether to include availability data
    location : str
        Location identifier (for malfunction filtering)
        
    Returns:
    --------
    dict
        Combined time interval with merged profiles and quality flags
    """
    all_heights_list = []
    ground_elevations = []
    
    for instrument_name, result in instruments_data.items():
        if result.get('wind_profiles'):
            all_heights_list.extend(result['wind_profiles'].keys())
        if result.get('turbulence_profiles'):
            all_heights_list.extend(result['turbulence_profiles'].keys())
        if result.get('ground_elevation'):
            ground_elevations.append(result['ground_elevation'])
    
    # A non-finite height would make max_height NaN and break the grid below.
    all_heights_list = [h for h in all_heights_list if np.isfinite(h)]
    if not all_heights_list:
        return None

    avg_ground_elevation = np.mean(ground_elevations) if ground_elevations else 0
    max_height = max(all_heights_list)
    
    # Grid: the anemometer measurement heights first (config anemometer_heights)
    # so each single-height sonic sits exactly where it measured, then 20 m
    # spacing through the surface layer and 30 m aloft, matching the native
    # lidar and radar range-gate resolution.
    heights_below_100m = np.concatenate([
        np.array(get_near_surface_levels(), dtype=float),
        np.arange(20.0, 100.0, 20.0)
    ])
    heights_above_100m = np.arange(100.0, max_height + 30, 30.0)
    target_heights = np.concatenate([heights_below_100m, heights_above_100m])

    interpolated_wind = {}
    interpolated_turbulence = {}
    for instrument_name, result in instruments_data.items():

        if result.get('wind_profiles'):
            wind_interp = interpolate_profiles_to_grid(
                result['wind_profiles'], target_heights, 'wind', max_gap=max_gap
            )
            if wind_interp:
                interpolated_wind[instrument_name] = wind_interp

        if result.get('turbulence_profiles'):
            turb_interp = interpolate_profiles_to_grid(
                result['turbulence_profiles'], target_heights, 'turbulence', max_gap=max_gap
            )
            if turb_interp:
                interpolated_turbulence[instrument_name] = turb_interp
     

    merged_wind = average_multi_instrument_profiles(
        interpolated_wind, target_heights, ['ws', 'wd', 'w'],
        location=location,          
        time_interval=time_interval 
    )
    merged_turbulence = average_multi_instrument_profiles(
        interpolated_turbulence, target_heights, ['ti', 'tke', 'std_u', 'std_v', 'std_w'],
        location=location,          
        time_interval=time_interval  
    )

    combined_interval = {
        'time': int(time_interval.timestamp()),
        'ground_elevation': avg_ground_elevation,
        'wind_profiles': {
            'merged': merged_wind['profiles'],
            'individual': interpolated_wind,
            'quality_flags': merged_wind['flags'],
            'contributing_instruments': list(interpolated_wind.keys())
        },
        'turbulence_profiles': {
            'merged': merged_turbulence['profiles'],
            'individual': interpolated_turbulence,
            'quality_flags': merged_turbulence['flags'],
            'contributing_instruments': list(interpolated_turbulence.keys())
        }
    }

    if include_availability:
        combined_interval['availability'] = extract_availability_data(instruments_data)

    total_heights_available = 0
    total_heights_included = 0
    total_heights_excluded = 0
    instrument_summaries = {}
    
    for instrument_name, result in instruments_data.items():
        if 'filtering_summary' in result:
            fs = result['filtering_summary']
            total_heights_available += fs['total_heights_available']
            total_heights_included += fs['heights_included']
            total_heights_excluded += fs['heights_excluded']
            instrument_summaries[instrument_name] = fs
    
    retention_rate = total_heights_included / total_heights_available if total_heights_available > 0 else 0
    
    combined_interval['filtering_summary'] = {
        'total_heights_available': total_heights_available,
        'heights_included': total_heights_included,
        'heights_excluded': total_heights_excluded,
        'retention_rate': retention_rate,
        'by_instrument': instrument_summaries
    }

    return combined_interval

def merge_all_instruments_and_times(all_results, include_availability, location='nantucket'):
    """Index all per-instrument result lists by timestamp, then call
    create_combined_results for each time step to produce the merged product.
    """
    all_data_by_time = {}

    for instrument_name, results_list in all_results.items():

        for result in results_list:

            time_interval = result['time']

            if time_interval not in all_data_by_time:
                all_data_by_time[time_interval] = {}
                
            all_data_by_time[time_interval][instrument_name] = result

    combined_intervals = []

    for time_interval in sorted(all_data_by_time.keys()):
        instruments_data = all_data_by_time[time_interval]

        combined_interval = create_combined_results(
            time_interval, instruments_data, include_availability,
            location=location  
        )

        if combined_interval:
            combined_intervals.append(combined_interval)

    return {
        'time_intervals': combined_intervals,
        'processing_summary': {
            'total_time_intervals': len(combined_intervals),
            'instruments_processed': list(all_results.keys()),
        }
    }

def convert_radar_results_to_combined_format(radar_results):
    """Reformat radar results to match the combined dict structure used by
    merge_all_instruments_and_times. Radar has no turbulence data, so
    turbulence_profiles is set to an empty dict.
    """
    converted_results = []
    for result in radar_results:
        converted_result = {
            'time': result['time'],
            'instrument_code': result.get('instrument_code', 'radar'),
            'wind_profiles': result.get('profiles', {}),
            'turbulence_profiles': {},
            'ground_elevation': result.get('ground_elevation'),
            'latitude': result.get('latitude'),
            'longitude': result.get('longitude')
        }
        converted_results.append(converted_result)
    return converted_results

def finalize_combined_results(combined_results):
    """Attach processing metadata (coordinate system, QC flag legend, surface
    met units) to the combined results dict before NetCDF export.
    """
    final_results = combined_results.copy()
    
    final_results['processing_info'] = {
        'height_coordinate': 'AGL',
        'time_format': 'Unix timestamp (seconds since 1970-01-01)',
        'quality_flags': {
            0: 'Good data',
            1: 'Suspect data (large inter-instrument differences)',
            2: 'Bad data (high turbulence ratio)',
            3: 'No data available'
        },
        'surface_met_info': {
            'variables': ['pressure', 'temperature', 'relative_humidity', 'precipitation'],
            'units': {
                'pressure': 'hPa',
                'temperature': '°C', 
                'relative_humidity': '%',
                'precipitation': 'mm'
            },
            'averaging_period': f'{TIME_WINDOW_MINUTES} minutes',
            'missing_value_treatment': 'Values of 99999 converted to NaN'
        }
    }
    
    return final_results

def process_combined_profiles(instrument_configs, start_time, end_time,
                             location='nantucket',
                             availability_threshold=0.5,
                             include_availability=False,
                             verbose=False,
                             **kwargs):
    """
    Process both wind and turbulence profiles from all instruments and combine into 
    best-estimate profiles for each time interval.

    Parameters:
    -----------
    instrument_configs : dict
        Dictionary of instrument configurations
    start_time, end_time : str or datetime
        Time range for processing
    location : str
        Location identifier ('nantucket', 'block island', etc.)
    availability_threshold : float
        Minimum data availability threshold (0.0-1.0)
    include_availability : bool
        Whether to include detailed availability data in output
    **kwargs : dict
        Additional parameters (ignored)
    Returns:
    --------
    dict
        Combined results with wind and turbulence profiles
    """
    location = normalize_location(location)
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)
    print(f"Processing combined wind and turbulence profiles from {start_time} to {end_time}")
    print(f"Location: {location}")
    
    all_results = {}
    surface_met_config = None
    for instrument_name, config in instrument_configs.items():
        if instrument_name == 'surface_met':
            surface_met_config = config
            continue
        print(f"\nProcessing {instrument_name}...")
        filename = config['filename']
        instrument_params = config.get('params', {})
        try:
            if instrument_name.startswith('lidar_'):
                instrument_code = instrument_name.split('_')[1]
                instrument_params['availability_threshold'] = availability_threshold
                results = process_lidar_time_series(
                    instrument_code, filename, start_time, end_time, location,
                    verbose=verbose, **instrument_params
                )

            elif instrument_name.startswith('met_'):
                instrument_code = instrument_name.split('_')[1]
                
                if location == 'cape_cod':
                    results = process_sonic_c1_time_series(
                        filename, start_time, end_time, location,
                        instrument=instrument_code, verbose=verbose, **instrument_params
                    )
                elif location == 'rhode_island':
                    # Rhode Island uses EddyPro CSV format
                    results = process_rhod_sonic_time_series(
                        filename, start_time, end_time, location,
                        instrument=instrument_code, verbose=verbose, **instrument_params
                    )
                else:
                    instrument_params['instrument'] = instrument_code 
                    results = process_anemometer_time_series(
                        filename, start_time, end_time, location,
                        verbose=verbose, **instrument_params
                    )

            elif instrument_name == 'radar':
                if location == 'rhode_island':
                    # Rhode Island radar is NetCDF.
                    if isinstance(filename, str):
                        filenames = [filename]
                    else:
                        filenames = filename
                        
                    results = process_radar_netcdf_time_series(
                        filenames, start_time, end_time, location,
                        verbose=verbose, **instrument_params
                    )
                    
                    results = convert_radar_results_to_combined_format(results)
                    
                else:
                    # Nantucket/Block Island use text format - filename is single file
                    results = process_radar_time_series(
                        filename, start_time, end_time, location,
                        verbose=verbose, **instrument_params
                    )
                    results = convert_radar_results_to_combined_format(results)
            else:
                print(f"Unknown instrument type: {instrument_name}")
                continue
            if results:
                all_results[instrument_name] = results
                print(f"Successfully processed {len(results)} time intervals for {instrument_name}")
            else:
                print(f"No valid data for {instrument_name}")
        except Exception as e:
            print(f"Error processing {instrument_name}: {e}")
            continue
        
    surface_met_data = {}
    if surface_met_config:
        print(f"\nProcessing surface meteorological data...")
        try:
            date_str = start_time.strftime('%Y-%m-%d')
            start_time_str = start_time.strftime('%H:%M:%S')
            end_time_str = end_time.strftime('%H:%M:%S')
            surface_met_data = process_surface_met_for_date(
                date_str,
                data_dir=str(Path(surface_met_config['filename']).parent.parent),
                start_time=start_time_str,
                end_time=end_time_str,
                location=location
            )
            print(f"Successfully processed {len(surface_met_data)} surface met time windows")
            
            if location == 'block_island':
                wind_from_surface = extract_wind_from_surface_met(surface_met_data, location)
                if wind_from_surface:
                    all_results['met_z01'] = wind_from_surface
                    print(f"Extracted wind profiles from surface met: {len(wind_from_surface)} intervals")
            
        except Exception as e:
            print(f"Error processing surface met data: {e}")
            surface_met_data = {}
    
    if not all_results:
        print("No valid instrument data found")
        return {'time_intervals': []}
    
    print(f"\nTotal instruments passed to combination: {len(all_results)}")
    print(f"\nCombining data from {len(all_results)} instruments...")
    combined_results = merge_all_instruments_and_times(
        all_results, include_availability, location=location
    )
    
    if surface_met_data:
        print(f"Distributing surface met data into {len(combined_results.get('time_intervals', []))} time intervals...")
        for interval in combined_results.get('time_intervals', []):
            interval_unix_time = interval['time']
            # Nearest match within 300 s: both window kinds are TIME_WINDOW_MINUTES
            # long, so this absorbs clock skew without a cross-window match.
            best_match = None
            min_time_diff = float('inf')

            for surf_unix_time, surf_data in surface_met_data.items():
                time_diff = abs(interval_unix_time - surf_unix_time)
                if time_diff < min_time_diff and time_diff <= 300:
                    min_time_diff = time_diff
                    best_match = surf_data
            if best_match:
                interval['surface_met'] = {
                    'pressure': best_match['pressure'],
                    'temperature': best_match['temperature'],
                    'relative_humidity': best_match['relative_humidity'],
                    'precipitation': best_match['precipitation']
                }
            else:
                interval['surface_met'] = {
                    'pressure': np.nan,
                    'temperature': np.nan,
                    'relative_humidity': np.nan,
                    'precipitation': np.nan
                }

        intervals_with_surface_data = sum(1 for interval in combined_results.get('time_intervals', [])
                                     if 'surface_met' in interval and
                                     not all(np.isnan(list(interval['surface_met'].values()))))
        print(f"Successfully matched surface met data to {intervals_with_surface_data} time intervals")
    
    final_results = combined_results.copy()
    print(f"\nSuccessfully processed {len(final_results.get('time_intervals', []))} combined time intervals")
    return final_results

RAIN_COUNTER_MAX_GAP_S = 600
RAIN_COUNTER_SPIKE_MM = 0.5

# Cape Cod WXT536 screen: (measurement range, single-reading jump). The ranges
# are the sensor's specified measurement ranges; a jump larger than this between
# neighboring readings seconds apart is not a change in the air.
CACO_WXT_SCREEN = {
    'pressure': ((600.0, 1100.0), 2.0),
    'temperature': ((-52.0, 60.0), 2.0),
    'relative_humidity': ((0.0, 100.0), 5.0),
}
SURFACE_SPIKE_MAX_GAP_S = 120


def single_reading_spikes(seconds, values, threshold, max_gap_s, require_agreement=False):
    """
    Flag readings that differ from both of their neighbors, in the same
    direction and by more than ``threshold``. Neighbors further than
    ``max_gap_s`` away cannot show that a reading is corrupt and are not used.
    With ``require_agreement`` the two neighbors must also agree with each
    other to within half the threshold, so a reading in a steady rise or fall
    is never flagged.

    Parameters
    ----------
    seconds : numpy.ndarray
        Integer report times (s) of the valid readings, ascending.
    values : numpy.ndarray
        The valid readings, aligned with ``seconds``.

    Returns
    -------
    numpy.ndarray of bool
        True where a reading is flagged. First and last readings are never
        flagged.
    """
    flagged = np.zeros(values.size, dtype=bool)
    if values.size < 3:
        return flagged
    left = values[1:-1] - values[:-2]
    right = values[1:-1] - values[2:]
    close = ((seconds[1:-1] - seconds[:-2] <= max_gap_s)
             & (seconds[2:] - seconds[1:-1] <= max_gap_s))
    spike = close & (((left > threshold) & (right > threshold))
                     | ((left < -threshold) & (right < -threshold)))
    if require_agreement:
        spike &= np.abs(values[2:] - values[:-2]) < threshold / 2
    flagged[1:-1] = spike
    return flagged


def screen_single_readings(times, values, threshold, valid_range):
    """
    Set to NaN any reading outside ``valid_range`` and any lone reading that
    ``single_reading_spikes`` flags, for an instrument whose telegrams are
    occasionally corrupted in transmission.
    """
    values = np.asarray(values, dtype=float).copy()
    lo, hi = valid_range
    with np.errstate(invalid='ignore'):
        values[(values < lo) | (values > hi)] = np.nan
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size >= 3:
        seconds = np.asarray(times[valid], dtype='datetime64[s]').astype('int64')
        spike = single_reading_spikes(seconds, values[valid], threshold,
                                      SURFACE_SPIKE_MAX_GAP_S, require_agreement=True)
        values[valid[spike]] = np.nan
    return values


def rain_counter_increments(times, counter):
    """
    Rain per report from a cumulative rain counter, aligned to ``times``.

    The counter only rises while rain falls, so the rain in each report
    interval is the rise in the counter since the previous report. Three
    departures from that are handled:

    - A single reading differing from both neighbors in the same direction
      by more than ``RAIN_COUNTER_SPIKE_MM`` is a corrupted telegram and is
      skipped.
    - A drop to below ``RAIN_COUNTER_SPIKE_MM`` is a counter restart; the new
      reading is the rain since the restart.
    - Any other drop is noise. The level is the running maximum since the
      last restart, so a dip and the recovery after it add no rain.

    Rises across a gap longer than ``RAIN_COUNTER_MAX_GAP_S`` are not placed,
    because rain that fell during an outage cannot be located in time. The
    first report of a file has no predecessor and carries no amount.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Report times.
    counter : numpy.ndarray
        Counter values (mm); missing reports are NaN or beyond the ±9999
        sentinel.

    Returns
    -------
    numpy.ndarray
        Rain (mm) at each report time, NaN where no amount can be assigned.
    """
    counter = np.asarray(counter, dtype=float).copy()
    counter[np.abs(counter) > 9999] = np.nan
    out = np.full(counter.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(counter))
    if valid.size < 2:
        return out
    values = counter[valid]
    seconds = np.asarray(times[valid], dtype='datetime64[s]').astype('int64')
    if values.size >= 3:
        keep = ~single_reading_spikes(seconds, values, RAIN_COUNTER_SPIKE_MM,
                                      RAIN_COUNTER_MAX_GAP_S)
        valid, values, seconds = valid[keep], values[keep], seconds[keep]
        if valid.size < 2:
            return out
    level = values[0]
    for k in range(1, values.size):
        if values[k] < RAIN_COUNTER_SPIKE_MM and values[k] < level - RAIN_COUNTER_SPIKE_MM:
            amount, level = values[k], values[k]
        else:
            amount = max(values[k] - level, 0.0)
            level = max(level, values[k])
        if seconds[k] - seconds[k - 1] <= RAIN_COUNTER_MAX_GAP_S:
            out[valid[k]] = amount
    return out


def get_surface_met_files_for_date(date_str, location, data_dir):
    """Get all surface met files for a date, handling different file structures"""
    date_obj = pd.to_datetime(date_str)
    date_formatted = date_obj.strftime('%Y%m%d')
    
    if location == 'rhode_island':
        # Rhode Island: hourly files
        pattern = f"rhod.met.z01.a0.{date_formatted}.*.nc"
        files = glob.glob(os.path.join(data_dir, '**', pattern), recursive=True)
        return sorted(files)
    else:
        # Other sites: single daily file
        patterns = get_surface_met_patterns(location)
        for pattern in patterns:
            formatted_pattern = pattern.format(date_formatted=date_formatted)
            matches = glob.glob(os.path.join(data_dir, '**', formatted_pattern), recursive=True)
            if matches:
                return [matches[0]]
    return []

def process_surface_met_for_date(date_str, data_dir=None, start_time=None, end_time=None,
                               file_path=None, location='nantucket', verbose=False):
    """
    Process surface meteorological data for specified date and time range.
    Handles both single daily files and multiple hourly files (Rhode Island).
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    data_dir : str, optional
        Path to data directory. If None, will try to extract from file_path
    start_time : str, optional
        Start time in 'HH:MM:SS' format
    end_time : str, optional
        End time in 'HH:MM:SS' format
    file_path : str, optional
        Direct path to surface met file. If provided, overrides data_dir search
    location : str
        Location identifier ('nantucket', 'block_island', 'rhode_island', 'cape_cod')
    verbose : bool, optional
        Print detailed processing information
        
    Returns:
    --------
    dict : Surface met data in 10-minute intervals
        {unix_timestamp: {'pressure': val, 'temperature': val,
                         'relative_humidity': val, 'precipitation': val}}
    """
    
    file_paths = []
    if file_path:
        file_paths = [file_path]
    else:
        if not data_dir:
            return {}
        file_paths = get_surface_met_files_for_date(date_str, location, data_dir)
    
    if not file_paths:
        if verbose:
            print(f"No surface met files found for {date_str} at {location}")
        return {}
    
    if verbose:
        print(f"Found {len(file_paths)} surface met files for {date_str}")
        for fp in file_paths[:3]:
            print(f"  {os.path.basename(fp)}")
        if len(file_paths) > 3:
            print(f"  ... and {len(file_paths)-3} more")
    
    all_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            if verbose:
                print(f"File not found: {file_path}")
            continue
            
        try:
            ds = xr.open_dataset(file_path)
            time_data = pd.to_datetime(ds.time.values)
            
            var_map = {
                'pressure': ['air_pressure', 'barometric_pressure', 'pressure', 'pres', 'atmos_pressure'],
                'temperature': ['ambient_air_temperature', 'air_temperature', 'temperature', 'temp'],
                'relative_humidity': ['relative_humidity', 'rh', 'humidity'],
            }

            if location not in ['rhode_island', 'rhod']:
                var_map['precipitation'] = ['precipitation', 'precip', 'rainfall']

            # Add wind variables for Block Island only (other surface mets don't include wind data)
            if location == 'block_island':
                var_map.update({
                    'wind_speed': ['wind_speed', 'ws'],
                    'wind_direction': ['wind_direction', 'wd', 'wind_dir']
                })
            
            variables = {}
            for var, possible_names in var_map.items():
                data = None
                for name in possible_names:
                    if name in ds.variables:
                        data = ds[name].values.astype(float)
                        # Sentinel cleaning: DOE ARM/NOAA met files encode missing
                        # values as ±9999 / ±99999; treat anything beyond that as NaN.
                        data[np.abs(data) > 9999] = np.nan
                        break
                variables[var] = data if data is not None else np.full(len(time_data), np.nan)

            if 'precipitation' not in variables:
                variables['precipitation'] = np.full(len(time_data), np.nan)

            # The Cape Cod WXT536 reports rain as a running counter, not an amount.
            if location in ('cape_cod', 'caco') and 'rain_accumulation' in ds.variables:
                variables['precipitation'] = rain_counter_increments(
                    time_data, ds['rain_accumulation'].values)

            # The same corrupted telegrams reach pressure, temperature and
            # humidity, where a single bad reading would bias a 10-minute mean.
            if location in ('cape_cod', 'caco'):
                for var, (limits, jump) in CACO_WXT_SCREEN.items():
                    variables[var] = screen_single_readings(time_data, variables[var], jump, limits)
            
            file_df = pd.DataFrame(variables, index=time_data)
            all_data.append(file_df)
            
            ds.close()
            
        except Exception as e:
            if verbose:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue
    
    if not all_data:
        if verbose:
            print(f"No valid data found in any files for {date_str}")
        return {}
    
    df = pd.concat(all_data, ignore_index=False).sort_index()
    
    df = df[~df.index.duplicated(keep='first')]
    
    if verbose:
        print(f"Combined data: {len(df)} time points from {df.index.min()} to {df.index.max()}")
    
    date_obj = pd.to_datetime(date_str)
    # Left-closed window edges for the day; [:-1] drops the next day's 00:00
    # so each window is labeled by its start time and days don't double-count.
    time_windows = pd.date_range(
        start=date_obj,
        end=date_obj + timedelta(days=1),
        freq=f'{TIME_WINDOW_MINUTES}min'
    )[:-1]
    
    # The integer backing a datetime64 is in the index's own resolution, which
    # is not guaranteed to be nanoseconds, so convert through the dtype.
    unix_times = time_windows.astype('datetime64[s]').astype('int64')
    
    result = {}
    valid_windows = 0
    
    for window_start, unix_time in zip(time_windows, unix_times):
        window_end = window_start + timedelta(minutes=TIME_WINDOW_MINUTES)
        window_mask = (df.index >= window_start) & (df.index < window_end)
        window_data = df[window_mask]
        
        if len(window_data) > 0:
            # An all-NaN window must yield NaN: np.nansum would return 0.0, which
            # for precipitation is indistinguishable from a genuinely dry interval.
            window_result = {
                'pressure': float(np.nanmean(window_data['pressure'])) if len(window_data['pressure'].dropna()) > 0 else np.nan,
                'temperature': float(np.nanmean(window_data['temperature'])) if len(window_data['temperature'].dropna()) > 0 else np.nan,
                'relative_humidity': float(np.nanmean(window_data['relative_humidity'])) if len(window_data['relative_humidity'].dropna()) > 0 else np.nan,
                'precipitation': float(np.nansum(window_data['precipitation'])) if len(window_data['precipitation'].dropna()) > 0 else np.nan
            }

            # Until 12 May 2024 the Cape Cod WXT536 sent its rain telegram only
            # while rain was falling, so a window with no rain report is dry if
            # the station was otherwise reporting, and missing if it was silent.
            if (location in ('cape_cod', 'caco') and np.isnan(window_result['precipitation'])
                    and any(np.isfinite(window_result[k])
                            for k in ('pressure', 'temperature', 'relative_humidity'))):
                window_result['precipitation'] = 0.0
            
            if location == 'block_island' and len(window_data) > 0:
                if 'wind_speed' in window_data.columns and 'wind_direction' in window_data.columns:
                    ws_mean = float(np.nanmean(window_data['wind_speed']))
                    wd_values = window_data['wind_direction'].dropna().values
                    wd_mean = float(wind_direction_average(wd_values)) if len(wd_values) > 0 else np.nan

                    if not (np.isnan(ws_mean) or np.isnan(wd_mean)):
                        # Keyed by the configured measurement height (10 m
                        # surface-met tower per NOAA PSL spec,
                        # https://psl.noaa.gov/data/obs/instruments/SurfaceMetDescription.html);
                        # downstream uses this key as the true AGL height.
                        surf_wind_height = float(LOCATION_CONFIG.get(location, {})
                                                 .get('anemometer_heights', {})
                                                 .get('surf_met', 10.0))
                        window_result['wind_profile'] = {
                            surf_wind_height: {
                                'ws': round(ws_mean, 2),
                                'wd': round(wd_mean, 1),
                            }
                        }
            
            if not all(np.isnan(v) for v in window_result.values() if isinstance(v, (int, float))):
                valid_windows += 1
                
        else:
            window_result = {
                'pressure': np.nan, 
                'temperature': np.nan,
                'relative_humidity': np.nan, 
                'precipitation': np.nan
            }
        
        result[int(unix_time)] = window_result
    
    if verbose:
        total_windows = len(time_windows)
        coverage_pct = (valid_windows / total_windows) * 100
        print(f"Surface met processing complete: {valid_windows}/{total_windows} windows ({coverage_pct:.1f}%) with valid data")
    
    return result