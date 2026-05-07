"""File discovery and CF-compliant NetCDF export

Searches the configured ``DATA_BASE_PATH`` and ``ARCHIVE_BASE_PATH`` to
assemble per-instrument file lists for a given site and date range, then
writes the merged products as CF-compliant NetCDF files with per-height
quality flags and global attributes documenting the campaign.

Entry points:

  - ``discover_instrument_files_for_date_range``: returns a dict mapping
    each active instrument to its list of input files for the given
    range, accommodating per-instrument file naming conventions.
  - ``save_results_to_netcdf``: writes the merged product to
    ``WINDPROF_RESULTS_PATH`` with the canonical filename
    ``{site}.windprof.z01.c1.{YYYYMMDD}.000000.nc``.

File-naming patterns are campaign-specific. Adapters should expect
to edit this module to match the file conventions of their own
data delivery; the per-site discovery functions are a reasonable
template but are not generic.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
from .config import (normalize_location, SITE_INSTRUMENT_MAPPINGS, TURBULENCE_INSTRUMENTS,
                     WIND_INSTRUMENTS, VAD_INSTRUMENTS, calculate_average_coordinates,
                     DATA_BASE_PATH, ARCHIVE_BASE_PATH)

#### File Discovery

def find_caco_file_for_date(target_date):
    """Locate the CACO z01 lidar workbook whose internal time range covers target_date.

    CACO z01 files are delivered as monthly-ish .xlsx workbooks spanning roughly
    30-40 days, so the filename date is not a reliable index — we open candidate
    files in the current and two prior months and check the actual first/last
    timestamps on Sheet1.
    """
    year = target_date.year
    month = target_date.month

    # Fetch current month plus two prior months to cover the ~30-40 day span.
    search_months = []
    for offset in [0, -1, -2]:
        search_month = month + offset
        search_year = year
        if search_month <= 0:
            search_month += 12
            search_year -= 1
        search_months.append((search_year, search_month))
    
    for search_year, search_month in search_months:
        pattern = f'{ARCHIVE_BASE_PATH}/caco.lidar.z01.00/{search_year}/{search_month:02d}/*/caco.lidar.z01.00.*.xlsx'
        files = glob.glob(pattern)

        for filepath in files:
            try:
                # header=41 skips the vendor metadata block; column 0 is the timestamp
                df_start = pd.read_excel(filepath, sheet_name='Sheet1', header=41, usecols=[0], nrows=1)
                df_end = pd.read_excel(filepath, sheet_name='Sheet1', header=41, usecols=[0]).tail(1)

                start_time = pd.to_datetime(df_start.iloc[0, 0])
                end_time = pd.to_datetime(df_end.iloc[0, 0])

                if start_time.date() <= target_date.date() <= end_time.date():
                    return filepath
            except Exception:
                continue

    return None

def discover_instrument_files_for_date_range(start_date, end_date, location='nantucket', verbose=False):
    """
    Discover available instrument data files for a date range.

    Uses a hybrid file-source strategy: high-volume instruments (scanning lidars)
    are pre-merged to a local directory for fast access, while low-volume daily
    files are read directly from the campaign archive tree (``ARCHIVE_BASE_PATH``,
    e.g. ``/data`` on the WFIP3 server) to avoid duplicating storage.

    Parameters:
    -----------
    start_date : str or datetime
        Start date (format: YYYY-MM-DD or datetime object)
    end_date : str or datetime
        End date (format: YYYY-MM-DD or datetime object)
    location : str
        Location identifier ('nantucket', 'nant', or 'block_island')
    verbose : bool, optional
        Print detailed file discovery information (default: False)
        
    Returns:
    --------
    dict
        Configuration dictionary organized by date:
        {date_str: {instrument_name: {'filename': path, 'params': defaults}}}
        
    """
    location = normalize_location(location)
    # Convert to datetime objects
    if isinstance(start_date, str):
        start_obj = pd.to_datetime(start_date)
    else:
        start_obj = start_date
    if isinstance(end_date, str):
        end_obj = pd.to_datetime(end_date)
    else:
        end_obj = end_date
    
    # Generate date range
    date_range = pd.date_range(start=start_obj, end=end_obj, freq='D')
    
    # Dynamically determine paths based on the date
    year = start_obj.year
    month = start_obj.month
    yyyymm = f"{year}{month:02d}/"    
    
    # Location-specific configurations
    if location == 'nantucket':
        site_prefix = 'nant'
        z03_ext = 'rtd'
        instruments = {
            # Merged files - look in local directory (high-volume instruments)
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z01.a0.{{date_formatted}}.*.nc', 'params': {'qc_threshold': -23, 'min_beams': 3}},
            'lidar_z02': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z02.a0.{{date_formatted}}.*.nc', 'params': {'qc_threshold': 0.008, 'min_beams': 3}},
            
            # Single daily files - look directly in ARCHIVE_BASE_PATH (low-volume instruments)
            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.{z03_ext}', 'params': {'cnr_threshold': -23, 'min_beams': 3}},
            'met_z02': {'path_template': f'{ARCHIVE_BASE_PATH}/nant.met.z02.a0/{{year}}/{{month:02d}}/{{day:02d}}/nant.met.z02.a0.{{date_formatted}}.*.nc', 'params': {'instrument': 'z02'}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z05.c1/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z05.c1.{{date_formatted}}.*.nc', 'params': {}},
            
            # Manually added files - local directory
            'radar': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.radar.{{date_formatted}}.txt', 'params': {}}
        }
    elif location == 'block_island':
        site_prefix = 'bloc'
        z03_ext = 'sta'
        instruments = {
            # Merged files - look in local directory
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z01.a0.{{date_formatted}}.*.nc', 'params': {'qc_threshold': -23, 'min_beams': 3}},
            
            # Single daily files - look directly in ARCHIVE_BASE_PATH
            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.{z03_ext}', 'params': {'cnr_threshold': -22, 'min_beams': 3}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z01.c1/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z01.c1.{{date_formatted}}.*.nc', 'params': {}},
            
            # Manually added files - local directory
            'radar': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.radar.{{date_formatted}}.txt', 'params': {}}
        }
    elif location == 'rhode_island':
        site_prefix = 'rhod'
        instruments = {
            # Merged files - look in local directory
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}/{site_prefix}.lidar.z01.a0.{{date_formatted}}.000000.nc', 'params': {'qc_threshold': 1.008, 'min_beams': 3}},

            # Single daily files - look directly in ARCHIVE_BASE_PATH
            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.csv', 'params': {'cnr_threshold': -23, 'min_beams': 3}},
            'met_z01': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.sonic.z01.c1/{{year}}/{{month:02d}}/{{day:02d}}/10/full_output/ver_3/{site_prefix}.sonic.z01.c1.{{date_formatted}}.000000.10.full_output.ver_3.csv', 'params': {}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z01.a0/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z01.a0.{{date_formatted}}.*.nc', 'params': {}}
        }
    elif location == 'cape_cod': 
        site_prefix = 'caco'
        # CACO z01 is delivered as multi-day .xlsx workbooks rather than per-date
        # files, so it is discovered by find_caco_file_for_date() outside the standard
        # glob loop below. All other CACO instruments follow the path_template pattern.
        instruments = {
            'lidar_z02': {'path_template': f'{DATA_BASE_PATH}{yyyymm}caco.lidar.z02.a0.{{date_formatted}}.*.nc', 'params': {}},
            'met_z01': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.sonic.z01.c1/{{year}}/{{month:02d}}/{{day:02d}}/10/full_output/ver_3/{site_prefix}.sonic.z01.c1.{{date_formatted}}.000000.10.full_output.ver_3.csv', 'params': {}},
            'met_z02': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.sonic.z02.c1/{{year}}/{{month:02d}}/{{day:02d}}/10/full_output/ver_3/{site_prefix}.sonic.z02.c1.{{date_formatted}}.000000.10.full_output.ver_3.csv', 'params': {}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/caco.met.z01.00/{{year}}/{{month:02d}}/{{day:02d}}/caco.met.z01.00.{{date_formatted}}.*.nc', 'params': {}},
        }
    else:
        raise ValueError(f"Unknown location: {location}. Supported: 'nantucket', 'block_island', 'rhode_island', 'cape_cod'")
    
    all_configs = {}
    for date_obj in date_range:
        date_str = date_obj.strftime('%Y-%m-%d')
        date_formatted = date_obj.strftime('%Y%m%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        if verbose:
            print(f"Looking for {location} files for {date_str}...")
        
        found_configs = {}

        # CACO z01 uses multi-day .xlsx workbooks, dispatched separately
        if location == 'cape_cod':
            filepath = find_caco_file_for_date(date_obj)
            if filepath:
                found_configs['lidar_z01'] = {'filename': filepath, 'params': {}}
                if verbose:
                    print(f"  ✓ lidar_z01: {filepath}")
            elif verbose:
                print(f"  ✗ lidar_z01: No CACO workbook covering {date_str}")

        # All other instruments follow the per-date glob pattern
        for instrument_name, config in instruments.items():
            pattern = config['path_template'].format(
                date_formatted=date_formatted,
                year=year,
                month=month,
                day=day
            )
            files = glob.glob(pattern)
            if files:
                found_configs[instrument_name] = {
                    'filename': files[0],
                    'params': config['params']
                }
                if verbose:
                    print(f"  ✓ {instrument_name}: {files[0]}")
            elif verbose:
                print(f"  ✗ {instrument_name}: {pattern}")
        
        if found_configs:
            all_configs[date_str] = found_configs

    return all_configs

def list_file_availability_for_date_range(start_date, end_date, location='nantucket', verbose=False):
    """Return {date_str: {instrument_name: filename}} for the date range.
    Convenience wrapper around discover_instrument_files_for_date_range that
    strips the params dicts from the result.
    """

    configs = discover_instrument_files_for_date_range(start_date, end_date, location=location, verbose=verbose)
    if not configs:
        print("No files found for the specified date range.")
        return {}
    
    result = {}
    for date_str, day_configs in configs.items():
        result[date_str] = {instrument: config['filename']
                          for instrument, config in day_configs.items()}
    return result

def discover_instrument_files_for_date(target_date, location='nantucket', verbose=False):
    """Single-date convenience wrapper around discover_instrument_files_for_date_range.
    Returns {instrument_name: {'filename': path, 'params': defaults}}.
    """

    configs = discover_instrument_files_for_date_range(target_date, target_date, location=location, verbose=verbose)
    if configs:
        date_str = list(configs.keys())[0]
        return configs[date_str]
    return {}

#### Saving and Loading netCDF

def save_results_to_netcdf(combined_results, output_path, location='nantucket', verbose=False):
    """Write combined wind/turbulence profile results to a tsdat-style NetCDF.

    Missing samples are stored as the -9999 sentinel (matching the `_FillValue`
    attribute on every variable); quality flags use 0=good, 1=suspect,
    2=bad, 3=no_data.
    """

    time_intervals = combined_results.get('time_intervals', [])
    if not time_intervals:
        print("No time intervals to save")
        return None
    if verbose:
        print(f"Saving {len(time_intervals)} time intervals to NetCDF...")
    
    # Get site config from centralized mappings
    location = normalize_location(location)
    site_config = SITE_INSTRUMENT_MAPPINGS.get(location, SITE_INSTRUMENT_MAPPINGS['nantucket'])
    site_prefix = site_config['prefix']
    instrument_mapping = site_config['instruments']
    
    # Extract heights, instruments, and times
    all_heights = set()
    all_instruments = set()
    
    for interval in time_intervals:
        if 'wind_profiles' in interval and 'merged' in interval['wind_profiles']:
            all_heights.update(interval['wind_profiles']['merged'].keys())
        if 'turbulence_profiles' in interval and 'merged' in interval['turbulence_profiles']:
            all_heights.update(interval['turbulence_profiles']['merged'].keys())
        if 'wind_profiles' in interval and 'individual' in interval['wind_profiles']:
            all_instruments.update(interval['wind_profiles']['individual'].keys())
        if 'turbulence_profiles' in interval and 'individual' in interval['turbulence_profiles']:
            all_instruments.update(interval['turbulence_profiles']['individual'].keys())
    
    heights = np.array(sorted(all_heights))
    instruments = sorted(all_instruments)
    times = np.array([interval['time'] for interval in time_intervals])
    
    # Convert to seconds since epoch
    time_seconds = times.astype('datetime64[s]').astype(np.int64)
    n_times = len(times)
    n_heights = len(heights)
    
    # Calculate average coordinates
    avg_lat, avg_lon = calculate_average_coordinates(location, instruments)
    
    # Initialize ALL possible instrument data arrays (even if not found)
    all_possible_instruments = list(instrument_mapping.keys())
    instrument_data = {}
    
    for instrument in all_possible_instruments:
        instrument_data[instrument] = {}
        
        # Wind variables for all instruments
        if instrument in WIND_INSTRUMENTS:
            instrument_data[instrument].update({
                'ws': np.full((n_times, n_heights), -9999.0),
                'wd': np.full((n_times, n_heights), -9999.0),
                'w': np.full((n_times, n_heights), -9999.0)
            })
            
            # VAD error variables
            if instrument in VAD_INSTRUMENTS:
                instrument_data[instrument].update({
                    'uerr': np.full((n_times, n_heights), -9999.0),
                    'verr': np.full((n_times, n_heights), -9999.0),
                    'werr': np.full((n_times, n_heights), -9999.0),
                    'wserr': np.full((n_times, n_heights), -9999.0),
                    'wderr': np.full((n_times, n_heights), -9999.0)
                })

        # Turbulence variables
        if instrument in TURBULENCE_INSTRUMENTS:
            instrument_data[instrument].update({
                'ti': np.full((n_times, n_heights), -9999.0),
                'tke': np.full((n_times, n_heights), -9999.0),
                'std_u': np.full((n_times, n_heights), -9999.0),
                'std_v': np.full((n_times, n_heights), -9999.0),
                'std_w': np.full((n_times, n_heights), -9999.0)
            })
    
    # Initialize merged data arrays
    ws_data = np.full((n_times, n_heights), -9999.0)
    wd_data = np.full((n_times, n_heights), -9999.0)
    w_data = np.full((n_times, n_heights), -9999.0)
    ti_data = np.full((n_times, n_heights), -9999.0)
    tke_data = np.full((n_times, n_heights), -9999.0)
    std_u_data = np.full((n_times, n_heights), -9999.0)
    std_v_data = np.full((n_times, n_heights), -9999.0)
    std_w_data = np.full((n_times, n_heights), -9999.0)
    # Per-parameter QC flags default to 3 ("no data"); the fill loop below promotes
    # cells to 0/1/2 wherever the merge step produced a usable value.
    ws_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    wd_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    w_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    ti_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    
    # Surface met and ground elevation
    surface_pressure = np.full(n_times, -9999.0)
    surface_temperature = np.full(n_times, -9999.0)
    surface_humidity = np.full(n_times, -9999.0)
    surface_precipitation = np.full(n_times, -9999.0)
    ground_elevation = np.full(n_times, -9999.0)
    
    # Instrument availability bitwise flags (per time step)
    instrument_flags = np.zeros(n_times, dtype=np.int32)
    instrument_flag_mapping = {inst: i for i, inst in enumerate(all_possible_instruments)}
    
    # Fill data arrays 
    for t_idx, interval in enumerate(time_intervals):
        # Track which instruments were found this time step
        found_instruments = set()
        if 'wind_profiles' in interval and 'individual' in interval['wind_profiles']:
            found_instruments.update(interval['wind_profiles']['individual'].keys())
        if 'turbulence_profiles' in interval and 'individual' in interval['turbulence_profiles']:
            found_instruments.update(interval['turbulence_profiles']['individual'].keys())
        
        # Set bitwise flags for instrument availability
        for instrument in found_instruments:
            if instrument in instrument_flag_mapping:
                instrument_flags[t_idx] |= (1 << instrument_flag_mapping[instrument])
        
        # Ground elevation
        if 'ground_elevation' in interval:
            ground_elevation[t_idx] = interval['ground_elevation']
            
        # Fill merged data arrays
        if 'wind_profiles' in interval and 'merged' in interval['wind_profiles']:
            for height, profile in interval['wind_profiles']['merged'].items():
                try:
                    h_idx = np.where(heights == height)[0][0]
                    if 'ws' in profile and not np.isnan(profile['ws']):
                        ws_data[t_idx, h_idx] = profile['ws']
                    if 'wd' in profile and not np.isnan(profile['wd']):
                        wd_data[t_idx, h_idx] = profile['wd']
                    if 'w' in profile and not np.isnan(profile['w']):
                        w_data[t_idx, h_idx] = profile['w']
                except IndexError:
                    continue
                    
        if 'turbulence_profiles' in interval and 'merged' in interval['turbulence_profiles']:
            for height, profile in interval['turbulence_profiles']['merged'].items():
                try:
                    h_idx = np.where(heights == height)[0][0]
                    if 'ti' in profile and not np.isnan(profile['ti']):
                        ti_data[t_idx, h_idx] = profile['ti']
                    if 'tke' in profile and not np.isnan(profile['tke']):
                        tke_data[t_idx, h_idx] = profile['tke']
                    if 'std_u' in profile and not np.isnan(profile['std_u']):
                        std_u_data[t_idx, h_idx] = profile['std_u']
                    if 'std_v' in profile and not np.isnan(profile['std_v']):
                        std_v_data[t_idx, h_idx] = profile['std_v']
                    if 'std_w' in profile and not np.isnan(profile['std_w']):
                        std_w_data[t_idx, h_idx] = profile['std_w']
                except IndexError:
                    continue

        # Fill quality flags (parameter-specific) 
        if 'wind_profiles' in interval and 'quality_flags' in interval['wind_profiles']:
            for height, flag_dict in interval['wind_profiles']['quality_flags'].items():
                try:
                    h_idx = np.where(heights == height)[0][0]
                    ws_quality_flags[t_idx, h_idx] = flag_dict.get('ws', 3)
                    wd_quality_flags[t_idx, h_idx] = flag_dict.get('wd', 3)
                    w_quality_flags[t_idx, h_idx] = flag_dict.get('w', 3)
                except IndexError:
                    continue
        
        if 'turbulence_profiles' in interval and 'quality_flags' in interval['turbulence_profiles']:
            for height, flag_dict in interval['turbulence_profiles']['quality_flags'].items():
                try:
                    h_idx = np.where(heights == height)[0][0]
                    ti_quality_flags[t_idx, h_idx] = flag_dict.get('ti', 3)
                except IndexError:
                    continue
                    
        # Fill individual instrument data arrays (only for instruments that were found in this interval)
        if 'wind_profiles' in interval and 'individual' in interval['wind_profiles']:
            for instrument, profiles in interval['wind_profiles']['individual'].items():
                if instrument in instrument_data:
                    for height, profile in profiles.items():
                        try:
                            h_idx = np.where(heights == height)[0][0]
                            for param in ['ws', 'wd', 'w', 'uerr', 'verr', 'werr', 'wserr', 'wderr']:
                                if param in profile and param in instrument_data[instrument] and not np.isnan(profile[param]):
                                    instrument_data[instrument][param][t_idx, h_idx] = profile[param]
                        except IndexError:
                            continue
                            
        if 'turbulence_profiles' in interval and 'individual' in interval['turbulence_profiles']:
            for instrument, profiles in interval['turbulence_profiles']['individual'].items():
                if instrument in instrument_data:
                    for height, profile in profiles.items():
                        try:
                            h_idx = np.where(heights == height)[0][0]
                            for param in ['ti', 'tke', 'std_u', 'std_v', 'std_w']:
                                if param in profile and param in instrument_data[instrument] and not np.isnan(profile[param]):
                                    instrument_data[instrument][param][t_idx, h_idx] = profile[param]
                        except IndexError:
                            continue
                            
        # Surface met data 
        if 'surface_met' in interval:
            surface_met = interval['surface_met']
            if 'pressure' in surface_met and not np.isnan(surface_met['pressure']):
                surface_pressure[t_idx] = surface_met['pressure']
            if 'temperature' in surface_met and not np.isnan(surface_met['temperature']):
                surface_temperature[t_idx] = surface_met['temperature']
            if 'relative_humidity' in surface_met and not np.isnan(surface_met['relative_humidity']):
                surface_humidity[t_idx] = surface_met['relative_humidity']
            if 'precipitation' in surface_met and not np.isnan(surface_met['precipitation']):
                surface_precipitation[t_idx] = surface_met['precipitation']
    
    # Create data variables dictionary 
    data_vars = {
        # Merged variables
        'wind_speed_merged': (['time', 'height'], ws_data, {
            'units': 'm/s', 
            'long_name': 'Merged Wind Speed', 
            '_FillValue': -9999.0
        }),
        'wind_direction_merged': (['time', 'height'], wd_data, {
            'units': 'degrees', 
            'long_name': 'Merged Wind Direction', 
            '_FillValue': -9999.0
        }),
        'vertical_velocity_merged': (['time', 'height'], w_data, {
            'units': 'm/s', 
            'long_name': 'Merged Vertical Wind Speed Component', 
            '_FillValue': -9999.0,
            'convention': 'Positive upward, negative downward (meteorological convention)'
        }),
        'turbulence_intensity_merged': (['time', 'height'], ti_data, {
            'units': '-', 
            'long_name': 'Merged Turbulence Intensity', 
            '_FillValue': -9999.0
        }),
        'turbulent_kinetic_energy_merged': (['time', 'height'], tke_data, {
            'units': 'm²/s²', 
            'long_name': 'Merged Turbulent Kinetic Energy', 
            '_FillValue': -9999.0
        }),
        'u_std_merged': (['time', 'height'], std_u_data, {
            'units': 'm/s', 
            'long_name': 'Merged U-Component Wind Speed Standard Deviation', 
            '_FillValue': -9999.0
        }),
        'v_std_merged': (['time', 'height'], std_v_data, {
            'units': 'm/s', 
            'long_name': 'Merged V-Component Wind Speed Standard Deviation', 
            '_FillValue': -9999.0
        }),
        'w_std_merged': (['time', 'height'], std_w_data, {
            'units': 'm/s', 
            'long_name': 'Merged W-Component Wind Speed Standard Deviation', 
            '_FillValue': -9999.0
        }),

        # Quality control variables for merged data
        'qc_wind_speed_merged': (['time', 'height'], ws_quality_flags, {
            'long_name': 'Quality control for merged wind speed',
            'description': 'Quality flags based on inter-instrument agreement and physical validation',
            'flag_values': [0, 1, 2, 3],
            'flag_meanings': 'good_data suspect_data bad_data no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data: inter-instrument disagreement (ws spread >max(1.5 m/s, 50% median))",
            'flag_2_description': "Bad data: exceeds physical limits (σᵤ/ū >1.0)",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),
        'qc_wind_direction_merged': (['time', 'height'], wd_quality_flags, {
            'long_name': 'Quality control for merged wind direction',
            'description': 'Quality flags based on inter-instrument agreement and physical validation',
            'flag_values': [0, 1, 2, 3],
            'flag_meanings': 'good_data suspect_data bad_data no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data: inter-instrument disagreement (wd spread >30°)", 
            'flag_2_description': "Bad data",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),
        'qc_vertical_velocity_merged': (['time', 'height'], w_quality_flags, {
            'long_name': 'Quality control for merged vertical velocity',
            'description': 'Quality flags based on inter-instrument agreement and physical validation',
            'flag_values': [0, 1, 2, 3],
            'flag_meanings': 'good_data suspect_data bad_data no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data: inter-instrument disagreement (w spread >2.0 m/s) or extreme values (|w| > 7 m/s)",
            'flag_2_description': "Bad data",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),
        'qc_turbulence_intensity_merged': (['time', 'height'], ti_quality_flags, {
            'long_name': 'Quality control for merged turbulence intensity',
            'description': 'Quality flags based on inter-instrument agreement and physical validation',
            'flag_values': [0, 1, 2, 3],
            'flag_meanings': 'good_data suspect_data bad_data no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data",
            'flag_2_description': "Bad data: exceeds physical limits (σᵤ/ū >1.0)",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),        
        
        # Surface variables
        'surface_pressure': (['time'], surface_pressure, {
            'units': 'hPa', 
            'long_name': 'Surface Air Pressure', 
            '_FillValue': -9999.0
        }),
        'surface_temperature': (['time'], surface_temperature, {
            'units': 'degrees_C', 
            'long_name': 'Surface Air Temperature', 
            '_FillValue': -9999.0
        }),
        'surface_humidity': (['time'], surface_humidity, {
            'units': '%', 
            'long_name': 'Surface Relative Humidity', 
            '_FillValue': -9999.0
        }),
        'surface_precipitation': (['time'], surface_precipitation, {
            'units': 'mm', 
            'long_name': 'Surface Precipitation', 
            '_FillValue': -9999.0
        }),
        'ground_elevation': (['time'], ground_elevation, {
            'units': 'm', 
            'long_name': 'Ground Elevation Above Sea Level', 
            '_FillValue': -9999.0
        }),

        'instrument_availability': (['time'], instrument_flags, {
            'long_name': 'Bitwise flag indicating available instruments',
            'flag_values': [2**i for i in range(len(all_possible_instruments))],
            'flag_meanings': ' '.join([f'flag_{i}_{inst}' for i, inst in enumerate(all_possible_instruments)]),
            '_FillValue': -9999 
        })

    }

    # Add individual instrument variables and their QC variables
    param_mappings = {
        'ws': ('wind_speed', 'm/s', 'Wind Speed'),
        'wd': ('wind_direction', 'degrees', 'Wind Direction'), 
        'w': ('vertical_velocity', 'm/s', 'Vertical Wind Speed Component'),
        'ti': ('turbulence_intensity', '1', 'Turbulence Intensity'),
        'tke': ('turbulent_kinetic_energy', 'm²/s²', 'Turbulent Kinetic Energy'),
        'std_u': ('u_std', 'm/s', 'U-Component Wind Speed Standard Deviation'),
        'std_v': ('v_std', 'm/s', 'V-Component Wind Speed Standard Deviation'),
        'std_w': ('w_std', 'm/s', 'W-Component Wind Speed Standard Deviation'),
        'uerr': ('u_error', 'm/s', 'VAD-derived U-Component Uncertainty'),
        'verr': ('v_error', 'm/s', 'VAD-derived V-Component Uncertainty'),
        'werr': ('w_error', 'm/s', 'VAD-derived W-Component Uncertainty'),
        'wserr': ('wind_speed_error', 'm/s', 'VAD-derived Wind Speed Uncertainty'),
        'wderr': ('wind_direction_error', 'degrees', 'VAD-derived Wind Direction Uncertainty (NaN for ws <2.0 m/s, capped at 180°)')
    }

    for instrument in all_possible_instruments:
        instrument_name = instrument_mapping[instrument]
        for param, data_array in instrument_data[instrument].items():
            var_name, units, long_name_base = param_mappings[param]
            
            # Data variable for this parameter and instrument
            var_key = f"{var_name}_{instrument}"
            data_vars[var_key] = (['time', 'height'], data_array, {
                'units': units,
                'long_name': f'{long_name_base} from {instrument_name}',
                '_FillValue': -9999.0
            })
            
            # Simple QC variable for individual instruments (availability-based)
            qc_data = np.where(data_array == -9999.0, 1, 0).astype(np.int32)
            qc_var_key = f"qc_{var_name}_{instrument}"  
            data_vars[qc_var_key] = (['time', 'height'], qc_data, {
                'long_name': f'Quality control for {long_name_base.lower()} from {instrument_name}',
                'description': 'Simple data availability flag: 0=data present, 1=no data',
                'flag_values': [0, 1],  
                'flag_meanings': 'good_data missing_data', 
                '_FillValue': -9999
            })
    
    # Create dataset!
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': ('time', time_seconds, {
                'units': 'Seconds since 1970-01-01 00:00:00 UTC',
                'long_name': 'Time'
            }),
            'height': ('height', heights, {
                'units': 'm', 
                'long_name': 'Height Above Ground Level'
            })
        },

        attrs={
            'title': 'Multi-Instrument Merged Wind and Turbulence Profiles',
            'description': 'Quality-controlled wind and turbulence profiles from multiple instruments with statistical inter-instrument validation and physics-based filtering',
            'instruments': ', '.join([instrument_mapping[inst] for inst in all_possible_instruments]),
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'time_convention': 'Timestamps indicate start time of averaging period',
            'latitude': float(avg_lat) if avg_lat is not None else -9999.0,
            'longitude': float(avg_lon) if avg_lon is not None else -9999.0,
            'quality_control': 'Physics-based filtering (TKE>30 m²/s², TI>300%, multi-parameter thresholds: TKE>15 m²/s² + component std>10 m/s), geometric constraints (heights<100m removed for scanning lidar, min 3 beams for VAD), statistical inter-instrument validation with flagging',
            'vad_error_processing': 'VAD wind direction errors set to NaN for wind speeds <2.0 m/s, capped at 180°. VAD fit fails for: insufficient data (<3 beams), wind speed error >2 m/s, or condition number >1000',
        }
    )
    
    # Update filename to tsdat format
    first_time = pd.to_datetime(times[0], unit='s', utc=True)
    date_str = first_time.strftime('%Y%m%d')
    time_str = first_time.strftime('%H%M%S')
    
    # Extract directory and create new filename
    output_dir = os.path.dirname(output_path)
    new_filename = f"{site_prefix}.windprof.z01.c1.{date_str}.{time_str}.nc"
    new_output_path = os.path.join(output_dir, new_filename)
    
    # Save to NetCDF
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(new_output_path):
        try:
            os.remove(new_output_path)
            if verbose:
                print(f"Removed existing file: {new_filename}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not remove existing file: {e}")
    
    ds.to_netcdf(new_output_path)
    print(f"NetCDF file saved: {new_output_path}")
    
    if verbose:
        print(f"Time range: {first_time} to {pd.to_datetime(times[-1], unit='s', utc=True)}")
        print(f"Height range: {heights.min():.1f} to {heights.max():.1f} m")
        print(f"Instruments included: {len(all_possible_instruments)}")
    
    return new_output_path