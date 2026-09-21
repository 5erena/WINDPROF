"""Sonic anemometer and surface meteorological station processing

Sonics are the only direct (non-remote-sensing) wind and turbulence
measurements in the merged product, and the only source below the minimum
range of the lidars and radar (typically 5-10 m AGL). Reynolds decomposition
of the high-rate (10-20 Hz) record gives 10-minute mean (u, v, w) and
component variances.

Entry points by file format encountered in WFIP3:

  - ``process_sonic_anemometer`` / ``process_anemometer_time_series``:
    Nantucket and Block Island NetCDF format, hourly files.
  - ``process_sonic_c1_csv`` / ``process_sonic_c1_time_series``:
    pre-aggregated C1 CSV format used at some sites.
  - ``process_rhod_sonic_csv`` / ``process_rhod_sonic_time_series``:
    Narragansett-specific CSV layout.
  - ``extract_wind_from_surface_met``: low-rate surface met wind extraction
    for sites without true sonic anemometers (Block Island only).

Tower heights and per-instrument anemometer corrections are config-driven via
``LOCATION_CONFIG[site]['anemometer_heights']`` and ``['anemometer_corrections']``.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from .config import (get_instrument_coordinates, get_anemometer_correction,
                     get_w_sign_correction, get_ground_elevation_from_config,
                     LOCATION_CONFIG, round_profile_values, TIME_WINDOW_MINUTES)
from .wind_analysis import calculate_turbulence_metrics, apply_physics_based_qc

### Anemometer Processing

def process_sonic_anemometer(filename, start_time, location, time_window=TIME_WINDOW_MINUTES,
                           instrument='z02', ground_elevation=None, verbose=False,
                           open_dataset=None):
    """
    Compute one wind + turbulence profile from a single 10-minute window of
    high-rate sonic anemometer data (NetCDF, hourly files).

    Pass ``open_dataset`` when iterating multiple windows in the same hour
    so the file is only opened once. See ``process_anemometer_time_series``.

    Parameters
    ----------
    filename : str
        Path to anemometer NC file (only used to derive the directory when
        ``open_dataset`` is None).
    start_time : datetime
        Start of the averaging window.
    location : str
        Site identifier ('nantucket', 'block island', etc.).
    time_window : int
        Window length in minutes.
    instrument : str
        Tower-instrument identifier (z02, z03, z04).
    ground_elevation : float, optional
        Looked up from config when None.
    open_dataset : xarray.Dataset, optional
        Pre-opened NetCDF dataset (for batch processing efficiency).

    Returns
    -------
    dict or None
        Combined wind + turbulence profile, or None on any failure.
    """

    def create_valid_mask(*arrays):
        combined = np.column_stack(arrays)
        return ~np.isnan(combined).any(axis=1)

    def apply_coordinate_correction(u_series, v_series, correction_deg):
        # Rotates (u, v) from the instrument frame into true north.
        if correction_deg == 0:
            return u_series, v_series

        theta = np.radians(correction_deg)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        u_corrected = u_series * cos_theta - v_series * sin_theta
        v_corrected = u_series * sin_theta + v_series * cos_theta

        return u_corrected, v_corrected

    try:
        if open_dataset is not None:
            ds = open_dataset
            should_close = False
        else:
            directory = os.path.dirname(filename)
            date_str = start_time.strftime('%Y%m%d')
            hour_str = start_time.strftime('%H0000')

            if location in ['nantucket', 'nant']:
                hourly_filename = f"{directory}/nant.met.{instrument}.a0.{date_str}.{hour_str}.nc"
            elif location in ['block_island', 'bloc']:
                hourly_filename = f"{directory}/bloc.met.{instrument}.a0.{date_str}.{hour_str}.nc"
            else:
                raise NotImplementedError(
                    f"process_sonic_anemometer: hourly NetCDF naming convention "
                    f"not implemented for location='{location}'. Currently "
                    f"supported: nantucket, block_island. For Cape Cod and "
                    f"Narragansett sonics, use process_sonic_c1_csv or "
                    f"process_rhod_sonic_csv instead. To extend, add a branch "
                    f"in anemometers.py:process_sonic_anemometer with the "
                    f"appropriate file-naming pattern.")

            if not os.path.exists(hourly_filename):
                if verbose:
                    print(f"Error: Hourly anemometer file not found: {hourly_filename}")
                return None

            ds = xr.open_dataset(hourly_filename)
            should_close = True

        end_time = start_time + pd.Timedelta(minutes=time_window)
        window_data = ds.sel(time=slice(start_time, end_time))

        # 100 samples ≈ 5 s at 20 Hz; below this the variance estimate is
        # too noisy to be useful for turbulence metrics.
        if len(window_data.time) < 100:
            if verbose:
                print(f"Insufficient anemometer samples: {len(window_data.time)} (need at least 100)")
            if should_close:
                ds.close()
            return None

        u_data = window_data.u_wind.values
        v_data = window_data.v_wind.values
        w_data = window_data.w_wind.values if 'w_wind' in window_data else None

        if should_close:
            ds.close()

        # Joint validity mask so the Reynolds decomposition operates on a
        # consistent sample set across components.
        arrays_to_check = [u_data, v_data]
        if w_data is not None:
            arrays_to_check.append(w_data)

        valid_mask = create_valid_mask(*arrays_to_check)

        u_series = u_data[valid_mask]
        v_series = v_data[valid_mask]
        w_series = w_data[valid_mask] if w_data is not None else None

        if len(u_series) < 100:
            if verbose:
                print(f"Insufficient valid anemometer samples after QC: {len(u_series)}")
            return None

        coords = get_instrument_coordinates(location, f'{instrument}_met')
        if not coords:
            if verbose:
                print(f"Warning: No coordinates found for {instrument} at {location}")
            coords = [np.nan, np.nan]

        ground_elevation = get_ground_elevation_from_config(location, f'{instrument}_met')

        measurement_height_agl = (
            LOCATION_CONFIG.get(location, {})
            .get('anemometer_heights', {})
            .get(instrument, 10)
        )

        results = {
            'time': start_time,
            'instrument_code': f'anemometer_{instrument}',
            'wind_profiles': {},
            'turbulence_profiles': {},
            'ground_elevation': ground_elevation,
            'latitude': coords[0],
            'longitude': coords[1]
        }

        correction_deg = get_anemometer_correction(location, instrument)
        u_series, v_series = apply_coordinate_correction(u_series, v_series, correction_deg)
        ws_series = np.sqrt(u_series**2 + v_series**2)

        # Some instrument frames define +w as down rather than up; flip
        # here so all sites share an "up is positive" convention.
        w_correction = get_w_sign_correction('sonic', location)
        w_series *= w_correction

        mean_u = np.mean(u_series)
        mean_v = np.mean(v_series)
        mean_ws = np.mean(ws_series)
        mean_w = np.mean(w_series) if w_series is not None else np.nan

        # Meteorological convention: 0° = wind FROM the north, increasing
        # clockwise. The "270 - atan2" form converts (u, v) east-north
        # components to that convention in one step.
        mean_wd = (270 - np.rad2deg(np.arctan2(mean_v, mean_u))) % 360

        results['wind_profiles'][measurement_height_agl] = round_profile_values({
            'ws': mean_ws,
            'wd': mean_wd,
            'w': mean_w,
        })

        turbulence_metrics = calculate_turbulence_metrics(
            u_series, v_series, w_series, ws_series, 'empirical'
        )

        if turbulence_metrics:
            results['turbulence_profiles'][measurement_height_agl] = round_profile_values(turbulence_metrics)

        if verbose:
            print(f"Successfully processed anemometer: {len(u_series)} samples at {measurement_height_agl}m AGL")
            print(f"Wind: WS={mean_ws:.2f} m/s, WD={mean_wd:.1f}°")
            if turbulence_metrics:
                print(f"Turbulence: TI={turbulence_metrics['ti']:.3f}, TKE={turbulence_metrics['tke']:.3f}")

        return results

    except Exception as e:
        if verbose:
            print(f"Error processing anemometer data: {e}")
        return None

def process_sonic_c1_csv(filename, start_time, location, time_window=TIME_WINDOW_MINUTES,
                        instrument='z01', ground_elevation=None, verbose=False):
    """
    Read one 10-minute averaged record from a .c1 CSV (Cape Cod sonic
    ingest). CSV timestamps mark the END of the averaging period, so we
    shift them back by 10 minutes before matching against ``start_time``.
    """
    try:
        df = pd.read_csv(filename)

        # Row 0 holds variable names, row 1 holds units.
        variable_names = df.iloc[0].values
        df.columns = variable_names
        df = df.iloc[2:].reset_index(drop=True)

        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['period_start'] = df['datetime'] - pd.Timedelta(minutes=10)

        target_time = start_time.replace(second=0, microsecond=0)
        time_diffs = abs(df['period_start'] - target_time)
        closest_idx = time_diffs.idxmin()

        # Tolerance is half the window length.
        if time_diffs.iloc[closest_idx] > pd.Timedelta(minutes=5):
            if verbose:
                print(f"No data within 5 minutes of {start_time}")
            return None

        row = df.iloc[closest_idx]

        def safe_numeric(val):
            # -9999 is the EddyPro sentinel for "missing/invalid".
            try:
                num_val = float(val)
                return np.nan if num_val == -9999 else num_val
            except (ValueError, TypeError):
                return np.nan

        wind_speed = safe_numeric(row['wind_speed'])
        wind_dir = safe_numeric(row['wind_dir'])
        w_vertical = safe_numeric(row['w_unrot'])
        tke = safe_numeric(row['TKE'])
        u_var = safe_numeric(row.get('u_var'))
        v_var = safe_numeric(row.get('v_var'))
        w_var = safe_numeric(row.get('w_var'))

        std_u = np.sqrt(u_var) if not np.isnan(u_var) and u_var >= 0 else np.nan
        std_v = np.sqrt(v_var) if not np.isnan(v_var) and v_var >= 0 else np.nan
        std_w = np.sqrt(w_var) if not np.isnan(w_var) and w_var >= 0 else np.nan

        if not np.isnan(wind_speed) and wind_speed > 0 and not np.isnan(std_u):
            turbulence_intensity = std_u / wind_speed
        else:
            turbulence_intensity = np.nan

        if np.isnan(wind_speed) or np.isnan(wind_dir):
            if verbose:
                print(f"Invalid wind data at {start_time}")
            return None

        coords = get_instrument_coordinates(location, f'{instrument}_met')
        if not coords:
            if verbose:
                print(f"Warning: No coordinates found for {instrument} at {location}")
            coords = [np.nan, np.nan]

        ground_elevation = get_ground_elevation_from_config(location, f'{instrument}_met')

        if location in LOCATION_CONFIG and 'anemometer_heights' in LOCATION_CONFIG[location]:
            measurement_height_agl = LOCATION_CONFIG[location]['anemometer_heights'].get(instrument)
            if measurement_height_agl is None:
                if verbose:
                    print(f"No height configured for {instrument} at {location} - skipping")
                return None
        else:
            if verbose:
                print(f"No anemometer height configuration found for location {location} - skipping")
            return None

        correction_deg = get_anemometer_correction(location, instrument)
        if correction_deg != 0:
            wind_dir = (wind_dir + correction_deg) % 360

        results = {
            'time': start_time,
            'instrument_code': f'anemometer_{instrument}',
            'wind_profiles': {
                measurement_height_agl: round_profile_values({
                    'ws': wind_speed,
                    'wd': wind_dir,
                    'w': w_vertical if not np.isnan(w_vertical) else np.nan
                })
            },
            'turbulence_profiles': {},
            'ground_elevation': ground_elevation,
            'latitude': coords[0],
            'longitude': coords[1]
        }

        turb_data = {}
        if not np.isnan(turbulence_intensity):
            turb_data['ti'] = turbulence_intensity
        if not np.isnan(tke):
            turb_data['tke'] = tke
        if not np.isnan(std_u):
            turb_data['std_u'] = std_u
        if not np.isnan(std_v):
            turb_data['std_v'] = std_v
        if not np.isnan(std_w):
            turb_data['std_w'] = std_w

        if turb_data:
            if not apply_physics_based_qc(turb_data):
                results['turbulence_profiles'][measurement_height_agl] = round_profile_values(turb_data)

        if verbose:
            print(f"C1 {instrument}: WS={wind_speed:.2f} m/s, WD={wind_dir:.1f}°, TKE={tke:.3f}")

        return results

    except Exception as e:
        if verbose:
            print(f"Error processing C1 CSV data: {e}")
            import traceback
            traceback.print_exc()
        return None

def process_rhod_sonic_csv(filename, start_time, location, time_window=TIME_WINDOW_MINUTES,
                          instrument='z01', ground_elevation=None, verbose=False, **kwargs):
    """
    Read one interval from a Rhode Island EddyPro daily CSV. Same
    end-of-period timestamp convention as the .c1 files, but variable
    names follow EddyPro's full-output schema.
    """

    def safe_numeric(val):
        try:
            num_val = float(val)
            return np.nan if num_val == -9999 else num_val
        except (ValueError, TypeError):
            return np.nan

    try:
        df = pd.read_csv(filename)

        if len(df) < 3:
            if verbose:
                print(f"Insufficient data rows in {filename}")
            return None

        variable_names = df.iloc[0].values
        df.columns = variable_names
        df = df.iloc[2:].reset_index(drop=True)

        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['period_start'] = df['datetime'] - pd.Timedelta(minutes=10)

        target_time = start_time.replace(second=0, microsecond=0)
        time_diffs = abs(df['period_start'] - target_time)

        if len(time_diffs) == 0:
            if verbose:
                print(f"No data available in file")
            return None

        closest_idx = time_diffs.idxmin()

        if time_diffs.iloc[closest_idx] > pd.Timedelta(minutes=5):
            if verbose:
                print(f"No data within 5 minutes of {start_time}")
            return None

        row = df.iloc[closest_idx]

        # EddyPro flags pathological intervals by writing a sentinel
        # filename string instead of a real path.
        if str(row.get('filename')).startswith('not_enough_data'):
            if verbose:
                print(f"Not enough data flag at {start_time}")
            return None

        wind_speed = safe_numeric(row.get('wind_speed'))
        wind_dir = safe_numeric(row.get('wind_dir'))

        tke = safe_numeric(row.get('TKE'))
        u_var = safe_numeric(row.get('u_var'))
        v_var = safe_numeric(row.get('v_var'))
        w_var = safe_numeric(row.get('w_var'))

        std_u = np.sqrt(u_var) if not np.isnan(u_var) and u_var >= 0 else np.nan
        std_v = np.sqrt(v_var) if not np.isnan(v_var) and v_var >= 0 else np.nan
        std_w = np.sqrt(w_var) if not np.isnan(w_var) and w_var >= 0 else np.nan

        if not np.isnan(wind_speed) and wind_speed > 0 and not np.isnan(std_u):
            turbulence_intensity = std_u / wind_speed
        else:
            turbulence_intensity = np.nan

        if np.isnan(wind_speed) or np.isnan(wind_dir):
            if verbose:
                print(f"Invalid wind data at {start_time}: WS={wind_speed}, WD={wind_dir}")
            return None

        coords = get_instrument_coordinates(location, f'{instrument}_met')
        if not coords:
            coords = [np.nan, np.nan]

        ground_elevation = get_ground_elevation_from_config(location, f'{instrument}_met')

        measurement_height_agl = 10.0
        if location in LOCATION_CONFIG and 'anemometer_heights' in LOCATION_CONFIG[location]:
            height = LOCATION_CONFIG[location]['anemometer_heights'].get(instrument)
            if height is not None:
                measurement_height_agl = height

        correction_deg = get_anemometer_correction(location, instrument)
        if correction_deg != 0:
            wind_dir = (wind_dir + correction_deg) % 360

        results = {
            'time': start_time,
            'instrument_code': f'met_{instrument}',
            'wind_profiles': {
                measurement_height_agl: round_profile_values({
                    'ws': wind_speed,
                    'wd': wind_dir
                    # EddyPro means do not include rotated vertical velocity
                })
            },
            'turbulence_profiles': {},
            'ground_elevation': ground_elevation,
            'latitude': coords[0],
            'longitude': coords[1]
        }

        turb_data = {}
        if not np.isnan(turbulence_intensity):
            turb_data['ti'] = turbulence_intensity
        if not np.isnan(tke):
            turb_data['tke'] = tke
        if not np.isnan(std_u):
            turb_data['std_u'] = std_u
        if not np.isnan(std_v):
            turb_data['std_v'] = std_v
        if not np.isnan(std_w):
            turb_data['std_w'] = std_w

        if turb_data:
            if not apply_physics_based_qc(turb_data):
                results['turbulence_profiles'][measurement_height_agl] = round_profile_values(turb_data)

        if verbose:
            print(f"RHOD {instrument}: WS={wind_speed:.2f} m/s, WD={wind_dir:.1f}°, TKE={tke:.3f}")

        return results

    except Exception as e:
        if verbose:
            print(f"Error processing Rhode Island CSV data: {e}")
            import traceback
            traceback.print_exc()
        return None

def extract_wind_from_surface_met(surface_met_data, location='block_island'):
    """
    Lift surface-met wind records into the standard profile dict format.
    Block Island only: at the other sites the surface met record is not used
    as a profile input.
    """

    if location not in ['block_island', 'bloc']:
        return []

    coords = get_instrument_coordinates(location, 'surf_met')
    if coords:
        lat, lon = coords
    else:
        lat, lon = 41.1667, -71.58  # site default

    ground_elevation_asl = get_ground_elevation_from_config(location, 'surf_met')

    wind_results = []
    for unix_time, data in surface_met_data.items():
        if 'wind_profile' in data:
            wind_result = {
                'time': pd.to_datetime(unix_time, unit='s'),
                'instrument_code': 'met_z01',
                'wind_profiles': dict(data['wind_profile']),
                'turbulence_profiles': {},
                'ground_elevation': ground_elevation_asl,
                'latitude': lat,
                'longitude': lon
            }
            wind_results.append(wind_result)

    return wind_results

### Time-Series Handling

def process_anemometer_time_series(filename, start_time, end_time, location,
                                  time_window=TIME_WINDOW_MINUTES, verbose=False, **kwargs):
    """
    Process a date range of sonic anemometer data into 10-minute profiles.

    Used for the Nantucket and Block Island sonic file convention (hourly
    NetCDF files at native 10-20 Hz sampling). Windows are grouped by hour
    so each hourly file is opened only once; ``xarray.open_dataset``
    otherwise dominates the runtime.

    Parameters
    ----------
    filename : str
        Path to one of the hourly NetCDF files in the target directory;
        used only to derive the directory path.
    start_time, end_time : datetime
        Inclusive/exclusive time range to process.
    location : str
        Site key as defined in ``config.LOCATION_CONFIG``.
    time_window : int
        Averaging window in minutes (default 10).
    **kwargs
        Forwarded to ``process_sonic_anemometer`` per window (e.g.,
        ``instrument='z02'``, ``ground_elevation``).

    Returns
    -------
    list of dict
        One profile dict per successfully processed window.
    """
    # Snap to the start of the containing window, e.g. 12:03 -> 12:00.
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    hourly_groups = {}

    while current_time < end_time:
        hour_key = current_time.replace(minute=0, second=0, microsecond=0)
        if hour_key not in hourly_groups:
            hourly_groups[hour_key] = []
        hourly_groups[hour_key].append(current_time)
        current_time += pd.Timedelta(minutes=time_window)

    if verbose:
        total_intervals = sum(len(intervals) for intervals in hourly_groups.values())
        print(f"Processing anemometer combined data: {total_intervals} time intervals ({len(hourly_groups)} hourly files)")

    results = []

    for hour_start, intervals in hourly_groups.items():
        try:
            directory = os.path.dirname(filename)
            date_str = hour_start.strftime('%Y%m%d')
            hour_str = hour_start.strftime('%H0000')
            instrument = kwargs.get('instrument', 'z02')

            if location in ['nantucket', 'nant']:
                hourly_filename = f"{directory}/nant.met.{instrument}.a0.{date_str}.{hour_str}.nc"
            elif location in ['block_island', 'bloc']:
                hourly_filename = f"{directory}/bloc.met.{instrument}.a0.{date_str}.{hour_str}.nc"
            else:
                continue

            if not os.path.exists(hourly_filename):
                if verbose:
                    print(f"Hourly file not found: {hourly_filename}")
                continue

            ds = xr.open_dataset(hourly_filename)

            for interval_time in intervals:
                try:
                    interval_result = process_sonic_anemometer(
                        filename, interval_time, location, time_window,
                        open_dataset=ds, verbose=verbose, **kwargs
                    )

                    if interval_result and (interval_result.get('wind_profiles') or interval_result.get('turbulence_profiles')):
                        results.append(interval_result)

                except Exception as e:
                    if verbose:
                        print(f"Error processing interval {interval_time}: {e}")
                    continue

            ds.close()

        except Exception as e:
            if verbose:
                print(f"Error processing hour {hour_start}: {e}")
            continue

    return results

def process_sonic_c1_time_series(filename, start_time, end_time, location,
                                time_window=TIME_WINDOW_MINUTES, instrument='z01', verbose=False, **kwargs):
    """Step through the time range, calling ``process_sonic_c1_csv`` per window.
    The CSV is one file per day so no per-hour grouping is needed."""
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    results = []

    while current_time < end_time:
        result = process_sonic_c1_csv(
            filename, current_time, location, time_window,
            instrument=instrument, verbose=verbose, **kwargs
        )

        if result and (result.get('wind_profiles') or result.get('turbulence_profiles')):
            results.append(result)

        current_time += pd.Timedelta(minutes=time_window)

    if verbose:
        print(f"C1 {instrument}: Processed {len(results)} intervals")

    return results

def process_rhod_sonic_time_series(filename, start_time, end_time, location,
                                  time_window=TIME_WINDOW_MINUTES, instrument='z01', verbose=False, **kwargs):
    """Step through the time range, calling ``process_rhod_sonic_csv`` per window."""
    current_time = start_time.replace(minute=(start_time.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES, second=0, microsecond=0)
    results = []

    while current_time < end_time:
        result = process_rhod_sonic_csv(
            filename, current_time, location, time_window,
            instrument=instrument, verbose=verbose, **kwargs
        )

        if result and (result.get('wind_profiles') or result.get('turbulence_profiles')):
            results.append(result)

        current_time += pd.Timedelta(minutes=time_window)

    if verbose:
        print(f"RHOD {instrument}: Processed {len(results)} intervals")

    return results
