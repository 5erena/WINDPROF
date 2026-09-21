"""Format-specific readers for the profiling lidar file conventions.

  - ``.rtd``: WindCube V2.1 native (Nantucket)
  - ``.sta``: WindCube V1 native (Block Island)
  - ``.csv``: WindCube V2-96 and ZephIR-300 export (Rhode Island)

``parse_profiling_lidar_file`` dispatches on extension. Cape Cod is separate
(``parse_caco_lidar_file``): its profiling lidar is z01, not z03.
"""

import numpy as np
import pandas as pd
import os
import re
from .config import get_instrument_coordinates

### z03 Auto-Detection and Parsing

def detect_profiling_lidar_file_type(filepath):
    """Return the format tag for a z03 file: 'sta', 'rtd', 'csv', or None."""
    if filepath.lower().endswith('.sta'):
        return 'sta'
    elif filepath.lower().endswith('.rtd'):
        return 'rtd'
    elif filepath.lower().endswith('.csv'):
        return 'csv'
    else:
        print(f"Unsupported file type for {filepath}. Expected .rtd, .sta or .csv")
        return None

def parse_profiling_lidar_file(filepath, verbose=False):
    """Auto-detect format and dispatch to the matching parser."""
    file_type = detect_profiling_lidar_file_type(filepath)

    if file_type == 'rtd':
        return parse_profiling_rtd_file(filepath, verbose)
    elif file_type == 'sta':
        return parse_profiling_sta_file(filepath, verbose)
    elif file_type == 'csv':
        return parse_profiling_csv_file(filepath, verbose)
    else:
        print(f"Unknown z03 file type: {file_type}")
        return None

def parse_profiling_rtd_file(filepath, verbose=False):
    """
    Parse a Leosphere RTD file (Nantucket-era z03). Header line positions are
    fixed; heights and the per-height column names are read from the header.
    """
    if not os.path.exists(filepath):
        print(f"Error: RTD file not found: {filepath}")
        return None
    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"Error: RTD file is empty: {filepath}")
            return None
    except OSError as e:
        print(f"Error accessing RTD file: {e}")
        return None

    try:
        # latin1 because the Leosphere headers contain an encoded degree symbol
        with open(filepath, 'r', encoding='latin1') as f:
            lines = f.readlines()

        heights = []
        latitude, longitude = None, None

        # Line 40 (index 39): "Altitudes (m)= 40 60 80 ..."
        if len(lines) > 39 and 'Altitudes (m)=' in lines[39]:
            heights = [float(h) for h in lines[39].split('=')[1].strip().split()]
        # Line 6 (index 5): "GPS Location=Lat:41.24N, Long:70.10W"
        if len(lines) > 5 and 'GPS Location=' in lines[5]:
            try:
                gps_part = lines[5].split('GPS Location=')[1].strip()
                lat_part, lon_part = gps_part.split(', ')
                latitude = float(lat_part.split('Lat:')[1].rstrip('N'))
                longitude = -float(lon_part.split('Long:')[1].rstrip('W'))
                if verbose:
                    print(f"Extracted GPS coordinates: {latitude}, {longitude}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse GPS coordinates: {e}, using fallback")

        if not heights:
            print("No heights found in header")
            return None

        # RTD lacks GPS in some header revisions, so fall back to the site config
        #  (RTD is only for Nantucket here).
        if latitude is None or longitude is None:
            fallback_coords = get_instrument_coordinates('nantucket', 'z03_lidar')
            if fallback_coords:
                latitude, longitude = fallback_coords[0], fallback_coords[1]

        if verbose:
            print(f"Found {len(heights)} heights: {heights}")

        def custom_parse_file_from_lines(lines):
            # First 42 lines are header; data starts at line 43.
            parsed_lines = []
            for line_num, line in enumerate(lines[42:], start=43):
                parts = line.strip().split()
                # Some rows have date and time in separate whitespace tokens
                # Re-merge them so the column count stays stable.
                if len(parts) > 1 and '/' in parts[0] and ':' in parts[1]:
                    parts = [f"{parts[0]} {parts[1]}"] + parts[2:]
                parsed_lines.append(parts)
            return pd.DataFrame(parsed_lines)

        try:
            data = custom_parse_file_from_lines(lines)
        except Exception as e:
            print(f"Failed to parse file: {e}")
            return None

        if verbose:
            print(f"Loaded data shape: {data.shape}")
            print(f"First few columns sample: {data.iloc[0, :10].tolist()}")

        # Each height contributes 8 columns in fixed order.
        column_names = ['timestamp', 'position', 'temperature', 'wiper']
        for h in range(len(heights)):
            column_names.extend([
                f'cnr_{h}', f'vr_{h}', f'disp_{h}',
                f'ws_{h}', f'wd_{h}', f'u_{h}',
                f'v_{h}', f'w_{h}'
            ])

        # Real data sometimes has trailing junk columns; clip rather than error out
        if len(data.columns) > len(column_names):
            data = data.iloc[:, :len(column_names)]

        data.columns = column_names[:len(data.columns)]

        # Try several timestamp formats: firmware revisions differ in separators.
        timestamp_formats = [
            '%Y/%m/%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]

        timestamp_converted = False
        for fmt in timestamp_formats:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'], format=fmt)
                timestamp_converted = True
                break
            except Exception as e:
                if verbose:
                    print(f"Timestamp conversion failed with {fmt}: {e}")

        if not timestamp_converted:
            print("Failed to convert timestamp")
            return None

        result = {
            'heights': heights,
            'time': data['timestamp'],
            'position': data['position'],
            'latitude': latitude,
            'longitude': longitude,
            'measurements': {},
            'file_type': 'rtd'
        }

        # Pivot to {height: {var: series}}, the shape every parser returns.
        for i, height in enumerate(heights):
            try:
                result['measurements'][height] = {
                    'cnr': pd.to_numeric(data[f'cnr_{i}'], errors='coerce'),
                    'vr': pd.to_numeric(data[f'vr_{i}'], errors='coerce'),
                    'disp': pd.to_numeric(data[f'disp_{i}'], errors='coerce'),
                    'ws': pd.to_numeric(data[f'ws_{i}'], errors='coerce'),
                    'wd': pd.to_numeric(data[f'wd_{i}'], errors='coerce'),
                    'u': pd.to_numeric(data[f'u_{i}'], errors='coerce'),
                    'v': pd.to_numeric(data[f'v_{i}'], errors='coerce'),
                    'w': pd.to_numeric(data[f'w_{i}'], errors='coerce')
                }
            except KeyError as ke:
                print(f"Missing column for height {height}: {ke}")
            except Exception as e:
                print(f"Error processing height {height}: {e}")

        if verbose:
            print(f"Processed data with {len(result['time'])} timestamps")
            print(f"Number of heights processed: {len(result['measurements'])}")

        return result

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def parse_profiling_sta_file(filepath, verbose=True):
    """
    Parse a Leosphere STA file (Block Island z03). STA carries pre-computed
    wind statistics and per-height standard deviations.

    The ``*_native`` variables are in the instrument's engineering frame: U
    south-positive, V west-positive, W down-positive. Reorientation to
    meteorological convention happens downstream.

    Returns
    -------
    dict or None
        Standard parser output with 'heights', 'time', 'measurements',
        'metadata' (with gps_coords), and 'file_type'. None on any failure.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        if verbose:
            print(f"Error reading file: {e}")
        return None

    altitudes = None
    gps_coords = None
    data_start = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('Altitudes(m)='):
            altitudes = [float(x) for x in line_stripped.split('=')[1].split()]
        elif line_stripped.startswith('GPS Localisation='):
            gps_coords = _parse_gps_coordinates(line_stripped.split('=')[1])
            if verbose and gps_coords:
                print(f"GPS coordinates found: {gps_coords[0]:.6f}, {gps_coords[1]:.6f}")
        elif 'Date' in line_stripped and 'WiperCount' in line_stripped:
            data_start = i + 1
            break

    if not altitudes or data_start is None:
        if verbose:
            print("Failed to find altitudes or data start marker")
        return None

    if not gps_coords:
        # STA is only used at Block Island here.
        fallback_coords = get_instrument_coordinates('block_island', 'z03_lidar')
        if fallback_coords:
            gps_coords = fallback_coords
        if verbose:
            print(f"Warning: GPS coordinates not found in file header, using fallback: {gps_coords}")

    times = []
    measurements = {alt: {'u_native': [], 'v_native': [], 'w_native': [], 'cnr': [], 'availability': [],
                         'std_u_native': [], 'std_v_native': [], 'std_w_native': [], 'vhm': [], 'std_vhm': []}
                   for alt in altitudes}

    for line in lines[data_start:]:
        parts = line.strip().split()
        # 20 = a conservative lower bound; real rows are 4 + 18*N_alt
        if len(parts) < 20:
            continue

        try:
            timestamp = pd.to_datetime(f"{parts[0]} {parts[1]}", format="%d/%m/%Y %H:%M:%S")
            times.append(timestamp)

            # Column structure: Date Time WiperCount Tm [18 columns per altitude]
            # Per-altitude block layout:
            #   0  Vhm                     1  dVh
            #   2  VhMax                   3  VhMin
            #   4  Azim                    5  um
            #   6  du                      7  vm
            #   8  dv                      9  wm
            #   10 dw                      11 CNRm
            #   12 dCNR                    13 CNRmax
            #   14 CNRmin                  15 spectral_broadening
            #   16 dspectral_broadening    17 Avail
            for i, alt in enumerate(altitudes):
                base_idx = 4 + i * 18  # skip Date, Time, WiperCount, Tm
                if base_idx + 17 >= len(parts):
                    break

                measurements[alt]['vhm'].append(_safe_float(parts[base_idx + 0]))
                measurements[alt]['std_vhm'].append(_safe_float(parts[base_idx + 1]))
                measurements[alt]['u_native'].append(_safe_float(parts[base_idx + 5]))
                measurements[alt]['std_u_native'].append(_safe_float(parts[base_idx + 6]))
                measurements[alt]['v_native'].append(_safe_float(parts[base_idx + 7]))
                measurements[alt]['std_v_native'].append(_safe_float(parts[base_idx + 8]))
                measurements[alt]['w_native'].append(_safe_float(parts[base_idx + 9]))
                measurements[alt]['std_w_native'].append(_safe_float(parts[base_idx + 10]))
                measurements[alt]['cnr'].append(_safe_float(parts[base_idx + 11]))
                measurements[alt]['availability'].append(_safe_float(parts[base_idx + 17]))

        except Exception as e:
            if verbose:
                print(f"Error parsing line: {e}")
            continue

    if verbose:
        print(f"Parsed {len(times)} timestamps, {len(altitudes)} altitudes")
        if gps_coords:
            print(f"GPS coordinates: Lat={gps_coords[0]:.6f}, Lon={gps_coords[1]:.6f}")

    return {
        'heights': altitudes,
        'time': times,
        'measurements': measurements,
        'metadata': {'gps_coords': gps_coords},
        'file_type': 'sta'
    }

def parse_profiling_csv_file(filepath, verbose=False):
    """
    Parse a Rhode Island z03 CSV. Columns hold pre-computed wind statistics,
    one column per height per variable.
    """
    if verbose:
        print(f"Parsing Z03 CSV file: {filepath}")

    try:
        # Row 0 is metadata, row 1 is the column header.
        df = pd.read_csv(filepath, skiprows=1)

        if verbose:
            print(f"Data shape: {df.shape}")
            print(f"First few columns: {list(df.columns[:10])}")

        timestamps = pd.to_datetime(df['Time and Date'])

        # GPS is repeated on every row; the first one is sufficient.
        gps_str = df['GPS'].iloc[0]  # e.g. "41.44215 -71.41789"
        try:
            lat, lon = map(float, gps_str.split())
        except (ValueError, AttributeError):
            lat, lon = None, None

        # Heights live in the metadata line that read_csv skipped.
        with open(filepath, 'r') as f:
            metadata_line = f.readline().strip()

        # Format: "... Measurement heights: 40m 60m 80m, ..."
        heights_text = metadata_line.split('Measurement heights: ')[1].split(',')[0]
        height_strings = heights_text.split()
        heights = []
        for h_str in height_strings:
            if 'm' in h_str:
                heights.append(float(h_str.replace('m', '')))

        if verbose:
            print(f"Found heights: {heights}")
            print(f"GPS coordinates: {lat}, {lon}")

        measurements = {}

        for height in heights:
            height_cols = {
                'wind_speed': f'Horizontal Wind Speed (m/s) at {int(height)}m',
                'wind_direction': f'Wind Direction (deg) at {int(height)}m',
                'w': f'Vertical Wind Speed (m/s) at {int(height)}m',
                'TI': f'TI at {int(height)}m',
                'std_dev': f'Horizontal Wind Speed Std. Dev. (m/s) at {int(height)}m',
                'packets': f'Packets in Average at {int(height)}m'
            }

            measurements[height] = {
                'time': timestamps,
                'wind_speed': df[height_cols['wind_speed']].values if height_cols['wind_speed'] in df.columns else np.full(len(df), np.nan),
                'wind_direction': df[height_cols['wind_direction']].values if height_cols['wind_direction'] in df.columns else np.full(len(df), np.nan),
                'w': df[height_cols['w']].values if height_cols['w'] in df.columns else np.full(len(df), np.nan),
                'TI': df[height_cols['TI']].values if height_cols['TI'] in df.columns else np.full(len(df), np.nan),
                'std_dev': df[height_cols['std_dev']].values if height_cols['std_dev'] in df.columns else np.full(len(df), np.nan),
                'packets': df[height_cols['packets']].values if height_cols['packets'] in df.columns else np.full(len(df), np.nan)
            }

        if verbose:
            print(f"Successfully parsed {len(timestamps)} time steps for {len(heights)} heights")

        return {
            'heights': sorted(heights),
            'time': timestamps,
            'measurements': measurements,
            'metadata': {
                'gps_coords': (lat, lon),
                'columns': list(df.columns),
                'shape': df.shape,
                'file_path': filepath,
                'original_df': df  # retained so the QC step can re-read raw values
            },
            'file_type': 'csv'
        }

    except Exception as e:
        if verbose:
            print(f"Error parsing Z03 CSV: {e}")
        return None

def _parse_gps_coordinates(gps_string):
    """
    Parse the STA-header GPS string into decimal lat/lon. The format is
    DDMM'SS.SS"N DDMM'SS.SS"W ALTm. Note that the degree symbol is
    missing from the source, so degrees and minutes are concatenated in
    the first integer group.
    """
    pattern = r"(\d+)'([\d.]+)\"([NS])\s+(\d+)'([\d.]+)\"([EW])"
    match = re.match(pattern, gps_string.strip())

    if not match:
        return None

    # "4110" is 41 deg 10 min: the last two characters are the minutes.
    lat_full = match.group(1)
    if len(lat_full) >= 3:
        lat_deg = float(lat_full[:-2])
        lat_min = float(lat_full[-2:])
    else:
        lat_deg = float(lat_full)
        lat_min = 0

    lat_sec = float(match.group(2))
    lat_dir = match.group(3)

    lon_full = match.group(4)
    if len(lon_full) >= 3:
        lon_deg = float(lon_full[:-2])
        lon_min = float(lon_full[-2:])
    else:
        lon_deg = float(lon_full)
        lon_min = 0

    lon_sec = float(match.group(5))
    lon_dir = match.group(6)

    latitude = lat_deg + lat_min/60 + lat_sec/3600
    longitude = lon_deg + lon_min/60 + lon_sec/3600

    if lat_dir == 'S': latitude = -latitude
    if lon_dir == 'W': longitude = -longitude

    return latitude, longitude

def _safe_float(value):
    """Float coercion that maps the string 'NaN' and any parse error to np.nan"""
    try:
        return float(value) if value != 'NaN' else np.nan
    except (ValueError, TypeError):
        return np.nan

### CACO Lidar Parsing

def clear_caco_cache():
    """Drop the in-process CACO file cache to free memory between runs"""
    global _caco_file_cache
    _caco_file_cache.clear()

# Cache CACO Excel parses by filepath. CACO ships hundreds of small
# files and pandas Excel reads dominate the runtime otherwise.
_caco_file_cache = {}

def parse_caco_lidar_file(filepath, use_cache=True):
    """
    Parse a CACO z01 Excel file (Windcube-v2-96 native export) into the shape
    every other lidar parser returns.

    Layout: heights on row 39, column headers on row 41, data from row 42.
    Per-height columns carry the integer altitude, e.g. "100m Wind Speed (m/s)".

    ``Wind Speed Dispersion`` is the vendor's within-window sigma(WS) and the
    only turbulence quantity read here. ``Z-wind Dispersion`` is not read: it is
    the vertical beam's radial-velocity standard deviation, not the VAD-fit
    sigma_w the rest of the pipeline reports, so mixing the two would break the
    TKE definition.
    """

    if use_cache and filepath in _caco_file_cache:
        return _caco_file_cache[filepath]

    df = pd.read_excel(filepath, sheet_name='Sheet1', header=41)
    df = df.rename(columns={df.columns[0]: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    header_df = pd.read_excel(filepath, sheet_name='Sheet1', header=None, nrows=40)
    altitudes_row = header_df.iloc[39]
    heights = []
    for val in altitudes_row[1:]:  # skip the timestamp column
        if pd.notna(val) and str(val).replace('.', '').isdigit():
            heights.append(int(float(val)))

    # CACO Excel headers carry no GPS; these are the surveyed site coordinates.
    latitude, longitude = 42.0324, -70.0535

    measurements = {}

    for height in heights:
        wind_speed_col = f'{height}m Wind Speed (m/s)'
        wind_dir_col = f'{height}m Wind Direction (°)'
        vertical_wind_col = f'{height}m Z-wind (m/s)'
        ws_dispersion_col = f'{height}m Wind Speed Dispersion (m/s)'
        cnr_col = f'{height}m CNR (dB)'
        availability_col = f'{height}m Data Availability (%)'

        measurements[height] = {
            'wind_speed': np.full(len(df), np.nan),
            'wind_direction': np.full(len(df), np.nan),
            'w': np.full(len(df), np.nan),
            'ws_dispersion': np.full(len(df), np.nan),
            'cnr': np.full(len(df), np.nan),
            'availability': np.full(len(df), np.nan)
        }

        if wind_speed_col in df.columns:
            measurements[height]['wind_speed'] = pd.to_numeric(df[wind_speed_col], errors='coerce').values
        if wind_dir_col in df.columns:
            measurements[height]['wind_direction'] = pd.to_numeric(df[wind_dir_col], errors='coerce').values
        if vertical_wind_col in df.columns:
            measurements[height]['w'] = pd.to_numeric(df[vertical_wind_col], errors='coerce').values
        if ws_dispersion_col in df.columns:
            measurements[height]['ws_dispersion'] = pd.to_numeric(df[ws_dispersion_col], errors='coerce').values
        if cnr_col in df.columns:
            measurements[height]['cnr'] = pd.to_numeric(df[cnr_col], errors='coerce').values
        if availability_col in df.columns:
            measurements[height]['availability'] = pd.to_numeric(df[availability_col], errors='coerce').values

    result = {
        'heights': sorted(heights),
        'time': df['timestamp'],
        'measurements': measurements,
        'metadata': {
            'gps_coords': (latitude, longitude),
            'site': 'CACO',
            'instrument': 'Windcube-v2-96'
        },
        'file_type': 'caco_excel',
        'latitude': latitude,
        'longitude': longitude
    }

    if use_cache:
        _caco_file_cache[filepath] = result

    return result
