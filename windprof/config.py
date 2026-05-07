"""Site, instrument, and processing configuration for WINDPROF best-estimate pipeline

Single source of truth for the parameters a user adapting WINDPROF for a
new campaign would need to edit:

  - ``LOCATION_CONFIG``: per-site instrument coordinates, ground
    elevations, azimuth corrections (true-north reference), vertical
    velocity sign corrections, anemometer corrections and tower heights.
  - ``LOCATION_ALIASES`` / ``SITE_CODES``: canonical site naming and
    short codes used in output filenames.
  - ``MALFUNCTION_PERIODS``: time ranges to exclude per instrument.
  - Per-instrument QC variable names and lookup helpers
    (``get_wind_correction``, ``get_w_sign_correction``, etc.).

Paths default to ``./data`` and ``./results`` relative to the current
working directory; override with the ``WINDPROF_DATA_PATH`` and
``WINDPROF_RESULTS_PATH`` environment variables.

All site-specific knowledge is intended to live in this file. Processing
modules read from ``config.py`` rather than embedding site assumptions,
so adapting WINDPROF for a similar campaign should mostly be a matter of
adding a ``LOCATION_CONFIG`` entry, plus any format-specific reader
hooks where the new campaign's instruments differ in file format from
those encountered in WFIP3 (see ``lidar_parsers.py``,
``anemometers.py`` and ``radars.py``).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

### Processing Parameters
TIME_WINDOW_MINUTES = 10  # minutes; can modify to adjust output time resolution

### Terrain and Coordinates

#### Location Configuration
LOCATION_CONFIG = {
    'nantucket': {
        # Average coords across active instruments is 41.243195125, -70.105867125. 
        # z03_met (10 m north-facing sonic) and z04_met (same) are retained here 
        # as documentation but NOT used in processing - the tower creates >20% 
        # wind-speed error under the dominant SW flow, so only the SW-facing z02 
        # 5 m sonic is kept. z03_met is also excluded due to sparse data coverage.
        'coordinates': {
            'z01': [41.24255, -70.107003],
            'z02': [41.24255, -70.107003],
            'z03': [41.242463, -70.107117],
            'z02_met': [41.242566, -70.107401],
            'z03_met': [41.242566, -70.107401],  # unused; see block comment above
            'z04_met': [41.242566, -70.107401],  # unused; see block comment above
            'radar': [41.245000, -70.098611],
            'surf_met': [41.2453, -70.105]
        },
        # Elevations above sea level (m), from USGS EPQS
        # https://apps.nationalmap.gov/epqs/
        'elevations': {
            'z01': 5.099825382,
            'z02': 5.099825382,
            'z03': 4.340077877,
            'z02_met': 4.310080528,
            'z03_met': 4.310080528,
            'z04_met': 4.310080528,
            'radar': 1.530035973,
            'surf_met': 5.840040207
        },
        'wind_corrections': {
            # z01: -80.8° from homepoint calibration (Newsom 2024, comms tower bearing)
            # + -16.34° empirical correction (z03/radar coincident bias, campaign-mean).
            'z01': -97.14,            
            'z02': -4,  # empirical match to z03 WindCube. Source: 3/28/25 WINDPROF Meeting.
            'z03': 0,
            'radar': 0
        },
        'anemometer_corrections': {
            # Physical orientation corrections were already applied upstream
            # in the ingest parsers.
            'z02': 0,
            'z03': 0,
            'z04': 0
        },
        'anemometer_heights': { # Measurement heights AGL (m)
            'z02': 5,
            'z03': 10,
            'z04': 10,
        },
        'w_sign_corrections': {
            'z01': 1,
            'z02': 1,
            'z03': -1,
            'radar': -1,
            'sonic': -1
        }
    },
    'block_island': {
        # Average coords across active instruments is 41.16670025, -71.5803045. 
        'coordinates': { 
            'z01': [41.16663, -71.58033],
            'z03': [41.166835, -71.580446],
            'radar': [41.166636, -71.580442],
            'surf_met': [41.1667, -71.58]
        },
        'elevations': { # ASL
            'z01': 35.176486969,
            'z03': 34.137886047,
            'radar': 35.152641296,
            'surf_met': 34.606922150
        },
        'wind_corrections': {
            'z01': 180, 
            'z02': 0,
            'z03': 0,
            'radar': 0
        },
        'anemometer_corrections': {
            'z01': 0
        },
        'anemometer_heights': { # Measurement heights AGL (m)
            'z01': 10,         # sonic anemometer on met tower
            'surf_met': 10.0,  # same instrument, alternate key used by surface-met path
        },
        'w_sign_corrections': {
            'z01': 1,
            'z02': 1,
            'z03': 1,
            'radar': -1,
            'sonic': 1 # all anemometers
        }
    },
    'rhode_island': {
        # Average coords across active instruments is 41.44811325, -71.432165.
        'coordinates': {
            'z01': [41.448043, -71.43225],
            'z03': [41.44809, -71.43233],  # from GPS in data file
            'z01_met': [41.44816, -71.43204],
            'surf_met': [41.44816, -71.43204]
        },
        'elevations': { # ASL
            'z01': 3.383681536,
            'z03': 3.371003866,
            'radar': 3.105409145,
            'z01_met': 3.579585552,
            'surf_met': 3.579585552
        },        
        'wind_corrections': {
            'z01': 180,     
            'z02': 0.0,
            # z03: empirically derived from z03-vs-radar bias-minimization
            # at RHOD (the ZephIR-300 ships with magnetic-declination
            # correction nominally applied at deployment, but the residual
            # may support a small additional rotation correction).
            'z03': 15,
            'radar': 0.0
        },
        'anemometer_corrections': { 
            'z01': 0 
        },
        'anemometer_heights': { # Measurement heights AGL (m)
            'z01': 4
    },
        'w_sign_corrections': { 
            'z01': 1,
            'z02': 1,
            'z03': 1,  
            'sonic': 1, # all anemometers
            'radar': 1
        }
    },
    'cape_cod': {
        # Average coords across active instruments is 42.031888, -70.052386. 
        'coordinates': { 
            'z01': [42.0324, -70.0535],
            'z02': [42.03242, -70.05305], 
            'z01_met': [42.03231, -70.05319],
            'z02_met': [42.03231, -70.05319],
            'surf_met': [42.03, -70.049]
        },
        'elevations': { # ASL
            'z01': 46.786033630,
            'z02': 47.064838409,
            'z01_met': 46.445487976,
            'z02_met': 46.445487976,            
            'surf_met': 46.193538666
        },        
        'wind_corrections': {
            'z01': 0.0,
            'z02': 180,
            'radar': 0.0
        },
        'anemometer_corrections': { 
            'z01': 0,
            'z02': 0
        },
        'anemometer_heights': { # Measurement heights AGL (m)
            'z01': 4,
            'z02': 10
    },
        'w_sign_corrections': { 
            'z01': -1,
            'z02': 1,
            'sonic': 1, # all anemometers
        }
    }
}

### Location Aliases and Normalization
LOCATION_ALIASES = {
    'nant': 'nantucket',
    'bloc': 'block_island',
    'bi': 'block_island',
    'blockisland': 'block_island',
    'block island': 'block_island',
    'rhod': 'rhode_island',
    'rhode island': 'rhode_island',
    'caco': 'cape_cod',
    'cape cod': 'cape_cod',
}

SITE_CODES = {
    'nantucket': 'nant',
    'block_island': 'bloc',
    'rhode_island': 'rhod',
    'cape_cod': 'caco',
}

def normalize_location(location):
    """Resolve location aliases to canonical name"""
    loc = location.lower().strip()
    return LOCATION_ALIASES.get(loc, loc)

### Base Paths
# Override via environment variables so the pipeline is portable across machines 
# without editing config.py. Defaults point into the current working directory.

DATA_BASE_PATH = os.environ.get('WINDPROF_DATA_PATH', str(Path.cwd() / 'data')) + '/'
RESULTS_BASE_PATH = os.environ.get('WINDPROF_RESULTS_PATH', str(Path.cwd() / 'results')) + '/'

# Available daily files (z03 profiling lidars, sonics, surface met) are read
# directly from the campaign archive tree to avoid duplicating storage. On the
# WFIP3 server this is /data; override via WINDPROF_ARCHIVE_PATH for other deployments. 
# Local files had to be manually downloaded or merged to produce one daily file.

ARCHIVE_BASE_PATH = os.environ.get('WINDPROF_ARCHIVE_PATH', '/data')

### Radar Malfunction Periods (excluded from merging)
MALFUNCTION_PERIODS = {
    'nantucket': [
        ('2025-05-10 00:00:00', '2025-05-16 16:00:00')
    ],
    'rhode_island': [
        ('2025-02-01 00:00:00', '2025-08-20 23:59:59')
    ],
}

### Instrument-to-Datastream Mappings (per site)
SITE_INSTRUMENT_MAPPINGS = {
    'nantucket': {
        'prefix': 'nant',
        'instruments': {
            'lidar_z01': 'nant.lidar.z01.a0',
            'lidar_z02': 'nant.lidar.z02.a0',
            'lidar_z03': 'nant.lidar.z03.00',
            'met_z02': 'nant.met.z02.a0',
            'surface_met': 'nant.met.z05.c1',
            'radar': 'NOAA PSL radar'
        }
    },
    'block_island': {
        'prefix': 'bloc',
        'instruments': {
            'lidar_z01': 'bloc.lidar.z01.a0',
            'lidar_z03': 'bloc.lidar.z03.00',
            'surface_met': 'bloc.met.z01.c1',
            'met_z01': 'bloc.met.z01.c1',
            'radar': 'NOAA PSL radar'
        }
    },
    'rhode_island': {
        'prefix': 'rhod',
        'instruments': {
            'lidar_z01': 'rhod.lidar.z01.a0',
            'lidar_z03': 'rhod.lidar.z03.00',
            'met_z01': 'rhod.sonic.z01.c0',
            'surface_met': 'rhod.met.z01.a0'
        }
    },
    'cape_cod': {
        'prefix': 'caco',
        'instruments': {
            'lidar_z01': 'caco.lidar.z01.00',
            'lidar_z02': 'caco.lidar.z02.a0',
            'met_z01': 'caco.sonic.z01.c1',
            'met_z02': 'caco.sonic.z02.c1',
            'surface_met': 'caco.met.z01.00'
        }
    }
}

# Instrument capability sets
TURBULENCE_INSTRUMENTS = {'lidar_z01', 'lidar_z02', 'lidar_z03', 'met_z01', 'met_z02', 'met_z03', 'met_z04'}
WIND_INSTRUMENTS = {'lidar_z01', 'lidar_z02', 'lidar_z03', 'radar', 'met_z01', 'met_z02', 'met_z03', 'met_z04'}
VAD_INSTRUMENTS = {'lidar_z01', 'lidar_z02', 'lidar_z03'} # Types of instruments that can produce VAD profiles 

def get_instrument_coordinates(location, instrument_type):
    """Get coordinates for specific instrument at location"""
    config = LOCATION_CONFIG.get(location, {})
    coords = config.get('coordinates', {})
    
    # Handle different naming conventions
    if instrument_type in coords:
        return coords[instrument_type]
    
    # Strip lidar_ prefix if present
    if instrument_type.startswith('lidar_'):
        stripped_name = instrument_type[6:]  # Remove 'lidar_'
        if stripped_name in coords:
            return coords[stripped_name]
    
    # Try with _lidar suffix for lidar instruments
    if f"{instrument_type}_lidar" in coords:
        return coords[f"{instrument_type}_lidar"]
    
    # Try with _met suffix for met instruments  
    if f"{instrument_type}_met" in coords:
        return coords[f"{instrument_type}_met"]
    
    # Try stripping _lidar/_met suffix and looking for base name
    if instrument_type.endswith('_lidar'):
        base_name = instrument_type[:-6]  # Remove '_lidar'
        if base_name in coords:
            return coords[base_name]
    
    if instrument_type.endswith('_met'):
        base_name = instrument_type[:-4]  # Remove '_met'
        if base_name in coords:
            return coords[base_name]
        
    return None

def calculate_average_coordinates(location, instruments):
    """Calculate average lat/lon weighted by instruments actually used in the data"""
    coords_list = []
    
    for instrument in instruments:
        coords = get_instrument_coordinates(location, instrument)
        if coords and len(coords) == 2 and not any(np.isnan(coords)):
            # Validate coordinates are reasonable (basic sanity check)
            lat, lon = coords
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords_list.append(coords)
    
    if coords_list:
        # Weight all instruments equally (could be refined to weight by data availability)
        avg_lat = sum(coord[0] for coord in coords_list) / len(coords_list)
        avg_lon = sum(coord[1] for coord in coords_list) / len(coords_list)
        return avg_lat, avg_lon
    
    # Fallback: Use site representative coordinates when instrument lookup fails
    site_defaults = {
        'cape_cod': [42.031888, -70.052386],    
        'block_island': [41.16670025, -71.5803045], 
        'nantucket': [41.243195125, -70.105867125],              
        'rhode_island': [41.44811325, -71.432165]           
    }
    
    default_coords = site_defaults.get(location)
    return default_coords if default_coords else (None, None)

def get_wind_correction(location, instrument):
    """Get wind direction correction for instrument at location"""
    config = LOCATION_CONFIG.get(location, {})
    corrections = config.get('wind_corrections', {})
    return corrections.get(instrument, 0)

def get_anemometer_correction(location, instrument):
    """Get anemometer coordinate correction for instrument at location"""
    config = LOCATION_CONFIG.get(location, {})
    corrections = config.get('anemometer_corrections', {})
    return corrections.get(instrument, 0)

def get_w_sign_correction(instrument, location):
    """Get w sign correction factor from location config"""
    try:
        return LOCATION_CONFIG[location]['w_sign_corrections'][instrument]
    except KeyError:
        return 1  # Default: no correction if not found

def get_instrument_type(instrument_name):
    """Get proper instrument type label"""
    if instrument_name == 'merged':
        return 'Multi-instrument'
    elif 'lidar' in instrument_name:
        return 'Lidar'
    elif 'radar' in instrument_name:
        return 'Radar'
    elif 'met' in instrument_name:
        return 'Anemometer'
    else:
        return 'Unknown'

def get_surface_met_patterns(location):
    """
    Get location-specific surface met file patterns
    Parameters:
    -----------
    location : str
        Location identifier ('nantucket', 'block_island', 'rhode_island', 'cape_cod')
    Returns:
    --------
    list
        List of file patterns for the location
    """
    location_patterns = {
        'nantucket': {
            'patterns': [
                "*met.z05*{date_formatted}*.nc",
                "nant.met.z05.c1.{date_formatted}.000000.nc"
            ]
        },
        'block_island': {
            'patterns': [
                "*met.z01*{date_formatted}*.nc", 
                "bloc.met.z01.c1.{date_formatted}.000000.nc"
            ]
        },
        'rhode_island': {
            'patterns': [
                "*met.z01*{date_formatted}*.nc",
                "rhod.met.z01.a0.{date_formatted}.000000.nc"
            ]
        },
        'cape_cod': {
            'patterns': [
                "*met.z01*{date_formatted}*.nc",
                "caco.met.z01.00.{date_formatted}.000000.nc"
            ]
        }
    }
    
    # Default to nantucket patterns if location not recognized
    return location_patterns.get(location, location_patterns['nantucket'])['patterns']

def get_location_display_name(location):
    """Convert location code to display name"""
    location_map = {
        'nantucket': 'Nantucket',
        'block_island': 'Block Island',
        'blockisland': 'Block Island',
        'bi': 'Block Island',
        'nant': 'Nantucket',
        'bloc': 'Block Island',
        'rhode_island': 'Rhode Island',
        'rhod': 'Rhode Island',
        'cape_cod': 'Cape Cod',
        'caco': 'Cape Cod'
    }
    if isinstance(location, str):
        return location_map.get(location.lower(), location.title())
    return 'Unknown Location'

def get_instrument_dataset_info(location):
    """Get instrument dataset information for metadata"""
    dataset_info = {
        'nantucket': 'nant.lidar.z01.a0, nant.lidar.z02.a0, nant.lidar.z03.00, nant.met.z02.00, nant.met.z03.00, nant.met.z04.00, nant.met.z05.c1, NOAA PSL radar',
        'block_island': 'bloc.lidar.z01.a0, bloc.lidar.z03.00, bloc.met.z01.c1, NOAA PSL radar',
        'rhode_island': 'rhod.lidar.z01.a0, rhod.lidar.z03.00, rhod.sonic.z01.c1, rhod.met.z01.a0, rhod.rwp.z01.a0',
        'cape_cod': 'caco.lidar.z01.00, caco.lidar.z02.a0, caco.sonic.z01.c1, caco.sonic.z02.c1, caco.met.z01.00'
    }
    return dataset_info.get(location.lower(), f'Instruments from {location}')

#### Ground Elevation Functions 

def get_ground_elevation_from_config(location, instrument):
    """Get ground elevation from config values (faster than API calls)"""
    elevations = LOCATION_CONFIG.get(location, {}).get('elevations', {})
    return elevations.get(instrument, 0)

#### Utility Functions

def round_profile_values(profile_data):
    """
    Round profile values based on parameter type
    """
    if not profile_data:
        return profile_data
    
    precision_map = {
        'ws': 2, 'wd': 1, 'w': 2,  # Wind parameters
        'ti': 3, 'tke': 3,          # Turbulence parameters  
        'std_u': 3, 'std_v': 3, 'std_w': 3,  # Standard Deviations
        'u': 2, 'v': 2,             # Wind components
        'uerr': 3, 'verr': 3, 'werr': 3,  # Wind component errors
        'wserr': 3, 'wderr': 3      # Wind speed/direction errors
    }
    
    rounded_data = {}
    for param, value in profile_data.items():
        if isinstance(value, (int, float)) and not (np.isnan(value) if isinstance(value, float) else False):
            precision = precision_map.get(param, 2)  # Default to 2 decimals
            if precision == 0:  # Integer values
                rounded_data[param] = int(value)
            else:
                rounded_data[param] = round(float(value), precision)
        else:
            rounded_data[param] = value  # Keep NaN, None, etc. as-is
    
    return rounded_data

def wind_direction_average(directions):
    """Calculate circular average of wind directions"""
    directions = np.array(directions)
    sin_avg = np.mean(np.sin(np.deg2rad(directions)))
    cos_avg = np.mean(np.cos(np.deg2rad(directions)))
    return np.rad2deg(np.arctan2(sin_avg, cos_avg)) % 360    
