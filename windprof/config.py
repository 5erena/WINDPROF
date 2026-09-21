"""Site, instrument, and processing configuration for the WINDPROF pipeline.

Single source of truth for everything site specific:

  - ``LOCATION_CONFIG``: per-site instrument coordinates, ground elevations,
    azimuth and vertical-velocity sign corrections, anemometer corrections
    and measurement heights.
  - ``LOCATION_ALIASES`` / ``SITE_CODES``: canonical site names and the short
    codes used in output filenames.
  - ``MALFUNCTION_PERIODS``: time ranges to exclude per instrument.
  - Per-instrument QC variable names and lookup helpers
    (``get_wind_correction``, ``get_w_sign_correction``, etc.).

Paths default to ``./data`` and ``./results`` relative to the current
working directory; override with the ``WINDPROF_DATA_PATH`` and
``WINDPROF_RESULTS_PATH`` environment variables.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

### Processing Parameters
TIME_WINDOW_MINUTES = 10  # output time resolution

### Terrain and Coordinates

LOCATION_CONFIG = {
    'nantucket': {
        # z03_met and z04_met (10 m, north facing) are documented but not
        # processed: the tower wakes them under the dominant SW flow (>20% wind
        # speed error), and z03_met is sparse. Only the SW facing z02 5 m is used.
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
            'z02': -4,  # empirical: aligns z02 direction with the co-located z03 WindCube
            'z03': 0,
            'radar': 0
        },
        'anemometer_corrections': {
            # Rotation from the instrument frame to true north, applied to (u, v)
            # before the direction is formed: a POSITIVE value here DECREASES the
            # direction, while the sonic paths at the other sites add theirs to
            # the direction and rotate the opposite way.
            # z02: the declination was entered at the instrument with the wrong
            # sign, so the residual is twice the declination.
            'z02': 28,
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
            'z01': 10,         # R.M. Young propeller-vane anemometer on the NOAA PSL met tower
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
            # z03: empirical, from z03-vs-radar bias minimization.
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
            # z02: 180 is a TO/FROM convention flip, not an azimuth offset; the
            # remainder rotates magnetic to true north (westerly declination).
            'z02': 165.5,
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

DATA_BASE_PATH = os.environ.get('WINDPROF_DATA_PATH', str(Path.cwd() / 'data')) + '/'
RESULTS_BASE_PATH = os.environ.get('WINDPROF_RESULTS_PATH', str(Path.cwd() / 'results')) + '/'

# Campaign archive tree holding the daily input files (z03 profiling lidars,
# sonics, surface met), read in place rather than copied.

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
            'met_z01': 'rhod.sonic.z01.c1',
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
VAD_INSTRUMENTS = {'lidar_z01', 'lidar_z02', 'lidar_z03'}

# Per-instrument QC: the single source of truth for signal screening.
# Thresholds are taken from each instrument's own raw-file headers where it
# publishes one (WindCube CNRThreshold=; ZephIR packet and rain columns) and
# from the scanning lidars' noise floor otherwise.
#
# qc_type:
#   'intensity'    Halo scanning: keep beams with intensity >= threshold
#                  (intensity = linear SNR + 1; 1.008 ~ the noise floor)
#   'cnr'          WindCube profiling: keep gates with CNR(dB) >= threshold
#   'samples_rain' ZephIR profiling: no CNR/SNR column, keep where
#                  Packets >= min_samples AND rain% == 0
#   'prefiltered'  QC already applied upstream by the instrument; no pipeline filter
#
# min_beams = 4: the wserr < 2 m/s gate needs dof >= 1, and a 3-beam VAD is
# exactly determined (dof = 0), so sigma_ws is 0/0 and the gate cannot apply.
QC_CONFIG = {
    'nantucket': {
        'z01': {'qc_type': 'intensity',    'threshold': 1.008, 'min_beams': 4},
        'z02': {'qc_type': 'intensity',    'threshold': 1.008, 'min_beams': 4}, 
        'z03': {'qc_type': 'cnr',          'threshold': -23},                    # header CNRThreshold=-23
    },
    'block_island': {
        'z01': {'qc_type': 'intensity',    'threshold': 1.008, 'min_beams': 4},
        'z03': {'qc_type': 'cnr',          'threshold': -22},                    # header CNRThreshold=-22
    },
    'rhode_island': {
        'z01': {'qc_type': 'intensity',    'threshold': 1.008, 'min_beams': 4},
        'z03': {'qc_type': 'samples_rain', 'min_samples': 20},                   # ZephIR pre-averaged; no SNR/CNR
    },
    'cape_cod': {
        # min_ti_availability gates turbulence only, not wind: the workbook's
        # speed dispersion is formed from whatever scans the vendor retained, so
        # a window with almost none is not a turbulence measurement. 10% is the
        # >= 3-valid-scan minimum of calculate_turbulence_metrics, as a fraction.
        'z01': {'qc_type': 'prefiltered', 'min_ti_availability': 10},            # WindCube V2-96, filtered upstream
        'z02': {'qc_type': 'intensity',    'threshold': 1.008, 'min_beams': 4},
    },
}

# Default beam minimum for every least-squares wind fit (scanning VAD and the
# RTD profiling fit); a QC_CONFIG entry overrides it per instrument.
MIN_VAD_BEAMS = 4


def get_qc_params(location, instrument):
    """QC_CONFIG lookup that fails loudly on unknown (site, instrument)
    pairs instead of silently defaulting: a typo here must not disable QC."""
    location = normalize_location(location)
    try:
        return QC_CONFIG[location][instrument]
    except KeyError:
        raise ValueError(
            f"No QC configuration for instrument '{instrument}' at "
            f"'{location}'. Known: "
            f"{ {loc: sorted(entries) for loc, entries in QC_CONFIG.items()} }")

def get_instrument_coordinates(location, instrument_type):
    """Get coordinates for specific instrument at location"""
    config = LOCATION_CONFIG.get(location, {})
    coords = config.get('coordinates', {})
    
    if instrument_type in coords:
        return coords[instrument_type]
    
    if instrument_type.startswith('lidar_'):
        stripped_name = instrument_type[6:]
        if stripped_name in coords:
            return coords[stripped_name]
    
    if f"{instrument_type}_lidar" in coords:
        return coords[f"{instrument_type}_lidar"]
    
    if f"{instrument_type}_met" in coords:
        return coords[f"{instrument_type}_met"]
    
    if instrument_type.endswith('_lidar'):
        base_name = instrument_type[:-6]
        if base_name in coords:
            return coords[base_name]
    
    if instrument_type.endswith('_met'):
        base_name = instrument_type[:-4]
        if base_name in coords:
            return coords[base_name]
        
    return None
    
def calculate_average_coordinates(location, instruments):
    """Unweighted mean lat/lon over instruments with configured coordinates,
    with a per-site fallback"""
    coords_list = []
    
    for instrument in instruments:
        coords = get_instrument_coordinates(location, instrument)
        if coords and len(coords) == 2 and not any(np.isnan(coords)):
            lat, lon = coords
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords_list.append(coords)
    
    if coords_list:
        avg_lat = sum(coord[0] for coord in coords_list) / len(coords_list)
        avg_lon = sum(coord[1] for coord in coords_list) / len(coords_list)
        return avg_lat, avg_lon
    
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
    """Sonic frame-to-true-north rotation in degrees for instrument at location"""
    config = LOCATION_CONFIG.get(location, {})
    corrections = config.get('anemometer_corrections', {})
    return corrections.get(instrument, 0)

def get_near_surface_levels():
    """Grid levels for single-height anemometers: the configured measurement
    heights across all sites (4/5/10 m in WFIP3), so each sonic sits at its own
    height. 0 m carries no instrument and is deliberately not a grid level."""
    levels = set()
    for site in LOCATION_CONFIG.values():
        for height in site.get('anemometer_heights', {}).values():
            levels.add(float(height))
    return tuple(sorted(levels))

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

def get_ground_elevation_from_config(location, instrument):
    """Ground elevation in m ASL for an instrument, or 0 if not configured"""
    elevations = LOCATION_CONFIG.get(location, {}).get('elevations', {})
    return elevations.get(instrument, 0)

def round_profile_values(profile_data):
    """Round profile values to the per-parameter precision in precision_map"""
    if not profile_data:
        return profile_data
    
    precision_map = {
        'ws': 2, 'wd': 1, 'w': 2,
        'ti': 3, 'tke': 3,
        'std_u': 3, 'std_v': 3, 'std_w': 3,
        'u': 2, 'v': 2,
        'uerr': 3, 'verr': 3, 'werr': 3,
        'wserr': 3, 'wderr': 3
    }
    
    rounded_data = {}
    for param, value in profile_data.items():
        if isinstance(value, (int, float)) and not (np.isnan(value) if isinstance(value, float) else False):
            precision = precision_map.get(param, 2)
            if precision == 0:
                rounded_data[param] = int(value)
            else:
                rounded_data[param] = round(float(value), precision)
        else:
            rounded_data[param] = value
    
    return rounded_data

def wind_direction_average(directions):
    """Calculate circular average of wind directions"""
    directions = np.array(directions)
    sin_avg = np.mean(np.sin(np.deg2rad(directions)))
    cos_avg = np.mean(np.cos(np.deg2rad(directions)))
    return np.rad2deg(np.arctan2(sin_avg, cos_avg)) % 360    
