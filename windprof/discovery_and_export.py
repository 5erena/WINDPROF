"""File discovery and CF-compliant NetCDF export

Searches the configured ``DATA_BASE_PATH`` and ``ARCHIVE_BASE_PATH`` to
assemble per-instrument file lists for a given site and date range, then
writes the merged products as CF-compliant NetCDF files with per-height
quality flags and global attributes documenting the campaign.

File-naming patterns are campaign-specific. Adapters should expect
to edit this module to match the file conventions of their own
data delivery; the per-site discovery functions are a reasonable
template but are not generic.
"""

import getpass
import socket

import numpy as np
import pandas as pd
import xarray as xr
import errno
import os
import glob
import re
import uuid
from .config import (normalize_location, SITE_INSTRUMENT_MAPPINGS, TURBULENCE_INSTRUMENTS,
                     WIND_INSTRUMENTS, VAD_INSTRUMENTS, calculate_average_coordinates,
                     DATA_BASE_PATH, ARCHIVE_BASE_PATH, LOCATION_CONFIG,
                     TIME_WINDOW_MINUTES, get_location_display_name, get_qc_params)

#### File Discovery

def find_caco_file_for_date(target_date):
    """Locate the CACO z01 lidar workbook whose internal time range covers target_date.

    CACO z01 files are delivered as monthly-ish .xlsx workbooks spanning roughly
    30-40 days, so the filename date is not a reliable index, so we open candidate
    files in the current and two prior months and check the actual first/last
    timestamps on Sheet1.
    """
    year = target_date.year
    month = target_date.month

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

    High-volume instruments (scanning lidars) are pre-merged to a local
    directory; low-volume daily files are read directly from the campaign
    archive tree (``ARCHIVE_BASE_PATH``) to avoid duplicating storage.

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
    if isinstance(start_date, str):
        start_obj = pd.to_datetime(start_date)
    else:
        start_obj = start_date
    if isinstance(end_date, str):
        end_obj = pd.to_datetime(end_date)
    else:
        end_obj = end_date
    
    date_range = pd.date_range(start=start_obj, end=end_obj, freq='D')
    
    year = start_obj.year
    month = start_obj.month
    yyyymm = f"{year}{month:02d}/"    
    
    # Location-specific file paths only; QC thresholds and beam minimums live
    # in config.QC_CONFIG and are looked up by the instrument processors.
    if location == 'nantucket':
        site_prefix = 'nant'
        z03_ext = 'rtd'
        instruments = {
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z01.a0.{{date_formatted}}.*.nc', 'params': {}},
            'lidar_z02': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z02.a0.{{date_formatted}}.*.nc', 'params': {}},
            
            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.{z03_ext}', 'params': {}},
            'met_z02': {'path_template': f'{ARCHIVE_BASE_PATH}/nant.met.z02.a0/{{year}}/{{month:02d}}/{{day:02d}}/nant.met.z02.a0.{{date_formatted}}.*.nc', 'params': {'instrument': 'z02'}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z05.c1/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z05.c1.{{date_formatted}}.*.nc', 'params': {}},
            
            # Manually added files - local directory
            'radar': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.radar.{{date_formatted}}.txt', 'params': {}}
        }
    elif location == 'block_island':
        site_prefix = 'bloc'
        z03_ext = 'sta'
        instruments = {
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.lidar.z01.a0.{{date_formatted}}.*.nc', 'params': {}},
            
            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.{z03_ext}', 'params': {}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z01.c1/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z01.c1.{{date_formatted}}.*.nc', 'params': {}},
            
            # Manually added files - local directory
            'radar': {'path_template': f'{DATA_BASE_PATH}{yyyymm}{site_prefix}.radar.{{date_formatted}}.txt', 'params': {}}
        }
    elif location == 'rhode_island':
        site_prefix = 'rhod'
        instruments = {
            'lidar_z01': {'path_template': f'{DATA_BASE_PATH}{yyyymm}/{site_prefix}.lidar.z01.a0.{{date_formatted}}.000000.nc', 'params': {}},

            'lidar_z03': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.lidar.z03.00/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.lidar.z03.00.{{date_formatted}}.*.csv', 'params': {}},
            'met_z01': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.sonic.z01.c1/{{year}}/{{month:02d}}/{{day:02d}}/10/full_output/ver_3/{site_prefix}.sonic.z01.c1.{{date_formatted}}.000000.10.full_output.ver_3.csv', 'params': {}},
            'surface_met': {'path_template': f'{ARCHIVE_BASE_PATH}/{site_prefix}.met.z01.a0/{{year}}/{{month:02d}}/{{day:02d}}/{site_prefix}.met.z01.a0.{{date_formatted}}.*.nc', 'params': {}}
        }
    elif location == 'cape_cod': 
        site_prefix = 'caco'
        # CACO z01 arrives as multi-day .xlsx workbooks, so it is absent here
        # and is discovered by find_caco_file_for_date() below.
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

        if location == 'cape_cod':
            filepath = find_caco_file_for_date(date_obj)
            if filepath:
                found_configs['lidar_z01'] = {'filename': filepath, 'params': {}}
                if verbose:
                    print(f"  ✓ lidar_z01: {filepath}")
            elif verbose:
                print(f"  ✗ lidar_z01: No CACO workbook covering {date_str}")

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

def preflight_check_inputs(start_date, end_date, location='nantucket', verbose=True,
                           max_probe_dates=40):
    """Resolve every configured datastream across the whole range before any
    day is written.

    A missing input is not an error to the per-day processors, so a run against
    an unreachable archive completes with every day reporting success while
    silently omitting whole instruments and overwriting good products. This
    check turns that into a refusal before the first write.

    The test is at the datastream level, not per day: per-day gaps are normal
    because instruments are installed and removed mid-campaign, but a
    configured datastream resolving nothing across the entire range is not.

    Dates are probed rather than enumerated.

    Returns
    -------
    (ok, report) : (bool, dict)
        ``ok`` is False if any configured datastream resolved nothing on any
        probed date. ``report`` maps instrument name -> number of probed dates
        that resolved a file, so the caller can name the missing stream rather
        than only report that something is missing.
    """
    location = normalize_location(location)
    expected = dict(SITE_INSTRUMENT_MAPPINGS.get(location, {}).get('instruments', {}))
    # The radar is not a path-resolved datastream: it is read through its own
    # ingest and has no glob here, so it cannot be checked this way.
    expected.pop('radar', None)

    all_dates = pd.date_range(start=pd.to_datetime(start_date),
                              end=pd.to_datetime(end_date), freq='D')
    # Spread the probes across the range so an instrument installed partway
    # through, or removed partway through, is still seen.
    if len(all_dates) <= max_probe_dates:
        probes = list(all_dates)
    else:
        idx = np.linspace(0, len(all_dates) - 1, max_probe_dates).astype(int)
        probes = [all_dates[i] for i in sorted(set(idx))]

    report = {inst: 0 for inst in expected}
    probed = 0
    for d in probes:
        probed += 1
        day = discover_instrument_files_for_date(d.strftime('%Y-%m-%d'),
                                                 location=location, verbose=False)
        for inst in day:
            if inst in report:
                report[inst] += 1
        if all(report.values()):
            break

    # One datastream can be configured under two instrument keys (Block
    # Island's met_z01 and surface_met name the same file), so a stream
    # resolved under any key satisfies every key that names it.
    reachable = {expected[i] for i, n in report.items() if n > 0}
    aliased = sorted(i for i, n in report.items()
                     if n == 0 and expected[i] in reachable)
    missing = sorted(i for i, n in report.items()
                     if n == 0 and expected[i] not in reachable)
    ok = not missing

    if verbose:
        print(f"Pre-flight: {location} {start_date} to {end_date} "
              f"({len(all_dates)} dates, {probed} probed)")
        for inst in sorted(report):
            n = report[inst]
            mark = 'ok' if n else ('ok' if inst in aliased else 'MISSING')
            note = (f"resolved on {n} of {probed} probed dates" if n else
                    (f"same datastream as {[k for k, v in expected.items() if v == expected[inst] and report[k]][0]}"
                     if inst in aliased else f"resolved on 0 of {probed} probed dates"))
            print(f"  {mark:>8}  {inst:<14} {expected[inst]:<24} {note}")
        if missing:
            print(f"  {len(report) - len(missing)} of {len(report)} datastreams reachable; "
                  f"nothing found across {probed} dates spanning the range for: "
                  f"{', '.join(missing)}")
            print("  A datastream resolving nothing anywhere in the range usually means its "
                  "archive is unreachable, not that the instrument was absent.")
    return ok, report


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

# Method strings written into the turbulence_intensity_* variable attributes.
# TI definitions deliberately differ by instrument (documented project
# decision) because the instruments deliver different raw quantities.
_TI_HYBRID = ('TI = sigma(WS) / WS_hybrid with WS_hybrid = 0.55*scalar_mean_WS '
              '+ 0.45*vector_mean_WS (Rosenbusch 2021 hybrid), computed from '
              'scan-level VAD retrievals within the 10-minute window')
_TI_HYBRID_SONIC = ('TI = sigma(WS) / WS_hybrid with WS_hybrid = '
                    '0.55*scalar_mean_WS + 0.45*vector_mean_WS (Rosenbusch 2021 '
                    'hybrid), computed from the high-rate sonic series within '
                    'the 10-minute window')
_TI_STA = ('TI = sqrt(TKE) / WS with TKE = 0.5*(sigma_u^2 + sigma_v^2 + '
           'sigma_w^2), from WindCube STA vendor 10-minute component statistics')
# The availability minimum is read from QC_CONFIG so the published method
# string cannot drift from the threshold the processor actually applies.
_TI_CACO_WORKBOOK = ('TI = sigma(WS) / WS from WindCube vendor 10-minute wind speed '
                     'dispersion (workbook delivery), computed where the vendor reports '
                     f"at least {get_qc_params('cape_cod', 'z01')['min_ti_availability']}% "
                     'within-window data availability')

TI_METHODS = {
    ('nantucket', 'lidar_z01'): _TI_HYBRID,
    ('nantucket', 'lidar_z02'): _TI_HYBRID,
    ('nantucket', 'lidar_z03'): _TI_HYBRID + ' (WindCube RTD high-frequency series)',
    ('nantucket', 'met_z02'): _TI_HYBRID_SONIC,
    ('block_island', 'lidar_z01'): _TI_HYBRID,
    ('block_island', 'lidar_z03'): _TI_STA,
    ('block_island', 'met_z01'): 'not reported (surface-met wind record; no turbulence data)',
    ('rhode_island', 'lidar_z01'): _TI_HYBRID,
    ('rhode_island', 'lidar_z03'): 'vendor-reported TI (ZephIR precomputed 10-minute statistics)',
    ('rhode_island', 'met_z01'): 'TI = sigma_u / U from EddyPro streamwise variance (10-minute records)',
    ('cape_cod', 'lidar_z01'): _TI_CACO_WORKBOOK,
    ('cape_cod', 'lidar_z02'): _TI_HYBRID,
    ('cape_cod', 'met_z01'): 'TI = sigma_u / U from EddyPro streamwise variance (10-minute records)',
    ('cape_cod', 'met_z02'): 'TI = sigma_u / U from EddyPro streamwise variance (10-minute records)',
}


# --- CF-1.10 conventions applied generically at export (no site logic) ---
# base variable name -> (standard_name or None, canonical UDUNITS, cell_methods).
# Names CF does not define (TI, TKE, the error/std fields) carry no
# standard_name, so a descriptive long_name plus correct units instead.
_CF_SCIENCE_META = {
    'wind_speed':               ('wind_speed',          'm s-1', 'time: mean'),
    'wind_direction':           ('wind_from_direction', 'degree', 'time: mean'),
    'vertical_velocity':        ('upward_air_velocity', 'm s-1', 'time: mean'),
    'turbulence_intensity':     (None,                  '1',     'time: mean'),
    # CF defines this one, with canonical units m2 s-2 matching what we store.
    'turbulent_kinetic_energy': ('specific_turbulent_kinetic_energy_of_air',
                                 'm2 s-2', 'time: mean'),
    'u_std':                    (None, 'm s-1', 'time: standard_deviation'),
    'v_std':                    (None, 'm s-1', 'time: standard_deviation'),
    'w_std':                    (None, 'm s-1', 'time: standard_deviation'),
    'u_error':                  (None, 'm s-1', None),
    'v_error':                  (None, 'm s-1', None),
    'w_error':                  (None, 'm s-1', None),
    'wind_speed_error':         (None, 'm s-1', None),
    'wind_direction_error':     (None, 'degree', None),
}

# Source description for the met_* instruments, so users can tell true
# sonic-anemometer records from surface-met-derived winds (the met_z01 label
# alone does not distinguish them across sites). BLOC identity per the NOAA
# PSL surface-met documentation for wfip3/bloc.met.z01.c1.
MET_SOURCES = {
    ('nantucket', 'met_z02'): '20 Hz sonic anemometer record, Reynolds-averaged per 10-minute window',
    ('block_island', 'met_z01'): ('surface meteorological station 10 m wind '
                                  'record from a propeller-vane anemometer '
                                  '(R.M. Young Wind Monitor, NOAA PSL); not '
                                  'a sonic-anemometer time series'),
    ('rhode_island', 'met_z01'): 'EddyPro 10-minute processed sonic-anemometer statistics',
    ('cape_cod', 'met_z01'): 'EddyPro 10-minute processed sonic-anemometer statistics',
    ('cape_cod', 'met_z02'): 'EddyPro 10-minute processed sonic-anemometer statistics',
}

# Dataset attribution and publication identity (ACDD-1.3). Publisher block
# per Wind Data Hub instruction; creator block confirmed by the dataset owner.
DATASET_ATTRIBUTION = {
    'creator_name': 'Serena Lipari',
    'creator_email': 'serena.lipari@pnnl.gov',
    'creator_institution': 'Pacific Northwest National Laboratory',
    'publisher_name': 'Wind Data Hub',
    'publisher_email': 'wdhteam@pnnl.gov',
    'publisher_url': 'https://wdh.energy.gov/',
    # SPDX identifier per WDH review.
    'license': 'CC0-1.0',
}

# surface (time-only) variables -> (standard_name, units-as-stored, cell_methods)
_CF_SURFACE_META = {
    'surface_pressure':      ('air_pressure',    'hPa',  'time: mean'),
    'surface_temperature':   ('air_temperature', 'degC', 'time: mean'),
    'surface_humidity':      ('relative_humidity', '%',  'time: mean'),
    # precipitation is a depth (mm), so the length-dimensioned standard name
    # (canonical unit m) is the compatible one, not precipitation_amount
    # (canonical kg m-2).
    'surface_precipitation': ('lwe_thickness_of_precipitation_amount', 'mm', 'time: sum'),
    'ground_elevation':      ('surface_altitude', 'm',   None),
}

# error/uncertainty partner for each primary wind variable, linked via
# ancillary_variables alongside the qc flag.
_CF_ERROR_PARTNER = {
    'wind_speed': 'wind_speed_error',
    'wind_direction': 'wind_direction_error',
    'vertical_velocity': 'w_error',
}


def _cf_base_name(var_name, instrument_suffixes):
    """Strip the instrument/merged suffix so 'wind_speed_lidar_z01' and
    'u_std_merged' resolve to their base name for the metadata lookup."""
    for suffix in instrument_suffixes:
        tail = '_' + suffix
        if var_name.endswith(tail):
            return var_name[:-len(tail)], suffix
    return var_name, None


def _apply_cf_conventions(ds, all_possible_instruments, instrument_mapping,
                          avg_lat, avg_lon,
                          dataset_id=None, platform_name=None):
    """Apply CF-1.10 + ACDD-1.3 metadata to the assembled dataset in place.

    Structure and metadata only: no existing data array is modified, so values
    round-trip bit-identically (time_bounds is added as a new variable).
    Generic across campaigns: everything is driven by variable-name
    patterns and the metadata tables above.
    """
    import numpy as _np

    # Longest suffixes first so 'lidar_z01' matches before a bare token.
    instrument_suffixes = sorted(['merged'] + list(instrument_mapping.keys()),
                                 key=len, reverse=True)

    # Scalar CRS (WGS84) referenced by every geophysical variable.
    ds['crs'] = ((), _np.int32(0), {
        'grid_mapping_name': 'latitude_longitude',
        'longitude_of_prime_meridian': 0.0,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
        'epsg_code': 'EPSG:4326',
    })

    have_lonlat = avg_lat is not None and avg_lon is not None
    if have_lonlat:
        ds['latitude'] = ((), float(avg_lat), {
            'standard_name': 'latitude', 'long_name': 'Nominal site latitude',
            'units': 'degrees_north', 'axis': 'Y'})
        ds['longitude'] = ((), float(avg_lon), {
            'standard_name': 'longitude', 'long_name': 'Nominal site longitude',
            'units': 'degrees_east', 'axis': 'X'})
    coord_ref = 'latitude longitude' if have_lonlat else None

    def _science_extras(var_name):
        attrs = {'grid_mapping': 'crs'}
        if coord_ref:
            attrs['coordinates'] = coord_ref
        return attrs

    def _with_interval(cell_methods):
        return f'{cell_methods} (interval: {TIME_WINDOW_MINUTES} minutes)'

    for name in list(ds.data_vars):
        var = ds[name]

        if name == 'crs':
            continue

        if name == 'instrument_availability':
            n = len(all_possible_instruments)
            masks = [1 << i for i in range(n)]
            var.attrs.pop('flag_values', None)
            var.attrs.pop('_FillValue', None)
            var.attrs.update({
                'long_name': 'Bitmask of instruments contributing at each time',
                'flag_masks': _np.array(masks, dtype=_np.int32),
                'flag_meanings': ' '.join(f'{inst}_available'
                                          for inst in all_possible_instruments),
                'valid_range': _np.array([0, sum(masks)], dtype=_np.int32),
                'coverage_content_type': 'referenceInformation',
            })
            continue

        if name.startswith('qc_'):
            # flag_values must share the flag variable's integer dtype.
            if 'flag_values' in var.attrs:
                var.attrs['flag_values'] = _np.array(var.attrs['flag_values'],
                                                     dtype=var.dtype)
            # All flag variables use the bare 'status_flag' standard name, not
            # the CF modifier form, per the Wind Data Hub review.
            var.attrs['standard_name'] = 'status_flag'
            var.attrs['coverage_content_type'] = 'qualityInformation'
            var.attrs['grid_mapping'] = 'crs'
            if coord_ref:
                var.attrs['coordinates'] = coord_ref
            continue

        if name in _CF_SURFACE_META:
            sn, units, cm = _CF_SURFACE_META[name]
            var.attrs['standard_name'] = sn
            var.attrs['units'] = units
            if cm:
                var.attrs['cell_methods'] = _with_interval(cm)
            var.attrs.update(_science_extras(name))
            if name == 'ground_elevation':
                # Site characteristic, not an averaged measurement.
                var.attrs['cell_methods'] = 'time: point'
                var.attrs['coverage_content_type'] = 'referenceInformation'
            else:
                var.attrs['coverage_content_type'] = 'physicalMeasurement'
            qc_name = f'qc_{name}'
            if qc_name in ds.variables:
                var.attrs['ancillary_variables'] = qc_name
            continue

        base, suffix = _cf_base_name(name, instrument_suffixes)
        if base in _CF_SCIENCE_META:
            sn, units, cm = _CF_SCIENCE_META[base]
            if sn:
                var.attrs['standard_name'] = sn
            var.attrs['units'] = units
            if cm:
                var.attrs['cell_methods'] = _with_interval(cm)
            var.attrs.update(_science_extras(name))
            # The component stds are real turbulence statistics, not
            # uncertainty, so they stay physicalMeasurement.
            var.attrs['coverage_content_type'] = (
                'qualityInformation' if base.endswith('_error')
                else 'physicalMeasurement')

            ancillary = []
            qc_name = f'qc_{name}'
            if qc_name in ds.variables:
                ancillary.append(qc_name)
            partner_base = _CF_ERROR_PARTNER.get(base)
            if partner_base and suffix:
                partner = f'{partner_base}_{suffix}'
                if partner in ds.variables:
                    ancillary.append(partner)
            if ancillary:
                var.attrs['ancillary_variables'] = ' '.join(ancillary)

    # Coordinate variables. time carries integer seconds, so its units live
    # in attrs (xarray only CF-encodes datetime64 vars); lowercase 'seconds'
    # is what CF parsers require. Coordinates must not carry _FillValue.
    ds['time'].attrs.update({
        'standard_name': 'time', 'long_name': 'Time', 'axis': 'T',
        'units': 'seconds since 1970-01-01T00:00:00Z',
        'calendar': 'proleptic_gregorian', 'bounds': 'time_bounds'})
    ds['height'].attrs.update({
        'standard_name': 'height', 'long_name': 'Height above ground level',
        'units': 'm', 'positive': 'up', 'axis': 'Z'})

    # Cell bounds: timestamps mark the START of each averaging window (the
    # documented product convention), so bounds run [t, t + window], not
    # center +/- half-window, which would assume center-stamped files.
    tvals = _np.asarray(ds['time'].values, dtype='int64')
    window_s = TIME_WINDOW_MINUTES * 60
    ds['time_bounds'] = (('time', 'nv'),
                         _np.column_stack([tvals, tvals + window_s]))

    for coord in ('time', 'height', 'time_bounds'):
        ds[coord].encoding['_FillValue'] = None
    for scalar in ('crs', 'latitude', 'longitude'):
        if scalar in ds.variables:
            ds[scalar].encoding['_FillValue'] = None

    # Global attributes: CF + the ACDD-1.3 identification/coverage set.
    created = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    start_iso = pd.to_datetime(int(tvals[0]), unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_iso = pd.to_datetime(int(tvals[-1]) + window_s, unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%SZ')
    heights_m = _np.asarray(ds['height'].values, dtype=float)
    ds.attrs.update({
        'Conventions': 'CF-1.10, ACDD-1.3',
        'history': (f'{created}: created by the WINDPROF processing pipeline, '
                    f'user {getpass.getuser()} on machine '
                    f'{socket.gethostname()}.'),
        'date_created': created,
        'standard_name_vocabulary': 'CF Standard Name Table v93',
        'naming_authority': 'gov.energy.wdh',
        'source': ('Merged 10-minute wind and turbulence profiles from '
                   'scanning and profiling Doppler lidar VAD retrievals, '
                   '915 MHz radar wind profiler consensus winds, and sonic '
                   'anemometer statistics.'),
        'processing_level': 'c1: calibrated, quality-controlled, merged best estimate',
        'project': 'Wind Forecast Improvement Project 3 (WFIP3)',
        'keywords': ('wind profile, atmospheric boundary layer, Doppler lidar, '
                     'radar wind profiler, sonic anemometer, turbulence, '
                     'offshore wind energy, WFIP3'),
        # dict.fromkeys dedupes while preserving order: one datastream can
        # feed two instrument slots.
        'instrument': ', '.join(dict.fromkeys(
            instrument_mapping[i] for i in all_possible_instruments)),
        'time_coverage_start': start_iso,
        'time_coverage_end': end_iso,
        'time_coverage_duration': pd.Timedelta(seconds=int(tvals[-1]) + window_s - int(tvals[0])).isoformat(),
        'time_coverage_resolution': f'PT{TIME_WINDOW_MINUTES}M',
        'geospatial_vertical_min': float(heights_m.min()),
        'geospatial_vertical_max': float(heights_m.max()),
        'geospatial_vertical_units': 'm',
        'geospatial_vertical_positive': 'up',
        'institution': 'Pacific Northwest National Laboratory',
        'creator_url': 'https://www.pnnl.gov/',
        'acknowledgment': ('Data produced as part of the Wind Forecast '
                           'Improvement Project 3 (WFIP3). Funding provided '
                           'by the U.S. Department of Energy Office of '
                           'Critical Minerals and Energy Innovation '
                           'Integrated Energy Systems Office and the NOAA '
                           'Atmospheric Science for Renewable Energy '
                           'Program. This research was supported by NOAA '
                           'cooperative agreement NA22OAR4320151, for the '
                           'Cooperative Institute for Earth System Research '
                           'and Data Science (CIESRDS), and by DOE contract '
                           'DE-EE0009424405 to the Woods Hole Oceanographic '
                           'Institution.'),
        'comment': ('Timestamps mark the start of each averaging window '
                    '(see time_bounds). Consult per-variable attributes for '
                    'QC flags, uncertainties, and per-instrument method '
                    'documentation. No geospatial_bounds_vertical_crs is '
                    'declared: the vertical coordinate is height above '
                    'ground level, for which no EPSG vertical CRS applies.'),
    })
    if have_lonlat:
        ds.attrs['geospatial_bounds'] = f'POINT ({float(avg_lon)} {float(avg_lat)})'
        ds.attrs['geospatial_bounds_crs'] = 'EPSG:4326'
    ds.attrs.update(DATASET_ATTRIBUTION)
    if dataset_id:
        ds.attrs['id'] = dataset_id
    if platform_name:
        ds.attrs['platform'] = platform_name
    ds.attrs.pop('creation_date', None)
    return ds


# Floor on free bytes before a daily product is written. Module level so a
# deployment, or a test, can rebind it.
MIN_FREE_BYTES_FLOOR = int(os.environ.get('WINDPROF_MIN_FREE_BYTES', 64 * 1024 ** 2))


def _require_free_space(directory, estimated_bytes):
    """Raise ``OSError(ENOSPC)`` if ``directory`` cannot plausibly hold one
    more daily product.

    Free space can change between this check and the write, so treat the
    answer as a hint. Uses ``f_bavail``, not ``f_bfree``, which counts blocks
    an unprivileged writer cannot use.
    """
    try:
        st = os.statvfs(directory)
    except OSError:
        # Filesystem does not support introspection; let the write itself be
        # the authority rather than refusing on no evidence.
        return
    free = st.f_bavail * st.f_frsize
    # 4x the in-memory size covers the temp file, the file it is about to
    # replace, and HDF5 metadata overhead.
    needed = max(MIN_FREE_BYTES_FLOOR, 4 * int(estimated_bytes))
    if free < needed:
        raise OSError(
            errno.ENOSPC,
            f'refusing to write in {directory}: {free / 1024 ** 2:.1f} MiB free, '
            f'need at least {needed / 1024 ** 2:.1f} MiB '
            f'(set WINDPROF_MIN_FREE_BYTES to override)')


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
    
    location = normalize_location(location)
    site_config = SITE_INSTRUMENT_MAPPINGS.get(location, SITE_INSTRUMENT_MAPPINGS['nantucket'])
    site_prefix = site_config['prefix']
    instrument_mapping = site_config['instruments']
    
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

    # A day whose intervals hold no merged values yields an empty height set.
    # The product carries only populated levels, so skip rather than write an
    # empty shell file.
    if heights.size == 0:
        print("No populated height levels for this day - no file written")
        return None

    times = np.array([interval['time'] for interval in time_intervals])
    
    time_seconds = times.astype('datetime64[s]').astype(np.int64)
    n_times = len(times)
    n_heights = len(heights)
    
    avg_lat, avg_lon = calculate_average_coordinates(location, instruments)
    
    # Initialize ALL possible instrument data arrays (even if not found)
    all_possible_instruments = list(instrument_mapping.keys())
    instrument_data = {}
    
    for instrument in all_possible_instruments:
        instrument_data[instrument] = {}
        
        if instrument in WIND_INSTRUMENTS:
            instrument_data[instrument].update({
                'ws': np.full((n_times, n_heights), -9999.0),
                'wd': np.full((n_times, n_heights), -9999.0),
                'w': np.full((n_times, n_heights), -9999.0)
            })
            
            if instrument in VAD_INSTRUMENTS:
                instrument_data[instrument].update({
                    'uerr': np.full((n_times, n_heights), -9999.0),
                    'verr': np.full((n_times, n_heights), -9999.0),
                    'werr': np.full((n_times, n_heights), -9999.0),
                    'wserr': np.full((n_times, n_heights), -9999.0),
                    'wderr': np.full((n_times, n_heights), -9999.0)
                })

        if instrument in TURBULENCE_INSTRUMENTS:
            instrument_data[instrument].update({
                'ti': np.full((n_times, n_heights), -9999.0),
                'tke': np.full((n_times, n_heights), -9999.0),
                'std_u': np.full((n_times, n_heights), -9999.0),
                'std_v': np.full((n_times, n_heights), -9999.0),
                'std_w': np.full((n_times, n_heights), -9999.0)
            })
    
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
    tke_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    std_u_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    std_v_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    std_w_quality_flags = np.full((n_times, n_heights), 3, dtype=np.int32)
    
    surface_pressure = np.full(n_times, -9999.0)
    surface_temperature = np.full(n_times, -9999.0)
    surface_humidity = np.full(n_times, -9999.0)
    surface_precipitation = np.full(n_times, -9999.0)
    ground_elevation = np.full(n_times, -9999.0)
    
    instrument_flags = np.zeros(n_times, dtype=np.int32)
    instrument_flag_mapping = {inst: i for i, inst in enumerate(all_possible_instruments)}
    
    for t_idx, interval in enumerate(time_intervals):
        found_instruments = set()
        if 'wind_profiles' in interval and 'individual' in interval['wind_profiles']:
            found_instruments.update(interval['wind_profiles']['individual'].keys())
        if 'turbulence_profiles' in interval and 'individual' in interval['turbulence_profiles']:
            found_instruments.update(interval['turbulence_profiles']['individual'].keys())
        
        for instrument in found_instruments:
            if instrument in instrument_flag_mapping:
                instrument_flags[t_idx] |= (1 << instrument_flag_mapping[instrument])
        
        if 'ground_elevation' in interval:
            ground_elevation[t_idx] = interval['ground_elevation']
            
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
                    # The turbulence merge returns a flag for each of the five
                    # parameters it merges.
                    ti_quality_flags[t_idx, h_idx] = flag_dict.get('ti', 3)
                    tke_quality_flags[t_idx, h_idx] = flag_dict.get('tke', 3)
                    std_u_quality_flags[t_idx, h_idx] = flag_dict.get('std_u', 3)
                    std_v_quality_flags[t_idx, h_idx] = flag_dict.get('std_v', 3)
                    std_w_quality_flags[t_idx, h_idx] = flag_dict.get('std_w', 3)
                except IndexError:
                    continue
                    
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
            '_FillValue': -9999.0,
            'comment': ('May combine instruments using different TI '
                        'definitions; see the per-instrument '
                        'turbulence_intensity_* variable attributes for the '
                        'method used by each instrument.')
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
            'flag_meanings': 'good suspect bad no_data',
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
            'flag_meanings': 'good suspect bad no_data',
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
            'flag_meanings': 'good suspect bad no_data',
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
            'flag_meanings': 'good suspect bad no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data",
            'flag_2_description': "Bad data: exceeds physical limits (σᵤ/ū >1.0)",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),
        # TKE is flagged separately from TI: the merge judges each parameter
        # on its own, and the sigma_u/u bound in the TI flag is not a TKE
        # criterion.
        'qc_turbulent_kinetic_energy_merged': (['time', 'height'], tke_quality_flags, {
            'long_name': 'Quality control for merged turbulent kinetic energy',
            'description': 'Quality flags based on inter-instrument agreement and physical validation',
            'flag_values': [0, 1, 2, 3],
            'flag_meanings': 'good suspect bad no_data',
            'flag_0_description': "Good data",
            'flag_1_description': "Suspect data: inter-instrument disagreement",
            'flag_2_description': "Bad data: exceeds physical limits",
            'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
            '_FillValue': -9999
        }),
        # Flagged separately for the same reason as merged TKE.
        **{
            f'qc_{_c}_std_merged': (['time', 'height'], _flags, {
                'long_name': f'Quality control for merged {_c}-component wind speed standard deviation',
                'description': 'Quality flags based on inter-instrument agreement and physical validation',
                'flag_values': [0, 1, 2, 3],
                'flag_meanings': 'good suspect bad no_data',
                'flag_0_description': "Good data",
                'flag_1_description': "Suspect data: inter-instrument disagreement",
                'flag_2_description': "Bad data: exceeds physical limits",
                'flag_3_description': "No data: measurements unavailable or removed by QC filtering (physics-based limits, geometric constraints, or insufficient instrument availability)",
                '_FillValue': -9999
            })
            for _c, _flags in (('u', std_u_quality_flags),
                               ('v', std_v_quality_flags),
                               ('w', std_w_quality_flags))
        },

        
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
        'wderr': ('wind_direction_error', 'degrees', 'VAD-derived Wind Direction Uncertainty (fill value for ws <2.0 m/s, capped at 180°)')
    }

    for instrument in all_possible_instruments:
        instrument_name = instrument_mapping[instrument]
        for param, data_array in instrument_data[instrument].items():
            var_name, units, long_name_base = param_mappings[param]
            
            var_key = f"{var_name}_{instrument}"
            var_attrs = {
                'units': units,
                'long_name': f'{long_name_base} from {instrument_name}',
                '_FillValue': -9999.0
            }
            if param == 'ti':
                var_attrs['method'] = TI_METHODS.get(
                    (location, instrument),
                    'see project documentation for this instrument')
            # Near-surface records occupy only their configured measurement
            # height; the attribute repeats it as in-file metadata.
            if instrument.startswith('met_'):
                sonic_code = instrument.split('_')[1]
                sonic_height = (LOCATION_CONFIG.get(location, {})
                                .get('anemometer_heights', {}).get(sonic_code))
                if sonic_height is not None:
                    var_attrs['measurement_height_agl'] = float(sonic_height)
                    var_attrs['comment'] = (
                        'Single-height near-surface wind record: values '
                        'occupy only the grid level equal to '
                        'measurement_height_agl (m AGL) and are not '
                        'interpolated toward neighboring levels.')
                met_source = MET_SOURCES.get((location, instrument))
                if met_source:
                    var_attrs['source'] = (
                        f'{met_source}; input datastream '
                        f'{instrument_mapping[instrument]}')
            data_vars[var_key] = (['time', 'height'], data_array, var_attrs)
            
            qc_data = np.where(data_array == -9999.0, 1, 0).astype(np.int32)
            qc_var_key = f"qc_{var_name}_{instrument}"  
            data_vars[qc_var_key] = (['time', 'height'], qc_data, {
                'long_name': f'Quality control for {long_name_base.lower()} from {instrument_name}',
                'description': 'Simple data availability flag: 0=data present, 1=no data',
                'flag_values': [0, 1],  
                'flag_meanings': 'good_data missing_data', 
                '_FillValue': -9999
            })
    
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
            'summary': 'Quality-controlled wind and turbulence profiles from multiple instruments with statistical inter-instrument validation and physics-based filtering',
            'time_convention': 'Timestamps indicate start time of averaging period',
            'geospatial_lat_min': float(avg_lat) if avg_lat is not None else -9999.0,
            'geospatial_lat_max': float(avg_lat) if avg_lat is not None else -9999.0,
            'geospatial_lon_min': float(avg_lon) if avg_lon is not None else -9999.0,
            'geospatial_lon_max': float(avg_lon) if avg_lon is not None else -9999.0,
            # Prose only: the machine-readable QC lives in each variable's
            # flag_values, flag_meanings and ancillary_variables.
            'quality_control': 'Physics-based filtering (TKE>30 m2 s-2, TI>300%, multi-parameter thresholds: TKE>15 m2 s-2 + component std>10 m/s), geometric constraints (heights<100m removed for scanning lidar, min 4 beams for VAD), statistical inter-instrument validation with flagging',
            'vad_error_processing': 'VAD wind direction errors set to the fill value (-9999) for wind speeds <2.0 m/s, capped at 180 deg. VAD fit requires >=4 beams (dof>=1; sigma_ws is undefined for an exactly determined 3-beam fit) and fails for wind speed error >2 m/s or condition number >1000',
        }
    )

    # Apply CF-1.10 and ACDD-1.3 conventions.
    _apply_cf_conventions(ds, all_possible_instruments, instrument_mapping,
                          avg_lat, avg_lon,
                          dataset_id=os.path.splitext(os.path.basename(output_path))[0],
                          platform_name=get_location_display_name(location))

    # The requested path is honored as-is: the caller builds the canonical
    # name from the requested date, never from the first data interval.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_dir = output_dir or '.'

    # Stop a long campaign on its first day rather than its last. What
    # protects an existing product is the rename below, not this check.
    _require_free_space(write_dir, ds.nbytes)

    # Only this day's temp files: workers never share a date.
    for leaked in glob.glob(os.path.join(
            write_dir, f'.{os.path.basename(output_path)}.*.tmp')):
        try:
            os.remove(leaked)
        except OSError:
            pass

    # Temp file then os.replace, so the product on disk is only ever the
    # complete old file or the complete new one. The name is hidden and does
    # not end in .nc, so a leaked one cannot pass as a product. uuid rather
    # than tempfile.mkstemp, which creates the file 0600 and would carry that
    # mode onto the product.
    tmp_path = os.path.join(
        write_dir,
        f'.{os.path.basename(output_path)}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        ds.to_netcdf(tmp_path)
        # Flush before the rename: a crash must not leave an empty product.
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, output_path)
    except BaseException:
        # BaseException deliberately: an interrupt must take the temp file too.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    # Stale same-day twins from earlier partial-day runs are cleared only once
    # the new product is in place, so a failed write never leaves the day empty.
    twin_match = re.match(r'(.+\.windprof\.z01\.c1)\.(\d{8})\.\d{6}\.nc$',
                          os.path.basename(output_path))
    if twin_match:
        for stale in glob.glob(os.path.join(
                write_dir, f"{twin_match.group(1)}.{twin_match.group(2)}.*.nc")):
            if os.path.abspath(stale) == os.path.abspath(output_path):
                continue
            try:
                os.remove(stale)
                if verbose:
                    print(f"Removed same-day twin: {os.path.basename(stale)}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not remove {os.path.basename(stale)}: {e}")

    print(f"NetCDF file saved: {output_path}")

    if verbose:
        first_time = pd.to_datetime(times[0], unit='s', utc=True)
        print(f"Time range: {first_time} to {pd.to_datetime(times[-1], unit='s', utc=True)}")
        print(f"Height range: {heights.min():.1f} to {heights.max():.1f} m")
        print(f"Instruments included: {len(all_possible_instruments)}")

    return output_path