"""Parallel driver for per-day WINDPROF reprocessing.

Runs one worker per date over a date range, skipping dates whose
canonical output NetCDF already exists.

Example::

    python -m windprof.process_parallel rhode_island \
        --start-date 2025-05-17 --end-date 2025-09-04 \
        --n-processes 4
"""

import os
import gc
import argparse
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from .config import SITE_CODES, RESULTS_BASE_PATH, DATA_BASE_PATH
from .discovery_and_export import preflight_check_inputs
from .pipeline import process_wind_profiles_for_date

def check_if_processed(date_str, location):
    """Return True if the canonical daily netCDF already exists for this date.

    Only the ``...000000.nc`` name counts: a partial-day file such as
    ``...013000.nc`` must not mark the day processed, since its output is
    incomplete.
    """
    site_code = SITE_CODES[location]
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    yyyymm = date_obj.strftime('%Y%m')
    yyyymmdd = date_obj.strftime('%Y%m%d')
    expected_file = f"{RESULTS_BASE_PATH}{site_code}/{yyyymm}/{yyyymmdd}/{site_code}.windprof.z01.c1.{yyyymmdd}.000000.nc"
    return os.path.exists(expected_file)

def get_dates_to_process(start_date, end_date, location, skip_existing=True):
    """Build (date_str, year, month) tuples for dates still needing processing."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates_to_process = []
    dates_skipped = []
    print("Scanning for existing files...")
    for date_obj in date_range:
        date_str = date_obj.strftime('%Y-%m-%d')
        if skip_existing and check_if_processed(date_str, location):
            dates_skipped.append(date_str)
            if len(dates_skipped) <= 10:
                print(f"  Skipping {date_str} - already processed")
            elif len(dates_skipped) == 11:
                print(f"  ... (suppressing further skip messages)")
        else:
            dates_to_process.append((date_str, date_obj.year, date_obj.month))
    print(f"Scan complete: {len(dates_skipped)} already processed, {len(dates_to_process)} need processing")
    return dates_to_process

# Location is passed via a module-level global rather than as a worker arg so
# Pool.map can use a single-argument callable and fork()-inherit the value.
_LOCATION = None

def process_single_date(args):
    """Worker entry point: process one date and return a (date, success, error) tuple."""
    date_str, year, month = args
    location = _LOCATION
    print(f"Processing {date_str} [PID: {os.getpid()}]")
    try:
        result = process_wind_profiles_for_date(
            date_str,
            base_dir=f"{DATA_BASE_PATH}{year}{month:02d}",
            location=location,
            max_plots=1,
            create_hovmoller=False,
            save_netcdf=True,
            verbose=False,
            force_reprocess=True
        )
        # Success requires a product on disk, not just merge output, so a day
        # whose NetCDF save failed is retried.
        if result and result.get('netcdf_file'):
            print(f"  Success: {date_str}")
            return (date_str, True, None)
        else:
            reason = (result or {}).get('processing_summary', {}).get('error') or "No results"
            print(f"  Failed: {date_str}")
            return (date_str, False, reason)
    except Exception as e:
        print(f"  Error {date_str}: {e}")
        return (date_str, False, str(e))
    finally:
        # Workers are long-lived across many dates; drop the day's
        # xarray/netCDF buffers before the next one.
        if 'result' in locals():
            del result
        gc.collect()

def process_site(location, start_date, end_date, n_processes=1, skip_existing=True,
                 allow_missing_inputs=False):
    """Fan out daily processing across a worker Pool for a single site.

    Refuses to start if a configured datastream resolves no files: a missing
    instrument is silently omitted rather than raised, so the run would
    overwrite good products with instrument-less ones and report success.
    ``allow_missing_inputs`` overrides.
    """
    global _LOCATION
    _LOCATION = location

    ok, _ = preflight_check_inputs(start_date, end_date, location=location)
    if not ok and not allow_missing_inputs:
        print("\nRefusing to start: at least one configured datastream resolved no files.")
        print("Fix the input path or archive mount, or pass --allow-missing-inputs "
              "to process without it.")
        return
    if not ok:
        print("\nProceeding without a missing datastream (--allow-missing-inputs).")

    print(f"Scanning for dates between {start_date} and {end_date}...")
    date_args = get_dates_to_process(start_date, end_date, location, skip_existing)
    if not date_args:
        print("No dates need processing - all appear to be already done!")
        return
    print(f"Found {len(date_args)} dates that need processing")
    print(f"Starting parallel processing using {n_processes} processes...")
    successful = 0
    failed = 0
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_date, date_args)
    for date_str, success, error in results:
        if success:
            successful += 1
        else:
            failed += 1
            if error:
                print(f"Final error summary - {date_str}: {error}")
    print(f"\n=== FINAL RESULTS ===")
    print(f"Processed: {len(date_args)} dates")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if len(date_args) > 0:
        print(f"Success rate: {successful/len(date_args)*100:.1f}%")

def process_multiple_periods(location, periods, n_processes=1):
    """Process multiple date ranges in sequence."""
    for start_date, end_date in periods:
        print(f"\n{'='*60}")
        print(f"Processing period: {start_date} to {end_date}")
        print(f"{'='*60}")
        process_site(location, start_date, end_date, n_processes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process wind profiles for a site in parallel')
    parser.add_argument('location', choices=list(SITE_CODES.keys()),
                        help='Site to process')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-processes', type=int, default=1, help='Number of parallel processes (default: 1)')
    parser.add_argument('--no-skip', action='store_true', help='Reprocess all dates even if output exists')
    parser.add_argument('--allow-missing-inputs', action='store_true',
                        help='Process even if a configured datastream resolves no files '
                             '(default is to refuse, since a missing instrument is silently '
                             'omitted rather than raising)')
    args = parser.parse_args()
    process_site(args.location, args.start_date, args.end_date,
                 n_processes=args.n_processes, skip_existing=not args.no_skip,
                 allow_missing_inputs=args.allow_missing_inputs)
