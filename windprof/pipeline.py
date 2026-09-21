"""Process a site over a date range into merged wind profiles, NetCDF
files, and diagnostic plots.

Public entry points (called by users and by ``process_parallel.py``):

  - ``process_wind_profiles(start_date, end_date, location)``: a date
    range, one NetCDF per processed day.
  - ``process_wind_profiles_for_date(date, ...)``: single-day processing.
  - ``process_full_day_with_hovmoller``: same plus diagnostic Hovmöller
    plots.
  - ``create_monthly_hovmoller_plots``: campaign-summary plotting.

Site- and instrument-specific behavior is read from ``config.py``; this
module contains no site logic itself.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from .config import (normalize_location, SITE_CODES, RESULTS_BASE_PATH,
                     DATA_BASE_PATH, get_location_display_name)
from .merging import process_combined_profiles
from .discovery_and_export import (discover_instrument_files_for_date_range,
                                   save_results_to_netcdf)
from .plotting import (plot_combined_profiles, create_daily_hovmoller_plots,
                       create_full_daily_surface_met_plot)

def process_and_plot_combined_profiles(instrument_configs, start_time, end_time,
                                 location='nantucket',
                                 output_dir='./combined_plots',
                                 availability_threshold=0.5,
                                 max_plots=None,
                                 show_availability=False,
                                 verbose=False,
                                 **kwargs):
    """Process all instruments and create per-interval profile plots.
    Thin wrapper combining process_combined_profiles + plot_combined_profiles
    with optional surface met plotting.

    Returns (combined_results, saved_plot_files).
    """
    combined_results = process_combined_profiles(
        instrument_configs, start_time, end_time, location=location,
        availability_threshold=availability_threshold,
        include_availability=show_availability,
        verbose=verbose,
        **kwargs
    )

    if not combined_results.get('time_intervals'):
        print("No combined data to plot")
        return combined_results, []

    plots = _create_plots_for_results(
        combined_results, instrument_configs, start_time, end_time,
        location=location, output_dir=output_dir,
        availability_threshold=availability_threshold,
        max_plots=max_plots, show_availability=show_availability,
        verbose=verbose)

    return combined_results, plots


def _create_plots_for_results(combined_results, instrument_configs, start_time, end_time,
                              location, output_dir, availability_threshold,
                              max_plots, show_availability, verbose):
    """Per-interval wind profile plots plus the optional daily surface-met
    plot. Kept separate from the merge so callers can save the NetCDF first."""
    wind_plots = plot_combined_profiles(
        combined_results, output_dir,
        max_plots=max_plots,
        show_availability=show_availability,
        availability_threshold=availability_threshold
    )
    plots = wind_plots

    if 'surface_met' in instrument_configs:
        try:
            if verbose:
                print("\nCreating full daily surface met plot...")

            if isinstance(start_time, str):
                start_dt = pd.to_datetime(start_time)
            else:
                start_dt = start_time
            if isinstance(end_time, str):
                end_dt = pd.to_datetime(end_time)
            else:
                end_dt = end_time

            date_str = start_dt.strftime('%Y-%m-%d')

            surface_met_file = instrument_configs['surface_met']['filename']
            base_dir = str(Path(surface_met_file).parent.parent)

            surface_plot = create_full_daily_surface_met_plot(
                date_str=date_str,
                base_dir=base_dir,
                output_dir=output_dir,
                location=location,
                analysis_start_time=start_dt,
                analysis_end_time=end_dt,
                verbose=verbose
            )

            if surface_plot:
                plots.append(surface_plot)
                if verbose:
                    print(f"Surface met plot saved: {surface_plot}")
            else:
                if verbose:
                    print("No surface met plot created (no data available)")

        except Exception as e:
            print(f"Failed to create full daily surface met plot: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    else:
        if verbose:
            print("No surface_met instrument found - skipping surface met plot")

    return plots

def process_wind_profiles_for_date(date, base_dir, start_time=None, end_time=None,
                             location='nantucket',
                             output_dir=None, availability_threshold=0.5,
                             show_availability=False, max_plots=None,
                             create_hovmoller=True, save_netcdf=True,
                             force_reprocess=False,
                             verbose=False, **kwargs):
    """
    Complete wind and turbulence profile processing for a specific date.

    Parameters:
    -----------
    date : str or datetime
        Target date (e.g., '2024-07-03').
    base_dir : str
        Root directory to search for instrument files.
    start_time, end_time : str or datetime, optional
        Analysis window, 'YYYY-MM-DD HH:MM:SS' or 'HH:MM:SS'.
        Default: the whole date, 00:00:00 to 23:59:00.
    location : str, optional
        Location name ('nantucket', 'block island', etc.).
    output_dir : str, optional
        Output directory for plots (default: './wind_profiles_{date}').
    availability_threshold : float, optional
        Minimum fraction of valid data required per interval (0-1).
    show_availability : bool, optional
        Include availability panel in plots (default: False).
    max_plots : int, optional
        Maximum number of plots to generate (default: all).
    verbose : bool, optional
        Print detailed processing information (default: False).
    **kwargs
        Additional parameters passed to processing functions.

    Returns:
    --------
    dict
        {
            'results': combined_results_dict,
            'plots': list_of_saved_plot_files,
            'hovmoller_plot': path_to_hovmoller_png or None,
            'netcdf_file': path_to_saved_product or None,
            'instruments_found': list_of_instrument_names,
            'processing_summary': summary_stats
        }
    """
    if isinstance(date, str):
        date_obj = pd.to_datetime(date).date()
        date_str = date_obj.strftime('%Y-%m-%d')
    else:
        date_obj = date.date() if hasattr(date, 'date') else date
        date_str = date_obj.strftime('%Y-%m-%d')

    # A bare 'HH:MM:SS' is qualified with the target date.
    if start_time is None:
        start_time = f"{date_str} 00:00:00"
    elif isinstance(start_time, str) and ':' in start_time and len(start_time) <= 8:
        start_time = f"{date_str} {start_time}"

    if end_time is None:
        end_time = f"{date_str} 23:59:00"
    elif isinstance(end_time, str) and ':' in end_time and len(end_time) <= 8:
        end_time = f"{date_str} {end_time}"

    location = normalize_location(location)
    site_code = SITE_CODES.get(location, location[:4])

    if output_dir is None:
        year = pd.to_datetime(date_str).year
        month = pd.to_datetime(date_str).month
        yyyymm = f"{year}{month:02d}"
        output_dir = os.path.join(RESULTS_BASE_PATH, site_code, yyyymm, date_str.replace('-', ''))

    if verbose:
        print(f"Processing Wind Profiles for {date_str}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Location: {get_location_display_name(location)}")
        print(f"Searching for files in: {base_dir}")
        print(f"Output directory: {output_dir}")

    default_summary = {
        'date_processed': date_str,
        'location': location,
        'time_range': f"{start_time} to {end_time}",
        'instruments_found': 0,
        'time_intervals_processed': 0,
        'surface_met_windows': 0,
        'wind_plots_created': 0,
        'surface_met_plots_created': 0,
        'plots_created': 0,
        'hovmoller_created': False,
        'netcdf_saved': False,
        'output_directory': output_dir,
        'status': 'unknown',
        'error': None
    }

    try:
        configs = discover_instrument_files_for_date_range(date_obj, date_obj, location=location)
        date_str = date_obj.strftime('%Y-%m-%d')
        if date_str in configs:
            configs = configs[date_str]
        else:
            configs = {}

        if not configs:
            summary = default_summary.copy()
            summary.update({
                'status': 'failed',
                'error': 'No instrument files found for date'
            })
            return {
                'results': None,
                'plots': [],
                'hovmoller_plot': None,
                'netcdf_file': None,
                'instruments_found': [],
                'processing_summary': summary
            }

        default_summary['instruments_found'] = len(configs)
        if verbose:
            print(f"Found {len(configs)} instruments: {list(configs.keys())}")

    except Exception as e:
        summary = default_summary.copy()
        summary.update({
            'status': 'failed',
            'error': f'File discovery failed: {e}'
        })
        return {
            'results': None,
            'plots': [],
            'hovmoller_plot': None,
            'netcdf_file': None,
            'instruments_found': [],
            'processing_summary': summary
        }

    # Merge, save, then plot, in that order: the NetCDF is the product, so
    # it must be on disk before anything that can fail cosmetically runs.
    try:
        results = process_combined_profiles(
            configs,
            start_time=start_time,
            end_time=end_time,
            location=location,
            availability_threshold=availability_threshold,
            include_availability=show_availability,
            verbose=verbose,
            **kwargs
        )

        hovmoller_plot = None
        netcdf_file = None
        save_error = None
        plots = []

        if save_netcdf and results and results.get('time_intervals'):
            try:
                if verbose:
                    print(f"\nSaving NetCDF file...")

                # Filename stamps the requested date at 000000, never the first
                # data interval, so a partial day cannot mint a second name.
                date_formatted = date_str.replace('-', '')
                netcdf_filename = f"{site_code}.windprof.z01.c1.{date_formatted}.000000.nc"
                netcdf_path = os.path.join(output_dir, netcdf_filename)

                saved_path = save_results_to_netcdf(results, netcdf_path, location=location, verbose=verbose)
                if saved_path:
                    netcdf_file = saved_path
                    default_summary['netcdf_saved'] = True
                    if verbose:
                        print(f"✓ NetCDF saved: {netcdf_filename}")
                else:
                    save_error = 'save_results_to_netcdf returned no path'
                    print("✗ NetCDF save failed")
            except Exception as e:
                save_error = str(e)
                print(f"Failed to save NetCDF: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        # Plotting is best-effort once the product is on disk.
        if results and results.get('time_intervals'):
            try:
                plots = _create_plots_for_results(
                    results, configs, start_time, end_time,
                    location=location, output_dir=output_dir,
                    availability_threshold=availability_threshold,
                    max_plots=max_plots, show_availability=show_availability,
                    verbose=verbose)
            except Exception as e:
                print(f"Plotting failed (product already saved): {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        if create_hovmoller and results and results.get('time_intervals'):
            try:
                if verbose:
                    print(f"\nCreating Hovmöller plot...")

                date_formatted = date_str.replace('-', '')
                hovmoller_plot = create_daily_hovmoller_plots(
                    results, output_dir, date_formatted
                )
                if hovmoller_plot:
                    default_summary['hovmoller_created'] = True
                    if verbose:
                        print(f"✓ Hovmöller plot saved: {os.path.basename(hovmoller_plot)}")
                else:
                    if verbose:
                        print("✗ Hovmöller plot creation failed")
            except Exception as e:
                print(f"Failed to create Hovmöller plot: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        surface_met_count = 0
        if results and results.get('time_intervals'):
            surface_met_count = sum(1 for interval in results.get('time_intervals', [])
                                   if 'surface_met' in interval and
                                   not all(np.isnan(list(interval['surface_met'].values()))))

        surface_met_plots = [p for p in plots if 'surface_met' in os.path.basename(p)]
        wind_plots = [p for p in plots if 'surface_met' not in os.path.basename(p)]

        summary = default_summary.copy()
        summary.update({
            'instruments_found': len(configs),
            'time_intervals_processed': len(results.get('time_intervals', [])) if results else 0,
            'surface_met_windows': surface_met_count,
            'wind_plots_created': len(wind_plots),
            'surface_met_plots_created': len(surface_met_plots),
            'plots_created': len(plots),
            'status': 'success'
        })

        # "Success" means the product exists on disk: a requested save that
        # produced no file fails the day however the merge went.
        if save_netcdf and results and results.get('time_intervals') and netcdf_file is None:
            summary['status'] = 'failed'
            summary['error'] = f'NetCDF save failed: {save_error}'

        if verbose:
            print(f"\nProcessing complete!")
            print(f"- Location: {get_location_display_name(location)}")
            print(f"- Instruments: {summary['instruments_found']}")
            print(f"- Time intervals: {summary['time_intervals_processed']}")
            print(f"- Surface met windows: {summary['surface_met_windows']}")
            print(f"- Wind plots: {summary['wind_plots_created']}")
            print(f"- Surface met plots: {summary['surface_met_plots_created']}")
            print(f"- Hovmöller created: {summary['hovmoller_created']}")
            print(f"- NetCDF saved: {summary['netcdf_saved']}")
            print(f"- Output directory: {summary['output_directory']}")

        return {
            'results': results,
            'plots': plots,
            'hovmoller_plot': hovmoller_plot,
            'netcdf_file': netcdf_file,
            'instruments_found': list(configs.keys()),
            'processing_summary': summary
        }

    except Exception as e:
        summary = default_summary.copy()
        summary.update({
            'instruments_found': len(configs),
            'status': 'failed',
            'error': f'Processing failed: {e}'
        })
        if verbose:
            print(f"\nProcessing failed: {e}")
            import traceback
            traceback.print_exc()

        return {
            'results': None,
            'plots': [],
            'hovmoller_plot': None,
            'netcdf_file': None,
            'instruments_found': list(configs.keys()),
            'processing_summary': summary
        }

def process_wind_profiles(start_date, end_date, location='nantucket',
                         base_dir=None, output_dir=None,
                         start_time=None, end_time=None,
                         availability_threshold=0.5, show_availability=False,
                         max_plots_per_day=None, create_hovmoller=True,
                         save_netcdf=True, force_reprocess=False,
                         verbose=False, **kwargs):
    """
    Process a date range one day at a time, releasing memory between days.

    Each day's results are written to NetCDF on disk rather than accumulated
    in memory, so multi-month campaigns are memory-efficient. 
    No aggregate return value; outputs live in output_dir.

    Supported locations: block_island, nantucket, rhode_island, cape_cod.
    """
    import gc

    if isinstance(start_date, str):
        start_obj = pd.to_datetime(start_date)
    else:
        start_obj = start_date
    if isinstance(end_date, str):
        end_obj = pd.to_datetime(end_date)
    else:
        end_obj = end_date

    # Default input and output paths use the START month of the range.
    year = start_obj.year
    month = start_obj.month
    yyyymm = f"{year}{month:02d}"

    location = normalize_location(location)
    loc_code = SITE_CODES.get(location, location[:4])

    if base_dir is None:
        base_dir = f"{DATA_BASE_PATH}{yyyymm}"

    date_range = pd.date_range(start=start_obj, end=end_obj, freq='D')

    if output_dir is None:
        output_dir = os.path.join(RESULTS_BASE_PATH, loc_code, yyyymm)

    print(f"Processing {len(date_range)} days individually for {location} (memory-efficient mode)")
    print(f"Input data from: {base_dir}")
    print(f"Results will be saved to NetCDF files in: {output_dir}")

    successful_days = 0
    failed_days = 0

    for date_obj in date_range:
        date_str = date_obj.strftime('%Y-%m-%d')
        print(f"\n{'='*50}")
        print(f"Processing {date_str} - {location} ({successful_days + failed_days + 1}/{len(date_range)})")
        print(f"{'='*50}")

        date_formatted = date_str.replace('-', '')
        date_output_dir = os.path.join(output_dir, f"{date_formatted}")

        try:
            day_result = process_wind_profiles_for_date(
                date_str,
                base_dir=base_dir,
                start_time=start_time,
                end_time=end_time,
                location=location,
                output_dir=date_output_dir,
                availability_threshold=availability_threshold,
                show_availability=show_availability,
                max_plots=max_plots_per_day,
                create_hovmoller=create_hovmoller,
                save_netcdf=save_netcdf,
                force_reprocess=force_reprocess,
                verbose=verbose,
                **kwargs
            )

            day_status = (day_result or {}).get('processing_summary', {}).get('status')
            if day_result and day_result.get('results') and day_status == 'success':
                successful_days += 1
                if verbose:
                    print(f"✓ {date_str} processed successfully")
                    if day_result.get('netcdf_file'):
                        print(f"  NetCDF saved: {os.path.basename(day_result['netcdf_file'])}")
                    if day_result.get('hovmoller_plot'):
                        print(f"  Hovmöller saved: {os.path.basename(day_result['hovmoller_plot'])}")
            else:
                failed_days += 1
                print(f"✗ {date_str} failed - no results generated")

            del day_result
            gc.collect()

        except Exception as e:
            failed_days += 1
            print(f"✗ {date_str} failed with error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Processing Summary for {location}:")
    print(f"Successful: {successful_days}/{len(date_range)} days")
    print(f"Failed: {failed_days}/{len(date_range)} days")
    print(f"All results saved to individual NetCDF files and plots")
    print(f"{'='*50}")

def process_full_day_with_hovmoller(date, base_dir, output_dir=None,
                              location='nantucket',
                              create_hovmoller=True,
                              availability_threshold=0.5, verbose=False, **kwargs):
    """Process a full day and optionally produce a Hovmöller diagram.
    Convenience wrapper around process_wind_profiles_for_date that suppresses
    per-interval profile plots (max_plots=0) and exposes the Hovmöller path
    directly in the return dict.
    """
    if isinstance(date, str):
        date_obj = pd.to_datetime(date)
    else:
        date_obj = date
    date_str = date_obj.strftime('%Y%m%d')

    if output_dir is None:
        output_dir = f"./wind_profiles_{date_str}"

    output = process_wind_profiles_for_date(
        date=date,
        base_dir=base_dir,
        start_time=None,  # Full day (00:00)
        end_time=None,    # Full day (23:59)
        location=location,
        output_dir=output_dir,
        availability_threshold=availability_threshold,
        show_availability=False,
        max_plots=0,  # suppress per-interval profile plots; Hovmoller is the product here
        verbose=verbose,
        **kwargs
    )

    hovmoller_plot = None
    if create_hovmoller and output['results']:
        try:
            if verbose:
                print(f"\nCreating Hovmöller plot...")
            hovmoller_plot = create_daily_hovmoller_plots(
                output['results'], output_dir, date_str
            )
            if hovmoller_plot and verbose:
                print(f"Hovmöller plot saved: {hovmoller_plot}")
        except Exception as e:
            print(f"Failed to create Hovmöller plot: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    return {
        'daily_results': output['results'],
        'individual_plots': output['plots'],
        'hovmoller_plot': hovmoller_plot,
        'instruments_found': output['instruments_found'],
        'processing_summary': output['processing_summary']
    }

def create_monthly_hovmoller_plots(start_date, end_date, output_dir=None, location='nantucket',
                                 figsize=(15, 20), dpi=150, verbose=False):
    """Create a time–height Hovmöller plot for each day in the date range.
    Runs the full pipeline per day but suppresses per-interval profile plots.
    Returns {date_str: path_to_hovmoller_png}.
    """
    all_configs = discover_instrument_files_for_date_range(start_date, end_date, location=location)

    if not all_configs:
        print("No files found for the specified date range.")
        return {}

    if output_dir is None:
        start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
        output_dir = f"./hovmoller_plots_{start_str}_to_{end_str}"

    os.makedirs(output_dir, exist_ok=True)
    hovmoller_plots = {}

    for date_str, day_configs in all_configs.items():
        print(f"Creating Hovmöller plot for {date_str}...")

        try:
            results, _ = process_and_plot_combined_profiles(
                day_configs,
                start_time=f"{date_str} 00:00:00",
                end_time=f"{date_str} 23:59:00",
                output_dir=output_dir,
                max_plots=0,  # Hovmoller-only pass; skip per-interval profile plots
                location=location,
                verbose=verbose
            )

            if results and results.get('time_intervals'):
                date_formatted = date_str.replace('-', '')
                hovmoller_plot = create_daily_hovmoller_plots(
                    results, output_dir, date_formatted,
                    figsize=figsize, dpi=dpi
                )
                if hovmoller_plot:
                    hovmoller_plots[date_str] = hovmoller_plot
            else:
                print(f"  No data available for {date_str}")

        except Exception as e:
            print(f"  Failed to create Hovmöller for {date_str}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    return hovmoller_plots
