"""Diagnostic and summary plotting for WINDPROF output.

Profile, Hovmoller and surface-met figures for sanity-checking processed
output and for campaign summaries, not publication-ready plots (those live
in analysis notebooks). All routines are side-effect-only: they write PNGs
to the output directory given and return the paths.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import os
from .config import get_location_display_name, TIME_WINDOW_MINUTES
from .merging import process_surface_met_for_date


def plot_parameter(ax, param, title, xlabel, merged_profile, individual_profiles,
                  contributing_instruments, colors):
    """Plot one profile parameter: per-instrument dashed scatter with the merged profile overlaid in black"""
    for i, instrument in enumerate(contributing_instruments):
        if instrument in individual_profiles:
            profile = individual_profiles[instrument]
            color = colors[i % len(colors)]

            sorted_heights = sorted(profile.keys())
            heights = np.array(sorted_heights)
            values = np.array([profile[h].get(param) for h in sorted_heights])

            # Profile dicts carry None for a missing parameter, which yields an object array.
            if values.dtype == 'object':
                values = np.array([np.nan if v is None else v for v in values.flat]).reshape(values.shape)
                values = values.astype(float)

            valid_mask = ~np.isnan(values)
            heights = heights[valid_mask]
            values = values[valid_mask]
            
            if len(heights) > 0:
                ax.scatter(values, heights, color=color, alpha=0.7,
                         label=instrument.upper(), s=20)
                ax.plot(values, heights, color=color, linestyle='--', alpha=0.5)
    
    sorted_heights = sorted(merged_profile.keys())
    heights = np.array(sorted_heights)
    values = np.array([merged_profile[h].get(param) for h in sorted_heights])

    if values.dtype == 'object':
        values = np.array([np.nan if v is None else v for v in values.flat]).reshape(values.shape)
        values = values.astype(float)
    
    valid_mask = ~np.isnan(values)
    heights = heights[valid_mask]
    values = values[valid_mask]
    
    if len(heights) > 0:
        ax.plot(values, heights, color='black', marker='o',
               linestyle='-', linewidth=2, markersize=3, label='AVERAGE')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Height (m AGL)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

def plot_availability_panel(ax, interval, colors, markers, availability_threshold):
    """Plot per-instrument data availability vs. height for one time interval"""
    availability_data = interval.get('availability', {})
    if not availability_data:
        ax.text(0.5, 0.5, 'No availability data',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Data Availability')
        return
    
    all_heights = []
    all_availabilities = []
    all_meets_threshold = []
    all_colors = []
    all_markers = []
    all_instruments = []
    
    for i, (instrument, instrument_avail) in enumerate(availability_data.items()):
        height_availability = instrument_avail.get('height_availability', {})
        if not height_availability:
            continue
        
        heights_list = list(height_availability.keys())
        availabilities_list = [info['availability'] for info in height_availability.values()]
        meets_threshold_list = [info['meets_threshold'] for info in height_availability.values()]
        
        all_heights.extend(heights_list)
        all_availabilities.extend(availabilities_list)
        all_meets_threshold.extend(meets_threshold_list)
        all_colors.extend([colors[i % len(colors)]] * len(heights_list))
        all_markers.extend([markers[i % len(markers)]] * len(heights_list))
        all_instruments.extend([instrument] * len(heights_list))
    
    if not all_heights:
        ax.text(0.5, 0.5, 'No height availability data',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Data Availability')
        return
    
    heights_arr = np.array(all_heights)
    avail_arr = np.array(all_availabilities)
    meets_arr = np.array(all_meets_threshold)
    colors_arr = np.array(all_colors)
    markers_arr = np.array(all_markers)
    instruments_arr = np.array(all_instruments)

    # Points meeting the availability threshold get full alpha; others fade to indicate rejection.
    alphas_arr = np.where(meets_arr, 0.7, 0.3)

    unique_markers = np.unique(markers_arr)
    unique_instruments = np.unique(instruments_arr)

    legend_added = set()

    for marker in unique_markers:
        marker_mask = markers_arr == marker
        if not np.any(marker_mask):
            continue

        # Markers are assigned per-instrument, so any masked row gives the instrument label.
        instruments_for_marker = instruments_arr[marker_mask]
        instrument = instruments_for_marker[0]

        scatter = ax.scatter(avail_arr[marker_mask], heights_arr[marker_mask],
                           c=colors_arr[marker_mask], marker=marker,
                           alpha=alphas_arr[marker_mask], s=20,
                           label=instrument.upper() if instrument not in legend_added else "")
        
        legend_added.add(instrument)
    
    ax.axvline(x=availability_threshold, color='gray', linestyle='--',
               alpha=0.7, label=f'Threshold ({availability_threshold:.0%})')

    ax.set_xlabel('Data Availability')
    ax.set_ylabel('Height (m AGL)')
    ax.set_title('Data Availability')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Flip x-axis so high-availability (good) points sit near the wind/turbulence panels on the left.
    ax.set_xlim(1.05, -0.05)

def plot_combined_profiles(combined_results, output_dir, max_plots=None,
                          show_availability=False, availability_threshold=0.5,
                          figsize=(25, 10), date_prefix=None):
    """Create per-interval 8-panel wind+turbulence profile figures (+ optional availability panel).

    max_plots=0 produces no figures; max_plots=None plots every interval. date_prefix is used
    when processing multiple days into a shared output directory to avoid filename collisions.
    """
    os.makedirs(output_dir, exist_ok=True)
    time_intervals = combined_results.get('time_intervals', [])

    if not time_intervals:
        print("No time intervals to plot")
        return []

    if max_plots is not None and max_plots == 0:
        intervals_to_plot = []
    elif max_plots is not None and max_plots > 0:
        intervals_to_plot = time_intervals[:max_plots]
    else:
        intervals_to_plot = time_intervals

    print(f"Creating combined plots for {len(intervals_to_plot)} time intervals...")
    
    # Fixed cycles so an instrument keeps the same color and marker across a run's figures.
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    saved_files = []

    # Optional 9th panel widens the figure proportionally to preserve per-panel aspect ratio.
    n_panels = 9 if show_availability else 8
    if show_availability:
        figsize = (figsize[0] * 9/8, figsize[1])
    
    for interval in intervals_to_plot:
        interval_time_epoch = interval['time']
        interval_time_utc = pd.to_datetime(interval_time_epoch, unit='s', utc=True)
        wind_data = interval['wind_profiles']
        turb_data = interval['turbulence_profiles']
        
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        plot_parameter(axes[0], 'ws', 'Wind Speed', 'WS (m/s)',
                      wind_data['merged'], wind_data['individual'],
                      wind_data['contributing_instruments'], colors)
        plot_parameter(axes[1], 'wd', 'Wind Direction', 'WD (degrees)',
                      wind_data['merged'], wind_data['individual'],
                      wind_data['contributing_instruments'], colors)
        plot_parameter(axes[2], 'w', 'Vertical Velocity', 'W (m/s)',
                      wind_data['merged'], wind_data['individual'],
                      wind_data['contributing_instruments'], colors)
        
        plot_parameter(axes[3], 'ti', 'Turbulence Intensity', 'TI (-)',
                      turb_data['merged'], turb_data['individual'],
                      turb_data['contributing_instruments'], colors)
        plot_parameter(axes[4], 'tke', 'Turbulent Kinetic Energy', 'TKE (m²/s²)',
                      turb_data['merged'], turb_data['individual'],
                      turb_data['contributing_instruments'], colors)
        plot_parameter(axes[5], 'std_u', 'U-Component Std Dev', 'σᵤ (m/s)',
                      turb_data['merged'], turb_data['individual'],
                      turb_data['contributing_instruments'], colors)
        plot_parameter(axes[6], 'std_v', 'V-Component Std Dev', 'σᵥ (m/s)',
                      turb_data['merged'], turb_data['individual'],
                      turb_data['contributing_instruments'], colors)
        plot_parameter(axes[7], 'std_w', 'W-Component Std Dev', 'σᵩ (m/s)',
                      turb_data['merged'], turb_data['individual'],
                      turb_data['contributing_instruments'], colors)
        
        if show_availability:
            plot_availability_panel(axes[8], interval, colors, markers, availability_threshold)

        # Share one y-range across panels so height comparisons are consistent.
        all_heights = set()
        all_heights.update(wind_data['merged'].keys())
        all_heights.update(turb_data['merged'].keys())

        if all_heights:
            min_height = min(all_heights)
            max_height = max(all_heights)
            y_padding = (max_height - min_height) * 0.1 if max_height > min_height else 100
            for ax in axes:
                ax.set_ylim(min_height - y_padding, max_height + y_padding)

        all_instruments = set(wind_data['contributing_instruments'] + turb_data['contributing_instruments'])
        title = f'Combined Wind & Turbulence Profiles - {interval_time_utc.strftime("%Y-%m-%d %H:%M")} UTC'
        subtitle = f'Instruments: {", ".join(sorted(all_instruments))}'
        plt.suptitle(f'{title}\n{subtitle}', fontsize=16, y=0.98)
        
        plt.tight_layout()

        # Leave headroom for the two-line suptitle (title + instrument list)
        plt.subplots_adjust(top=0.88)

        if date_prefix:
            filename = f"{date_prefix}_combined_profile_{interval_time_utc.strftime('%Y-%m-%d_%H%M')}.png"
        else:
            filename = f"wind_profile_{interval_time_utc.strftime('%Y-%m-%d_%H%M')}.png"
        
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_files.append(filepath)

    return saved_files

def plot_surface_met_timeseries(surface_met_data, start_time, end_time,
                                location='nantucket', save_path=None, figsize=(12, 10)):
    """Four-panel (pressure, temperature, RH, precipitation) surface-met time series for one analysis window"""
    if not surface_met_data:
        print("No surface met data to plot")
        return None

    filtered_data = []
    for time_key, data in surface_met_data.items():
        if start_time <= time_key <= end_time:
            filtered_data.append({
                'datetime': data['time'],
                'pressure': data['pressure'],
                'temperature': data['temperature'],
                'relative_humidity': data['relative_humidity'],
                'precipitation': data['precipitation']
            })

    if not filtered_data:
        print("No surface met data in specified time range")
        return None

    df = pd.DataFrame(filtered_data).set_index('datetime')

    fig, axes = plt.subplots(4, 1, figsize=figsize)
    date_str = start_time.strftime('%Y-%m-%d')
    time_range_str = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} UTC"
    
    location_name = get_location_display_name(location)
    fig.suptitle(f'Surface Meteorological Conditions - {location_name}\n{date_str} {time_range_str}',
                 fontsize=14, fontweight='bold')

    axes[0].plot(df.index, df['pressure'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_ylabel('Pressure\n(hPa)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Surface Air Pressure')

    axes[1].plot(df.index, df['temperature'], 'r-', linewidth=2, marker='o', markersize=4)
    axes[1].set_ylabel('Temperature\n(°C)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Air Temperature')

    axes[2].plot(df.index, df['relative_humidity'], 'g-', linewidth=2, marker='o', markersize=4)
    axes[2].set_ylabel('Relative\nHumidity (%)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Relative Humidity')

    # Bars are 8 min wide against the 10 min reporting interval so they don't touch.
    axes[3].bar(df.index, df['precipitation'], width=pd.Timedelta(minutes=8),
               alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    axes[3].set_ylabel('Precipitation\n(mm)', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Precipitation')
    axes[3].set_xlabel('Time (UTC)', fontweight='bold')

    # 30-minute ticks match the typical analysis-window length.
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Surface met plot saved to: {save_path}")

    plt.close()
    return save_path

def create_daily_hovmoller_plots(daily_results, output_dir, date_str,
                            figsize=(12, 16), dpi=200, min_timepoints=5, height_buffer=50):
    """Daily Hovmöller (time-height) diagrams for wind speed, direction, vertical velocity (w), and TI.

    Parameters
    ----------
    min_timepoints : int
        Minimum number of time intervals at a height before that height is considered
        well-sampled; drives the adaptive y-axis maximum so that the top of the plot
        isn't dominated by sparse, noisy high-altitude bins.
    height_buffer : float
        Padding (m) added above the highest well-sampled height so the topmost data
        contours don't sit flush against the axis.
    """

    os.makedirs(output_dir, exist_ok=True)
    time_intervals = daily_results.get('time_intervals', [])
    if not time_intervals:
        print("No time intervals found in daily results")
        return None
    print(f"Creating Hovmöller plot for {len(time_intervals)} time intervals")
    
    # Union of heights: wind and turbulence coverage can differ for the same instrument.
    all_heights = set()
    for interval in time_intervals:
        if interval.get('wind_profiles', {}).get('merged'):
            all_heights.update(interval['wind_profiles']['merged'].keys())
        if interval.get('turbulence_profiles', {}).get('merged'):
            all_heights.update(interval['turbulence_profiles']['merged'].keys())
    
    if not all_heights:
        print("No height data found")
        return None
    
    heights = np.array(sorted(all_heights))

    n_periods = (24 * 60) // TIME_WINDOW_MINUTES
    start_time = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00", utc=True)
    time_grid = pd.date_range(start_time, periods=n_periods, freq=f'{TIME_WINDOW_MINUTES}min')

    # Data matrices indexed as [time, height]; NaN = no valid data for that cell.
    n_times = len(time_grid)
    n_heights = len(heights)
    ws_data = np.full((n_times, n_heights), np.nan)
    wd_data = np.full((n_times, n_heights), np.nan)
    w_data = np.full((n_times, n_heights), np.nan)
    ti_data = np.full((n_times, n_heights), np.nan)
    
    # One record per (time, height) so wind and turbulence variables share a row.
    data_records = []

    for interval in time_intervals:
        interval_time_epoch = interval['time']
        interval_time = pd.to_datetime(interval_time_epoch, unit='s', utc=True)

        all_heights_for_interval = set()
        wind_merged = interval.get('wind_profiles', {}).get('merged', {})
        turb_merged = interval.get('turbulence_profiles', {}).get('merged', {})

        all_heights_for_interval.update(wind_merged.keys())
        all_heights_for_interval.update(turb_merged.keys())

        for height in all_heights_for_interval:
            record = {
                'time': interval_time,
                'height': height,
                'ws': wind_merged.get(height, {}).get('ws', np.nan),
                'wd': wind_merged.get(height, {}).get('wd', np.nan),
                'w': wind_merged.get(height, {}).get('w', np.nan),
                'ti': turb_merged.get(height, {}).get('ti', np.nan)
            }
            data_records.append(record)
    
    if not data_records:
        print("No data records found")
        return None
    
    df = pd.DataFrame(data_records)

    # For every record, find the nearest time-grid bin and keep only those within one window.
    time_diffs_matrix = np.abs((time_grid.values[:, np.newaxis] - df['time'].values).astype('timedelta64[s]').astype(int))
    time_indices = np.argmin(time_diffs_matrix, axis=0)
    min_time_diffs = np.min(time_diffs_matrix, axis=0)

    valid_time_mask = min_time_diffs <= TIME_WINDOW_MINUTES * 60

    # searchsorted returns an insertion index, so confirm the height there really matches.
    height_indices = np.searchsorted(heights, df['height'].values)
    valid_height_mask = ((height_indices < len(heights)) &
                        (heights[height_indices] == df['height'].values))

    valid_mask = valid_time_mask & valid_height_mask
    
    if np.any(valid_mask):
        valid_time_idx = time_indices[valid_mask]
        valid_height_idx = height_indices[valid_mask]

        valid_ws = df['ws'].values[valid_mask]
        valid_wd = df['wd'].values[valid_mask]
        valid_w = df['w'].values[valid_mask]
        valid_ti = df['ti'].values[valid_mask]

        # Assign each variable separately so a cell keeps NaN for the variables it lacks.
        ws_valid_mask = ~np.isnan(valid_ws)
        if np.any(ws_valid_mask):
            ws_data[valid_time_idx[ws_valid_mask], valid_height_idx[ws_valid_mask]] = valid_ws[ws_valid_mask]
            
        wd_valid_mask = ~np.isnan(valid_wd)
        if np.any(wd_valid_mask):
            wd_data[valid_time_idx[wd_valid_mask], valid_height_idx[wd_valid_mask]] = valid_wd[wd_valid_mask]
            
        w_valid_mask = ~np.isnan(valid_w)
        if np.any(w_valid_mask):
            w_data[valid_time_idx[w_valid_mask], valid_height_idx[w_valid_mask]] = valid_w[w_valid_mask]
            
        ti_valid_mask = ~np.isnan(valid_ti)
        if np.any(ti_valid_mask):
            ti_data[valid_time_idx[ti_valid_mask], valid_height_idx[ti_valid_mask]] = valid_ti[ti_valid_mask]
    
    # Adaptive y-axis: crop to the tallest height sampled at min_timepoints or more intervals.
    print(f"\nCalculating optimal y-axis maximum (min {min_timepoints} timepoints coverage)...")

    coverage_counts = []
    for var_name, data_matrix in [('WS', ws_data), ('WD', wd_data), ('W', w_data), ('TI', ti_data)]:
        valid_per_height = np.sum(~np.isnan(data_matrix), axis=0)
        coverage_counts.append(valid_per_height)

        if np.any(valid_per_height >= min_timepoints):
            max_ht_idx = np.where(valid_per_height >= min_timepoints)[0][-1]
            max_ht = heights[max_ht_idx]
            n_points = valid_per_height[max_ht_idx]
            print(f"  {var_name}: max height = {max_ht:.0f}m ({n_points}/{n_times} timepoints)")

    # Use the best-covered variable at each height so losing one variable doesn't crop the plot.
    total_coverage = np.max(coverage_counts, axis=0)

    heights_with_sufficient_data = heights[total_coverage >= min_timepoints]

    if len(heights_with_sufficient_data) > 0:
        max_data_height = heights_with_sufficient_data[-1]
        ymax = max_data_height + height_buffer
        print(f"  -> Using ymax = {ymax:.0f}m (max data: {max_data_height:.0f}m + {height_buffer}m buffer)")
    else:
        # Fallback when no height meets the coverage bar; 1000 m keeps the figure readable.
        ymax = 1000
        print(f"  -> WARNING: Insufficient data, using fallback ymax = {ymax}m")

    # Suptitle coverage fraction is wind speed only, over the visible height band.
    height_mask = heights <= ymax
    overall_coverage = np.sum(~np.isnan(ws_data[:, height_mask])) / (n_times * np.sum(height_mask)) * 100
    print(f"  -> Overall data coverage in visible range: {overall_coverage:.1f}%")
    
    # Percentile colorbar stretch over the visible band so outliers don't wash out the map.
    def get_colorbar_range(data, heights, ymax, percentiles=(2, 98), var_name=''):
        """Return (vmin, vmax) from the 2nd/98th percentile of the visible data."""
        height_mask = heights <= ymax
        data_visible = data[:, height_mask]
        valid_data = data_visible[~np.isnan(data_visible)]

        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, percentiles[0])
            vmax = np.percentile(valid_data, percentiles[1])
            print(f"  {var_name}: colorbar range = [{vmin:.2f}, {vmax:.2f}] "
                  f"({len(valid_data)} valid points)")
            return vmin, vmax
        else:
            print(f"  {var_name}: no valid data, using defaults")
            return None, None

    print("\nCalculating adaptive colorbar ranges (2nd-98th percentile)...")

    ws_vmin, ws_vmax = get_colorbar_range(ws_data, heights, ymax, var_name='WS')
    # Wind direction is circular (0-360 deg) - a percentile stretch would be strange.
    wd_vmin, wd_vmax = 0, 360
    w_vmin, w_vmax = get_colorbar_range(w_data, heights, ymax, var_name='W')
    # Symmetric range so the diverging RdBu_r colormap centers on zero.
    if w_vmin is not None and w_vmax is not None:
        w_absmax = max(abs(w_vmin), abs(w_vmax))
        w_vmin, w_vmax = -w_absmax, w_absmax
    ti_vmin, ti_vmax = get_colorbar_range(ti_data, heights, ymax, var_name='TI')

    # Fallbacks span plausible offshore boundary-layer values for a nearly empty day.
    if ws_vmin is None: ws_vmin, ws_vmax = 0, 20
    if w_vmin is None: w_vmin, w_vmax = -2, 2
    if ti_vmin is None: ti_vmin, ti_vmax = 0, 0.5
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    time_mpl = mdates.date2num(time_grid)

    mask1 = ~np.isnan(ws_data.T)
    if np.any(mask1):
        im1 = axes[0].contourf(time_mpl, heights, ws_data.T, levels=20, cmap='viridis',
                               vmin=ws_vmin, vmax=ws_vmax, extend='both')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Wind Speed (m/s)')
    axes[0].set_ylabel('Height (m AGL)')
    axes[0].set_title('Wind Speed (m/s)')
    axes[0].set_ylim(0, ymax)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    mask2 = ~np.isnan(wd_data.T)
    if np.any(mask2):
        im2 = axes[1].contourf(time_mpl, heights, wd_data.T, levels=20, cmap='hsv',
                              vmin=wd_vmin, vmax=wd_vmax, extend='neither')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Wind Direction (°)')
    axes[1].set_ylabel('Height (m AGL)')
    axes[1].set_title('Wind Direction (degrees)')
    axes[1].set_ylim(0, ymax)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    mask3 = ~np.isnan(w_data.T)
    if np.any(mask3):
        im3 = axes[2].contourf(time_mpl, heights, w_data.T, levels=20, cmap='RdBu_r',
                              vmin=w_vmin, vmax=w_vmax, extend='both')
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label('Vertical Velocity (m/s)')
    axes[2].set_ylabel('Height (m AGL)')
    axes[2].set_title('Vertical Velocity (m/s)')
    axes[2].set_ylim(0, ymax)
    axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    mask4 = ~np.isnan(ti_data.T)
    if np.any(mask4):
        im4 = axes[3].contourf(time_mpl, heights, ti_data.T, levels=20, cmap='plasma',
                              vmin=ti_vmin, vmax=ti_vmax, extend='both')
        cbar4 = plt.colorbar(im4, ax=axes[3])
        cbar4.set_label('Turbulence Intensity (-)')
    axes[3].set_ylabel('Height (m AGL)')
    axes[3].set_title('Turbulence Intensity (-)')
    axes[3].set_ylim(0, ymax)
    axes[3].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[3].xaxis.set_minor_locator(mdates.HourLocator())
    axes[3].set_xlabel('Time (UTC)')
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

    date_formatted = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    fig.suptitle(f'Daily Wind & Turbulence Profiles - {date_formatted} UTC\n'
                 f'Y-max: {ymax:.0f}m | Coverage: {overall_coverage:.1f}%', 
                 fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    filename = f"hovmoller_{date_str}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Hovmöller plot: {filename}")
    return filepath

def create_daily_surface_met_summary(combined_results, output_dir, date_str):
    """Daily quick-look surface-met plot built from per-interval entries in combined_results"""

    time_intervals = combined_results.get('time_intervals', [])
    if not time_intervals:
        return None

    times, pressure, temp, rh, precip = [], [], [], [], []
    for interval in time_intervals:
        if 'surface_met' in interval:
            interval_time = pd.to_datetime(interval['time'], unit='s', utc=True)
            met = interval['surface_met']

            # Skip intervals where every variable is missing to avoid empty line segments
            if not all(np.isnan([met['pressure'], met['temperature'],
                                met['relative_humidity'], met['precipitation']])):
                times.append(interval_time)
                pressure.append(met['pressure'])
                temp.append(met['temperature'])
                rh.append(met['relative_humidity'])
                precip.append(met['precipitation'])

    if not times:
        print("No surface met data found in time intervals")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    date_formatted = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    fig.suptitle(f'Surface Met Summary - {date_formatted} UTC',
                 fontsize=14, fontweight='bold')

    axes[0].plot(times, pressure, 'b-o', markersize=2)
    axes[0].set_ylabel('Pressure (hPa)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Surface Air Pressure')

    axes[1].plot(times, temp, 'r-o', markersize=2)
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Air Temperature')

    axes[2].plot(times, rh, 'g-o', markersize=2)
    axes[2].set_ylabel('RH (%)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Relative Humidity')

    axes[3].bar(times, precip, width=pd.Timedelta(minutes=8), alpha=0.7, color='purple')
    axes[3].set_ylabel('Precip (mm)')
    axes[3].set_xlabel('Time (UTC)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Precipitation')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"surface_met_daily_summary_{date_str.replace('-', '')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved daily surface met summary: {filename}")
    return filepath

def create_full_daily_surface_met_plot(date_str, base_dir, output_dir,
                                 location='nantucket',
                                 analysis_start_time=None, analysis_end_time=None,
                                 figsize=(15, 12), dpi=300, verbose=False):
    """Full 24-hour surface-met plot for date_str, shading the analysis window when
    start/end times are given so a period can be read against the full day.

    Parameters
    ----------
    date_str : str
        Date string in 'YYYY-MM-DD' format
    base_dir : str
        Base directory to search for surface met files
    output_dir : str
        Directory to save the plot
    location : str, optional
        Location name ('nantucket', 'block island', etc.)
    analysis_start_time : str or datetime, optional
        Start time of analysis window to highlight
    analysis_end_time : str or datetime, optional
        End time of analysis window to highlight
    """
    if verbose:
        print(f"Creating full daily surface met plot for {date_str}...")

    # None start/end requests the whole UTC day from the loader
    try:
        surface_met_data = process_surface_met_for_date(
            date_str,
            data_dir=base_dir,
            start_time=None,
            end_time=None,
            location=location,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"Error loading surface met data: {e}")
        return None

    if not surface_met_data:
        if verbose:
            print("No surface met data available for the full day")
        return None

    df_data = []
    for unix_time, data in surface_met_data.items():
        df_data.append({
            'datetime': pd.to_datetime(unix_time, unit='s', utc=True),
            'pressure': data['pressure'],
            'temperature': data['temperature'],
            'relative_humidity': data['relative_humidity'],
            'precipitation': data['precipitation']
        })

    df = pd.DataFrame(df_data).set_index('datetime').sort_index()
    df = df.dropna(how='all')
    if df.empty:
        if verbose:
            print("No valid surface met data for plotting")
        return None

    fig, axes = plt.subplots(4, 1, figsize=figsize)
    date_formatted = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    location_name = get_location_display_name(location)

    title = f'Daily Surface Meteorological Summary - {location_name}\n{date_formatted} UTC'
    if analysis_start_time and analysis_end_time:
        if isinstance(analysis_start_time, str):
            analysis_start_time = pd.to_datetime(analysis_start_time)
        if isinstance(analysis_end_time, str):
            analysis_end_time = pd.to_datetime(analysis_end_time)

        start_str = analysis_start_time.strftime('%H:%M')
        end_str = analysis_end_time.strftime('%H:%M')
        title += f'\n(Analysis Period: {start_str} - {end_str} highlighted)'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    axes[0].plot(df.index, df['pressure'], 'b-', linewidth=2, marker='o', markersize=3)
    axes[0].set_ylabel('Pressure\n(hPa)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Surface Air Pressure')

    axes[1].plot(df.index, df['temperature'], 'r-', linewidth=2, marker='o', markersize=3)
    axes[1].set_ylabel('Temperature\n(°C)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Air Temperature')

    axes[2].plot(df.index, df['relative_humidity'], 'g-', linewidth=2, marker='o', markersize=3)
    axes[2].set_ylabel('Relative\nHumidity (%)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Relative Humidity')

    ax4a = axes[3]
    ax4b = ax4a.twinx()

    bars = ax4a.bar(df.index, df['precipitation'], width=pd.Timedelta(minutes=8),
                   alpha=0.6, color='purple', label='10-min Rate')

    # NaN -> 0 before cumsum so a single missing interval doesn't freeze the cumulative line.
    precip_cumsum = df['precipitation'].fillna(0).cumsum()
    line = ax4b.plot(df.index, precip_cumsum, 'navy', linewidth=2,
                    marker='o', markersize=2, label='Cumulative')

    ax4a.set_ylabel('Precip Rate\n(mm/10min)', color='purple', fontweight='bold')
    ax4b.set_ylabel('Cumulative\nPrecip (mm)', color='navy', fontweight='bold')
    ax4a.set_title('Precipitation')
    ax4a.grid(True, alpha=0.3)

    if analysis_start_time and analysis_end_time:
        for ax in axes:
            ax.axvspan(analysis_start_time, analysis_end_time,
                      alpha=0.2, color='yellow', label='Analysis Window')
        axes[0].legend(loc='upper right', fontsize=10)

    # Skip axes[3] here; its twin right axis needs formatting applied to the left (ax4a) instead
    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax4a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4a.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax4a.xaxis.set_minor_locator(mdates.HourLocator())
    plt.setp(ax4a.xaxis.get_majorticklabels(), rotation=45)
    axes[3].set_xlabel('Time (UTC)', fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"surface_met_daily_full_{date_str.replace('-', '')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Daily Surface Met Summary for {date_formatted}:")
        print("-" * 50)
        for var in ['pressure', 'temperature', 'relative_humidity', 'precipitation']:
            valid_data = df[var].dropna()
            if len(valid_data) > 0:
                print(f"{var.replace('_', ' ').title()}:")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Min:  {valid_data.min():.2f}")
                print(f"  Max:  {valid_data.max():.2f}")
                if var == 'precipitation':
                    print(f"  Total: {valid_data.sum():.2f} mm")
            else:
                print(f"{var.replace('_', ' ').title()}: No valid data")
            print()

        print(f"Saved full daily surface met plot: {filename}")

    return filepath

def create_full_daily_surface_met_plot_from_file(surface_met_file, date_str, output_dir,
                                               analysis_start_time=None, analysis_end_time=None, location='nantucket',
                                               figsize=(15, 12), dpi=300, verbose=False):
    """Full 24-hour surface-met plot built from an explicit NetCDF file path rather than the loader.

    Parameters
    ----------
    surface_met_file : str
        Path to the surface met NetCDF file
    date_str : str
        Date string in 'YYYY-MM-DD' format
    output_dir : str
        Directory to save the plot
    analysis_start_time : str or datetime, optional
        Start time of analysis window to highlight
    analysis_end_time : str or datetime, optional
        End time of analysis window to highlight
    """
    if verbose:
        print(f"Creating full daily surface met plot for {date_str}...")
        print(f"Using file: {surface_met_file}")
    
    try:
        ds = xr.open_dataset(surface_met_file)

        time_values = pd.to_datetime(ds.time.values)
        n_times = len(time_values)

        # Site surface-met NetCDFs use inconsistent variable names, so fall back through aliases.
        def get_var_values(ds, primary_name, fallback_names=None):
            if fallback_names is None:
                fallback_names = []
            
            if primary_name in ds:
                return ds[primary_name].values
            else:
                for fallback in fallback_names:
                    if fallback in ds:
                        return ds[fallback].values
            return np.full(n_times, np.nan)

        df = pd.DataFrame({
            'datetime': time_values,
            'pressure': get_var_values(ds, 'pressure', ['atmos_pressure', 'air_pressure']),
            'temperature': get_var_values(ds, 'temperature', ['air_temperature', 'temp']),
            'relative_humidity': get_var_values(ds, 'relative_humidity', ['rh', 'humidity']),
            'precipitation': get_var_values(ds, 'precipitation', ['precip', 'rain'])
        }).set_index('datetime').sort_index()
        
        ds.close()

        df = df.dropna(how='all')
        
    except Exception as e:
        if verbose:
            print(f"Error loading surface met data from {surface_met_file}: {e}")
        return None
    
    if df.empty:
        if verbose:
            print("No valid surface met data for plotting")
        return None
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    date_formatted = pd.to_datetime(date_str).strftime('%Y-%m-%d')

    location_name = get_location_display_name(location)
    title = f'Daily Surface Meteorological Summary - {location_name}\n{date_formatted} UTC'
    if analysis_start_time and analysis_end_time:
        if isinstance(analysis_start_time, str):
            analysis_start_time = pd.to_datetime(analysis_start_time)
        if isinstance(analysis_end_time, str):
            analysis_end_time = pd.to_datetime(analysis_end_time)
        start_str = analysis_start_time.strftime('%H:%M')
        end_str = analysis_end_time.strftime('%H:%M')
        title += f'\n(Analysis Period: {start_str} - {end_str} highlighted)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plot_configs = [
        ('pressure', 'b-', 'Pressure\n(hPa)', 'Surface Air Pressure'),
        ('temperature', 'r-', 'Temperature\n(°C)', 'Air Temperature'),
        ('relative_humidity', 'g-', 'Relative\nHumidity (%)', 'Relative Humidity')
    ]
    
    for i, (var, style, ylabel, title_text) in enumerate(plot_configs):
        if not df[var].isna().all():
            axes[i].plot(df.index, df[var], style, linewidth=2, marker='o', markersize=3)
        axes[i].set_ylabel(ylabel, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(title_text)

    if not df['precipitation'].isna().all():
        ax4a = axes[3]
        ax4b = ax4a.twinx()

        bars = ax4a.bar(df.index, df['precipitation'], width=pd.Timedelta(minutes=8),
                       alpha=0.6, color='purple', label='10-min Rate')

        precip_cumsum = df['precipitation'].fillna(0).cumsum()
        line = ax4b.plot(df.index, precip_cumsum, 'navy', linewidth=2,
                        marker='o', markersize=2, label='Cumulative')
        
        ax4a.set_ylabel('Precip Rate\n(mm/10min)', color='purple', fontweight='bold')
        ax4b.set_ylabel('Cumulative\nPrecip (mm)', color='navy', fontweight='bold')
    else:
        axes[3].text(0.5, 0.5, 'No precipitation data', transform=axes[3].transAxes, 
                    ha='center', va='center', fontsize=12)
    
    axes[3].set_title('Precipitation')
    axes[3].grid(True, alpha=0.3)

    if analysis_start_time and analysis_end_time:
        for ax in axes:
            ax.axvspan(analysis_start_time, analysis_end_time,
                      alpha=0.2, color='yellow', label='Analysis Window')
        axes[0].legend(loc='upper right', fontsize=10)

    time_formatter = mdates.DateFormatter('%H:%M')
    major_locator = mdates.HourLocator(interval=2)
    minor_locator = mdates.HourLocator()

    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(time_formatter)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    axes[3].xaxis.set_major_formatter(time_formatter)
    axes[3].xaxis.set_major_locator(major_locator)
    axes[3].xaxis.set_minor_locator(minor_locator)
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)
    axes[3].set_xlabel('Time (UTC)', fontweight='bold')
    
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"surface_met_{date_str.replace('-', '')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved full daily surface met plot: {filename}")
    
    return filepath
