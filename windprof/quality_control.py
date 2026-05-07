"""Per-instrument quality control and inter-instrument agreement flags

Two QC layers operate at different stages of processing:

  1. **Per-beam QC** (``filter_data_by_qc_criteria``) is applied before VAD
     fitting. Each instrument has its own QC signal (CNR, SNR, or
     intensity) and threshold, set empirically per instrument documentation or 
     instrument manager recommendation; gates are dropped if fewer than ``min_beams`` beams pass.
  2. **Inter-instrument agreement flags** (``calculate_quality_flag_for_height``)
     are applied during merging. Output flags follow the convention
     0 = good, 1 = suspect, 2 = bad, 3 = no data.

Per-instrument default thresholds (CNR/SNR/intensity, beam minimums) are
defined inline in the module and named for direct adaptation. 
Adjust thresholds for new campaigns as needed.
"""

import numpy as np

### Quality Control and Data Validation

def filter_data_by_qc_criteria(data, instrument=None, qc_params=None, verbose=True, return_summary=False):
    """
    Apply per-beam QC to a single-scan lidar dataset.

    Each instrument has its own QC signal (CNR, SNR, or intensity) and
    threshold. A height gate is dropped entirely if fewer than ``min_beams`` 
    beams pass, since downstream VAD fit needs at least 3 valid beams.

    Parameters
    ----------
    data : dict
        Unfiltered scan data with 'heights', 'time', 'position', and
        per-height 'measurements'.
    instrument : str
        Instrument key (z01, z02, z03) used to look up default QC params.
    qc_params : dict, optional
        Override the per-instrument defaults with a custom rule set.
    verbose : bool
        Print per-height pass statistics for debugging.
    return_summary : bool
        If True, also return the qc_stats dict alongside the filtered data.

    Returns
    -------
    dict or None
        Filtered data with the same structure as the input, or None if no
        heights survived QC.
    """
    # Instrument-specific QC rules. Thresholds were tuned for the WFIP3 campaign 
    # —> do not use as generic defaults elsewhere.
    default_qc_params = {
        'z03': {
            'qc_type': 'cnr',
            'threshold': -23,
            'min_beams': 3,
            'az_range_threshold': None,
            'min_height': None
        },
        'z02': {
            'qc_type': 'intensity',
            'threshold': 1.008,
            'min_beams': 3,
            'az_range_threshold': None,
            'min_height': 100  # drop scanning lidar range gates below 100 m (near-field noise)
        },
        'z01': {
            'qc_type': 'intensity',
            'threshold': 1.008,
            'min_beams': 3,
            'az_range_threshold': None,
            'min_height': 100  # drop scanning lidar range gates below 100 m (near-field noise)
        }
    }

    if qc_params is None:
        qc_params = default_qc_params.get(instrument, {})

    if not qc_params:
        raise ValueError(f"Unsupported instrument: {instrument}")

    qc_type = qc_params.get('qc_type')
    threshold = qc_params.get('threshold')
    min_beams = qc_params.get('min_beams', 3)

    filtered_data = {
        'heights': data['heights'],
        'time': data['time'],
        'position': data['position'],
        'measurements': {}
    }

    qc_stats = {
        'total_heights': len(data['measurements']),
        'heights_passed': 0,
        'pass_percentages': [],
        'passed_heights': [],
        'low_pass_heights': []
    }

    if verbose:
        print(f"--- QC DEBUG for {instrument} ---")
        print(f"Input heights: {len(data['heights'])}")
        print(f"QC params: {qc_params}")
        print(f"About to process {len(data['measurements'])} heights with measurements")

    for height in data['measurements'].keys():
        try:
            if qc_type == 'cnr':
                qc_values = data['measurements'][height]['cnr']
            elif qc_type == 'snr':
                qc_values = data['measurements'][height].get('snr', data['measurements'][height].get('SNR'))
            elif qc_type == 'intensity':
                qc_values = data['measurements'][height]['intensity']
            else:
                raise ValueError(f"Unsupported QC type: {qc_type}")
        except KeyError:
            if verbose:
                print(f"Warning: Could not find {qc_type} for height {height}")
            continue

        qc_mask = qc_values >= threshold

        n_total = len(qc_values)
        n_passed = np.sum(qc_mask)
        pass_percentage = n_passed / n_total * 100

        if verbose and height in list(data['heights'])[:3]:
            print(f"  Height {height}m:")
            print(f"    QC > {threshold}: {n_passed}/{n_total}")
            vr_data = data['measurements'][height].get('vr', data['measurements'][height].get('radial_wind_speed', []))
            vr_valid = ~np.isnan(vr_data)
            print(f"    Valid VR: {np.sum(vr_valid)}/{len(vr_data)}")
            combined_valid = qc_mask & vr_valid
            print(f"    Combined valid: {np.sum(combined_valid)}")
            print(f"    Min beams needed: {min_beams}")

        # Drop the whole height if VAD can't solve with the survivors
        if n_passed < min_beams:
            qc_stats['low_pass_heights'].append((height, pass_percentage))
            continue

        qc_stats['heights_passed'] += 1
        qc_stats['pass_percentages'].append(pass_percentage)
        qc_stats['passed_heights'].append(height)

        filtered_data['measurements'][height] = {}

        # Apply the same mask to every variable at this height so all
        # arrays stay aligned for the downstream solve.
        for key, values in data['measurements'][height].items():
            filtered_values = values[qc_mask]
            filtered_data['measurements'][height][key] = filtered_values

        # Position may be a per-beam azimuth array (z01/z02) or a shared
        # scalar (z03). Filter only in the former case.
        if hasattr(data['position'], '__len__') and len(data['position']) == len(qc_values):
            filtered_data['measurements'][height]['position'] = data['position'][qc_mask]
        else:
            filtered_data['measurements'][height]['position'] = data['position']

    if verbose:
        print(f"--- QC Summary: {qc_stats['heights_passed']} passed, {qc_stats['total_heights'] - qc_stats['heights_passed']} failed ---")
        if qc_stats['pass_percentages']:
            pass_dist = {
                '100%': sum(1 for p in qc_stats['pass_percentages'] if p == 100),
                '>50%': sum(1 for p in qc_stats['pass_percentages'] if p > 50),
                '<50%': sum(1 for p in qc_stats['pass_percentages'] if p <= 50)
            }
            print("Height Distribution by QC Pass Percentage:")
            for category, count in pass_dist.items():
                print(f"{category}: {count} heights")

    if not filtered_data['measurements']:
        if verbose:
            print("No heights passed QC criteria")
        return None

    if return_summary:
        return filtered_data, qc_stats
    return filtered_data

def calculate_data_availability(qc_filtered_data, original_data, availability_threshold=0.5):
    """
    Per-height data availability as the fraction of raw beams that would
    pass QC, computed against the unfiltered input. i.e. "what fraction of 
    what the instrument recorded was usable".

    Parameters
    ----------
    qc_filtered_data : dict
        QC-filtered data, used only for its height list.
    original_data : dict
        Unfiltered data used for the availability calculation.
    availability_threshold : float
        Per-height availability at or above this value counts as "meets_threshold".

    Returns
    -------
    dict
        'total_availability', per-height breakdown, and a boolean 'meets_threshold'.
    """
    availability = {
        'total_availability': 0,
        'height_availability': {},
        'meets_threshold': False
    }
    for height in qc_filtered_data['measurements'].keys():
        if height in original_data['measurements']:
            # Auto-detect QC signal since this function is called on data
            # from any of the three lidars without knowing which upstream.
            qc_values = None
            qc_threshold = None
            if 'snr' in original_data['measurements'][height] or 'SNR' in original_data['measurements'][height]:
                qc_values = original_data['measurements'][height].get('snr',
                           original_data['measurements'][height].get('SNR'))
                qc_threshold = -23
            elif 'intensity' in original_data['measurements'][height]:
                qc_values = original_data['measurements'][height]['intensity']
                qc_threshold = 1.008
            elif 'cnr' in original_data['measurements'][height]:
                qc_values = original_data['measurements'][height]['cnr']
                qc_threshold = -23

            if qc_values is None:
                continue

            total_measurements = len(qc_values)
            valid_measurements = np.sum(qc_values >= qc_threshold)
            availability_percentage = valid_measurements / total_measurements if total_measurements > 0 else 0

            availability['height_availability'][height] = {
                'availability': round(availability_percentage, 3),
                'meets_threshold': availability_percentage >= availability_threshold
            }
            availability['total_availability'] += availability_percentage

    if availability['height_availability']:
        availability['total_availability'] = round(
            availability['total_availability'] / len(availability['height_availability']), 3
        )
        availability['meets_threshold'] = any(
            info['meets_threshold'] for info in availability['height_availability'].values()
        )

    return availability

def summarize_vad_fit_failures(qc_failure_reasons):
    """
    Aggregate per-scan VAD fit failures into a single summary dict for
    post-campaign diagnostics — counts per failure mode, affected height
    ranges, and the raw details from each rejected fit.
    """
    if not qc_failure_reasons or 'chi_square_fit' not in qc_failure_reasons or not qc_failure_reasons['chi_square_fit']:
        return None

    summary = {
        'total_failures': 0,
        'failure_modes': {},
        'height_ranges': {
            'overall': [float('inf'), float('-inf')],
        }
    }

    for failure in qc_failure_reasons['chi_square_fit']:
        summary['total_failures'] += 1

        failure_reason = failure.get('failure_reason', 'unknown')
        if failure_reason not in summary['failure_modes']:
            summary['failure_modes'][failure_reason] = {
                'count': 0,
                'heights': [],
                'details': []
            }

        details = {k: v for k, v in failure.items() if k not in ['height', 'failure_reason']}
        summary['failure_modes'][failure_reason]['count'] += 1
        summary['failure_modes'][failure_reason]['heights'].append(failure['height'])
        summary['failure_modes'][failure_reason]['details'].append(details)

        height = failure['height']
        summary['height_ranges']['overall'][0] = min(summary['height_ranges']['overall'][0], height)
        summary['height_ranges']['overall'][1] = max(summary['height_ranges']['overall'][1], height)

    summary['height_ranges']['overall'] = tuple(summary['height_ranges']['overall'])

    return summary

#### Quality Flag Assignment

def calculate_wind_direction_flag(wind_directions, sigma_threshold=2.0):
    """
    Flag a height as 'suspect' when one instrument's wind direction is an
    outlier relative to the others at the same level. Uses circular statistics
    (arctan2 of mean sin/cos) so the 360°/0° wrap does not create false outliers.

    Returns 0 (good) or 1 (suspect). Returns 0 when the spread is below 1e-3°
    to avoid flagging on floating-point noise with well-agreeing instruments.
    """
    if len(wind_directions) < 2:
        return 0

    directions_rad = np.deg2rad(wind_directions)

    sin_mean = np.mean(np.sin(directions_rad))
    cos_mean = np.mean(np.cos(directions_rad))
    circular_mean = np.arctan2(sin_mean, cos_mean)

    # Circular distance from each observation to the mean, wrapped to [0, pi].
    circular_diffs = []
    for direction_rad in directions_rad:
        diff = np.abs(np.arctan2(np.sin(direction_rad - circular_mean),
                                np.cos(direction_rad - circular_mean)))
        circular_diffs.append(diff)

    circular_diffs_deg = np.rad2deg(circular_diffs)

    if len(circular_diffs_deg) < 2:
        return 0

    std_circular_diff = np.std(circular_diffs_deg)
    max_circular_diff = max(circular_diffs_deg)

    if std_circular_diff < 1e-3:
        return 0

    return 1 if max_circular_diff > sigma_threshold * std_circular_diff else 0

def check_excessive_turbulence_flag(interpolated_data, height):
    """
    Flag σ_u/|ū| > 1 at a given height as 'bad' (flag=2).

    This ratio captures "velocity fluctuations larger than the mean flow",
    which is physically implausible for sustained horizontal wind — usually
    indicates a broken retrieval rather than real turbulence. We skip the
    check when |ū| < 0.1 m/s since the ratio blows up for calm conditions.
    """
    u_components = []
    std_u_values = []

    for instrument, profile in interpolated_data.items():
        if height in profile:
            if 'ws' in profile[height] and 'wd' in profile[height]:
                ws = profile[height]['ws']
                wd = profile[height]['wd']
                u_comp = -ws * np.sin(np.deg2rad(wd))  # meteorological convention
                u_components.append(u_comp)

            if 'std_u' in profile[height]:
                std_u_values.append(profile[height]['std_u'])

    if not u_components or not std_u_values:
        return 0

    mean_u = np.mean(u_components)
    mean_std_u = np.mean(std_u_values)

    if abs(mean_u) < 0.1:
        return 0

    sigma_u_over_u = mean_std_u / abs(mean_u)

    return 2 if sigma_u_over_u > 1.0 else 0

def check_extreme_vertical_velocity_flag(interpolated_data, height):
    """
    Flag |w| > 7 m/s as 'suspect' (flag=1). 7 m/s is well above the
    vertical velocities expected in the marine boundary layer outside of
    deep convection, typically indicates a fit/contamination issue.
    """
    for instrument, profile in interpolated_data.items():
        if height in profile and 'w' in profile[height]:
            w_val = profile[height]['w']
            if w_val is not None and not np.isnan(w_val):
                if abs(w_val) > 7.0:
                    return 1

    return 0

def calculate_circular_spread(wind_directions):
    """
    Maximum pairwise circular distance among a list of wind directions,
    in degrees on [0, 180]. O(N²) where N is the number of instruments at
    one height (typically 2–4), so it stays cheap.
    """
    if len(wind_directions) < 2:
        return 0.0

    directions_rad = np.deg2rad(wind_directions)

    max_spread = 0.0
    for i in range(len(directions_rad)):
        for j in range(i + 1, len(directions_rad)):
            diff = np.abs(np.arctan2(np.sin(directions_rad[i] - directions_rad[j]),
                                   np.cos(directions_rad[i] - directions_rad[j])))
            max_spread = max(max_spread, diff)

    return np.rad2deg(max_spread)

def check_inter_instrument_agreement(instruments_data, height, param_name, sigma_threshold=None):
    """
    Flag a height as 'suspect' when independent instruments disagree
    on wind speed, direction, or vertical velocity beyond a tolerance. 
    Turbulence parameters are not checked here — their valid
    across-instrument spread is much larger and is handled separately.

    Thresholds:
        wd: > 30° circular spread
        w:  > 2 m/s range
        ws: > max(1.5 m/s, 0.5 x median ws) — scale with ambient flow

    The ``sigma_threshold`` argument is retained for backward compatibility
    with residual older callers but is no longer used.
    """
    if param_name not in ['ws', 'wd', 'w']:
        return 0

    values = []
    for instrument, profile in instruments_data.items():
        if height in profile and param_name in profile[height]:
            val = profile[height][param_name]
            if val is not None and not np.isnan(val):
                values.append(val)

    if len(values) < 2:
        return 0

    if param_name == 'wd':
        spread = calculate_circular_spread(values)
        return 1 if spread > 30.0 else 0

    elif param_name == 'w':
        spread = max(values) - min(values)
        return 1 if spread > 2.0 else 0

    elif param_name == 'ws':
        spread = max(values) - min(values)
        median_ws = np.median(values)
        threshold = max(1.5, 0.5 * median_ws)
        return 1 if spread > threshold else 0

    return 0

def calculate_quality_flag_for_height(height, interpolated_data, parameters):
    """
    Assemble per-parameter quality flags for one height level by running
    each parameter through the applicable checks (inter-instrument
    agreement + parameter-specific sanity checks). Returns 3 (missing)
    when the height is absent from every instrument's profile.

    Parameters
    ----------
    height : float
    interpolated_data : dict
        {instrument: {height: {param: value}}}
    parameters : list
        Parameter names to evaluate.

    Returns
    -------
    dict or int
        {param: flag} mapping, or 3 if the height is missing everywhere.
        Flag values: 0 = good, 1 = suspect, 2 = bad.
    """
    if not interpolated_data:
        return 3

    height_exists = False
    for instrument, profile in interpolated_data.items():
        if height in profile:
            height_exists = True
            break

    if not height_exists:
        return 3

    param_flags = {}

    for param in parameters:
        flag = 0

        inter_flag = check_inter_instrument_agreement(interpolated_data, height, param)
        if inter_flag == 1:
            flag = 1

        if param == 'w':
            extreme_w_flag = check_extreme_vertical_velocity_flag(interpolated_data, height)
            if extreme_w_flag == 1:
                flag = max(flag, 1)

        # σ_u/|ū| is a wind-speed-only check; wind direction has its own inter-instrument logic
        if param == 'ws':
            sigma_u_flag = check_excessive_turbulence_flag(interpolated_data, height)
            if sigma_u_flag == 2:
                flag = 2

        param_flags[param] = flag

    return param_flags
