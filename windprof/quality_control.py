"""Per-instrument quality control and inter-instrument agreement flags

Two QC layers run at different stages: per-beam QC
(``filter_data_by_qc_criteria``) before VAD fitting, and inter-instrument
agreement flags (``calculate_quality_flag_for_height``) during merging.
Output flags are 0 = good, 1 = suspect, 2 = bad, 3 = no data.

Per-instrument QC rules (signal type, threshold, beam minimum) live in
``config.QC_CONFIG`` and are looked up per (site, instrument).
"""

import numpy as np

from .config import get_qc_params, MIN_VAD_BEAMS

def filter_data_by_qc_criteria(data, instrument=None, qc_params=None, location=None,
                               verbose=True, return_summary=False):
    """
    Apply per-beam QC to a single-scan lidar dataset.

    A height gate is dropped entirely if fewer than ``min_beams`` beams
    pass: the VAD fit needs at least 4 beams (dof >= 1) for its error
    bound to be defined.

    Parameters
    ----------
    data : dict
        Unfiltered scan data with 'heights', 'time', 'position', and
        per-height 'measurements'.
    instrument : str
        Instrument key (z01, z02, z03).
    qc_params : dict, optional
        QC rule set (a QC_CONFIG entry). Looked up from ``location`` +
        ``instrument`` when omitted.
    location : str, optional
        Site key for the QC_CONFIG lookup when ``qc_params`` is omitted.
    verbose : bool
        Print per-height pass statistics for debugging.
    return_summary : bool
        If True, also return the qc_stats dict alongside the filtered data.

    Returns
    -------
    dict or None
        Filtered data with the same structure as the input, or None if no
        heights survived QC. 'prefiltered' instruments pass through
        unchanged (QC already applied upstream).
    """
    if qc_params is None:
        if location is None:
            raise ValueError(
                "filter_data_by_qc_criteria needs qc_params or location+instrument "
                "to look up QC_CONFIG")
        qc_params = get_qc_params(location, instrument)

    qc_type = qc_params.get('qc_type')
    threshold = qc_params.get('threshold')
    min_beams = qc_params.get('min_beams', MIN_VAD_BEAMS)

    # QC already applied upstream by the instrument: nothing to filter.
    if qc_type == 'prefiltered':
        if return_summary:
            return data, {'total_heights': len(data['measurements']),
                          'heights_passed': len(data['measurements']),
                          'pass_percentages': [], 'passed_heights': list(data['measurements']),
                          'low_pass_heights': []}
        return data

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

        if n_passed < min_beams:
            qc_stats['low_pass_heights'].append((height, pass_percentage))
            continue

        qc_stats['heights_passed'] += 1
        qc_stats['pass_percentages'].append(pass_percentage)
        qc_stats['passed_heights'].append(height)

        filtered_data['measurements'][height] = {}

        # Mask every variable at this height so the per-beam arrays stay aligned.
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

def calculate_data_availability(qc_filtered_data, original_data, availability_threshold=0.5,
                                qc_type=None, threshold=None):
    """
    Per-height data availability as the fraction of raw beams that would
    pass QC, computed against the unfiltered input: "what fraction of what
    the instrument recorded was usable".

    ``qc_type`` and ``threshold`` must be the QC_CONFIG values the beam
    filter used; inferring them here would let the two silently disagree.

    Parameters
    ----------
    qc_filtered_data : dict
        QC-filtered data, used only for its height list.
    original_data : dict
        Unfiltered data used for the availability calculation.
    availability_threshold : float
        Per-height availability at or above this value counts as "meets_threshold".
    qc_type : str
        QC signal name ('intensity', 'cnr', 'snr') from QC_CONFIG.
    threshold : float
        The matching QC_CONFIG threshold.

    Returns
    -------
    dict
        'total_availability', per-height breakdown, and a boolean 'meets_threshold'.
    """
    if qc_type is None or threshold is None:
        raise ValueError(
            "calculate_data_availability requires the QC_CONFIG qc_type and "
            "threshold so availability uses the same rule as the beam filter")

    availability = {
        'total_availability': 0,
        'height_availability': {},
        'meets_threshold': False
    }
    for height in qc_filtered_data['measurements'].keys():
        if height in original_data['measurements']:
            fields = original_data['measurements'][height]
            qc_values = fields.get(qc_type, fields.get(qc_type.upper()))
            if qc_values is None:
                continue

            total_measurements = len(qc_values)
            valid_measurements = np.sum(qc_values >= threshold)
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
    post-campaign diagnostics: counts per failure mode, affected height
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
    Flag σ_u/|ū| > 1 at a given height as 'bad' (flag=2): fluctuations
    larger than the mean flow are implausible for sustained horizontal
    wind and usually mean a broken retrieval. Skipped when |ū| < 0.1 m/s,
    where the ratio blows up.
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
    in degrees on [0, 180].
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

def check_inter_instrument_agreement(instruments_data, height, param_name):
    """
    Flag a height as 'suspect' when independent instruments disagree on
    wind speed, direction, or vertical velocity beyond a tolerance; the
    wind-speed tolerance scales with the ambient flow. Turbulence
    parameters are excluded, their across-instrument spread is
    legitimately much larger.
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
    agreement plus parameter-specific sanity checks).

    Parameters
    ----------
    height : float
    interpolated_data : dict
        {instrument: {height: {param: value}}}, restricted by the caller to
        the instruments that contributed to the merged value at this height.
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

        if param == 'ws':
            sigma_u_flag = check_excessive_turbulence_flag(interpolated_data, height)
            if sigma_u_flag == 2:
                flag = 2

        param_flags[param] = flag

    return param_flags
