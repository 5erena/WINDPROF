"""VAD wind retrieval, turbulence estimation, and uncertainty propagation

Provides the geometry-agnostic computational core used by both scanning
and profiling lidar processing:

  - ``fit_chi_winds``: chi-square least-squares fit of (u, v, w) to a single
    conical scan, with covariance-based component uncertainties.
  - ``calculate_turbulence_metrics``: Reynolds decomposition, hybrid
    wind-speed averaging (Rosenbusch 2021), TKE and turbulence intensity.
  - ``apply_physics_based_qc``: physics-bound rejection of unphysical
    turbulence retrievals (TKE, TI).

Inputs are arrays of (azimuth, elevation, radial velocity) for one scan
or one 10-minute window; instrument-specific ingestion happens upstream
in ``lidars.py``, ``radars.py``, and ``anemometers.py``.
"""

import numpy as np
from numpy.linalg import pinv, norm

### Wind Analysis (VAD Fitting)

def fit_chi_winds(az, el, rv, wind_speed_error=2, min_ws_for_direction=2.0):
    """
    Retrieve (u, v, w) wind components from a single conical scan using a
    chi-square least-squares fit to the VAD equation.

    For each beam at azimuth α and elevation θ, the radial velocity satisfies
        V_r = u sin(α) cos(θ) + v cos(α) cos(θ) + w sin(θ)
    We solve this over-determined linear system (N beams, 3 unknowns) via the
    normal equations and propagate the fit residuals into component standard errors.

    Parameters
    ----------
    az : array-like
        Beam azimuth angles over the range-ring being fit (degrees).
    el : array-like
        Beam elevation angles (degrees).
    rv : array-like
        Radial velocities for the corresponding beams (m/s).
    wind_speed_error : float
        Reject the fit if the propagated wind speed error exceeds this value (m/s).
        The fit may be numerically valid but too noisy to be useful.
    min_ws_for_direction : float
        Minimum wind speed for reporting a wind direction error (m/s).
        The direction error propagation has a WS^-2 dependence, so at
        low wind speeds it blows up unphysically.

    Returns
    -------
    dict
        u, v, w and their errors, wind speed, wind direction, their errors,
        chi-square residual, R², condition number, and a diagnostics sub-dict
        explaining any rejection.
    """
    # Diagnostics accumulate at each failure path so callers can attribute
    # rejections to specific causes during post-campaign analysis.
    diagnostics = {
        'status': 'success',
        'failure_reason': None,
        'min_ws_threshold': min_ws_for_direction
    }

    # Drop beams with NaN radial velocity (failed instrument-level QC
    # upstream — low SNR/intensity) before the solve.
    mask = ~np.isnan(rv)
    az = np.array(az)[mask]
    el = np.array(el)[mask]
    rv = np.array(rv)[mask]

    # Need >= 3 valid beams to solve for (u, v, w) — system is
    # under-determined otherwise.
    if len(rv) < 3:
        diagnostics.update({
            'status': 'failed',
            'failure_reason': 'insufficient_data',
            'data_points': len(rv)
        })
        return {**_return_nans(), 'diagnostics': diagnostics}

    # Direction cosines project each beam onto (u, v, w) axes via 
    # VAD equation above.
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    cos_el = np.cos(el_rad)
    x1 = np.sin(az_rad) * cos_el
    x2 = np.cos(az_rad) * cos_el
    x3 = np.sin(el_rad)

    # Normal equations for the least-squares system X · [u, v, w]^T ≈ rv.
    X = np.column_stack([x1, x2, x3])
    a = X.T @ X
    b = X.T @ rv

    # Flag ill-posed scan geometries via the condition number of the normal
    # matrix (e.g., azimuthal sampling too narrow to separate u from v, or
    # a near-vertical stare that collapses the horizontal basis). 
    # CN >= 1000: practical threshold for ill-conditioning. Higher values
    # indicate noise amplification may degrade the solution.
    ainv = pinv(a)
    CN = norm(a) * norm(ainv)

    if CN >= 1000:
        diagnostics.update({
            'status': 'failed',
            'failure_reason': 'high_condition_number',
            'condition_number': round(CN, 2)
        })
        return {**_return_nans(), 'diagnostics': diagnostics}

    # Solve for wind components
    c = ainv @ b
    u, v, w = c

    rv_fit = X @ c

    # Chi-square captures systematic residuals between observed and fitted
    # radial velocities; it feeds into the component error calculation below
    # via the reduced chi-square s² = χ²/dof.
    chisq = np.sum((rv_fit - rv)**2)
    residual = np.sqrt(chisq/len(rv))

    r2 = _calculate_r2(rv, rv_fit)

    # Three free parameters (u, v, w), so dof = N_beams - 3
    dof = len(rv) - 3
    if dof <= 0:
        return _return_nans()

    # Component standard errors from the least-squares covariance matrix
    # (X^T X)^-1, scaled by the reduced chi-square.
    try:
        uerr = np.sqrt((chisq/dof) * ainv[0,0])
        verr = np.sqrt((chisq/dof) * ainv[1,1])
        werr = np.sqrt((chisq/dof) * ainv[2,2])
    except (ValueError, RuntimeWarning):
        return _return_nans()

    ws = np.sqrt(u**2 + v**2)

    # Wind direction via quadrant-aware arccos, avoiding atan2 sign
    # ambiguities and producing meteorological convention directly
    # (0° = N, clockwise positive).
    if ws == 0:
        wd = np.nan
    else:
        if (u >= 0) and (v >= 0):
            wd = np.rad2deg(np.arccos(abs(v)/ws))
        elif (u >= 0) and (v <= 0):
            wd = np.rad2deg(np.arccos(abs(u)/ws)) + 90
        elif (u <= 0) and (v <= 0):
            wd = np.rad2deg(np.arccos(abs(v)/ws)) + 180
        elif (u <= 0) and (v >= 0):
            wd = np.rad2deg(np.arccos(abs(u)/ws)) + 270
        else:
            wd = np.nan

    # Wind speed error propagated from component errors
    wserr = np.sqrt((u*uerr)**2 + (v*verr)**2)/ws if ws > 0 else np.nan

    # Wind direction error has a WS^-2 dependence (see docstring). Below
    # min_ws_for_direction we return NaN; above we still cap at the
    # physical maximum of 180° to handle numerical edge cases.
    if ws < min_ws_for_direction:
        wderr = np.nan
        diagnostics['low_wind_speed_flag'] = True
        diagnostics['ws_value'] = round(ws, 3)
        diagnostics['wd_error_reason'] = f'wind_speed_below_{min_ws_for_direction}_ms'
    elif ws > 0:
        wderr_raw = (180/np.pi) * np.sqrt((u*verr)**2 + (v*uerr)**2)/(ws**2)
        wderr = min(wderr_raw, 180.0)

        if wderr_raw > 180.0:
            diagnostics['direction_error_capped'] = True
            diagnostics['raw_wd_error'] = round(wderr_raw, 1)
    else:
        wderr = np.nan

    # Reject the entire fit if the component-propagated wind speed error exceeds
    # ceiling — fit is technically valid but too noisy for downstream use.
    if wserr > wind_speed_error:
        diagnostics.update({
            'status': 'failed',
            'failure_reason': 'high_wind_speed_error',
            'ws_error': round(wserr, 3)
        })
        return {**_return_nans(), 'diagnostics': diagnostics}

    return {
        'u': round(u, 2), 'v': round(v, 2), 'w': round(w, 2),
        'uerr': round(uerr, 2), 'verr': round(verr, 2), 'werr': round(werr, 2),
        'residual': round(residual, 3),
        'ws': round(ws, 2), 'wd': round(wd, 2),
        'r2': round(r2, 3),
        'wserr': round(wserr, 3), 'wderr': round(wderr, 3),
        'cond': round(CN, 2),
        'diagnostics': diagnostics
    }

def _return_nans():
    """Return a dict of NaN values with the same shape as a successful fit.
    Used by every failure path in fit_chi_winds so callers can rely on
    consistent output keys regardless of whether the fit succeeded."""
    return {
        'u': np.nan, 'v': np.nan, 'w': np.nan,
        'uerr': np.nan, 'verr': np.nan, 'werr': np.nan,
        'residual': np.nan,
        'ws': np.nan, 'wd': np.nan,
        'r2': np.nan,
        'wserr': np.nan, 'wderr': np.nan,
        'diagnostics': {
            'status': 'failed',
            'failure_reason': None
        }
    }

def _calculate_r2(measured, predicted):
    """
    Coefficient of determination (R²) between measured and predicted arrays.
    Clamped to [0, 1] to match the MATLAB reference implementation — negative
    R² can arise from poor fits but isn't meaningful as a fit-quality metric.
    """
    valid = ~(np.isnan(measured) | np.isnan(predicted))
    measured = measured[valid]
    predicted = predicted[valid]

    if len(measured) < 3:
        return np.nan

    ss_res = np.sum((measured - predicted)**2)
    ss_tot = np.sum((measured - np.mean(measured))**2)

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
        r2 = max(0, min(1, r2))
    else:
        r2 = np.nan

    return r2

### Turbulence Analysis

def apply_physics_based_qc(turbulence_data):
    """
    Reject turbulence records whose metrics exceed physically plausible
    limits for the marine/coastal boundary layer. These bounds were chosen
    from published maxima rather than tuned to this campaign.

    Parameters
    ----------
    turbulence_data : dict
        Must contain at least 'tke', 'ti', and 'std_u'/'std_v'/'std_w'.

    Returns
    -------
    bool
        True if the record should be rejected, False if it's acceptable.
    """
    # TKE > 30 m²/s² exceeds maxima reported for well-mixed convective
    # boundary layers (Stull 1988; Kaimal & Finnigan 1994).
    if turbulence_data.get('tke', 0) > 30.0:
        return True

    # TI > 3.0 is inconsistent with IEC 61400-1 (2019) design envelopes
    # and observed offshore atmospheric conditions.
    if turbulence_data.get('ti', 0) > 3.0:
        return True

    # Multi-parameter consistency check — catches artifacts where a
    # moderate TKE is paired with a single blown-up component std, typically
    # from a lone outlier scan dominating the variance within a 10-min window.
    if (turbulence_data.get('tke', 0) > 15.0 and
        any(turbulence_data.get(f'std_{comp}', 0) > 10.0 for comp in ['u', 'v', 'w'])):
        return True

    return False

def calculate_turbulence_metrics(u_series, v_series, w_series, ws_series, hybrid_method='empirical'):
    """
    Compute TKE, TI, and component standard deviations from a 10-minute
    time series of scan-level wind retrievals.

    Uses the Rosenbusch (2021) hybrid wind speed formulation for TI, which
    combines scalar- and vector-averaged wind speeds to cancel the opposing
    biases each exhibits under turbulence (scalar averaging overestimates,
    vector averaging underestimates the cup-equivalent wind speed).

    Parameters
    ----------
    u_series, v_series, w_series : array-like
        Time series of wind component measurements. ``w_series`` may be
        None for instruments that don't resolve vertical velocity; in that
        case TKE and σ_w are returned as NaN rather than approximated.
    ws_series : array-like
        Time series of scalar wind speed measurements.
    hybrid_method : {'empirical', 'theoretical'}
        'empirical' uses α = 0.55 (Rosenbusch 2021, cross-site ensemble
        mean). 'theoretical' uses α = 2/3 (Rosenbusch 2021, from
        IEC-consistent isotropy assumptions).

    Returns
    -------
    dict or None
        Keys: ti, tke, std_u, std_v, std_w. Returns None if the record has
        fewer than 3 valid samples or fails physics-based QC.
    """
    u_series = np.array(u_series)
    v_series = np.array(v_series)
    w_series = np.array(w_series) if w_series is not None else None
    ws_series = np.array(ws_series)

    # Need at least 3 scans to compute a meaningful variance
    if len(u_series) < 3:
        return None

    # Unified validity mask across all series — dropping a scan from one
    # variable means dropping it from all, so Reynolds decomposition
    # operates on a consistent sample set.
    valid_mask = ~np.isnan(u_series) & ~np.isnan(v_series) & ~np.isnan(ws_series)
    if w_series is not None:
        valid_mask = valid_mask & ~np.isnan(w_series)

    u_valid = u_series[valid_mask]
    v_valid = v_series[valid_mask]
    ws_valid = ws_series[valid_mask]

    if len(u_valid) < 3:
        return None

    # === Hybrid Wind Speed (Rosenbusch 2021) ===
    u_mean = np.mean(u_valid)
    v_mean = np.mean(v_valid)
    vector_avg = np.sqrt(u_mean**2 + v_mean**2)
    scalar_avg = np.mean(ws_valid)

    if hybrid_method == 'empirical':
        hybrid_ws = (0.55 * scalar_avg) + (0.45 * vector_avg)
    elif hybrid_method == 'theoretical':
        hybrid_ws = (2/3 * scalar_avg) + (1/3 * vector_avg)
    else:
        raise ValueError("Method must be 'empirical' or 'theoretical'")

    # === Turbulence Intensity ===
    # TI is undefined as WS -> 0 (division by zero). 0.1 m/s floor
    # protects the calculation without affecting any realistic flow.
    if hybrid_ws < 0.1:
        ti = np.nan
    else:
        std_ws = np.std(ws_valid)
        ti = std_ws / hybrid_ws

    # === Component Standard Deviations ===
    # Note: these are computed in the instrument-fixed frame (east-north-up),
    # not rotated to streamwise/cross-stream. TKE and TI are rotationally
    # invariant, but individual σ_u / σ_v are not — users comparing components
    # across instruments with different orientations should account for this.
    std_u = np.std(u_valid)
    std_v = np.std(v_valid)

    # === Turbulent Kinetic Energy ===
    u_prime = u_valid - u_mean
    v_prime = v_valid - v_mean
    u_prime_sq = u_prime**2
    v_prime_sq = v_prime**2

    if w_series is not None:
        w_valid = w_series[valid_mask]
        w_mean = np.mean(w_valid)
        w_prime = w_valid - w_mean
        std_w = np.std(w_valid)
        # TKE = 1/2 * (σ_u² + σ_v² + σ_w²)
        tke = 0.5 * (np.mean(u_prime_sq) + np.mean(v_prime_sq) + np.mean(w_prime**2))
    else:
        # Without vertical velocity (some anemometer configs) we can't
        # compute the full 3D TKE. Return NaN rather than a 2D approximation.
        std_w = np.nan
        tke = np.nan

    # If any headline metric failed, null the component stds too so
    # downstream filtering treats the whole record as bad rather than
    # accidentally keeping partial data.
    if np.isnan(ti) or np.isnan(tke) or np.isnan(hybrid_ws):
        std_u = np.nan
        std_v = np.nan
        std_w = np.nan

    result = {
        'ti': ti,
        'tke': tke,
        'std_u': std_u,
        'std_v': std_v,
        'std_w': std_w,
    }

    if apply_physics_based_qc(result):
        return None

    return result
