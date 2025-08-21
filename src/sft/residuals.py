import numpy as np

from .utils import (
    clip_with_penalty,
    safe_divide,
    safe_log,
    safe_power,
    safe_residual_term,
)


def residual(beta: list, x: tuple, series_idx: int) -> np.ndarray:
    """
    Compute objective function for orthogonal distance regression.

    Scalar residuals returned of list of length of x.

    Parameters
    ----------
    beta : list
        [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]
    x : tuple
        (Gi, Gii)
    series_idx : int
        Index of the series the datapoint belongs to.
    """
    # Unpack data point vectors
    Gi, Gii = x

    # Compute per-series bounds based on data ranges
    series_idx_array = np.asarray(series_idx)
    bounds = {}

    for series in range(3):
        mask = series_idx_array == series
        if np.any(mask):
            gi_max = float(np.max(Gi[mask]))
            gii_max = float(np.max(Gii[mask]))
            bounds[f"GIc_{series + 1}"] = (0, gi_max)
            bounds[f"GIIc_{series + 1}"] = (0, gii_max)
        else:
            # Fallback if series has no data
            bounds[f"GIc_{series + 1}"] = (0, float(np.max(Gi)))
            bounds[f"GIIc_{series + 1}"] = (0, float(np.max(Gii)))

    # Exponent bounds based on overall data range
    bounds["n"] = (1e-6, 1)
    bounds["m"] = (1e-6, 1)

    # Unpack constrained inputs
    constrained_beta, penalties = clip_with_penalty(beta, bounds)
    GIc1, GIIc1, GIc2, GIIc2, GIc3, GIIc3, n, m = constrained_beta

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc1, GIc2, GIc3])
    GIIc = np.choose(series_idx, [GIIc1, GIIc2, GIIc3])

    # Safely compute residual terms
    mode_I_term = safe_residual_term(Gi, GIc, n)
    mode_II_term = safe_residual_term(Gii, GIIc, m)

    # Compute residuals (Gi/GIc)^(1/n) + (Gii/GIIc)^(1/m) - 1
    residuals = mode_I_term + mode_II_term - 1.0

    # Add penalty
    total_penalty = float(np.sum(penalties))
    if total_penalty > 0:
        # Scale penalties relative to residual magnitude to reduce intercept bias
        scale = np.maximum(1.0, np.mean(np.abs(residuals)) + 1e-12)
        residuals += total_penalty / scale

    # Catch all remaining NaNs and Infs
    residuals = np.where(~np.isfinite(residuals), 1e6, residuals)

    return residuals


def param_jacobian(beta: list, x: tuple, series_idx: list[int], *args):
    """
    Compute derivates with respect to parameters (GIc, GIIc, n, m).

    Jacobian = [df/dGIc, df/dGIIc, df/dn, df/dm].T

    Parameters
    ----------
    beta : list[float]
        Model parameters (GIc, GIIc, n, m).
    x : list[float]
        Variables (Gi, Gii).
    series_idx : list[int]
        Index of the series the datapoint belongs to.

    Returns
    -------
    np.ndarray
        Jacobian matrix.
    """
    # Unpack data point vectors
    Gi, Gii = x
    Gi = np.maximum(Gi, 0.0)
    Gii = np.maximum(Gii, 0.0)

    # Compute per-series bounds based on data ranges
    series_idx_array = np.asarray(series_idx)
    bounds = {}

    for series in range(3):
        mask = series_idx_array == series
        if np.any(mask):
            gi_max = float(np.max(Gi[mask]))
            gii_max = float(np.max(Gii[mask]))
            bounds[f"GIc_{series + 1}"] = (0, gi_max)
            bounds[f"GIIc_{series + 1}"] = (0, gii_max)
        else:
            # Fallback if series has no data
            bounds[f"GIc_{series + 1}"] = (0, float(np.max(Gi)))
            bounds[f"GIIc_{series + 1}"] = (0, float(np.max(Gii)))

    # Exponent bounds based on overall data range
    bounds["n"] = (1e-6, 1)
    bounds["m"] = (1e-6, 1)

    # Unpack constrained inputs
    constrained_beta, _ = clip_with_penalty(beta, bounds)
    GIc1, GIIc1, GIc2, GIIc2, GIc3, GIIc3, n, m = constrained_beta

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc1, GIc2, GIc3])
    GIIc = np.choose(series_idx, [GIIc1, GIIc2, GIIc3])

    # Safely compute jacobian terms
    mode_I_term = safe_residual_term(Gi, GIc, n)
    mode_II_term = safe_residual_term(Gii, GIIc, m)
    mode_I_exp = safe_divide(1.0, n)
    mode_II_exp = safe_divide(1.0, m)
    mode_I_log = safe_log(safe_divide(Gi, GIc))
    mode_II_log = safe_log(safe_divide(Gii, GIIc))

    # Calculate derivatives (each np.array of length of x)
    with np.errstate(invalid="ignore"):
        # dGIc = -(((Gi / GIc) ** (1 / n) * (1 / n)) / GIc)
        dGIc = -safe_divide(mode_I_term * mode_I_exp, GIc)
        # dGIIc = -(((Gii / GIIc) ** (1 / m) * (1 / m)) / GIIc)
        dGIIc = -safe_divide(mode_II_term * mode_II_exp, GIIc)
        # dn = (Gi / GIc) ** (1 / n) * np.log(Gi / GIc) * (-1 / n**2)
        dn = mode_I_term * mode_I_log * safe_divide(-1.0, safe_power(n, 2.0))
        # dm = (Gii / GIIc) ** (1 / m) * np.log(Gii / GIIc) * (-1 / m**2)
        dm = mode_II_term * mode_II_log * safe_divide(-1, safe_power(m, 2.0))

    # Gradient is 0 for all series except the one being fit
    dGIc1 = np.where(series_idx == 0, dGIc, 0)
    dGIIc1 = np.where(series_idx == 0, dGIIc, 0)
    dGIc2 = np.where(series_idx == 1, dGIc, 0)
    dGIIc2 = np.where(series_idx == 1, dGIIc, 0)
    dGIc3 = np.where(series_idx == 2, dGIc, 0)
    dGIIc3 = np.where(series_idx == 2, dGIIc, 0)

    # Stack derivaties (number of rowns corresponds to of length of beta)
    return np.vstack([dGIc1, dGIIc1, dGIc2, dGIIc2, dGIc3, dGIIc3, dn, dm])


def value_jacobian(beta: list, x: tuple, series_idx: list[int], *args):
    """
    Compute derivates with respect to function arguments (Gi, Gii).

    Jacobian = [df/dGi, df/dGii].T

    Parameters
    ----------
    beta : list[float]
        Model parameters (GIc, GIIC, n, m).
    x : list[float]
        Variables (Gi, Gii).
    series_idx : list[int]
        Index of the series the datapoint belongs to.

    Returns
    -------
    np.ndarray
        Jacobian matrix.

    Raises
    ------
    NotImplementedError
        If residual variant is not implemented.
    """
    # Unpack data point vectors
    Gi, Gii = x
    Gi = np.maximum(Gi, 0.0)
    Gii = np.maximum(Gii, 0.0)

    # Compute per-series bounds based on data ranges
    series_idx_array = np.asarray(series_idx)
    bounds = {}

    for series in range(3):
        mask = series_idx_array == series
        if np.any(mask):
            gi_max = float(np.max(Gi[mask]))
            gii_max = float(np.max(Gii[mask]))
            bounds[f"GIc_{series + 1}"] = (0, gi_max)
            bounds[f"GIIc_{series + 1}"] = (0, gii_max)
        else:
            # Fallback if series has no data
            bounds[f"GIc_{series + 1}"] = (0, float(np.max(Gi)))
            bounds[f"GIIc_{series + 1}"] = (0, float(np.max(Gii)))

    # Exponent bounds based on overall data range
    bounds["n"] = (1e-6, 1)
    bounds["m"] = (1e-6, 1)

    # Unpack constrained inputs
    constrained_beta, _ = clip_with_penalty(beta, bounds)
    GIc1, GIIc1, GIc2, GIIc2, GIc3, GIIc3, n, m = constrained_beta

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc1, GIc2, GIc3])
    GIIc = np.choose(series_idx, [GIIc1, GIIc2, GIIc3])

    # Calculate derivatives (each np.array of length of x)
    # dGi = ((Gi / GIc) ** (1 / n) * (1 / n)) / Gi
    dGi = safe_divide(safe_residual_term(Gi, GIc, n) * safe_divide(1.0, n), Gi)
    # dGii = ((Gii / GIIc) ** (1 / m) * (1 / m)) / Gii
    dGii = safe_divide(safe_residual_term(Gii, GIIc, m) * safe_divide(1.0, m), Gii)

    # Stack derivaties
    return np.vstack([dGi, dGii])
