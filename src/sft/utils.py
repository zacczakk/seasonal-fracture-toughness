"""
Safe mathematical operations for robust ODR fitting.

This module provides numerically stable mathematical operations that prevent
NaN and Inf values in optimization routines.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int]


# Default numerical safety parameters
DEFAULT_EPS = 1e-12
DEFAULT_MAX_VALUE = 1e10
DEFAULT_MIN_POSITIVE = 1e-12


def safe_power(
    base: ArrayLike, exponent: ArrayLike, max_result: float = DEFAULT_MAX_VALUE
):
    """
    Compute power operation with protection against numerical issues.

    Parameters
    ----------
    base : np.ndarray
        Base values
    exponent : np.ndarray
        Exponent values
    max_result : float, default DEFAULT_MAX_VALUE
        Maximum allowed result value

    Returns
    -------
    np.ndarray
        Safe power result without inf
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.power(base, exponent)
    # Catch anything larger than max_result
    result = np.clip(result, -max_result, max_result)

    return result


def safe_divide(
    numerator: ArrayLike, denominator: ArrayLike, max_result: float = DEFAULT_MAX_VALUE
):
    """
    Divide with protection against division by zero and finite overflow (no nan, no inf).

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values
    denominator : np.ndarray
        Denominator values
    max_result : float, default DEFAULT_MAX_VALUE
        Maximum allowed result magnitude

    Returns
    -------
    np.ndarray
        Division result capped at max_result
    """
    # Compute devision ignoring nan and inf warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)

    # Catch NaNs and anything larger than max_result (e.g., inf and -inf)
    result = np.where(np.isnan(result), max_result, result)
    result = np.clip(result, -max_result, max_result)

    # Propagate numerator NaNs
    result = np.where(np.isnan(numerator), np.full_like(result, np.nan), result)

    return result


def safe_log(x: ArrayLike, max_result: float = DEFAULT_MAX_VALUE) -> np.ndarray:
    """
    Compute natural logarithm with protection against invalid inputs.

    Parameters
    ----------
    x : np.ndarray
        Input values
    max_result : float, default DEFAULT_MAX_VALUE
        Maximum allowed result magnitude

    Returns
    -------
    np.ndarray
        Safe logarithm result
    """
    # Compute log
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(x)
    # Catch anything larger than max_result
    result = np.clip(result, -max_result, max_result)

    return result


def quadratic_penalty(x: "ArrayLike", lower: float, upper: float) -> np.ndarray:
    """
    Quadratic penalty function for constraint violations.

    Returns 0 inside [lower, upper], and grows quadratically outside the bounds.

    Parameters
    ----------
    x : np.ndarray, float, or int
        Values to check against bounds
    lower : float
        Lower bound
    upper : float
        Upper bound

    Returns
    -------
    np.ndarray
        Penalty values (0 inside bounds, positive outside)
    """
    x = np.asarray(x)
    lower_penalty = np.where(x < lower, (lower - x) ** 2, 0)
    upper_penalty = np.where(x > upper, (x - upper) ** 2, 0)
    return lower_penalty + upper_penalty


def ensure_finite(x: ArrayLike, replacement: float = DEFAULT_MAX_VALUE) -> np.ndarray:
    """
    Replace any NaN or Inf values with finite replacements.

    Parameters
    ----------
    x : np.ndarray
        Input array
    replacement : float, default DEFAULT_MAX_VALUE
        Finite value to use as replacement

    Returns
    -------
    np.ndarray
        Array with all finite values
    """
    x = np.asarray(x)

    # Replace NaN with replacement value
    result = np.where(np.isnan(x), replacement, x)

    # Replace +Inf with replacement value
    result = np.where(np.isposinf(result), replacement, result)

    # Replace -Inf with negative replacement value
    result = np.where(np.isneginf(result), -replacement, result)

    return result


def clip_with_penalty(
    beta: ArrayLike,
    bounds: Dict[str, Tuple[float, float]],
    penalty_scale: float = 1e2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip values to bounds while computing smooth penalty.

    This function enforces hard bounds (for numerical stability) and computes
    smooth penalties (for optimization) with per-series granularity.

    Parameters
    ----------
    beta : np.ndarray
        Parameter vector [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]
    bounds : dict
        Dictionary with keys 'GIc_1', 'GIIc_1', 'GIc_2', 'GIIc_2', 'GIc_3', 'GIIc_3', 'n', 'm'
        Each value is a (lower, upper) tuple for that parameter
    penalty_scale : float, default 1.0
        Scale factor for penalty computation

    Returns
    -------
    clipped_values : np.ndarray
        Values clipped to their respective bounds
    penalty : np.ndarray
        Smooth penalty for bound violations
    """
    beta = np.asarray(beta)

    # Set default lower and upper bounds for each parameter if not already
    # specified in 'bounds'. For example, GIc_1 is bounded between 1e-6 and 10.0,
    # n and m between 1e-6 and 1.0. The user-supplied 'bounds' dictionary can
    # override any of these defaults.
    default_bounds = {
        "GIc_1": (1e-6, 10.0),
        "GIIc_1": (1e-6, 10.0),
        "GIc_2": (1e-6, 10.0),
        "GIIc_2": (1e-6, 10.0),
        "GIc_3": (1e-6, 10.0),
        "GIIc_3": (1e-6, 10.0),
        "n": (1e-6, 1.0),
        "m": (1e-6, 1.0),
    }
    bounds = {**default_bounds, **bounds}

    # Parameter order: [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]
    param_names = ["GIc_1", "GIIc_1", "GIc_2", "GIIc_2", "GIc_3", "GIIc_3", "n", "m"]
    clipped_beta = np.empty_like(beta)
    penalty = np.zeros_like(beta)

    for i, param_name in enumerate(param_names):
        lower, upper = bounds[param_name]
        clipped_beta[i] = np.clip(beta[i], lower, upper)
        penalty[i] = penalty_scale * quadratic_penalty(beta[i], lower, upper)

    return clipped_beta, penalty


def safe_residual_term(
    numerator: ArrayLike,
    denominator: ArrayLike,
    exponent_value: ArrayLike,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """
    Compute (numerator/denominator)^(1/exponent_value) term safely.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values
    denominator : np.ndarray
        Denominator values
    exponent_value : float or np.ndarray
        Exponent values
    eps : float, default DEFAULT_EPS
        Numerical protection parameter

    Returns
    -------
    np.ndarray
        Robust computation of (numerator/denominator)^(1/exponent_value)
    """
    # Ensure positive values
    safe_numerator = np.maximum(numerator, 0)  # >= 0
    safe_denominator = np.maximum(denominator, eps)  # >= eps
    safe_exponent = safe_divide(1.0, exponent_value)  # >= 1

    # Compute fraction and power with exact-zero preservation for numerator == 0
    fraction = safe_divide(safe_numerator, safe_denominator)
    result = safe_power(fraction, safe_exponent)

    return result
