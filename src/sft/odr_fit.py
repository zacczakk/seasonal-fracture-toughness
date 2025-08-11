"""
Orthogonal Distance Regression fitting for fracture toughness data.

This module provides ODR fitting functionality for the fracture toughness model:
    ((Gi / GIc) ** (1 / n) + (Gii / GIIc) ** (1 / m)) - 1 = 0

The implementation handles MultiIndex DataFrames with (source, series) indexing
and uncertainties.ufloat objects, supporting both fixed and optimized parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import odr, stats
from uncertainties import unumpy as unp
from uncertainties.core import Variable


@dataclass
class FractureToughnessResult:
    """Results from ODR fitting of fracture toughness model."""

    # Fitted parameters
    GIc_1: float
    GIc_2: float
    GIc_3: float
    GIIc_1: float
    GIIc_2: float
    GIIc_3: float
    n: float
    m: float

    # Parameter uncertainties
    GIc_1_err: float
    GIc_2_err: float
    GIc_3_err: float
    GIIc_1_err: float
    GIIc_2_err: float
    GIIc_3_err: float
    n_err: float
    m_err: float

    # Fit statistics
    reduced_chi_squared: float
    chi_squared: float
    p_value: float
    r_squared: float
    degrees_of_freedom: int
    n_data_points: int

    # Additional info
    convergence_info: Dict[str, Any]
    residuals: np.ndarray


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame structure and content."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex")

    if df.index.names != ["source", "series"]:
        raise ValueError("MultiIndex must have levels ['source', 'series']")

    if not all(col in df.columns for col in ["GIc", "GIIc"]):
        raise ValueError("DataFrame must contain 'GIc' and 'GIIc' columns")

    # Check for ufloat objects
    sample_gic = df["GIc"].iloc[0] if len(df) > 0 else None
    if sample_gic is not None and not isinstance(sample_gic, Variable):
        raise ValueError(
            "GIc and GIIc columns must contain uncertainties.ufloat objects"
        )


def _extract_data_for_odr(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data arrays from DataFrame for ODR fitting.

    Returns
    -------
    gi_values : np.ndarray
        Nominal values of GIc measurements
    gii_values : np.ndarray
        Nominal values of GIIc measurements
    gi_errors : np.ndarray
        Standard deviations of GIc measurements
    gii_errors : np.ndarray
        Standard deviations of GIIc measurements
    """
    gi_values = unp.nominal_values(df["GIc"].to_numpy())
    gii_values = unp.nominal_values(df["GIIc"].to_numpy())
    gi_errors = unp.std_devs(df["GIc"].to_numpy())
    gii_errors = unp.std_devs(df["GIIc"].to_numpy())

    return gi_values, gii_values, gi_errors, gii_errors


def _create_series_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """Create mapping from series names to indices."""
    series_names = df.index.get_level_values("series").unique().sort_values()
    return {name: i for i, name in enumerate(series_names)}


def _fracture_toughness_residual(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Residual function for ODR fitting.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]
    x : np.ndarray
        Data array with shape (3, N) containing:
        - x[0]: Gi values
        - x[1]: Gii values
        - x[2]: Series indices (0, 1, 2)

    Returns
    -------
    residuals : np.ndarray
        Residual values for the model equation
    """
    # Extract parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = params

    # Extract data
    gi_vals = x[0]
    gii_vals = x[1]
    series_idx = x[2].astype(int)  # (!) TODO: consider passing extra_args to odr.Model

    # Create parameter arrays based on series
    GIc_params = np.array([GIc_1, GIc_2, GIc_3])[series_idx]
    GIIc_params = np.array([GIIc_1, GIIc_2, GIIc_3])[series_idx]

    # Ensure positive parameters to avoid numerical issues  # TODO: can add a penalty for out-of-bounds parameters, but cannot manipulate data
    GIc_params = np.maximum(GIc_params, 1e-10)  # (!) TODO: cannot manipulate data
    GIIc_params = np.maximum(GIIc_params, 1e-10)  # (!) TODO: cannot manipulate data
    n = max(n, 0.1)  # (!) TODO: cannot manipulate data, must obey n > 0 and n <= 1
    m = max(m, 0.1)  # (!) TODO: cannot manipulate data, must obey m > 0 and m <= 1

    # Calculate model residuals: ((Gi/GIc)^(1/n) + (Gii/GIIc)^(1/m)) - 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        term1 = (gi_vals / GIc_params) ** (1.0 / n)
        term2 = (gii_vals / GIIc_params) ** (1.0 / m)
        residuals = term1 + term2 - 1.0

    # Handle any NaN or inf values
    residuals = np.where(
        np.isfinite(residuals), residuals, 1e6
    )  # TODO: treat parameter bounds violations similarly

    # For implicit models in scipy.odr, return 2D array of shape (N,)
    return residuals


def _estimate_initial_parameters(df: pd.DataFrame) -> np.ndarray:
    """Estimate initial parameters for ODR fitting."""
    series_mapping = _create_series_mapping(df)

    # Initialize parameter array
    params = np.zeros(8)

    # Estimate GIc and GIIc for each series using maximum values
    for series_name, series_idx in series_mapping.items():
        series_data = df.xs(series_name, level="series")

        gi_vals = unp.nominal_values(series_data["GIc"].to_numpy())
        gii_vals = unp.nominal_values(series_data["GIIc"].to_numpy())

        # Use 90th percentile as initial estimate
        params[series_idx * 2] = np.percentile(gi_vals, 90)  # GIc
        params[series_idx * 2 + 1] = np.percentile(gii_vals, 90)  # GIIc

    # Initial estimates for n and m
    params[6] = 2.0  # n  # TODO: 0 < n <= 1
    params[7] = 2.0  # m  # TODO: 0 < m <= 1

    return params


def _validate_parameters(
    params: np.ndarray, bounds: Optional[Dict] = None
) -> np.ndarray:
    """Validate and constrain parameters within reasonable bounds."""
    # Default bounds if not provided
    if bounds is None:
        bounds = {
            "GIc_min": 1e-6,
            "GIc_max": 10.0,
            "GIIc_min": 1e-6,
            "GIIc_max": 10.0,
            "n_min": 1e-6,  # TODO: 0 < n <= 1
            "n_max": 1,  # TODO: 0 < n <= 1
            "m_min": 1e-6,  # TODO: 0 < m <= 1
            "m_max": 1,  # TODO: 0 < m <= 1
        }

    # Apply bounds to GIc parameters (indices 0, 2, 4)
    for i in [0, 2, 4]:
        params[i] = np.clip(
            params[i], bounds.get("GIc_min", 1e-6), bounds.get("GIc_max", 10.0)
        )

    # Apply bounds to GIIc parameters (indices 1, 3, 5)
    for i in [1, 3, 5]:
        params[i] = np.clip(
            params[i], bounds.get("GIIc_min", 1e-6), bounds.get("GIIc_max", 10.0)
        )

    # Apply bounds to n and m
    params[6] = np.clip(params[6], bounds.get("n_min", 1e-6), bounds.get("n_max", 1))
    params[7] = np.clip(params[7], bounds.get("m_min", 1e-6), bounds.get("m_max", 1))

    return params


def _calculate_r_squared(residuals: np.ndarray, data: np.ndarray) -> float:
    """Calculate R-squared for the fit."""
    # For implicit model, calculate R² differently
    # Use correlation between predicted and observed values
    gi_vals = data[0]
    gii_vals = data[1]

    # Calculate mean of dependent variable (using combined metric)
    combined_vals = np.sqrt(gi_vals**2 + gii_vals**2)
    ss_tot = np.sum((combined_vals - np.mean(combined_vals)) ** 2)
    ss_res = np.sum(residuals**2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return max(0.0, min(1.0, r_squared))  # Clamp between 0 and 1


def fit_fracture_toughness_model(
    df: pd.DataFrame,
    *,
    n: Optional[float] = None,
    m: Optional[float] = None,
    bounds: Optional[Dict] = None,
    grid_search: bool = False,
    grid_points: int = 10,
) -> FractureToughnessResult:
    """
    Fit fracture toughness model using Orthogonal Distance Regression.

    Fits the model: ((Gi / GIc) ** (1 / n) + (Gii / GIIc) ** (1 / m)) - 1 = 0

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with levels ['source', 'series'] and columns ['GIc', 'GIIc']
        containing uncertainties.ufloat objects.
    n : float, optional
        Fixed value for n parameter. If None, will be optimized.
    m : float, optional
        Fixed value for m parameter. If None, will be optimized.
    bounds : dict, optional
        Parameter bounds dictionary with keys like 'GIc_min', 'GIc_max', etc.
    grid_search : bool, default False
        If True and n/m are not fixed, perform grid search for initialization.
    grid_points : int, default 10
        Number of grid points per dimension for grid search.

    Returns
    -------
    FractureToughnessResult
        Fitted parameters, uncertainties, and fit statistics.
    """
    # Validate input
    _validate_dataframe(df)

    if len(df) < 8:  # Need at least 8 data points for 8 parameters
        raise ValueError("Need at least 8 data points for fitting")

    # Extract data for ODR
    gi_values, gii_values, gi_errors, gii_errors = _extract_data_for_odr(df)

    # Create series index array
    series_mapping = _create_series_mapping(df)
    series_indices = np.array(
        [
            series_mapping[series_name]
            for series_name in df.index.get_level_values("series")
        ]
    )

    # Prepare data array for ODR
    data_array = np.vstack([gi_values, gii_values, series_indices])

    # Create ODR data object for implicit fitting
    # For implicit models, y=1 indicates 1-dimensional implicit constraint
    odr_data = odr.RealData(
        x=data_array,
        y=1,  # Integer indicating dimensionality of implicit constraint
        sx=np.vstack([gi_errors, gii_errors, np.zeros_like(gi_errors)]),
    )

    # Estimate initial parameters
    initial_params = _estimate_initial_parameters(df)

    # Fix parameters if requested
    if n is not None:
        initial_params[6] = n
    if m is not None:
        initial_params[7] = m

    # Apply parameter bounds
    initial_params = _validate_parameters(initial_params, bounds)

    # Set up parameter fixing
    fix_list = [0, 0, 0, 0, 0, 0, 0, 0]  # 0 = free, 1 = fixed
    if n is not None:
        fix_list[6] = 1
    if m is not None:
        fix_list[7] = 1

    # Create ODR model
    odr_model = odr.Model(_fracture_toughness_residual, implicit=True)

    # Perform ODR fitting
    odr_obj = odr.ODR(odr_data, odr_model, beta0=initial_params, ifixb=fix_list)
    odr_output = odr_obj.run()

    # Extract results
    fitted_params = odr_output.beta
    param_errors = odr_output.sd_beta

    # Calculate fit statistics
    residuals = _fracture_toughness_residual(fitted_params, data_array)
    n_data = len(residuals)
    n_params = np.sum(np.array(fix_list) == 0)  # Count free parameters
    dof = n_data - n_params

    chi_squared = np.sum(residuals**2)
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf

    # Calculate p-value
    p_value = 1.0 - stats.chi2.cdf(chi_squared, dof) if dof > 0 else 0.0

    # Calculate R-squared
    r_squared = _calculate_r_squared(residuals, data_array)

    # Create result object
    result = FractureToughnessResult(
        GIc_1=fitted_params[0],
        GIIc_1=fitted_params[1],
        GIc_2=fitted_params[2],
        GIIc_2=fitted_params[3],
        GIc_3=fitted_params[4],
        GIIc_3=fitted_params[5],
        n=fitted_params[6],
        m=fitted_params[7],
        GIc_1_err=param_errors[0],
        GIIc_1_err=param_errors[1],
        GIc_2_err=param_errors[2],
        GIIc_2_err=param_errors[3],
        GIc_3_err=param_errors[4],
        GIIc_3_err=param_errors[5],
        n_err=param_errors[6],
        m_err=param_errors[7],
        reduced_chi_squared=reduced_chi_squared,
        chi_squared=chi_squared,
        p_value=p_value,
        r_squared=r_squared,
        degrees_of_freedom=dof,
        n_data_points=n_data,
        convergence_info={
            "info": odr_output.info,
            "stopreason": odr_output.stopreason,
            "iterations": getattr(odr_output, "iwork", [0])[0]
            if hasattr(odr_output, "iwork")
            else 0,
        },
        residuals=residuals,
    )

    return result
