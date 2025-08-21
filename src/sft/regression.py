from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, Output, RealData
from scipy.stats import distributions
from uncertainties import unumpy as unp

from .residuals import param_jacobian, residual, value_jacobian


def _assemble_data(
    df: pd.DataFrame, source: str
) -> tuple[RealData, list[str], np.ndarray]:
    """
    Compile ODR pack data object from data frame.

    See https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.odr.RealData.html

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (source, series) DataFrame with columns 'GIc','GIIc' holding
        Gi, Gii as ufloats.
    source : str
        Source of the cut-length data (e.g. 'manual', 'video')

    Returns
    -------
    data : scipy.odr.RealData
        ODR pack data object.
    series_names : list[str]
        Names of the measurement series.
    series_idx : np.ndarray
        Indices of the measurement series.
    """
    # Get the dataframe for the given source
    source_df = df.xs(source, level="source")

    # Convert series names to series indices ('1', '2', '3') -> (0, 1, 2)
    series_names = sorted(source_df.index.get_level_values("series").unique())  # str
    series2idx = {s: i for i, s in enumerate(series_names)}  # int

    # Get data to fit from dataframe
    Gi = unp.nominal_values(source_df["GIc"].to_numpy())  # Gi from 'GIc' column
    Gii = unp.nominal_values(source_df["GIIc"].to_numpy())  # Gii from 'GIIc' column
    Gi_err = unp.std_devs(source_df["GIc"].to_numpy())
    Gii_err = unp.std_devs(source_df["GIIc"].to_numpy())

    # Map series names to indices
    series_idx = source_df.index.get_level_values("series").map(series2idx).to_numpy()

    # Stack data and uncertainties as input array
    x = np.vstack([Gi, Gii])
    sx = np.vstack([Gi_err, Gii_err])

    # Pack data in scipy ODR format. Scalar input for y implies
    # that the model to be used on the data is implicit
    data = RealData(x, y=1, sx=sx)

    # Return data object and series indices
    return data, series_names, series_idx


def _run_regression(
    data,
    model,
    beta0,
    sstol=1e-12,
    partol=1e-12,
    maxit=1000,
    ndigit=12,
    ifixb=[1, 1, 1, 1, 1, 1, 0, 0],
    fit_type=1,
    deriv=3,
    init=0,
    iteration=0,
    final=0,
):
    """
    Setup ODR object and run regression.

    See https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.odr.ODR.html
        scipy.odr.ODR.set_job.html
        scipy.odr.ODR.set_iprint.html

    Parameters
    ----------
    data : ODRData
        Scipy ODRpack data object.
    model : ODRmodel
        Scipy ODRpack model object.
    beta0 : list[float]
        List of initial parameter guesses.
    sstol : float, optional
        Tolerance for residual convergence (<1). Default is 1e-12.
    partol : float, optional
        Tolerance for parameter convergence (<1). Default is 1e-12.
    maxit : int, optional
        Maximum number of iterations. Default is 1000.
    ndigit : int, optional
        Number of reliable digits. Default is 12.
    ifixb : list[int], optional
        0 parameter fixed, 1 parameter free. Default is [1, 1, 1, 1, 1, 1, 0, 0].
    fit_type : int, optional
        0 explicit ODR, 1 implicit ODR. Default is 1.
    deriv : int, optional
        0 finite differences, 3 jacobians. Default is 0.
    init : int, optional
        No, short, or long initialization report. Default is 0.
    iteration : int, optional
        No, short, or long iteration report. Default is 0.
    final : int, optional
        No, short, or long final report. Default is 0.

    Returns
    -------
    scipy.odr.Output
        Optimization results object.
    """

    # Setup ODR object
    odr = ODR(
        data,  # Input data
        model,  # Model
        beta0=beta0,  # Initial parameter guess
        sstol=sstol,  # Tolerance for residual convergence (<1)
        partol=partol,  # Tolerance for parameter convergence (<1)
        maxit=maxit,  # Maximum number of iterations
        ndigit=ndigit,  # Number of reliable digits
        ifixb=ifixb,  # 0 parameter fixed, 1 parameter free
    )

    # Set job options
    odr.set_job(
        fit_type=fit_type,  # 0 explicit ODR, 1 implicit ODR
        deriv=deriv,  # 0 finite differences, 3 jacobians
    )

    # Define outputs
    odr.set_iprint(
        init=init,  # No, short, or long initialization report
        iter=iteration,  # No, short, or long iteration report
        final=final,  # No, short, or long final report
    )

    # Run optimization
    return odr.run()


def _calc_fit_statistics(final: Output) -> Dict[str, Any]:
    r"""
    Complement fit results dictionary with goodness of fit info.

    Check the scipy user forum (https://scipy-user.scipy.narkive.com/
    ZOHix6nj/scipy-odr-goodness-of-fit-and-parameter-estimation-for-
    explicit-orthogonal-distance-regression) for an explanation of ODR's
    goodness of fit estimation and Wikipedia https://en.wikipedia.org/
    wiki/Reduced_chi-squared_statistic) for an explanation of the
    reduced chi_nu^2 goodness-of-fit indicator.

    As a rule of thumb, when the variance of the measurement error is
    known a priori, a $\chi _{\nu }^{2}\gg 1$ indicates a poor model
    fit. A $\chi _{\nu }^{2}>1$ indicates that the fit has not fully
    captured the data (or that the error variance has been underestimated).
    In principle, a value of $\chi _{\nu }^{2}$ around 1 indicates that
    the extent of the match between observations and estimates is in
    accord with the error variance. A $\chi _{\nu }^{2}<1$ indicates that
    the model is "over-fitting" the data: either the model is improperly
    fitting noise, or the error variance has been overestimated.

    Parameters
    ----------
    final : scipy.odr.Output
        Optimization results object.

    Returns
    -------
    dict
        Fit results dictionary.
    """
    # Unpack parameters and their standard deviations
    p_values = getattr(final, "beta", np.nan)
    p_stddevs = getattr(final, "sd_beta", np.nan)

    # Unpack residuals
    eps = getattr(final, "eps", np.nan)

    # Calculate number of degrees of freedom
    p_free = int(np.count_nonzero(np.asarray(p_stddevs) > 0))  # no. of free params
    n_used = int(np.size(eps))  # no. of response residuals with nonzero weight
    ndof = n_used - p_free if n_used > p_free else np.nan

    # ODR tries to minimize both the residual values (sum_square_eps) and the distance
    # data points would need to be shifted (delta_Gi, delta_Gii) to fall exactly onto
    # the curve (sum_square_delta). They are summed up as sum_square. For implicit fits,
    # delta can become very large, i.e., sum_square_delta >> sum_square_eps, which
    # means estimating the number of degrees of freedom from the total sum of squares
    # is fragile. Therefore, we use the sum of squares of the residuals (sum_square_eps)
    # to get reduced chi squared and to compute the number of degrees of freedom.
    chi2 = getattr(final, "sum_square", np.nan)
    chi2_red = chi2 / ndof if (np.isfinite(ndof) and ndof > 0) else np.nan

    # Optimiation objective
    S_total = getattr(final, "sum_square", np.nan)

    # Guard p-value and R^2 to avoid division by zero / invalid dof
    if np.isfinite(ndof) and ndof > 0:
        p_val = distributions.chi2.sf(chi2, ndof)
        R2 = 1 - chi2 / (ndof + chi2)
    else:
        p_val = np.nan
        R2 = np.nan

    return {
        "params": p_values,
        "stddev": p_stddevs,
        "reduced_chi_squared": chi2_red,
        "chi_squared": chi2,
        "ndof": ndof,
        "S_total": S_total,
        "p_value": p_val,
        "R_squared": R2,
        "final": final,
    }


def _results(fit: Dict[str, Any]) -> None:
    r"""
    Print fit results to console.

    As a rule of thumb, when the variance of the measurement error is
    known a priori, a $\chi _{\nu }^{2}\gg 1$ indicates a poor model
    fit. A $\chi _{\nu }^{2}>1$ indicates that the fit has not fully
    captured the data (or that the error variance has been underestimated).
    In principle, a value of $\chi _{\nu }^{2}$ around 1 indicates that
    the extent of the match between observations and estimates is in
    accord with the error variance. A $\chi _{\nu }^{2}<1$ indicates that
    the model is "over-fitting" the data: either the model is improperly
    fitting noise, or the error variance has been overestimated.

    Parameters
    ----------
    fit : dict
        Dictionary with optimization results.
    """

    # Unpack variables
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = fit["final"].beta
    chi2 = fit["reduced_chi_squared"]
    pval = fit["p_value"]
    R2 = fit["R_squared"]
    ndof = fit["ndof"]
    S_total = fit["S_total"]

    # Handling np.nan gracefully
    def fmt(val, fmtstr="7.3f"):
        return f"{val:{fmtstr}}" if np.isfinite(val) else "nan"

    def rule(width: int = 50):
        return "-" * width

    # Print fit parameters in compact table format
    print(f"{'GIc':<5}   {'GIIc':<5}   DESCRIPTION")
    print(rule(45))
    print(f"{GIc_1:5.3f}   {GIIc_1:5.3f}   Series 1 fracture toughnesses")
    print(f"{GIc_2:5.3f}   {GIIc_2:5.3f}   Series 2 fracture toughnesses")
    print(f"{GIc_3:5.3f}   {GIIc_3:5.3f}   Series 3 fracture toughnesses")
    print()

    print("EXPONENTS   DESCRIPTION")
    print(rule(45))
    print(f"n  {n:6.3f}   Mode I interaction-law exponent")
    print(f"m  {m:6.3f}   Mode II interaction-law exponent")
    print()

    print("STATISTICS")
    print(rule(70))
    print(f"chi^2_red  {fmt(chi2)}   Residual sum of squares per DOF (goodness of fit)")
    print(f"p-value    {fmt(pval)}   p-value (significant if below 0.05)")
    print(f"R^2        {fmt(R2)}   R-squared (not valid for nonlinear regression)")
    print(f"nDOF       {ndof:7.0f}   n(DOF) as n(used obs) - n(free params)")
    print(f"S_total    {S_total:7.0f}   Total sum of squares (optimization objective)")
    print()


def assemble_initial_guesses(
    series_names,
    gc_list: list[float] = [0.1],
    nm_list: list[float] | list[tuple[float, float]] = np.linspace(0.1, 1.0, 10),
    tie_nm: bool = False,
):
    """
    Assemble initial guess arrays for regression.

    Parameters
    ----------
    series_names : list
        List of series names (for number of GIc/GIIc).
    gc_list : list of float
        List of values to use for both GIc and GIIc (always tied).
    nm_list : list of tuple or list of float
        List of (n, m) tuples if tie_nm is False, or list of n values if tie_nm is True.
    tie_nm : bool
        If True, use n=m for all guesses; if False, use (n, m) pairs from nm_list.

    Returns
    -------
    guesses : list
        List of initial guess arrays.
    """
    guesses = []
    for gc in gc_list:
        gIc0 = np.full(len(series_names), gc)
        gIIc0 = np.full(len(series_names), gc)
        if tie_nm:
            for n in nm_list:
                guesses.append(np.concatenate([gIc0, gIIc0, [n, n]]))
        else:
            for n in nm_list:
                for m in nm_list:
                    guesses.append(np.concatenate([gIc0, gIIc0, [n, m]]))
    return guesses


def odr(df, source, print_results=True):
    """
    Perform orthogonal distance regression (ODR) on the data frame.

    Scipy.odr is a wrapper around a much older FORTRAN77 package known as
    ODRPACK. The documentation for ODRPACK can actually be found on the
    scipy website: https://docs.scipy.org/doc/external/odrpack_guide.pdf

    See also https://docs.scipy.org/doc/scipy/reference/
            generated/scipy.odr.Model.html

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with energy release rates.
    source : str
        Source of the cut-length data (e.g. 'manual', 'video')
    print_results : bool
        If True, print fit results to console. Default is True.
    """
    # Assemble ODR pack data object
    data, series_names, series_idx = _assemble_data(df, source)

    # Compile scipy ODR models
    model = Model(
        fcn=residual,
        fjacb=param_jacobian,
        fjacd=value_jacobian,
        implicit=True,
        extra_args=(series_idx,),
    )

    # Assemble initial guesses
    guess = assemble_initial_guesses(
        series_names,
        gc_list=[0.3],
        nm_list=np.linspace(0.05, 1, 20),
        tie_nm=False,
    )

    # Run regressions with all start values
    fixed_params = [1, 1, 1, 1, 1, 1, 0, 0]
    all_runs = [_run_regression(data, model, g, ifixb=fixed_params) for g in guess]

    # Check results
    fit_results = [_calc_fit_statistics(r) for r in all_runs if r.info <= 3]
    if len(fit_results) == 0:
        print("No runs converged")
        return None
    else:
        filtered_results = [r for r in fit_results if r["reduced_chi_squared"] > 0]
        best_result = min(filtered_results, key=lambda r: r["S_total"])

        if print_results:
            _results(best_result)

    return best_result


def _run_single_guess_process(args):
    """Worker: build model/data in-process and run ODR for a single guess."""
    Gi, Gii, Gi_err, Gii_err, series_idx, beta0, fixed_params = args
    try:
        x = np.vstack([Gi, Gii])
        sx = np.vstack([Gi_err, Gii_err])
        data = RealData(x, y=1, sx=sx)
        model = Model(
            fcn=residual,
            fjacb=param_jacobian,
            fjacd=value_jacobian,
            implicit=True,
            extra_args=(series_idx,),
        )
        odr = ODR(
            data,
            model,
            beta0=beta0,
            sstol=1e-12,
            partol=1e-12,
            maxit=1000,
            ndigit=12,
            ifixb=fixed_params,
        )
        odr.set_job(fit_type=1, deriv=3)
        return odr.run()

    except Exception:
        return None


def parallel_odr(df, source, print_results=True, max_workers: int = 12):
    """
    Parallel version of odr(): runs all initial guesses concurrently using processes.
    """
    # Prepare arrays for pickling to workers
    source_df = df.xs(source, level="source")
    Gi = unp.nominal_values(source_df["GIc"].to_numpy())
    Gii = unp.nominal_values(source_df["GIIc"].to_numpy())
    Gi_err = unp.std_devs(source_df["GIc"].to_numpy())
    Gii_err = unp.std_devs(source_df["GIIc"].to_numpy())
    series_names = sorted(source_df.index.get_level_values("series").unique())
    series2idx = {s: i for i, s in enumerate(series_names)}
    series_idx = source_df.index.get_level_values("series").map(series2idx).to_numpy()

    # Set gc_list to span from min(min(Gi), min(Gii)) to max(max(Gi), max(Gii)), ensuring min > 0
    gc_min = max(min(np.min(Gi), np.min(Gii)), 1e-6)
    gc_max = max(np.max(Gi), np.max(Gii))
    gc_list = np.linspace(gc_min, gc_max, 5)

    # Initial guesses
    guess = assemble_initial_guesses(
        series_names,
        gc_list=gc_list,
        # nm_list=np.linspace(0.02, 1, 50),
        nm_list=np.linspace(0.02, 1, 100),
        tie_nm=False,
    )
    fixed_params = [1, 1, 1, 1, 1, 1, 0, 0]
    tasks = [(Gi, Gii, Gi_err, Gii_err, series_idx, g, fixed_params) for g in guess]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        all_runs = list(ex.map(_run_single_guess_process, tasks))

    # Check results
    fit_results = [_calc_fit_statistics(r) for r in all_runs if r.info <= 3]
    if len(fit_results) == 0:
        print("No runs converged")
        return None
    else:
        filtered_results = [r for r in fit_results if r["reduced_chi_squared"] > 0]
        best_result = min(filtered_results, key=lambda r: r["S_total"])

        if print_results:
            _results(best_result)

    return best_result
