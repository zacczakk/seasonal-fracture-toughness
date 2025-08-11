import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, Output, RealData
from scipy.stats import distributions
from uncertainties import unumpy as unp

logger = logging.getLogger(__name__)
# Avoid "No handler found" warnings if the app doesn't configure logging
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _arr_stats(name: str, arr: np.ndarray) -> str:
    """Return a compact one-line summary of an array's finiteness and range."""
    arr = np.asarray(arr)
    total = arr.size
    nonfinite = int(np.sum(~np.isfinite(arr)))
    try:
        amin = float(np.nanmin(arr))
        amax = float(np.nanmax(arr))
    except ValueError:
        amin = float("nan")
        amax = float("nan")
    return f"{name}: nonfinite={nonfinite}/{total}, min={amin}, max={amax}"


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

    def softplus(z):
        """Smoothly map to (0, inf)."""
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def smooth_violation(x, lo, hi, k=1e2):
        """≈ 0 inside [lo, hi]; grows smoothly outside; scales with k."""
        return softplus(k * (lo - x)) + softplus(k * (x - hi))

    # Unpack multidimensional inputs
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi, Gii = x  # vector with all datapoints for all series

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc_1, GIc_2, GIc_3])
    GIIc = np.choose(series_idx, [GIIc_1, GIIc_2, GIIc_3])

    # Parameter bounds
    eps = 1e-6
    Gc_max = 3

    # Penalty for violations of bounds
    penalty = 1e3 * (
        smooth_violation(GIc, 0, 3, k=1e2)  # Don't force, only nudge
        + smooth_violation(GIIc, 0, 3, k=1e2)  # Don't force, only nudge
        # + smooth_violation(n, 0, 1.1, k=1e3)
        # + smooth_violation(m, 0, 1.1, k=1e3)
    )

    # Prevent NaNs by strictly enforcing bounds before power-law
    # operations and penalize violations later
    GIc = np.clip(GIc, eps, Gc_max)
    GIIc = np.clip(GIIc, eps, Gc_max)
    n = np.clip(n, eps, 1)
    m = np.clip(m, eps, 1)

    # Data points may become negative because of uncertainties
    Gi = np.where(np.isfinite(Gi), Gi, eps)
    Gii = np.where(np.isfinite(Gii), Gii, eps)
    Gi = np.maximum(Gi, eps)
    Gii = np.maximum(Gii, eps)

    mode_I_residuals = np.maximum(Gi / GIc, 1e-6)
    mode_II_residuals = np.maximum(Gii / GIIc, 1e-6)

    if not (
        np.isfinite(mode_I_residuals).all() and np.isfinite(mode_II_residuals).all()
    ):
        logger.debug(
            "non-finite base detected | %s | %s | n=%s m=%s | %s | %s",
            _arr_stats("base1", mode_I_residuals),
            _arr_stats("base2", mode_II_residuals),
            n,
            m,
            _arr_stats("GIc", GIc),
            _arr_stats("GIIc", GIIc),
        )

    with np.errstate(all="ignore"):
        residuals = mode_I_residuals ** (1.0 / n) + mode_II_residuals ** (1.0 / m) - 1.0
        # residuals = (Gi / GIc) ** (1.0 / n) + (Gii / GIIc) ** (1.0 / m) - 1.0

    residuals += penalty

    # Log if nans or infs in residuals
    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        logger.debug(
            "non-finite residuals | beta=%s | nans=%d infs=%d",
            beta,
            int(np.sum(np.isnan(residuals))),
            int(np.sum(np.isinf(residuals))),
        )

    # Smoothly penalize NaNs and Infs
    # penalty = 1e3 * np.tanh(np.abs(residuals))
    # residuals = np.where(~np.isfinite(residuals), penalty, residuals)
    residuals = np.where(~np.isfinite(residuals), 1e6, residuals)

    # Optional: debug when penalties are applied substantially
    # if logger.isEnabledFor(logging.DEBUG):
    #     pen_sum = float(np.nansum(penalty))
    #     if pen_sum > 0:
    #         logger.debug("penalty_sum=%s", pen_sum)

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

    # Unpack multidimensional inputs
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi, Gii = x  # vector with all datapoints for all series

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc_1, GIc_2, GIc_3])
    GIIc = np.choose(series_idx, [GIIc_1, GIIc_2, GIIc_3])

    # Parameter bounds
    eps = 1e-6
    Gc_max = 3

    # Prevent NaNs by strictly enforcing bounds before power-law
    # operations and penalize violations later
    GIc = np.clip(GIc, eps, Gc_max)
    GIIc = np.clip(GIIc, eps, Gc_max)
    n = np.clip(n, eps, 1)
    m = np.clip(m, eps, 1)

    # Data points may become negative because of uncertainties
    Gi = np.where(np.isfinite(Gi), Gi, eps)
    Gii = np.where(np.isfinite(Gii), Gii, eps)
    Gi = np.maximum(Gi, eps)
    Gii = np.maximum(Gii, eps)

    # Calculate derivatives (each np.array of length of x)
    with np.errstate(invalid="ignore"):
        dGIc = -(((Gi / GIc) ** (1 / n) * (1 / n)) / GIc)
        dGIIc = -(((Gii / GIIc) ** (1 / m) * (1 / m)) / GIIc)
        dn = (Gi / GIc) ** (1 / n) * np.log(Gi / GIc) * (-1 / n**2)
        dm = (Gii / GIIc) ** (1 / m) * np.log(Gii / GIIc) * (-1 / m**2)

    # Gradient is 0 for all series except the one being fit
    dGIc1 = np.where(series_idx == 0, dGIc, 0)
    dGIIc1 = np.where(series_idx == 0, dGIIc, 0)
    dGIc2 = np.where(series_idx == 1, dGIc, 0)
    dGIIc2 = np.where(series_idx == 1, dGIIc, 0)
    dGIc3 = np.where(series_idx == 2, dGIc, 0)
    dGIIc3 = np.where(series_idx == 2, dGIIc, 0)

    # Stack derivaties (number of rowns corresponds to of length of beta)
    return np.row_stack([dGIc1, dGIIc1, dGIc2, dGIIc2, dGIc3, dGIIc3, dn, dm])


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
    # Unpack multidimensional inputs
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi, Gii = x  # vector with all datapoints for all series

    # Choose which GIc and GIIc to use based on series index
    GIc = np.choose(series_idx, [GIc_1, GIc_2, GIc_3])
    GIIc = np.choose(series_idx, [GIIc_1, GIIc_2, GIIc_3])

    # Parameter bounds
    eps = 1e-6
    Gc_max = 3

    # Prevent NaNs by strictly enforcing bounds before power-law
    # operations and penalize violations later
    GIc = np.clip(GIc, eps, Gc_max)
    GIIc = np.clip(GIIc, eps, Gc_max)
    n = np.clip(n, eps, 1)
    m = np.clip(m, eps, 1)

    # Data points may become negative because of uncertainties
    Gi = np.where(np.isfinite(Gi), Gi, eps)
    Gii = np.where(np.isfinite(Gii), Gii, eps)
    Gi = np.maximum(Gi, eps)
    Gii = np.maximum(Gii, eps)

    # Calculate derivatives (each np.array of length of x)
    dGi = ((Gi / GIc) ** (1 / n) * (1 / n)) / Gi
    dGii = ((Gii / GIIc) ** (1 / m) * (1 / m)) / Gii
    # Stack derivaties
    return np.row_stack([dGi, dGii])


def assemble_data(
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
    try:
        source_df = df.xs(source, level="source")
    except KeyError as exc:
        raise ValueError(
            f"Source '{source}' not found in DataFrame MultiIndex level 'source'."
        ) from exc

    # Get the names and indices of all measurement series
    series_names = sorted(
        source_df.index.get_level_values("series").unique()
    )  # '1', '2', '3'
    series_to_idx = {s: i for i, s in enumerate(series_names)}  # 0, 1, 2

    # Get data to fit from dataframe
    Gi = unp.nominal_values(source_df["GIc"].to_numpy())  # Gi from 'GIc'
    Gii = unp.nominal_values(source_df["GIIc"].to_numpy())  # Gii from 'GIIc'
    Gi_err = unp.std_devs(source_df["GIc"].to_numpy())
    Gii_err = unp.std_devs(source_df["GIIc"].to_numpy())

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "assemble_data | source=%s | %s | %s | %s | %s",
            source,
            _arr_stats("Gi", Gi),
            _arr_stats("Gii", Gii),
            _arr_stats("Gi_err", Gi_err),
            _arr_stats("Gii_err", Gii_err),
        )

    # Map series names to indices
    series_idx = (
        source_df.index.get_level_values("series").map(series_to_idx).to_numpy()
    )

    # Stack data and uncertainties as input array
    x = np.row_stack([Gi, Gii])
    sx = np.row_stack([Gi_err, Gii_err])

    # Pack data in scipy ODR format. Scalar input for y implies
    # that the model to be used on the data is implicit
    data = RealData(x, y=1, sx=sx)

    # Return data object and series indices
    return data, series_names, series_idx


def run_regression(
    data,
    model,
    beta0,
    sstol=1e-12,
    partol=1e-12,
    maxit=1000,
    ndigit=12,
    ifixb=[1, 1, 1, 1, 1, 1, 0, 0],
    fit_type=1,
    deriv=3,  # should not use jacobians when fitting 3 series simultaneously
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

    # Logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "run_regression | beta0=%s | sstol=%s partol=%s maxit=%s ndigit=%s ifixb=%s fit_type=%s deriv=%s",
            beta0,
            sstol,
            partol,
            maxit,
            ndigit,
            ifixb,
            fit_type,
            deriv,
        )

    # Run optimization
    out = odr.run()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "run_done | info=%s stop=%s sumsq=%s res_var=%s",
            getattr(out, "info", None),
            getattr(out, "stopreason", None),
            getattr(out, "sum_square", None),
            getattr(out, "res_var", None),
        )
    return out


def calc_fit_statistics(final: Output) -> Dict[str, Any]:
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
    ndof : int
        Number of degrees of freedom.
    fit : dict
        Dictoinary to store fit data. Default is defaultdict(dict).

    Returns
    -------
    fit : dict
        Updated dictionary.
    """
    chi2 = final.sum_square  # type: ignore
    chi2_red = getattr(final, "res_var", np.nan)  # type: ignore
    if chi2_red and chi2_red != 0:
        ndof = chi2 / chi2_red  # type: ignore
    else:
        ndof = float("nan")

    # Guard p-value and R^2 to avoid division by zero / invalid dof
    if np.isfinite(ndof) and ndof > 0:
        p_val = distributions.chi2.sf(chi2, ndof)
        R2 = 1 - chi2 / (ndof + chi2)
    else:
        p_val = 1.0 if chi2 == 0 else float("nan")
        R2 = 1.0 if chi2 == 0 else 0.0

    return {
        "params": final.beta,
        "stddev": final.sd_beta,
        "reduced_chi_squared": chi2_red,
        "chi_squared": chi2,
        "ndof": ndof,
        "p_value": p_val,
        "R_squared": R2,
        "final": final,
    }


def results(fit: Dict[str, Any]) -> None:
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

    # Define the header and horizontal rules
    header = "Variable   Value   Description".upper()
    rule = "---".join(["-" * s for s in [8, 5, 50]])

    # Print the header
    print(header)
    print(rule)

    # Print fit paramters
    print(f"GIc_1      {GIc_1:5.3f}   Mode I fracture toughness")
    print(f"GIIc_1     {GIIc_1:5.3f}   Mode II fracture toughness")
    print(f"GIc_2      {GIc_2:5.3f}   Mode I fracture toughness")
    print(f"GIIc_2     {GIIc_2:5.3f}   Mode II fracture toughness")
    print(f"GIc_3      {GIc_3:5.3f}   Mode I fracture toughness")
    print(f"GIIc_3     {GIIc_3:5.3f}   Mode II fracture toughness")
    print(f"n          {n:5.2f}   Interaction-law exponent")
    print(f"m          {m:5.2f}   Interaction-law exponent")
    print(rule)

    # Print goodness-of-fit indicators
    print(f"chi2       {chi2:5.3f}   Reduced chi^2 per DOF (goodness of fit)")
    print(f"p-value    {pval:5.3f}   p-value (statistically significant if below 0.05)")
    print(f"R2         {R2:5.3f}   R-squared (not valid for nonlinear regression)")
    print()


def odr(
    df,
    source,
    print_results=True,
    log_level: int | None = None,
    log_file: str | None = None,
):
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
    # Optionally set logger level for this call
    prev_level = None
    prev_propagate = None
    file_handler: logging.Handler | None = None
    if log_level is not None:
        prev_level = logger.level
        logger.setLevel(log_level)
    if log_file is not None:
        try:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            effective_level = (
                log_level if log_level is not None else logger.getEffectiveLevel()
            )
            file_handler.setLevel(effective_level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(file_handler)
            # Ensure logs don't propagate to root handlers (e.g., Jupyter console)
            prev_propagate = logger.propagate
            logger.propagate = False
        except Exception:  # fallback silently if file handler creation fails
            file_handler = None

    # Assemble ODR pack data object
    data, series_names, series_idx = assemble_data(df, source)

    # Compile scipy ODR models
    model = Model(
        fcn=residual,
        fjacb=param_jacobian,
        fjacd=value_jacobian,
        implicit=True,
        extra_args=(series_idx,),
    )

    # Get the dataframe for the given source
    source_df = df.xs(source, level="source")

    # Get initial guesses for the fracture toughnesses
    gIc0 = [
        np.nanmedian(unp.nominal_values(source_df.loc[s, "GIc"])) for s in series_names
    ]
    gIIc0 = [
        np.nanmedian(unp.nominal_values(source_df.loc[s, "GIIc"])) for s in series_names
    ]

    # Create a grid of n and m values strictly within (0, 1], avoiding 0
    # to prevent division by zero or undefined exponents.
    nm_values = np.linspace(0.1, 1, 10)
    guess = [
        [val for pair in zip(gIc0, gIIc0) for val in pair] + [n, m]
        for n in nm_values
        for m in nm_values
    ]
    logger.info(
        "odr | source=%s | series=%s | n_grid=%d | guesses=%d | gIc0=%s | gIIc0=%s",
        source,
        ",".join(map(str, series_names)),
        len(nm_values),
        len(guess),
        np.array2string(np.asarray(gIc0), precision=3),
        np.array2string(np.asarray(gIIc0), precision=3),
    )
    # guess = [
    #     [val for pair in zip([0.11, 0.11, 0.11], [0.22, 0.22, 0.22]) for val in pair]
    #     + [n, n]
    #     for n in nm_values
    # ]

    # Run regression for all guesses. Prefer successful runs (info <= 3),
    # but fall back to the best run overall if none converge.
    runs_all = []
    for g in guess:
        try:
            runs_all.append(run_regression(data, model, g))
        except Exception:
            # Skip runs that error out completely
            continue
    if not runs_all:
        raise ValueError(
            "ODR failed to start for all initializations. Check input data for NaNs/Inf or invalid structure."
        )
    runs_ok = [r for r in runs_all if getattr(r, "info", 9) <= 3]
    logger.info("runs_ok=%d total_runs=%d", len(runs_ok), len(runs_all))
    if len(runs_ok) > 0:
        candidates = runs_ok
        # Determine run with smallest sum of squared errors among candidates
        final = min(candidates, key=lambda r: getattr(r, "sum_square", np.inf))  # type: ignore

        # Compile fit results dictionary with goodness-of-fit info
        fit = calc_fit_statistics(final)

        # Print fit results to console
        if print_results:
            results(fit)
    else:
        candidates = runs_all
        # Log info attribute for runs that failed to converge
        for r in runs_all[:5]:
            logger.warning(
                "nonconverged | info=%s stop=%s",
                getattr(r, "info", None),
                getattr(r, "stopreason", None),
            )
        # Determine run with smallest sum of squared errors among candidates
        final = min(candidates, key=lambda r: getattr(r, "sum_square", np.inf))  # type: ignore

        # Compile fit results dictionary with goodness-of-fit info
        fit = calc_fit_statistics(final)

    # Restore previous logger level if modified
    if prev_level is not None:
        logger.setLevel(prev_level)
    if file_handler is not None:
        try:
            if prev_propagate is not None:
                logger.propagate = prev_propagate
            logger.removeHandler(file_handler)
            file_handler.close()
        except Exception:
            pass

    # Return fit results dictionary
    return fit
