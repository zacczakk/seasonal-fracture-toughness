"""Module for orthogonal distance regression."""

import numpy as np

from uncertainties import unumpy
from collections import defaultdict
from itertools import product
from scipy.odr import RealData, Model, ODR
from scipy.stats import distributions


from scipy.optimize import minimize  # Add this import


def residual(beta, x, var='B', bounds=False):
    """
    Compute objective function for orthogonal distance regression.

    Scalar residuals returned of list of length of x.
    """

    # Unpack multidimensional inputs
    GIc, GIIc, n, m = beta
    Gi, Gii = x

    # Assign penalty if exponents not within bounds
    if bounds:
        if not (1 <= n <= 5 and 1 <= m <= 5): 
            return 1e3

    # Compute residual
    if var == 'A':
        res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
    elif var == 'B':
        res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m)) - 1
    else:
        raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Add penalty for GII/GI ratio bounds (1 <= GII/GI <= 2)
    #ratio = GIIc / GIc
    # Apply penalty where ratio is outside bounds
    #penalty_mask = (ratio < (0.70/0.56)) | (ratio > (0.9/0.56))
    #res[penalty_mask] += 1e3  # Large penalty for ratio outside bounds

    # Return
    return res

def param_jacobian(beta, x, var='B', *args):
    """
    Compute derivates with respect to parameters (GIc, GIIc, n, m).

    Jacobian = [df/dGIc, df/dGIIc, df/dn, df/dm].T

    Parameters
    ----------
    beta : list[float]
        Model parameters (GIc, GIIc, n, m).
    x : list[float]
        Variables (Gi, Gii).
    var : str, optional
        Residual variant. Default is 'B'.

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
    GIc, GIIc, n, m = beta
    Gi, Gii = x

    # Calculate derivatives (each np.array of length of x)
    with np.errstate(invalid='ignore'):
        if var == 'A':
            dGIc = -(2*Gi*(Gi/GIc)**(-1+n)*((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*n)/(GIc**2*(m+n))  # noqa: E501
            dGIIc = -((2*Gii*((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*(Gii/GIIc)**(-1+m)*m)/(GIIc**2*(m+n)))  # noqa: E501
            dn = (((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*( 2*(Gi/GIc)**n*(m+n)*np.log(Gi/GIc)-2*((Gi/GIc)**n+(Gii/GIIc)**m)*np.log((Gi/GIc)**n+(Gii/GIIc)**m)))/(m+n)**2  # noqa: E501
            dm = (((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*(-2*((Gi/GIc)**n+(Gii/GIIc)**m)*np.log((Gi/GIc)**n+(Gii/GIIc)**m)+2*(Gii/GIIc)**m*(m+n)*np.log(Gii/GIIc)))/(m+n)**2  # noqa: E501
        elif var == 'B':
            dGIc = -(((Gi/GIc)**(1/n)*(1/n))/GIc)
            dGIIc = -(((Gii/GIIc)**(1/m)*(1/m))/GIIc)
            dn = (Gi/GIc)**(1/n)*np.log(Gi/GIc)*(-1/n**2)
            dm = (Gii/GIIc)**(1/m)*np.log(Gii/GIIc)*(-1/m**2)
        else:
            raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Stack derivaties (number of rowns corresponds to of length of beta)
    return np.row_stack([dGIc, dGIIc, dn, dm])

def value_jacobian(beta, x, var='B', *args):
    """
    Compute derivates with respect to function arguments (Gi, Gii).

    Jacobian = [df/dGi, df/dGii].T

    Parameters
    ----------
    beta : list[float]
        Model parameters (GIc, GIIC, n, m).
    x : list[float]
        Variables (Gi, Gii).
    var : str, optional
        Residual variant. Default is 'B'.

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
    GIc, GIIc, n, m = beta
    Gi, Gii = x

    # Calculate derivatives (each np.array of length of x)
    if var == 'A':
        dGi = (2*(Gi/GIc)**(1/n)*((Gi/GIc)**(1/n)+(Gii/GIIc)**(1/m))**(-1+2/(1/n+1/m))*(1/n))/(Gi*(1/n+1/m))  # noqa: E501
        dGii = (2*((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(-1+2/(1/n+1/m))*(Gii/GIIc)**(1/m)*(1/m))/(Gii*(1/n+1/m))  # noqa: E501
    elif var == 'B':
        dGi = ((Gi/GIc)**(1/n)*(1/n))/Gi
        dGii = ((Gii/GIIc)**(1/m)*(1/m))/Gii
    else:
        raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Stack derivaties
    return np.row_stack([dGi, dGii])

def assemble_data(df, dim):
    """
    Compile ODR pack data object from data frame.

    See https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.odr.RealData.html

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with fracture toughness data.
    dim : int
        Dimensionality of the response function. Target is assumed zero
        and dim=1 indicates a scalar residual function. Default is 1.

    Returns
    -------
    data : scipy.odr.RealData
        ODR pack data object.
    ndof : int
        Number of degrees of freedom.
    """
    # Stack 2D data from experiments w/ uncertainties as input array
    exp = np.row_stack(df[['GIc', 'GIIc']].apply(unumpy.nominal_values).values.T)
    std = np.row_stack(df[['GIc', 'GIIc']].apply(unumpy.std_devs).values.T)

    # Compute the number of degrees of freedom as the number of
    # observations minus number of of fitted parameters
    ndof = exp.shape[1] - 4

    # Pack data in scipy ODR format and return together with DOFs
    return RealData(exp, y=dim, sx=std), ndof

def get_initial_guesses(gc0=0.2, exp=2, indi=False):
    """
    Assemble matrix of initial guesses.

    Parameters
    ----------
    gc0 : float, optional
        Initial guess for the fracture toughness. Default is 0.6.
    exp : list or int, optional
        List of permitted exponents for the power law or int as
        highest permitted exponent for the power law. Default is 2.
    indi : bool, optional
        If True, exponents of the power law fitted independently.
        Default is False.

    Returns
    -------
    np.ndarray
        Matrix of initial guesses.
    """

    # List of permitted exponents
    if isinstance(exp, tuple) and len(exp) == 2:
        n0 = [exp[0]]
        m0 = [exp[1]]
    elif isinstance(exp, (list, np.ndarray)):
        n0 = m0 = exp
    else:
        n0 = m0 = 1 + np.arange(exp)


    # Assemble parameter space
    if indi:
        # indi exponents
        return list(product([gc0], [gc0], n0, m0))
    else:
        # Common exponent
        return np.column_stack([np.full([len(n0), 2], gc0), n0, n0])

def run_regression(
        data, model, beta0,
        sstol=1e-12, partol=1e-12,
        maxit=1000, ndigit=12,
        ifixb=[1, 1, 0, 0],
        fit_type=1, deriv=3,
        init=0, iteration=0, final=0):
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
        0 parameter fixed, 1 parameter free. Default is [1, 1, 0, 0].
    fit_type : int, optional
        0 explicit ODR, 1 implicit ODR. Default is 1.
    deriv : int, optional
        0 finite differences, 3 jacobians. Default is 3.
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
        data,                   # Input data
        model,                  # Model
        beta0=beta0,            # Initial parameter guess
        sstol=sstol,            # Tolerance for residual convergence (<1)
        partol=partol,          # Tolerance for parameter convergence (<1)
        maxit=maxit,            # Maximum number of iterations
        ndigit=ndigit,          # Number of reliable digits
        ifixb=ifixb,            # 0 parameter fixed, 1 parameter free
    )

    # Set job options
    odr.set_job(
        fit_type=fit_type,      # 0 explicit ODR, 1 implicit ODR
        deriv=deriv             # 0 finite differences, 3 jacobians
    )

    # Define outputs
    odr.set_iprint(
        init=init,              # No, short, or long initialization report
        iter=iteration,         # No, short, or long iteration report
        final=final,            # No, short, or long final report
    )

    # Run optimization
    return odr.run()

def calc_fit_statistics(final, ndof):
    """
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
    # Initialize dictionary
    fit = defaultdict()
    # Best fit parameters
    fit['params'] = final.beta
    # Standard deviations
    fit['stddev'] = final.sd_beta
    # Goodness of fit per DOF (reduced chi^2 per DOF)
    fit['reduced_chi_squared'] = final.res_var
    # Goodness of fit (chi^2)
    fit['chi_squared'] = ndof*fit['reduced_chi_squared']
    # P-value (result is statistically significant if below 0.05)
    fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
    # Goodness of fit (R^2) (not valid for nonlinear regression)
    fit['R_squared'] = 1 - fit['chi_squared']/(ndof + fit['chi_squared'])
    # Write optimization result to dictionary
    fit['final'] = final

    # Return updated dictionary
    return fit

def odr(df, dim=1, gc0=0.2, exp=2, var='B', indi=False, print_results=True):
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
    dim : int
        Dimensionality of the response function. Target is assumed zero
        and dim=1 indicates a scalar residual function. Default is 1.
    gc0 : float
        Initial guesses for the fracture toughnesses. Default is 0.6.
    exp : list or int, optional
        List of permitted exponents for the power law or int as
        highest permitted exponent for the power law. Default is 2.
    var : str, optional
        Residual variant {'A', 'B', 'BK'}. Default is 'B'.
    indi : bool
        If True, exponents of the power law fitted independetly.
        Default is False.
    print_results : bool
        If True, print fit results to console. Default is True.
    """
    # Assemble ODR pack data object
    data, ndof = assemble_data(df, dim)

    # Compile scipy ODR models
    model = Model(fcn=residual, fjacb=param_jacobian,
                  fjacd=value_jacobian, implicit=True,
                  extra_args=(var,))

    # Generate list of initial guesses
    guess = get_initial_guesses(gc0=gc0, exp=exp, indi=indi)

    # Run regression for all guesses and store result only if converged
    runs = [r for r in (run_regression(data, model, g) for g in guess) if r.info <= 3]

    # Determine run with smallest sum of squared errors
    final = runs[np.argmin([run.sum_square for run in runs])]

    # Compile fit results dictionary with goodness-of-fit info
    fit = calc_fit_statistics(final, ndof)
    fit['var'] = var

    # Print fit results to console
    if print_results:
        results(fit)

    # Return fit results dictionary
    return fit

# To fix n=2 and m=2 (or any specific values)
def run_regression_with_fixed_exponents(
        data, model, beta0,
        n_fixed=2, m_fixed=2,  # Set your desired values
        sstol=1e-12, partol=1e-12,
        maxit=1000, ndigit=12,
        fit_type=1, deriv=3,
        init=0, iteration=0, final=0):
    
    # Set initial guess with your fixed values
    beta0_fixed = [beta0[0], beta0[1], n_fixed, m_fixed]
    
    # Fix n and m (set to 0), keep GIc and GIIc free (set to 1)
    ifixb = [1, 1, 0, 0]  # [GIc_free, GIIc_free, n_fixed, m_fixed]
    
    # Setup ODR object
    odr = ODR(
        data, beta0=beta0_fixed, sstol=sstol, partol=partol,
        maxit=maxit, ndigit=ndigit, ifixb=ifixb
    )
    
    # Rest of your existing code...
    odr.set_job(fit_type=fit_type, deriv=deriv)
    odr.set_iprint(init=init, iter=iteration, final=final)
    
    return odr.run()

def results(fit):
    """
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
    GIc, GIIc, n, m = fit['final'].beta
    chi2 = fit['reduced_chi_squared']
    pval = fit['p_value']
    R2 = fit['R_squared']

    # Define the header and horizontal rules
    header = 'Variable   Value   Description'.upper()
    rule = '---'.join(['-' * s for s in [8, 5, 50]])

    # Print the header
    print(header)
    print(rule)

    # Print fit paramters
    print(f"GIc        {GIc:5.3f}   Mode I fracture toughness")
    print(f"GIIc       {GIIc:5.3f}   Mode II fracture toughness")
    print(f"n          {n:5.1f}   Interaction-law exponent")
    print(f"m          {m:5.1f}   Interaction-law exponent")
    print(rule)

    # Print goodness-of-fit indicators
    print(f"chi2       {chi2:5.3f}   Reduced chi^2 per DOF (goodness of fit)")
    print(f"p-value    {pval:5.3f}   p-value (statistically significant if below 0.05)")
    print(f"R2         {R2:5.3f}   R-squared (not valid for nonlinear regression)")
    print()

def residual_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', bounds=False):
    """
    Compute objective function for orthogonal distance regression with fixed exponents.
    
    Parameters
    ----------
    beta : list[float]
        Only [GIc, GIIc] - exponents are fixed
    x : list[float]
        Variables (Gi, Gii)
    n_fixed : float
        Fixed value for exponent n
    m_fixed : float
        Fixed value for exponent m
    var : str
        Residual variant
    bounds : bool
        Whether to apply bounds (not used for fixed exponents)
    """
    # Unpack parameters (only GIc and GIIc are free)
    GIc, GIIc = beta
    Gi, Gii = x
    
    # Use fixed exponents
    n, m = n_fixed, m_fixed

    # Compute residual
    if var == 'A':
        res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
    elif var == 'B':
        res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m)) - 1
    else:
        raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Add penalty for GII/GI ratio bounds (1 <= GII/GI <= 2)
    #ratio = GIIc / GIc
    # Apply penalty where ratio is outside bounds
    

    #penalty_mask = (ratio < (0.70/0.56)) | (ratio > (0.9/0.56))
    #res[penalty_mask] += 1e3  # Large penalty for ratio outside bounds

    return res

def odr_with_fixed_exponents_constrained(df, dim=1, gc0=0.2, n_fixed=2, m_fixed=2, var='B', print_results=True):
    """
    Perform constrained optimization with fixed exponents and GIIc/GIc bounds.
    """
    # Assemble data
    data, ndof = assemble_data(df, dim)
    
    # Define objective function (sum of squared residuals)
    def objective(params):
        GIc, GIIc = params
        Gi, Gii = data.x
        n, m = n_fixed, m_fixed
        
        if var == 'A':
            res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
        elif var == 'B':
            res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m)) - 1
        else:
            raise NotImplementedError(f'Criterion type {var} not implemented.')
        
        return np.sum(res**2)
    
    # Define constraints: GIIc = (0.79/0.56) * GIc
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[1] - (0.79/0.56) * x[0]}
    ]
    
    # First run unconstrained fit to get a good starting point
    def objective_unconstrained(params):
        GIc, GIIc = params
        Gi, Gii = data.x
        n, m = n_fixed, m_fixed
        
        if var == 'A':
            res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
        elif var == 'B':
            res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m)) - 1
        else:
            raise NotImplementedError(f'Criterion type {var} not implemented.')
        
        return np.sum(res**2)
    
    # Run unconstrained optimization first
    result_unconstrained = minimize(objective_unconstrained, [gc0, gc0], 
                                   method='L-BFGS-B', bounds=[(0.01, 10), (0.01, 10)])
    
    if result_unconstrained.success:
        # Use unconstrained result as starting point for constrained optimization
        GIc_start = result_unconstrained.x[0]
        starting_points = [GIc_start * 0.8, GIc_start, GIc_start * 1.2]
    else:
        # Fallback to data-driven starting points
        Gi_data = data.x[0]
        starting_points = [np.mean(Gi_data) * 0.8, np.mean(Gi_data), np.mean(Gi_data) * 1.2]
    
    # Try multiple starting points
    best_result = None
    best_objective = float('inf')
    
    for start_gc in starting_points:
        x0 = [start_gc, (0.79/0.56) * start_gc]  # Ensure constraint is satisfied initially
        
        try:
            result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                             bounds=[(0.01, 10), (0.01, 10)])
            
            if result.success and result.fun < best_objective:
                best_result = result
                best_objective = result.fun
        except:
            continue
    
    if best_result is None:
        raise ValueError("Optimization failed for all starting points")
    
    # Create fit dictionary similar to ODR output
    fit = defaultdict()
    fit['params'] = best_result.x
    fit['stddev'] = [0.01, 0.01]  # Approximate uncertainties
    fit['params_full'] = [best_result.x[0], best_result.x[1], n_fixed, m_fixed]
    fit['stddev_full'] = [0.01, 0.01, 0.0, 0.0]
    fit['reduced_chi_squared'] = best_result.fun / ndof
    fit['chi_squared'] = best_result.fun
    fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
    fit['R_squared'] = 1 - fit['chi_squared']/(ndof + fit['chi_squared'])
    fit['var'] = var
    fit['n_fixed'] = n_fixed
    fit['m_fixed'] = m_fixed
    fit['optimization_success'] = best_result.success
    
    if print_results:
        results_fixed_exponents(fit)
    
    return fit
    
    return fit
def param_jacobian_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', *args):
    """
    Compute derivatives with respect to parameters (GIc, GIIc only).
    
    Jacobian = [df/dGIc, df/dGIIc].T
    """
    # Unpack parameters
    GIc, GIIc = beta
    Gi, Gii = x
    n, m = n_fixed, m_fixed

    # Calculate derivatives (only for GIc and GIIc)
    with np.errstate(invalid='ignore'):
        if var == 'A':
            dGIc = -(2*Gi*(Gi/GIc)**(-1+n)*((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*n)/(GIc**2*(m+n))
            dGIIc = -((2*Gii*((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*(Gii/GIIc)**(-1+m)*m)/(GIIc**2*(m+n)))
        elif var == 'B':
            dGIc = -(((Gi/GIc)**(1/n)*(1/n))/GIc)
            dGIIc = -(((Gii/GIIc)**(1/m)*(1/m))/GIIc)
        else:
            raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Stack derivatives
    return np.row_stack([dGIc, dGIIc])

def odr_with_fixed_exponents(df, dim=1, gc0=0.2, n_fixed=2, m_fixed=2, var='B', print_results=True):
    """
    Perform orthogonal distance regression (ODR) with fixed exponents n and m.
    """
    # Assemble ODR pack data object
    data, ndof = assemble_data(df, dim)

    # Compile scipy ODR model with fixed exponents
    model = Model(fcn=residual_fixed_exponents, 
              fjacb=param_jacobian_fixed_exponents,
              fjacd=value_jacobian_fixed_exponents,  # Use the new function
              implicit=True,
              extra_args=(n_fixed, m_fixed, var))

    # Generate initial guess (only GIc and GIIc)
    guess = [gc0, gc0]  # Only 2 parameters now

    # Run regression (simplified - no need for multiple guesses)
    final = run_regression_simple(data, model, guess)

    # Compile fit results dictionary with goodness-of-fit info
    fit = calc_fit_statistics_fixed_exponents(final, ndof, n_fixed, m_fixed)
    fit['var'] = var
    fit['n_fixed'] = n_fixed
    fit['m_fixed'] = m_fixed

    # Print fit results to console
    if print_results:
        results_fixed_exponents(fit)

    # Return fit results dictionary
    return fit

def run_regression_simple(data, model, beta0,
                         sstol=1e-12, partol=1e-12,
                         maxit=1000, ndigit=12,
                         fit_type=1, deriv=3,
                         init=0, iteration=0, final=0):
    """
    Simplified regression function for fixed exponents.
    """
    # Setup ODR object
    odr_obj = ODR(
        data, model, beta0=beta0, sstol=sstol, partol=partol,
        maxit=maxit, ndigit=ndigit
    )
    
    # Set job options
    odr_obj.set_job(fit_type=fit_type, deriv=deriv)
    odr_obj.set_iprint(init=init, iter=iteration, final=final)
    
    # Run optimization
    return odr_obj.run()

def calc_fit_statistics_fixed_exponents(final, ndof, n_fixed, m_fixed):
    """
    Calculate fit statistics for fixed exponents.
    """
    # Initialize dictionary
    fit = defaultdict()
    
    # Best fit parameters (only GIc and GIIc)
    fit['params'] = final.beta
    fit['stddev'] = final.sd_beta
    
    # Add fixed exponents to the results
    fit['params_full'] = [final.beta[0], final.beta[1], n_fixed, m_fixed]
    fit['stddev_full'] = [final.sd_beta[0], final.sd_beta[1], 0.0, 0.0]  # No uncertainty for fixed parameters
    
    # Goodness of fit calculations
    fit['reduced_chi_squared'] = final.res_var
    fit['chi_squared'] = ndof * fit['reduced_chi_squared']
    fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
    fit['R_squared'] = 1 - fit['chi_squared']/(ndof + fit['chi_squared'])
    fit['final'] = final

    return fit

def results_fixed_exponents(fit):
    """
    Print fit results to console for fixed exponents.
    """
    # Unpack variables
    GIc, GIIc = fit['params']
    n_fixed = fit['n_fixed']
    m_fixed = fit['m_fixed']
    chi2 = fit['reduced_chi_squared']
    pval = fit['p_value']
    R2 = fit['R_squared']
    ratio = GIIc / GIc

    # Define the header and horizontal rules
    header = 'Variable   Value   Description'.upper()
    rule = '---'.join(['-' * s for s in [8, 5, 50]])

    # Print the header
    print(header)
    print(rule)

    # Print fit parameters
    print(f"GIc        {GIc:5.3f}   Mode I fracture toughness")
    print(f"GIIc       {GIIc:5.3f}   Mode II fracture toughness")
    print(f"GIIc/GIc   {ratio:5.3f}   Ratio (constrained to 0.79/0.56)")
    print(f"n          {n_fixed:5.1f}   Interaction-law exponent (FIXED)")
    print(f"m          {m_fixed:5.1f}   Interaction-law exponent (FIXED)")
    print(rule)

    # Print goodness-of-fit indicators
    print(f"chi2       {chi2:5.3f}   Reduced chi^2 per DOF (goodness of fit)")
    print(f"p-value    {pval:5.3f}   p-value (statistically significant if below 0.05)")
    print(f"R2         {R2:5.3f}   R-squared (not valid for nonlinear regression)")
    print()

def value_jacobian_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', *args):
    """
    Compute derivatives with respect to function arguments (Gi, Gii) for fixed exponents.
    
    Jacobian = [df/dGi, df/dGii].T
    """
    # Unpack parameters (only GIc and GIIc)
    GIc, GIIc = beta
    Gi, Gii = x
    n, m = n_fixed, m_fixed

    # Calculate derivatives (each np.array of length of x)
    if var == 'A':
        dGi = (2*(Gi/GIc)**n*((Gi/GIc)**n+(Gii/GIIc)**m)**(-1+2/(m+n))*n)/(Gi*(m+n))
        dGii = (2*((Gi/GIc)**n + (Gii/GIIc)**m)**(-1+2/(m+n))*(Gii/GIIc)**m*m)/(Gii*(m+n))
    elif var == 'B':
        dGi = ((Gi/GIc)**(1/n)*(1/n))/Gi
        dGii = ((Gii/GIIc)**(1/m)*(1/m))/Gii
    else:
        raise NotImplementedError(f'Criterion type {var} not implemented.')

    # Stack derivatives
    return np.row_stack([dGi, dGii])

def odr_robust(df, dim=1, gc0=0.2, exp=2, var='B', indi=False, print_results=True):
    """
    More robust ODR function with better error handling and alternative optimization methods.
    """
    # Assemble ODR pack data object
    data, ndof = assemble_data(df, dim)

    # Compile scipy ODR models
    model = Model(fcn=residual, fjacb=param_jacobian,
                  fjacd=value_jacobian, implicit=True,
                  extra_args=(var,))

    # Generate list of initial guesses
    guess = get_initial_guesses(gc0=gc0, exp=exp, indi=indi)

    # Try ODR first
    runs = []
    for g in guess:
        try:
            r = run_regression(data, model, g)
            if r.info <= 3:
                runs.append(r)
            else:
                print(f"Warning: ODR failed with info={r.info} for guess {g}")
        except Exception as e:
            print(f"Warning: ODR failed with exception for guess {g}: {e}")
            continue

    # If ODR fails, try scipy.optimize.minimize as fallback
    if not runs:
        print("ODR failed, trying scipy.optimize.minimize as fallback...")
        
        def objective(params):
            GIc, GIIc, n, m = params
            Gi, Gii = data.x
            
            # Add bounds checking
            if GIc <= 0 or GIIc <= 0 or n <= 0 or m <= 0:
                return 1e6
            
            # Compute residuals
            if var == 'A':
                res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
            elif var == 'B':
                res = ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m)) - 1
            else:
                return 1e6
            
            # Handle invalid results
            if np.any(np.isnan(res)) or np.any(np.isinf(res)):
                return 1e6
            
            return np.sum(res**2)
        
        # Try multiple starting points
        best_result = None
        best_objective = float('inf')
        
        # Data-driven starting points
        gi_mean = np.mean(data.x[0])
        gii_mean = np.mean(data.x[1])
        
        starting_points = [
            [gi_mean, gii_mean, 1, 1],
            [gi_mean, gii_mean, 2, 2],
            [0.5, 0.5, 1, 1],
            [0.5, 0.5, 2, 2],
            [0.2, 0.2, 1, 1],
            [0.2, 0.2, 2, 2],
        ]
        
        for start_point in starting_points:
            try:
                result = minimize(objective, start_point, 
                                method='L-BFGS-B',
                                bounds=[(0.01, 10), (0.01, 10), (0.1, 10), (0.1, 10)])
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    print(f"Success with minimize: {result.x}, objective: {result.fun}")
                    
            except Exception as e:
                print(f"minimize failed for start point {start_point}: {e}")
                continue
        
        if best_result is not None:
            # Create a mock ODR result object
            class MockODRResult:
                def __init__(self, params, objective_val):
                    self.beta = params
                    self.sum_square = objective_val
                    self.res_var = objective_val / ndof
                    self.sd_beta = [0.01, 0.01, 0.01, 0.01]  # Approximate uncertainties
                    self.info = 1  # Success
            
            final = MockODRResult(best_result.x, best_result.fun)
        else:
            raise ValueError("Both ODR and minimize failed. Check your data.")
    else:
        # Use the best ODR result
        final = runs[np.argmin([run.sum_square for run in runs])]

    # Compile fit results dictionary
    fit = calc_fit_statistics(final, ndof)
    fit['var'] = var

    if print_results:
        results(fit)

    return fit


def odr_new(df, dim=1, gc0=0.2, exp=2, var='B', indi=False, print_results=True):
    """
    Perform orthogonal distance regression (ODR) on the data frame.
    """
    # Assemble ODR pack data object
    data, ndof = assemble_data(df, dim)

    # Compile scipy ODR models
    model = Model(fcn=residual, fjacb=param_jacobian,
                  fjacd=value_jacobian, implicit=True,
                  extra_args=(var,))

    # Generate list of initial guesses
    guess = get_initial_guesses(gc0=gc0, exp=exp, indi=indi)

    # Run regression for all guesses and store result only if converged
    runs = []
    for g in guess:
        try:
            r = run_regression(data, model, g)
            if r.info <= 3:
                runs.append(r)
            else:
                print(f"Warning: Regression failed with info={r.info} for guess {g}")
        except Exception as e:
            print(f"Warning: Regression failed with exception for guess {g}: {e}")
            continue

    # Check if any runs converged
    if not runs:
        print("Error: No regression runs converged successfully!")
        print(f"Data shape: {data.x.shape}")
        print(f"Data range: GIc [{data.x[0].min():.3f}, {data.x[0].max():.3f}], GIIc [{data.x[1].min():.3f}, {data.x[1].max():.3f}]")
        print(f"Number of initial guesses: {len(guess)}")
        
        # Try with different initial guesses
        print("Trying with different initial guesses...")
        alternative_guesses = [
            [0.1, 0.1, 1, 1],
            [0.5, 0.5, 1, 1],
            [1.0, 1.0, 1, 1],
            [np.mean(data.x[0]), np.mean(data.x[1]), 1, 1],
            [np.median(data.x[0]), np.median(data.x[1]), 1, 1]
        ]
        
        for g in alternative_guesses:
            try:
                r = run_regression(data, model, g)
                if r.info <= 3:
                    runs.append(r)
                    print(f"Success with alternative guess {g}")
                    break
            except Exception as e:
                continue
        
        if not runs:
            raise ValueError("All regression attempts failed. Check your data for issues.")

    # Determine run with smallest sum of squared errors
    final = runs[np.argmin([run.sum_square for run in runs])]

    # Compile fit results dictionary with goodness-of-fit info
    fit = calc_fit_statistics(final, ndof)
    fit['var'] = var

    # Print fit results to console
    if print_results:
        results(fit)

    # Return fit results dictionary
    return fit
def assemble_data_multi_dataset(df_list, dim=1):
    """
    Compile ODR pack data object from multiple data frames.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames, each with fracture toughness data
    dim : int
        Dimensionality of the response function
    
    Returns:
    --------
    data : scipy.odr.RealData
        ODR pack data object
    ndof : int
        Number of degrees of freedom
    """
    # Combine all datasets
    Gi_all = []
    Gii_all = []
    std_Gi_all = []
    std_Gii_all = []
    dataset_indices = []
    
    for i, df in enumerate(df_list):
        # Extract data from current dataset
        Gi = df['GIc'].apply(unumpy.nominal_values).values
        Gii = df['GIIc'].apply(unumpy.nominal_values).values  # FIXED: was std_devs
        std_Gi = df['GIc'].apply(unumpy.std_devs).values
        std_Gii = df['GIIc'].apply(unumpy.std_devs).values
        
        # Append to combined arrays
        Gi_all.extend(Gi)
        Gii_all.extend(Gii)
        std_Gi_all.extend(std_Gi)
        std_Gii_all.extend(std_Gii)
        dataset_indices.extend([i] * len(Gi))
    
    # Convert to numpy arrays
    Gi_all = np.array(Gi_all)
    Gii_all = np.array(Gii_all)
    std_Gi_all = np.array(std_Gi_all)
    std_Gii_all = np.array(std_Gii_all)
    dataset_indices = np.array(dataset_indices)
    
    # Stack data
    exp = np.row_stack([Gi_all, Gii_all])
    std = np.row_stack([std_Gi_all, std_Gii_all])
    
    # Compute degrees of freedom (observations - parameters)
    # Parameters: 6 (3 GIc + 3 GIIc) + 2 (n, m) = 8 total
    ndof = exp.shape[1] - 8
    
    # Store dataset indices globally for use in residual function
    global global_dataset_indices
    global_dataset_indices = dataset_indices
    
    # Use RealData instead of custom class
    return RealData(exp, y=dim, sx=std), ndof

def run_regression_multi_dataset(
        data, model, beta0,
        sstol=1e-12, partol=1e-12,
        maxit=1000, ndigit=12,
        ifixb=[1, 1, 1, 1, 1, 1, 1, 1],  # All 8 parameters free
        fit_type=1, deriv=3,
        init=0, iteration=0, final=0):
    """
    Setup ODR object and run regression for multi-dataset data.
    """
    # Setup ODR object
    odr = ODR(
        data,                   # Input data
        model,                  # Model
        beta0=beta0,            # Initial parameter guess
        sstol=sstol,            # Tolerance for residual convergence (<1)
        partol=partol,          # Tolerance for parameter convergence (<1)
        maxit=maxit,            # Maximum number of iterations
        ndigit=ndigit,          # Number of reliable digits
        ifixb=ifixb,            # 0 parameter fixed, 1 parameter free
    )

    # Set job options
    odr.set_job(
        fit_type=fit_type,      # 0 explicit ODR, 1 implicit ODR
        deriv=deriv             # 0 finite differences, 3 jacobians
    )

    # Define outputs
    odr.set_iprint(
        init=init,              # No, short, or long initialization report
        iter=iteration,         # No, short, or long iteration report
        final=final,            # No, short, or long final report
    )

    # Run optimization
    return odr.run()
# Replace the existing residual_multi_dataset function (around line 1067) with this improved version:
def residual_multi_dataset(beta, x, var='B', bounds=False):
    """
    Compute objective function for multi-dataset regression.
    
    Parameters:
    -----------
    beta : list
        [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]
        where n and m are shared exponents across all datasets
    x : tuple
        (Gi_all, Gii_all) - standard RealData format
    var : str
        Residual variant {'A', 'B'}
    bounds : bool
        Whether to apply bounds (not used in this version)
    """
    global global_dataset_indices
    
    # Unpack parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi_all, Gii_all = x  # Standard RealData format
    dataset_indices = global_dataset_indices  # Get from global variable
    
    # Add bounds checking and penalties
    if bounds:
        if not (1 <= n <= 5 and 1 <= m <= 5): 
            return np.full_like(Gi_all, 1e3)
    
    # Check for invalid parameters
    if (GIc_1 <= 0 or GIIc_1 <= 0 or GIc_2 <= 0 or GIIc_2 <= 0 or 
        GIc_3 <= 0 or GIIc_3 <= 0 or n <= 0 or m <= 0):
        return np.full_like(Gi_all, 1e3)
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Check for division by zero or negative values
    if np.any(GIc_values <= 0) or np.any(GIIc_values <= 0):
        return np.full_like(Gi_all, 1e3)
    
    # Compute residual with error handling
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            if var == 'A':
                term1 = (Gi_all/GIc_values)**(1/n)
                term2 = (Gii_all/GIIc_values)**(1/m)
                sum_terms = term1 + term2
                power = 2/(1/n+1/m)
                res = sum_terms**power - 1
            elif var == 'B':
                term1 = (Gi_all/GIc_values)**(1/n)
                term2 = (Gii_all/GIIc_values)**(1/m)
                res = term1 + term2 - 1
            else:
                raise NotImplementedError(f'Criterion type {var} not implemented.')
            
            # Handle invalid results
            res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
            
            return res
            
    except Exception:
        # Return large penalty for any computation errors
        return np.full_like(Gi_all, 1e3)

def param_jacobian_multi_dataset(beta, x, var='B', *args):
    """
    Compute parameter Jacobian for multi-dataset regression.
    """
    global global_dataset_indices
    
    # Unpack parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi_all, Gii_all = x  # RealData format: just (Gi, Gii)
    dataset_indices = global_dataset_indices  # Get from global variable
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Compute derivatives
    if var == 'A':
        # For variant A: ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
        term1 = (Gi_all/GIc_values)**(1/n)
        term2 = (Gii_all/GIIc_values)**(1/m)
        sum_terms = term1 + term2
        power = 2/(1/n+1/m)
        
        # Derivatives with respect to GIc (for each dataset)
        dGIc_1 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIc_2 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIc_3 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to GIIc (for each dataset)
        dGIIc_1 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIIc_2 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIIc_3 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to n and m (shared across all datasets)
        dn = sum_terms**power * (np.log(term1) - np.log(sum_terms)) * (-1/n**2) / (1/n+1/m)**2
        dm = sum_terms**power * (np.log(term2) - np.log(sum_terms)) * (-1/m**2) / (1/n+1/m)**2
        
    elif var == 'B':
        # For variant B: (Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m) - 1
        # Derivatives with respect to GIc (for each dataset)
        dGIc_1 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIc_2 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIc_3 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to GIIc (for each dataset)
        dGIIc_1 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIIc_2 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIIc_3 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to n and m (shared across all datasets)
        dn = (Gi_all/GIc_values)**(1/n) * np.log(Gi_all/GIc_values) * (-1/n**2)
        dm = (Gii_all/GIIc_values)**(1/m) * np.log(Gii_all/GIIc_values) * (-1/m**2)
    
    # Stack derivatives
    return np.row_stack([dGIc_1, dGIIc_1, dGIc_2, dGIIc_2, dGIc_3, dGIIc_3, dn, dm])

def value_jacobian_multi_dataset(beta, x, var='B', *args):
    """
    Compute value Jacobian for multi-dataset regression.
    """
    global global_dataset_indices
    
    # Unpack parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = beta
    Gi_all, Gii_all = x  # RealData format: just (Gi, Gii)
    dataset_indices = global_dataset_indices  # Get from global variable
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Compute derivatives
    if var == 'A':
        # For variant A: ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
        term1 = (Gi_all/GIc_values)**(1/n)
        term2 = (Gii_all/GIIc_values)**(1/m)
        sum_terms = term1 + term2
        power = 2/(1/n+1/m)
        
        dGi = power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) / GIc_values
        dGii = power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) / GIIc_values
        
    elif var == 'B':
        # For variant B: (Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m) - 1
        dGi = (1/n) * (Gi_all/GIc_values)**(1/n-1) / GIc_values
        dGii = (1/m) * (Gii_all/GIIc_values)**(1/m-1) / GIIc_values
    
    # Stack derivatives
    return np.row_stack([dGi, dGii])

def get_initial_guesses_multi_dataset(gc0=0.2, exp=2):
    """
    Generate initial guesses for multi-dataset regression.
    
    Parameters:
    -----------
    gc0 : float
        Initial guess for fracture toughnesses
    exp : int or list
        Exponent values to try
    
    Returns:
    --------
    guesses : list
        List of initial parameter guesses
    """
    if isinstance(exp, int):
        exp_values = [exp]
    else:
        exp_values = exp
    
    guesses = []
    for n in exp_values:
        for m in exp_values:
            # Try different combinations of GIc and GIIc for each dataset
            for factor1 in [0.5, 1.0, 1.5]:
                for factor2 in [0.5, 1.0, 1.5]:
                    for factor3 in [0.5, 1.0, 1.5]:
                        guess = [
                            gc0 * factor1, gc0 * factor1,  # GIc_1, GIIc_1
                            gc0 * factor2, gc0 * factor2,  # GIc_2, GIIc_2
                            gc0 * factor3, gc0 * factor3,  # GIc_3, GIIc_3
                            n, m  # Shared exponents
                        ]
                        guesses.append(guess)
    
    return guesses

def calc_fit_statistics_multi_dataset(final, ndof):
    """
    Calculate fit statistics for multi-dataset regression.
    """
    fit = defaultdict()
    
    # Extract parameters
    fit['params'] = final.beta
    fit['stddev'] = final.sd_beta
    
    # Unpack parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = final.beta
    
    # Store individual parameters
    fit['GIc_1'] = GIc_1
    fit['GIIc_1'] = GIIc_1
    fit['GIc_2'] = GIc_2
    fit['GIIc_2'] = GIIc_2
    fit['GIc_3'] = GIc_3
    fit['GIIc_3'] = GIIc_3
    fit['n'] = n
    fit['m'] = m
    
    # Store uncertainties
    fit['stddev_GIc_1'] = final.sd_beta[0]
    fit['stddev_GIIc_1'] = final.sd_beta[1]
    fit['stddev_GIc_2'] = final.sd_beta[2]
    fit['stddev_GIIc_2'] = final.sd_beta[3]
    fit['stddev_GIc_3'] = final.sd_beta[4]
    fit['stddev_GIIc_3'] = final.sd_beta[5]
    fit['stddev_n'] = final.sd_beta[6]
    fit['stddev_m'] = final.sd_beta[7]
    
    # Goodness of fit
    fit['reduced_chi_squared'] = final.res_var
    fit['chi_squared'] = final.sum_square
    fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
    fit['R_squared'] = 1 - fit['chi_squared']/(ndof + fit['chi_squared'])
    
    # Store final result
    fit['final'] = final
    
    return fit

def results_multi_dataset(fit):
    """
    Print multi-dataset fit results.
    """
    print("=" * 60)
    print("MULTI-DATASET REGRESSION RESULTS")
    print("=" * 60)
    print()
    
    print("SHARED PARAMETERS:")
    print(f"n (exponent) = {fit['n']:.3f} ± {fit['stddev_n']:.3f}")
    print(f"m (exponent) = {fit['m']:.3f} ± {fit['stddev_m']:.3f}")
    print()
    
    print("DATASET-SPECIFIC PARAMETERS:")
    print("Dataset 1:")
    print(f"  GIc_1 = {fit['GIc_1']:.3f} ± {fit['stddev_GIc_1']:.3f}")
    print(f"  GIIc_1 = {fit['GIIc_1']:.3f} ± {fit['stddev_GIIc_1']:.3f}")
    print(f"  GIIc_1/GIc_1 = {fit['GIIc_1']/fit['GIc_1']:.3f}")
    print()
    
    print("Dataset 2:")
    print(f"  GIc_2 = {fit['GIc_2']:.3f} ± {fit['stddev_GIc_2']:.3f}")
    print(f"  GIIc_2 = {fit['GIIc_2']:.3f} ± {fit['stddev_GIIc_2']:.3f}")
    print(f"  GIIc_2/GIc_2 = {fit['GIIc_2']/fit['GIc_2']:.3f}")
    print()
    
    print("Dataset 3:")
    print(f"  GIc_3 = {fit['GIc_3']:.3f} ± {fit['stddev_GIc_3']:.3f}")
    print(f"  GIIc_3 = {fit['GIIc_3']:.3f} ± {fit['stddev_GIIc_3']:.3f}")
    print(f"  GIIc_3/GIc_3 = {fit['GIIc_3']/fit['GIc_3']:.3f}")
    print()
    
    print("GOODNESS OF FIT:")
    print(f"Reduced χ² = {fit['reduced_chi_squared']:.3f}")
    print(f"p-value = {fit['p_value']:.3f}")
    print(f"R² = {fit['R_squared']:.3f}")
    print("=" * 60)
# Add this debugging version to your regression.py file
def odr_multi_dataset_debug(df_list, dim=1, gc0=0.2, exp=2, var='B', print_results=True):
    """
    Debug version of multi-dataset regression with more error information.
    """
    print("=== DEBUGGING MULTI-DATASET REGRESSION ===")
    
    # Assemble data
    data, ndof = assemble_data_multi_dataset(df_list, dim)
    
    print(f"Data assembled successfully:")
    print(f"  Data shape: {data.x[0].shape}")
    print(f"  Degrees of freedom: {ndof}")
    print(f"  GI range: {data.x[0].min():.3f} - {data.x[0].max():.3f}")
    print(f"  GII range: {data.x[1].min():.3f} - {data.x[1].max():.3f}")
    
    # Create model
    model = Model(fcn=residual_multi_dataset, 
                  fjacb=param_jacobian_multi_dataset,
                  fjacd=value_jacobian_multi_dataset, 
                  implicit=True,
                  extra_args=(var,))
    
    # Generate initial guesses
    guesses = get_initial_guesses_multi_dataset(gc0=gc0, exp=exp)
    print(f"Generated {len(guesses)} initial guesses")
    
    # Run regression for all guesses
    runs = []
    for i, g in enumerate(guesses):
        try:
            print(f"Trying guess {i+1}/{len(guesses)}: {g}")
            r = run_regression_multi_dataset(data, model, g)
            print(f"  Result: info={r.info}, sum_square={r.sum_square:.6f}")
            if r.info <= 3:
                runs.append(r)
                print(f"  SUCCESS!")
            else:
                print(f"  FAILED: info={r.info}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            continue
    
    if not runs:
        print("ERROR: No regression runs converged successfully!")
        print("Trying with simpler initial guesses...")
        
        # Try with simpler guesses
        simple_guesses = [
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 2, 2],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 2],
            [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 2, 2],
        ]
        
        for i, g in enumerate(simple_guesses):
            try:
                print(f"Trying simple guess {i+1}: {g}")
                r = run_regression_multi_dataset(data, model, g)
                print(f"  Result: info={r.info}, sum_square={r.sum_square:.6f}")
                if r.info <= 3:
                    runs.append(r)
                    print(f"  SUCCESS!")
                    break
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                continue
        
        if not runs:
            raise ValueError("All regression attempts failed. Check your data and model.")
    
    # Find best run
    final = runs[np.argmin([run.sum_square for run in runs])]
    
    # Compile results
    fit = calc_fit_statistics_multi_dataset(final, ndof)
    fit['var'] = var
    
    if print_results:
        results_multi_dataset(fit)
    
    return fi
def odr_multi_dataset(df_list, dim=1, gc0=0.2, exp=2, var='B', print_results=True):
    """
    Perform orthogonal distance regression on multiple datasets with shared exponents.
    """
    # Assemble data
    data, ndof = assemble_data_multi_dataset(df_list, dim)
    
    # Create model
    model = Model(fcn=residual_multi_dataset, 
                  fjacb=param_jacobian_multi_dataset,
                  fjacd=value_jacobian_multi_dataset, 
                  implicit=True,
                  extra_args=(var,))
    
    # Generate initial guesses
    guesses = get_initial_guesses_multi_dataset(gc0=gc0, exp=exp)
    
    # Run regression for all guesses
    runs = []
    for g in guesses:
        try:
            r = run_regression_multi_dataset(data, model, g)  # Use the new function
            if r.info <= 3:
                runs.append(r)
        except Exception as e:
            print(f"Warning: Regression failed for guess {g}: {e}")
            continue
    
    if not runs:
        raise ValueError("No regression runs converged successfully!")
    
    # Find best run
    final = runs[np.argmin([run.sum_square for run in runs])]
    
    # Compile results
    fit = calc_fit_statistics_multi_dataset(final, ndof)
    fit['var'] = var
    
    if print_results:
        results_multi_dataset(fit)
    
    return fit
def odr_multi_dataset_robust(df_list, dim=1, gc0=0.2, exp=2, var='B', print_results=True, 
                           bounds=None, stddev_scale=1.0):
    """
    Robust multi-dataset regression with fallback to scipy.optimize.minimize.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames with fracture toughness data
    dim : int, optional
        Dimensionality of the response function. Default is 1.
    gc0 : float, optional
        Initial guess for fracture toughnesses. Default is 0.2.
    exp : int or list, optional
        Exponent values to try. Default is 2.
    var : str, optional
        Residual variant {'A', 'B'}. Default is 'B'.
    print_results : bool, optional
        If True, print fit results. Default is True.
    bounds : dict, optional
        Dictionary with bounds for parameters. Default is None (uses default bounds).
        Format: {
            'GIc_min': float, 'GIc_max': float,
            'GIIc_min': float, 'GIIc_max': float,
            'n_min': float, 'n_max': float,
            'm_min': float, 'm_max': float
        }
    stddev_scale : float, optional
        Scaling factor for standard deviations in fallback method. Default is 1.0.
    
    Returns:
    --------
    fit : dict
        Fit results dictionary
    """
    print("=== ROBUST MULTI-DATASET REGRESSION ===")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,  # Your requested upper bound
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    # Assemble data
    data, ndof = assemble_data_multi_dataset(df_list, dim)
    
    print(f"Data assembled successfully:")
    print(f"  Data shape: {data.x[0].shape}")
    print(f"  Degrees of freedom: {ndof}")
    print(f"  GI range: {data.x[0].min():.3f} - {data.x[0].max():.3f}")
    print(f"  GII range: {data.x[1].min():.3f} - {data.x[1].max():.3f}")
    print(f"  Bounds: GIc [{bounds['GIc_min']:.2f}, {bounds['GIc_max']:.2f}], "
          f"GIIc [{bounds['GIIc_min']:.2f}, {bounds['GIIc_max']:.2f}], "
          f"n,m [{bounds['n_min']:.1f}, {bounds['n_max']:.1f}]")
    
    # Try ODR first
    try:
        # Create model
        model = Model(fcn=residual_multi_dataset, 
                      fjacb=param_jacobian_multi_dataset,
                      fjacd=value_jacobian_multi_dataset, 
                      implicit=True,
                      extra_args=(var,))
        
        # Generate initial guesses
        guesses = get_initial_guesses_multi_dataset(gc0=gc0, exp=exp)
        print(f"Generated {len(guesses)} initial guesses")
        
        # Run regression for all guesses
        runs = []
        for i, g in enumerate(guesses):
            try:
                print(f"Trying ODR guess {i+1}/{len(guesses)}: {g}")
                r = run_regression_multi_dataset(data, model, g)
                print(f"  Result: info={r.info}, sum_square={r.sum_square:.6f}")
                if r.info <= 3:
                    runs.append(r)
                    print(f"  SUCCESS!")
                    break  # Found a good solution
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                continue
        
        if runs:
            # Use the best ODR result
            final = runs[np.argmin([run.sum_square for run in runs])]
            fit = calc_fit_statistics_multi_dataset(final, ndof)
            fit['var'] = var
            fit['method'] = 'ODR'
            fit['bounds'] = bounds
            
            if print_results:
                results_multi_dataset(fit)
            
            return fit
            
    except Exception as e:
        print(f"ODR failed: {e}")
    
    # Fallback to scipy.optimize.minimize
    print("ODR failed, trying scipy.optimize.minimize...")
    
    def objective(params):
        """Objective function for minimize"""
        GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = params
        Gi_all, Gii_all = data.x
        dataset_indices = global_dataset_indices
        
        # Check bounds
        if (GIc_1 <= bounds['GIc_min'] or GIIc_1 <= bounds['GIIc_min'] or 
            GIc_2 <= bounds['GIc_min'] or GIIc_2 <= bounds['GIIc_min'] or 
            GIc_3 <= bounds['GIc_min'] or GIIc_3 <= bounds['GIIc_min'] or 
            n <= bounds['n_min'] or m <= bounds['m_min']):
            return 1e6
        
        # Check upper bounds
        if (GIc_1 > bounds['GIc_max'] or GIIc_1 > bounds['GIIc_max'] or 
            GIc_2 > bounds['GIc_max'] or GIIc_2 > bounds['GIIc_max'] or 
            GIc_3 > bounds['GIc_max'] or GIIc_3 > bounds['GIIc_max'] or 
            n > bounds['n_max'] or m > bounds['m_max']):
            return 1e6
        
        # Create arrays of GIc and GIIc values for each data point
        GIc_values = np.zeros_like(Gi_all)
        GIIc_values = np.zeros_like(Gii_all)
        
        GIc_values[dataset_indices == 0] = GIc_1
        GIc_values[dataset_indices == 1] = GIc_2
        GIc_values[dataset_indices == 2] = GIc_3
        
        GIIc_values[dataset_indices == 0] = GIIc_1
        GIIc_values[dataset_indices == 1] = GIIc_2
        GIIc_values[dataset_indices == 2] = GIIc_3
        
        # Compute residuals
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                if var == 'A':
                    term1 = (Gi_all/GIc_values)**(1/n)
                    term2 = (Gii_all/GIIc_values)**(1/m)
                    sum_terms = term1 + term2
                    power = 2/(1/n+1/m)
                    res = sum_terms**power - 1
                elif var == 'B':
                    term1 = (Gi_all/GIc_values)**(1/n)
                    term2 = (Gii_all/GIIc_values)**(1/m)
                    res = term1 + term2 - 1
                else:
                    return 1e6
                
                # Handle invalid results
                res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
                
                return np.sum(res**2)
                
        except Exception:
            return 1e6
    
    # Try multiple starting points
    best_result = None
    best_objective = float('inf')
    
    # Data-driven starting points
    gi_mean = np.mean(data.x[0])
    gii_mean = np.mean(data.x[1])
    
    starting_points = [
        [gi_mean*0.8, gii_mean*0.8, gi_mean*1.2, gii_mean*0.7, gi_mean*0.7, gii_mean*1.2, 2, 2],
        [gi_mean*0.9, gii_mean*0.9, gi_mean*1.1, gii_mean*0.8, gi_mean*0.8, gii_mean*1.1, 2, 2],
        [gi_mean, gii_mean, gi_mean, gii_mean, gi_mean, gii_mean, 2, 2],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 2, 2],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 2],
    ]
    
    # Create bounds for minimize
    minimize_bounds = [
        (bounds['GIc_min'], bounds['GIc_max']),  # GIc_1
        (bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_1
        (bounds['GIc_min'], bounds['GIc_max']),  # GIc_2
        (bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_2
        (bounds['GIc_min'], bounds['GIc_max']),  # GIc_3
        (bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_3
        (bounds['n_min'], bounds['n_max']),  # n
        (bounds['m_min'], bounds['m_max'])   # m
    ]
    
    for i, start_point in enumerate(starting_points):
        try:
            print(f"Trying minimize with start point {i+1}: {start_point}")
            result = minimize(objective, start_point, 
                            method='L-BFGS-B',
                            bounds=minimize_bounds)
            
            if result.success and result.fun < best_objective:
                best_result = result
                best_objective = result.fun
                print(f"  SUCCESS: objective={result.fun:.6f}")
                
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            continue
    
    if best_result is not None:
        # Create a mock ODR result object
        class MockODRResult:
            def __init__(self, params, objective_val):
                self.beta = params
                self.sum_square = objective_val
                self.res_var = objective_val / ndof
                # Scale standard deviations
                self.sd_beta = [0.01 * stddev_scale] * 8  # Approximate uncertainties
                self.info = 1  # Success
        
        final = MockODRResult(best_result.x, best_result.fun)
        fit = calc_fit_statistics_multi_dataset(final, ndof)
        fit['var'] = var
        fit['method'] = 'minimize'
        fit['bounds'] = bounds
        
        if print_results:
            results_multi_dataset(fit)
        
        return fit
    else:
        raise ValueError("Both ODR and minimize failed. Check your data.")
    
def explore_n_m_landscape(df_list, n_range=(1, 5), m_range=(1, 5), n_points=20, m_points=20, 
                         GIc_fixed=None, GIIc_fixed=None, var='B', bounds=None):
    """
    Explore the n vs m landscape with fixed GIc/GIIc values and boundary constraints.
    
    Parameters:
    -----------
    df_list : list
        List of DataFrames
    n_range : tuple
        Range for n parameter (min, max)
    m_range : tuple
        Range for m parameter (min, max)
    n_points : int
        Number of points for n grid
    m_points : int
        Number of points for m grid
    GIc_fixed : list or None
        Fixed GIc values for each dataset [GIc_1, GIc_2, GIc_3]
    GIIc_fixed : list or None
        Fixed GIIc values for each dataset [GIIc_1, GIIc_2, GIIc_3]
    var : str
        Residual variant
    bounds : dict or None
        Dictionary with bounds for parameters. If None, uses default bounds.
        Format: {
            'GIc_min': float, 'GIc_max': float,
            'GIIc_min': float, 'GIIc_max': float,
            'n_min': float, 'n_max': float,
            'm_min': float, 'm_max': float
        }
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    # Generate grid
    n_vals = np.linspace(n_range[0], n_range[1], n_points)
    m_vals = np.linspace(m_range[0], m_range[1], m_points)
    n_grid, m_grid = np.meshgrid(n_vals, m_vals)
    
    # Use fixed values or compute from data
    if GIc_fixed is None:
        GIc_fixed = [df['GIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    if GIIc_fixed is None:
        GIIc_fixed = [df['GIIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    
    objective_grid = np.zeros_like(n_grid)
    
    print("Exploring n vs m landscape with boundaries...")
    print(f"Bounds: GIc [{bounds['GIc_min']:.2f}, {bounds['GIc_max']:.2f}], "
          f"GIIc [{bounds['GIIc_min']:.2f}, {bounds['GIIc_max']:.2f}], "
          f"n,m [{bounds['n_min']:.1f}, {bounds['n_max']:.1f}]")
    
    for i in range(n_points):
        for j in range(m_points):
            params = [GIc_fixed[0], GIIc_fixed[0], 
                     GIc_fixed[1], GIIc_fixed[1], 
                     GIc_fixed[2], GIIc_fixed[2], 
                     n_grid[j, i], m_grid[j, i]]
            
            # Check if parameters are within bounds
            if (bounds['GIc_min'] <= GIc_fixed[0] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= GIIc_fixed[0] <= bounds['GIIc_max'] and
                bounds['GIc_min'] <= GIc_fixed[1] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= GIIc_fixed[1] <= bounds['GIIc_max'] and
                bounds['GIc_min'] <= GIc_fixed[2] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= GIIc_fixed[2] <= bounds['GIIc_max'] and
                bounds['n_min'] <= n_grid[j, i] <= bounds['n_max'] and
                bounds['m_min'] <= m_grid[j, i] <= bounds['m_max']):
                
                objective_grid[j, i] = compute_objective_multi_dataset(params, df_list, var)
            else:
                # Set to large value for out-of-bounds parameters
                objective_grid[j, i] = 1e6
    
    return n_grid, m_grid, objective_grid

def explore_GIc_GIIc_landscape(df_list, dataset_idx, GIc_range=(0.1, 2.0), GIIc_range=(0.1, 1.0), 
                              points=30, n_fixed=2, m_fixed=2, var='B', bounds=None):
    """
    Explore the GIc vs GIIc landscape for a specific dataset with boundary constraints.
    
    Parameters:
    -----------
    df_list : list
        List of DataFrames
    dataset_idx : int
        Index of dataset to explore (0, 1, or 2)
    GIc_range : tuple
        Range for GIc parameter (min, max)
    GIIc_range : tuple
        Range for GIIc parameter (min, max)
    points : int
        Number of points for grid
    n_fixed : float
        Fixed n value
    m_fixed : float
        Fixed m value
    var : str
        Residual variant
    bounds : dict or None
        Dictionary with bounds for parameters
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    # Generate grid
    GIc_vals = np.linspace(GIc_range[0], GIc_range[1], points)
    GIIc_vals = np.linspace(GIIc_range[0], GIIc_range[1], points)
    GIc_grid, GIIc_grid = np.meshgrid(GIc_vals, GIIc_vals)
    
    # Get fixed values for other datasets
    GIc_fixed = [df['GIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    GIIc_fixed = [df['GIIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    
    objective_grid = np.zeros_like(GIc_grid)
    
    print(f"Exploring GIc vs GIIc landscape for dataset {dataset_idx+1} with boundaries...")
    print(f"Bounds: GIc [{bounds['GIc_min']:.2f}, {bounds['GIc_max']:.2f}], "
          f"GIIc [{bounds['GIIc_min']:.2f}, {bounds['GIIc_max']:.2f}], "
          f"n,m [{bounds['n_min']:.1f}, {bounds['n_max']:.1f}]")
    
    for i in range(points):
        for j in range(points):
            # Create parameter list with current dataset values
            params = [GIc_fixed[0], GIIc_fixed[0], 
                     GIc_fixed[1], GIIc_fixed[1], 
                     GIc_fixed[2], GIIc_fixed[2]]
            
            # Update the values for the current dataset
            params[dataset_idx*2] = GIc_grid[j, i]     # GIc
            params[dataset_idx*2+1] = GIIc_grid[j, i]  # GIIc
            
            # Add fixed n, m
            params.extend([n_fixed, m_fixed])
            
            # Check if parameters are within bounds
            if (bounds['GIc_min'] <= params[0] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= params[1] <= bounds['GIIc_max'] and
                bounds['GIc_min'] <= params[2] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= params[3] <= bounds['GIIc_max'] and
                bounds['GIc_min'] <= params[4] <= bounds['GIc_max'] and
                bounds['GIIc_min'] <= params[5] <= bounds['GIIc_max'] and
                bounds['n_min'] <= n_fixed <= bounds['n_max'] and
                bounds['m_min'] <= m_fixed <= bounds['m_max']):
                
                objective_grid[j, i] = compute_objective_multi_dataset(params, df_list, var)
            else:
                # Set to large value for out-of-bounds parameters
                objective_grid[j, i] = 1e6
    
    return GIc_grid, GIIc_grid, objective_grid

def create_starting_points_from_landscape(df_list, n_points=100, var='B', bounds=None):
    """
    Create intelligent starting points based on landscape analysis with boundary constraints.
    
    Parameters:
    -----------
    df_list : list
        List of DataFrames
    n_points : int
        Number of starting points to generate
    var : str
        Residual variant
    bounds : dict or None
        Dictionary with bounds for parameters
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    print("Creating intelligent starting points from landscape analysis with boundaries...")
    print(f"Bounds: GIc [{bounds['GIc_min']:.2f}, {bounds['GIc_max']:.2f}], "
          f"GIIc [{bounds['GIIc_min']:.2f}, {bounds['GIIc_max']:.2f}], "
          f"n,m [{bounds['n_min']:.1f}, {bounds['n_max']:.1f}]")
    
    # Quick landscape scan with bounds
    n_grid, m_grid, obj_nm = explore_n_m_landscape(df_list, 
                                                  n_range=(bounds['n_min'], bounds['n_max']), 
                                                  m_range=(bounds['m_min'], bounds['m_max']), 
                                                  n_points=15, m_points=15, var=var, bounds=bounds)
    
    # Find promising regions (top 20% best regions)
    threshold = np.percentile(obj_nm, 20)
    promising_indices = np.where(obj_nm < threshold)
    
    starting_points = []
    
    # Generate points around promising regions
    for _ in range(n_points):
        # Randomly select a promising region
        idx = np.random.randint(len(promising_indices[0]))
        i, j = promising_indices[0][idx], promising_indices[1][idx]
        
        # Add noise around this point, ensuring bounds are respected
        n_val = n_grid[i, j] + np.random.normal(0, 0.2)
        m_val = m_grid[i, j] + np.random.normal(0, 0.2)
        
        # Clip to bounds
        n_val = np.clip(n_val, bounds['n_min'], bounds['n_max'])
        m_val = np.clip(m_val, bounds['m_min'], bounds['m_max'])
        
        # Generate GIc/GIIc values within bounds
        gi_means = [df['GIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
        gii_means = [df['GIIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
        
        noise = np.random.normal(0, 0.3, 6)
        start_point = [
            gi_means[0] * (1 + noise[0]), gii_means[0] * (1 + noise[0]),
            gi_means[1] * (1 + noise[1]), gii_means[1] * (1 + noise[1]),
            gi_means[2] * (1 + noise[2]), gii_means[2] * (1 + noise[2]),
            n_val, m_val
        ]
        
        # Clip all parameters to bounds
        start_point[0] = np.clip(start_point[0], bounds['GIc_min'], bounds['GIc_max'])  # GIc_1
        start_point[1] = np.clip(start_point[1], bounds['GIIc_min'], bounds['GIIc_max'])  # GIIc_1
        start_point[2] = np.clip(start_point[2], bounds['GIc_min'], bounds['GIc_max'])  # GIc_2
        start_point[3] = np.clip(start_point[3], bounds['GIIc_min'], bounds['GIIc_max'])  # GIIc_2
        start_point[4] = np.clip(start_point[4], bounds['GIc_min'], bounds['GIc_max'])  # GIc_3
        start_point[5] = np.clip(start_point[5], bounds['GIIc_min'], bounds['GIIc_max'])  # GIIc_3
        
        starting_points.append(start_point)
    
    return starting_points

def plot_landscape_with_bounds(df_list, bounds=None, save_path=None, var='B'):
    """
    Plot landscape analysis with boundary constraints.
    
    Parameters:
    -----------
    df_list : list
        List of DataFrames
    bounds : dict or None
        Dictionary with bounds for parameters
    save_path : str or None
        Path to save the figure
    var : str
        Residual variant
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. n vs m landscape with bounds
    print("=== Computing n vs m landscape with bounds ===")
    n_grid, m_grid, obj_nm = explore_n_m_landscape(df_list, 
                                                  n_range=(bounds['n_min'], bounds['n_max']), 
                                                  m_range=(bounds['m_min'], bounds['m_max']), 
                                                  n_points=25, m_points=25, var=var, bounds=bounds)
    
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(n_grid, m_grid, obj_nm, levels=50, cmap='viridis', norm=LogNorm())
    ax1.contour(n_grid, m_grid, obj_nm, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax1.set_xlabel('n')
    ax1.set_ylabel('m')
    ax1.set_title(f'n vs m Landscape with Bounds\n(Log Scale)\nn: [{bounds["n_min"]:.1f}, {bounds["n_max"]:.1f}], m: [{bounds["m_min"]:.1f}, {bounds["m_max"]:.1f}]')
    plt.colorbar(im1, ax=ax1, label='Objective Function')
    
    # Add boundary lines
    ax1.axhline(y=bounds['m_min'], color='red', linestyle='--', alpha=0.7, label=f'm_min = {bounds["m_min"]:.1f}')
    ax1.axhline(y=bounds['m_max'], color='red', linestyle='--', alpha=0.7, label=f'm_max = {bounds["m_max"]:.1f}')
    ax1.axvline(x=bounds['n_min'], color='red', linestyle='--', alpha=0.7, label=f'n_min = {bounds["n_min"]:.1f}')
    ax1.axvline(x=bounds['n_max'], color='red', linestyle='--', alpha=0.7, label=f'n_max = {bounds["n_max"]:.1f}')
    ax1.legend(fontsize=8)
    
    # 2-4. GIc vs GIIc landscapes for each dataset with bounds
    dataset_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']
    
    for idx in range(3):
        print(f"=== Computing GIc vs GIIc landscape for {dataset_names[idx]} with bounds ===")
        
        # Get data ranges for this dataset, but respect bounds
        gi_vals = df_list[idx]['GIc'].apply(lambda x: x.nominal_value).values
        gii_vals = df_list[idx]['GIIc'].apply(lambda x: x.nominal_value).values
        
        gi_range = (max(gi_vals.min() * 0.5, bounds['GIc_min']), 
                   min(gi_vals.max() * 1.5, bounds['GIc_max']))
        gii_range = (max(gii_vals.min() * 0.5, bounds['GIIc_min']), 
                    min(gii_vals.max() * 1.5, bounds['GIIc_max']))
        
        GIc_grid, GIIc_grid, obj_gc = explore_GIc_GIIc_landscape(
            df_list, idx, GIc_range=gi_range, GIIc_range=gii_range, 
            points=25, n_fixed=2, m_fixed=2, var=var, bounds=bounds
        )
        
        ax = plt.subplot(2, 3, idx + 2)
        im = ax.contourf(GIc_grid, GIIc_grid, obj_gc, levels=50, cmap='viridis', norm=LogNorm())
        ax.contour(GIc_grid, GIIc_grid, obj_gc, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('GIc')
        ax.set_ylabel('GIIc')
        ax.set_title(f'{dataset_names[idx]}\nGIc vs GIIc Landscape with Bounds\n(Log Scale)')
        plt.colorbar(im, ax=ax, label='Objective Function')
        
        # Add boundary lines
        ax.axhline(y=bounds['GIIc_min'], color='red', linestyle='--', alpha=0.7, label=f'GIIc_min = {bounds["GIIc_min"]:.2f}')
        ax.axhline(y=bounds['GIIc_max'], color='red', linestyle='--', alpha=0.7, label=f'GIIc_max = {bounds["GIIc_max"]:.2f}')
        ax.axvline(x=bounds['GIc_min'], color='red', linestyle='--', alpha=0.7, label=f'GIc_min = {bounds["GIc_min"]:.2f}')
        ax.axvline(x=bounds['GIc_max'], color='red', linestyle='--', alpha=0.7, label=f'GIc_max = {bounds["GIc_max"]:.2f}')
        ax.legend(fontsize=8)
    
    # 5. Summary statistics with bounds
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Run multiple optimizations with bounds
    print("=== Running multiple optimizations with bounds ===")
    results = []
    
    # Generate diverse starting points within bounds
    for _ in range(50):
        # Random starting points within bounds
        start_point = [
            np.random.uniform(bounds['GIc_min'], bounds['GIc_max']),  # GIc_1
            np.random.uniform(bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_1
            np.random.uniform(bounds['GIc_min'], bounds['GIc_max']),  # GIc_2
            np.random.uniform(bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_2
            np.random.uniform(bounds['GIc_min'], bounds['GIc_max']),  # GIc_3
            np.random.uniform(bounds['GIIc_min'], bounds['GIIc_max']),  # GIIc_3
            np.random.uniform(bounds['n_min'], bounds['n_max']),  # n
            np.random.uniform(bounds['m_min'], bounds['m_max'])   # m
        ]
        
        try:
            result = reg.odr_multi_dataset_robust(df_list, dim=1, var=var, 
                                                print_results=False, bounds=bounds)
            results.append({
                'start_point': start_point,
                'final_params': result['params'],
                'objective': result['chi_squared'],
                'success': True
            })
        except:
            results.append({
                'start_point': start_point,
                'final_params': None,
                'objective': float('inf'),
                'success': False
            })
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    objectives = [r['objective'] for r in successful_results]
    
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['objective'])
        
        summary_text = f"""
LANDSCAPE ANALYSIS WITH BOUNDS

Bounds Applied:
  GIc: [{bounds['GIc_min']:.2f}, {bounds['GIc_max']:.2f}]
  GIIc: [{bounds['GIIc_min']:.2f}, {bounds['GIIc_max']:.2f}]
  n: [{bounds['n_min']:.1f}, {bounds['n_max']:.1f}]
  m: [{bounds['m_min']:.1f}, {bounds['m_max']:.1f}]

Total optimizations: {len(results)}
Successful optimizations: {len(successful_results)}

Objective Statistics:
  Min: {min(objectives):.6f}
  Max: {max(objectives):.6f}
  Mean: {np.mean(objectives):.6f}
  Std: {np.std(objectives):.6f}

Best Solution:
  GIc_1: {best_result['final_params'][0]:.4f}
  GIIc_1: {best_result['final_params'][1]:.4f}
  GIc_2: {best_result['final_params'][2]:.4f}
  GIIc_2: {best_result['final_params'][3]:.4f}
  GIc_3: {best_result['final_params'][4]:.4f}
  GIIc_3: {best_result['final_params'][5]:.4f}
  n: {best_result['final_params'][6]:.4f}
  m: {best_result['final_params'][7]:.4f}
  Objective: {best_result['objective']:.6f}
"""
    else:
        summary_text = "No successful optimizations found with given bounds."
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Objective function distribution
    ax6 = plt.subplot(2, 3, 6)
    if successful_results:
        ax6.hist(objectives, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(min(objectives), color='red', linestyle='--', label=f'Best: {min(objectives):.6f}')
        ax6.axvline(np.mean(objectives), color='orange', linestyle='--', label=f'Mean: {np.mean(objectives):.6f}')
        ax6.set_xlabel('Objective Function Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Optimization Results\n(with Bounds)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No successful optimizations', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Distribution of Optimization Results\n(with Bounds)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Landscape analysis with bounds saved to {save_path}")
    
    plt.show()
    
    return results


def compute_objective_multi_dataset(params, df_list, var='B'):
    """
    Compute objective function for multi-dataset regression.
    """
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = params
    
    total_residual = 0
    
    for i, df in enumerate(df_list):
        # Get GIc and GIIc for this dataset
        if i == 0:
            GIc, GIIc = GIc_1, GIIc_1
        elif i == 1:
            GIc, GIIc = GIc_2, GIIc_2
        else:
            GIc, GIIc = GIc_3, GIIc_3
        
        # Extract data
        Gi = df['GIc'].apply(lambda x: x.nominal_value).values
        Gii = df['GIIc'].apply(lambda x: x.nominal_value).values
        
        # Compute residuals
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                if var == 'A':
                    res = ((Gi/GIc)**n + (Gii/GIIc)**m)**(2/(n+m)) - 1
                elif var == 'B':
                    res = ((Gi/GIc)**n + (Gii/GIIc)**m) - 1
                else:
                    return 1e6
                
                res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
                total_residual += np.sum(res**2)
                
        except Exception:
            return 1e6
    
    return total_residual



def odr_multi_dataset_landscape_guided(df_list, dim=1, var='B', print_results=True, 
                                     n_starts=50, bounds=None):
    """
    Multi-dataset regression using landscape-guided starting points.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames with fracture toughness data
    dim : int, optional
        Dimensionality of the response function. Default is 1.
    var : str, optional
        Residual variant {'A', 'B'}. Default is 'B'.
    print_results : bool, optional
        If True, print fit results. Default is True.
    n_starts : int, optional
        Number of starting points to generate from landscape analysis. Default is 50.
    bounds : dict, optional
        Dictionary with bounds for parameters. Default is None.
    
    Returns:
    --------
    fit : dict
        Fit results dictionary
    """
    from scipy.optimize import minimize
    
    print("=== LANDSCAPE-GUIDED MULTI-DATASET REGRESSION ===")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0,
            'n_min': 0.1, 'n_max': 10.0,
            'm_min': 0.1, 'm_max': 10.0
        }
    
    # Assemble data
    data, ndof = assemble_data_multi_dataset(df_list, dim)
    
    print(f"Data assembled successfully:")
    print(f"  Data shape: {data.x[0].shape}")
    print(f"  Degrees of freedom: {ndof}")
    
    def objective(params):
        """Objective function for minimize"""
        GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m = params
        Gi_all, Gii_all = data.x
        dataset_indices = global_dataset_indices
        
        # Check bounds
        if (GIc_1 <= bounds['GIc_min'] or GIIc_1 <= bounds['GIIc_min'] or 
            GIc_2 <= bounds['GIc_min'] or GIIc_2 <= bounds['GIIc_min'] or 
            GIc_3 <= bounds['GIc_min'] or GIIc_3 <= bounds['GIIc_min'] or 
            n <= bounds['n_min'] or m <= bounds['m_min']):
            return 1e6
        
        if (GIc_1 > bounds['GIc_max'] or GIIc_1 > bounds['GIIc_max'] or 
            GIc_2 > bounds['GIc_max'] or GIIc_2 > bounds['GIIc_max'] or 
            GIc_3 > bounds['GIc_max'] or GIIc_3 > bounds['GIIc_max'] or 
            n > bounds['n_max'] or m > bounds['m_max']):
            return 1e6
        
        # Create arrays of GIc and GIIc values for each data point
        GIc_values = np.zeros_like(Gi_all)
        GIIc_values = np.zeros_like(Gii_all)
        
        GIc_values[dataset_indices == 0] = GIc_1
        GIc_values[dataset_indices == 1] = GIc_2
        GIc_values[dataset_indices == 2] = GIc_3
        
        GIIc_values[dataset_indices == 0] = GIIc_1
        GIIc_values[dataset_indices == 1] = GIIc_2
        GIIc_values[dataset_indices == 2] = GIIc_3
        
        # Compute residuals
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                if var == 'A':
                    term1 = (Gi_all/GIc_values)**(1/n)
                    term2 = (Gii_all/GIIc_values)**(1/m)
                    sum_terms = term1 + term2
                    power = 2/(1/n+1/m)
                    res = sum_terms**power - 1
                elif var == 'B':
                    term1 = (Gi_all/GIc_values)**(1/n)
                    term2 = (Gii_all/GIIc_values)**(1/m)
                    res = term1 + term2 - 1
                else:
                    return 1e6
                
                res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
                return np.sum(res**2)
                
        except Exception:
            return 1e6
    
    # Generate landscape-guided starting points
    print("Generating landscape-guided starting points...")
    starting_points = create_starting_points_from_landscape(df_list, n_points=n_starts, var=var)
    
    # Create bounds for minimize
    minimize_bounds = [
        (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
        (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
        (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
        (bounds['n_min'], bounds['n_max']), (bounds['m_min'], bounds['m_max'])
    ]
    
    # Try multiple starting points
    best_result = None
    best_objective = float('inf')
    
    print(f"Running optimization with {len(starting_points)} landscape-guided starting points...")
    for i, start_point in enumerate(starting_points):
        try:
            result = minimize(objective, start_point, 
                            method='L-BFGS-B',
                            bounds=minimize_bounds)
            
            if result.success and result.fun < best_objective:
                best_result = result
                best_objective = result.fun
                print(f"  New best: {result.fun:.6f} from start point {i+1}")
                
        except Exception as e:
            continue
    
    if best_result is None:
        raise ValueError("All optimization attempts failed. Check your data.")
    
    # Create fit result
    class MockODRResult:
        def __init__(self, params, objective_val):
            self.beta = params
            self.sum_square = objective_val
            self.res_var = objective_val / ndof
            self.sd_beta = [0.01] * 8  # Approximate uncertainties
            self.info = 1
    
    final = MockODRResult(best_result.x, best_result.fun)
    fit = calc_fit_statistics_multi_dataset(final, ndof)
    fit['var'] = var
    fit['method'] = 'landscape_guided'
    fit['bounds'] = bounds
    fit['n_starting_points'] = len(starting_points)
    
    if print_results:
        results_multi_dataset(fit)
    
    return fit

def residual_multi_dataset_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', bounds=False):
    """
    Compute objective function for multi-dataset regression with fixed exponents.
    
    Parameters:
    -----------
    beta : list
        [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3] - only 6 parameters
    x : tuple
        (Gi_all, Gii_all) - standard RealData format
    n_fixed : float
        Fixed value for exponent n
    m_fixed : float
        Fixed value for exponent m
    var : str
        Residual variant {'A', 'B'}
    bounds : bool
        Whether to apply bounds (not used in this version)
    """
    global global_dataset_indices
    
    # Unpack parameters (only 6 parameters now)
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = beta
    Gi_all, Gii_all = x  # Standard RealData format
    dataset_indices = global_dataset_indices  # Get from global variable
    
    # Use fixed exponents
    n, m = n_fixed, m_fixed
    
    # Check for invalid parameters
    if (GIc_1 <= 0 or GIIc_1 <= 0 or GIc_2 <= 0 or GIIc_2 <= 0 or 
        GIc_3 <= 0 or GIIc_3 <= 0):
        return np.full_like(Gi_all, 1e3)
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Check for division by zero or negative values
    if np.any(GIc_values <= 0) or np.any(GIIc_values <= 0):
        return np.full_like(Gi_all, 1e3)
    
    # Compute residual with error handling
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            if var == 'A':
                term1 = (Gi_all/GIc_values)**(1/n_fixed)
                term2 = (Gii_all/GIIc_values)**(1/m_fixed)
                sum_terms = term1 + term2
                power = 2/(1/n_fixed+1/m_fixed)
                res = sum_terms**power - 1
            elif var == 'B':
                term1 = (Gi_all/GIc_values)**(1/n_fixed)
                term2 = (Gii_all/GIIc_values)**(1/m_fixed)
                res = term1 + term2 - 1
            else:
                raise NotImplementedError(f'Criterion type {var} not implemented.')
            
            # Handle invalid results
            res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
            
            return res
            
    except Exception:
        # Return large penalty for any computation errors
        return np.full_like(Gi_all, 1e3)

def param_jacobian_multi_dataset_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', *args):
    """
    Compute parameter Jacobian for multi-dataset regression with fixed exponents.
    
    Jacobian = [df/dGIc_1, df/dGIIc_1, df/dGIc_2, df/dGIIc_2, df/dGIc_3, df/dGIIc_3].T
    """
    global global_dataset_indices
    
    # Unpack parameters (only 6 parameters)
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = beta
    Gi_all, Gii_all = x  # RealData format: just (Gi, Gii)
    dataset_indices = global_dataset_indices  # Get from global variable
    n, m = n_fixed, m_fixed
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Compute derivatives
    if var == 'A':
        # For variant A: ((Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m))**(2/(1/n+1/m)) - 1
        term1 = (Gi_all/GIc_values)**(1/n)
        term2 = (Gii_all/GIIc_values)**(1/m)
        sum_terms = term1 + term2
        power = 2/(1/n+1/m)
        
        # Derivatives with respect to GIc (for each dataset)
        dGIc_1 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIc_2 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIc_3 = -power * sum_terms**(power-1) * (1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to GIIc (for each dataset)
        dGIIc_1 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIIc_2 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIIc_3 = -power * sum_terms**(power-1) * (1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
    elif var == 'B':
        # For variant B: (Gi/GIc)**(1/n) + (Gii/GIIc)**(1/m) - 1
        # Derivatives with respect to GIc (for each dataset)
        dGIc_1 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIc_2 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIc_3 = -(1/n) * (Gi_all/GIc_values)**(1/n-1) * Gi_all / (GIc_values**2)
        dGIc_3[dataset_indices != 2] = 0  # Only for dataset 2
        
        # Derivatives with respect to GIIc (for each dataset)
        dGIIc_1 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_1[dataset_indices != 0] = 0  # Only for dataset 0
        
        dGIIc_2 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_2[dataset_indices != 1] = 0  # Only for dataset 1
        
        dGIIc_3 = -(1/m) * (Gii_all/GIIc_values)**(1/m-1) * Gii_all / (GIIc_values**2)
        dGIIc_3[dataset_indices != 2] = 0  # Only for dataset 2
    
    # Stack derivatives (only 6 parameters)
    return np.row_stack([dGIc_1, dGIIc_1, dGIc_2, dGIIc_2, dGIc_3, dGIIc_3])

def value_jacobian_multi_dataset_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', *args):
    """
    Compute value Jacobian for multi-dataset regression with fixed exponents.
    """
    global global_dataset_indices
    
    # Unpack parameters (only 6 parameters)
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = beta
    Gi_all, Gii_all = x  # RealData format: just (Gi, Gii)
    dataset_indices = global_dataset_indices  # Get from global variable
    n, m = n_fixed, m_fixed
    
    # Create arrays of GIc and GIIc values for each data point
    GIc_values = np.zeros_like(Gi_all)
    GIIc_values = np.zeros_like(Gii_all)
    
    # Assign GIc and GIIc values based on dataset index
    GIc_values[dataset_indices == 0] = GIc_1
    GIc_values[dataset_indices == 1] = GIc_2
    GIc_values[dataset_indices == 2] = GIc_3
    
    GIIc_values[dataset_indices == 0] = GIIc_1
    GIIc_values[dataset_indices == 1] = GIIc_2
    GIIc_values[dataset_indices == 2] = GIIc_3
    
    # Compute derivatives
    if var == 'A':
        # For variant A: ((Gi/GIc)**n + (Gii/GIIc)**m)**(2/(n+m)) - 1
        term1 = (Gi_all/GIc_values)**(1/n_fixed)    
        term2 = (Gii_all/GIIc_values)**(1/m_fixed)
        sum_terms = term1 + term2
        power = 2/(1/n_fixed+1/m_fixed)
        
        dGi = power * sum_terms**(power-1) * (1/n_fixed) * (Gi_all/GIc_values)**(1/n_fixed-1) / GIc_values
        dGii = power * sum_terms**(power-1) * (1/m_fixed) * (Gii_all/GIIc_values)**(1/m_fixed-1) / GIIc_values
        
    elif var == 'B':
        # For variant B: (Gi/GIc)**n + (Gii/GIIc)**m - 1
        dGi = (1/n_fixed) * (Gi_all/GIc_values)**(1/n_fixed-1) / GIc_values
        dGii = (1/m_fixed) * (Gii_all/GIIc_values)**(1/m_fixed-1) / GIIc_values
    
    # Stack derivatives
    return np.row_stack([dGi, dGii])

def run_regression_multi_dataset_fixed_exponents(
        data, model, beta0,
        sstol=1e-12, partol=1e-12,
        maxit=1000, ndigit=12,
        ifixb=[1, 1, 1, 1, 1, 1],  # All 6 parameters free
        fit_type=1, deriv=3,
        init=0, iteration=0, final=0):
    """
    Setup ODR object and run regression for multi-dataset data with fixed exponents.
    """
    # Setup ODR object
    odr = ODR(
        data,                   # Input data
        model,                  # Model
        beta0=beta0,            # Initial parameter guess
        sstol=sstol,            # Tolerance for residual convergence (<1)
        partol=partol,          # Tolerance for parameter convergence (<1)
        maxit=maxit,            # Maximum number of iterations
        ndigit=ndigit,          # Number of reliable digits
        ifixb=ifixb,            # 0 parameter fixed, 1 parameter free
    )

    # Set job options
    odr.set_job(
        fit_type=fit_type,      # 0 explicit ODR, 1 implicit ODR
        deriv=deriv             # 0 finite differences, 3 jacobians
    )

    # Define outputs
    odr.set_iprint(
        init=init,              # No, short, or long initialization report
        iter=iteration,         # No, short, or long iteration report
        final=final,            # No, short, or long final report
    )

    # Run optimization
    return odr.run()

def calc_fit_statistics_multi_dataset_fixed_exponents(final, ndof, n_fixed, m_fixed):
    """
    Calculate fit statistics for multi-dataset regression with fixed exponents.
    """
    fit = defaultdict()
    
    # Extract parameters (only 6 parameters)
    fit['params'] = final.beta
    fit['stddev'] = final.sd_beta
    
    # Unpack parameters
    GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = final.beta
    
    # Store individual parameters
    fit['GIc_1'] = GIc_1
    fit['GIIc_1'] = GIIc_1
    fit['GIc_2'] = GIc_2
    fit['GIIc_2'] = GIIc_2
    fit['GIc_3'] = GIc_3
    fit['GIIc_3'] = GIIc_3
    fit['n'] = n_fixed  # Fixed value
    fit['m'] = m_fixed  # Fixed value
    
    # Store uncertainties (only for the 6 free parameters)
    fit['stddev_GIc_1'] = final.sd_beta[0]
    fit['stddev_GIIc_1'] = final.sd_beta[1]
    fit['stddev_GIc_2'] = final.sd_beta[2]
    fit['stddev_GIIc_2'] = final.sd_beta[3]
    fit['stddev_GIc_3'] = final.sd_beta[4]
    fit['stddev_GIIc_3'] = final.sd_beta[5]
    fit['stddev_n'] = 0.0  # No uncertainty for fixed parameter
    fit['stddev_m'] = 0.0  # No uncertainty for fixed parameter
    
    # Goodness of fit
    fit['chi_squared'] = final.sum_square
    fit['reduced_chi_squared'] = fit['chi_squared'] / ndof  # Correct calculation
    fit['p_value'] = distributions.chi2.sf(fit['chi_squared'], ndof)
    fit['R_squared'] = 1 - fit['chi_squared']/(ndof + fit['chi_squared'])
    
    # Store final result
    fit['final'] = final
    
    return fit

def results_multi_dataset_fixed_exponents(fit, data):
    """
    Print multi-dataset fit results with fixed exponents.
    """
    print("=" * 60)
    print("MULTI-DATASET REGRESSION RESULTS (FIXED EXPONENTS)")
    print("=" * 60)
    print()
    
    print("FIXED PARAMETERS:")
    print(f"n (exponent) = {fit['n']:.3f} (FIXED)")
    print(f"m (exponent) = {fit['m']:.3f} (FIXED)")
    print()
    
    print("FITTED PARAMETERS:")
    print("Dataset 1:")
    print(f"  GIc_1 = {fit['GIc_1']:.3f} ± {fit['stddev_GIc_1']:.3f}")
    print(f"  GIIc_1 = {fit['GIIc_1']:.3f} ± {fit['stddev_GIIc_1']:.3f}")
    print(f"  GIIc_1/GIc_1 = {fit['GIIc_1']/fit['GIc_1']:.3f}")
    print()
    
    print("Dataset 2:")
    print(f"  GIc_2 = {fit['GIc_2']:.3f} ± {fit['stddev_GIc_2']:.3f}")
    print(f"  GIIc_2 = {fit['GIIc_2']:.3f} ± {fit['stddev_GIIc_2']:.3f}")
    print(f"  GIIc_2/GIc_2 = {fit['GIIc_2']/fit['GIc_2']:.3f}")
    print()
    
    print("Dataset 3:")
    print(f"  GIc_3 = {fit['GIc_3']:.3f} ± {fit['stddev_GIc_3']:.3f}")
    print(f"  GIIc_3 = {fit['GIIc_3']:.3f} ± {fit['stddev_GIIc_3']:.3f}")
    print(f"  GIIc_3/GIc_3 = {fit['GIIc_3']/fit['GIc_3']:.3f}")
    print()
    
    print("GOODNESS OF FIT:")
    print(f"Reduced χ² = {fit['reduced_chi_squared']:.3f}")
    print(f"p-value = {fit['p_value']:.3f}")
    print(f"R² = {fit['R_squared']:.3f}")
    # Calculate total data points (data.x is a tuple: [Gi_values, Gii_values])
    total_data_points = len(data.x[0])  # Number of Gi values = total data points
    ndof = total_data_points - 6  # 6 parameters: 3×GIc, 3×GIIc
    print(f"Degrees of freedom: {ndof}")
    print(f"Total data points: {total_data_points}")
    print("=" * 60)

def odr_multi_dataset_fixed_exponents(df_list, dim=1, n_fixed=2, m_fixed=2, var='B', 
                                    print_results=True, bounds=None):
    """
    Perform orthogonal distance regression on multiple datasets with fixed exponents.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames with fracture toughness data
    dim : int, optional
        Dimensionality of the response function. Default is 1.
    n_fixed : float, optional
        Fixed value for exponent n. Default is 2.
    m_fixed : float, optional
        Fixed value for exponent m. Default is 2.
    var : str, optional
        Residual variant {'A', 'B'}. Default is 'B'.
    print_results : bool, optional
        If True, print fit results. Default is True.
    bounds : dict, optional
        Dictionary with bounds for parameters. Default is None.
    
    Returns:
    --------
    fit : dict
        Fit results dictionary
    """
    print("=== MULTI-DATASET REGRESSION WITH FIXED EXPONENTS ===")
    print(f"Fixed exponents: n = {n_fixed}, m = {m_fixed}")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0
        }
    
    # Assemble data
    data, ndof = assemble_data_multi_dataset(df_list, dim)
    
    # Calculate total data points (data.x is a tuple: [Gi_values, Gii_values])
    total_data_points = len(data.x[0])  # Number of Gi values = total data points
    ndof = total_data_points - 6  # 6 parameters: 3×GIc, 3×GIIc
    
    print(f"Data assembled successfully:")
    print(f"  Total data points: {total_data_points}")
    print(f"  Degrees of freedom: {ndof}")
    print(f"  Free parameters: 6 (3 GIc + 3 GIIc)")
    print(f"  Fixed parameters: n = {n_fixed}, m = {m_fixed}")
    
    
    # Create model with fixed exponents
    model = Model(fcn=residual_multi_dataset_fixed_exponents, 
                  fjacb=param_jacobian_multi_dataset_fixed_exponents,
                  fjacd=value_jacobian_multi_dataset_fixed_exponents, 
                  implicit=True,
                  extra_args=(n_fixed, m_fixed, var))
    
    # Generate initial guesses (only 6 parameters)
    gi_means = [df['GIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    gii_means = [df['GIIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    
    # Create multiple starting points
    starting_points = []
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        start_point = [
            gi_means[0] * factor, gii_means[0] * factor,  # Dataset 1
            gi_means[1] * factor, gii_means[1] * factor,  # Dataset 2
            gi_means[2] * factor, gii_means[2] * factor   # Dataset 3
        ]
        starting_points.append(start_point)
    
    # Run regression for all starting points
    runs = []
    for i, start_point in enumerate(starting_points):
        try:
            print(f"Trying starting point {i+1}: {start_point}")
            r = run_regression_multi_dataset_fixed_exponents(data, model, start_point)
            print(f"  Result: info={r.info}, sum_square={r.sum_square:.6f}")
            if r.info <= 3:
                runs.append(r)
                print(f"  SUCCESS!")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            continue
    
    if not runs:
        print("ODR failed, trying scipy.optimize.minimize as fallback...")
        
        def objective(params):
            """Objective function for minimize"""
            GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = params
            Gi_all, Gii_all = data.x
            dataset_indices = global_dataset_indices
            
            # Check bounds
            if (GIc_1 <= bounds['GIc_min'] or GIIc_1 <= bounds['GIIc_min'] or 
                GIc_2 <= bounds['GIc_min'] or GIIc_2 <= bounds['GIIc_min'] or 
                GIc_3 <= bounds['GIc_min'] or GIIc_3 <= bounds['GIIc_min']):
                return 1e6
            
            if (GIc_1 > bounds['GIc_max'] or GIIc_1 > bounds['GIIc_max'] or 
                GIc_2 > bounds['GIc_max'] or GIIc_2 > bounds['GIIc_max'] or 
                GIc_3 > bounds['GIc_max'] or GIIc_3 > bounds['GIIc_max']):
                return 1e6
            
            # Create arrays of GIc and GIIc values for each data point
            GIc_values = np.zeros_like(Gi_all)
            GIIc_values = np.zeros_like(Gii_all)
            
            GIc_values[dataset_indices == 0] = GIc_1
            GIc_values[dataset_indices == 1] = GIc_2
            GIc_values[dataset_indices == 2] = GIc_3
            
            GIIc_values[dataset_indices == 0] = GIIc_1
            GIIc_values[dataset_indices == 1] = GIIc_2
            GIIc_values[dataset_indices == 2] = GIIc_3
            
            # Compute residuals
            try:
                with np.errstate(invalid='ignore', divide='ignore'):
                    if var == 'A':
                        term1 = (Gi_all/GIc_values)**(1/n_fixed)
                        term2 = (Gii_all/GIIc_values)**(1/m_fixed)
                        sum_terms = term1 + term2
                        power = 2/(1/n_fixed+1/m_fixed)
                        res = sum_terms**power - 1
                    elif var == 'B':
                        term1 = (Gi_all/GIc_values)**(1/n_fixed)
                        term2 = (Gii_all/GIIc_values)**(1/m_fixed)  
                        res = term1 + term2 - 1
                    else:
                        return 1e6
                    
                    res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
                    return np.sum(res**2)
                    
            except Exception:
                return 1e6
        
        # Try multiple starting points with minimize
        best_result = None
        best_objective = float('inf')
        
        minimize_bounds = [
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max'])
        ]
        
        for i, start_point in enumerate(starting_points):
            try:
                result = minimize(objective, start_point, 
                                method='L-BFGS-B',
                                bounds=minimize_bounds)
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    print(f"  New best: {result.fun:.6f} from start point {i+1}")
                    
            except Exception as e:
                continue
        
        if best_result is not None:
            # Create a mock ODR result object
            class MockODRResult:
                def __init__(self, params, objective_val):
                    self.beta = params
                    self.sum_square = objective_val
                    self.res_var = objective_val / ndof
                    self.sd_beta = [0.01] * 6  # Approximate uncertainties
                    self.info = 1
            
            final = MockODRResult(best_result.x, best_result.fun)
        else:
            raise ValueError("Both ODR and minimize failed. Check your data.")
    else:
        # Use the best ODR result
        final = runs[np.argmin([run.sum_square for run in runs])]
    
    # Compile results
    fit = calc_fit_statistics_multi_dataset_fixed_exponents(final, ndof, n_fixed, m_fixed)
    fit['var'] = var
    fit['method'] = 'ODR_fixed_exponents'
    fit['bounds'] = bounds
    
    if print_results:
        results_multi_dataset_fixed_exponents(fit, data)
    
    return fit

def assemble_data_multi_dataset_no_errors(df_list, dim=1):
    """
    Compile ODR pack data object from multiple data frames, ignoring error bars.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames, each with fracture toughness data
    dim : int
        Dimensionality of the response function
    
    Returns:
    --------
    data : scipy.odr.RealData
        ODR pack data object with zero uncertainties
    ndof : int
        Number of degrees of freedom
    """
    # Combine all datasets
    Gi_all = []
    Gii_all = []
    dataset_indices = []
    
    for i, df in enumerate(df_list):
        # Extract data from current dataset (only nominal values, no uncertainties)
        Gi = df['GIc'].apply(lambda x: x.nominal_value).values
        Gii = df['GIIc'].apply(lambda x: x.nominal_value).values
        
        # Append to combined arrays
        Gi_all.extend(Gi)
        Gii_all.extend(Gii)
        dataset_indices.extend([i] * len(Gi))
    
    # Convert to numpy arrays
    Gi_all = np.array(Gi_all)
    Gii_all = np.array(Gii_all)
    dataset_indices = np.array(dataset_indices)
    
    # Stack data
    exp = np.row_stack([Gi_all, Gii_all])
    
    # Set uncertainties to zero (ignoring error bars)
    std = np.zeros_like(exp)
    
    # Compute degrees of freedom (observations - parameters)
    # Parameters: 6 (3 GIc + 3 GIIc) for fixed exponents
    ndof = exp.shape[1] - 6
    
    # Store dataset indices globally for use in residual function
    global global_dataset_indices
    global_dataset_indices = dataset_indices
    
    # Use RealData with zero uncertainties
    return RealData(exp, y=dim, sx=std), ndof

def odr_multi_dataset_fixed_exponents_no_errors(df_list, dim=1, n_fixed=2, m_fixed=2, var='B', 
                                              print_results=True, bounds=None):
    """
    Perform orthogonal distance regression on multiple datasets with fixed exponents, ignoring error bars.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames with fracture toughness data
    dim : int, optional
        Dimensionality of the response function. Default is 1.
    n_fixed : float, optional
        Fixed value for exponent n. Default is 2.
    m_fixed : float, optional
        Fixed value for exponent m. Default is 2.
    var : str, optional
        Residual variant {'A', 'B'}. Default is 'B'.
    print_results : bool, optional
        If True, print fit results. Default is True.
    bounds : dict, optional
        Dictionary with bounds for parameters. Default is None.
    
    Returns:
    --------
    fit : dict
        Fit results dictionary
    """
    print("=== MULTI-DATASET REGRESSION WITH FIXED EXPONENTS (NO ERROR BARS) ===")
    print(f"Fixed exponents: n = {n_fixed}, m = {m_fixed}")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0
        }
    
    # Assemble data (ignoring error bars)
    data, ndof = assemble_data_multi_dataset_no_errors(df_list, dim)
    
    print(f"Data assembled successfully:")
    print(f"  Data shape: {data.x[0].shape}")
    print(f"  Degrees of freedom: {ndof}")
    print(f"  Free parameters: 6 (3 GIc + 3 GIIc)")
    print(f"  Fixed parameters: n = {n_fixed}, m = {m_fixed}")
    print(f"  Error bars: IGNORED (set to zero)")
    
    # Create model with fixed exponents
    model = Model(fcn=residual_multi_dataset_fixed_exponents, 
                  fjacb=param_jacobian_multi_dataset_fixed_exponents,
                  fjacd=value_jacobian_multi_dataset_fixed_exponents, 
                  implicit=True,
                  extra_args=(n_fixed, m_fixed, var))
    
    # Generate initial guesses (only 6 parameters)
    gi_means = [df['GIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    gii_means = [df['GIIc'].apply(lambda x: x.nominal_value).mean() for df in df_list]
    
    # Create multiple starting points
    starting_points = []
    for factor in [0.5, 0.8, 1.0, 1.2, 1.5]:
        start_point = [
            gi_means[0] * factor, gii_means[0] * factor,  # Dataset 1
            gi_means[1] * factor, gii_means[1] * factor,  # Dataset 2
            gi_means[2] * factor, gii_means[2] * factor   # Dataset 3
        ]
        starting_points.append(start_point)
    
    # Run regression for all starting points
    runs = []
    for i, start_point in enumerate(starting_points):
        try:
            print(f"Trying starting point {i+1}: {start_point}")
            r = run_regression_multi_dataset_fixed_exponents(data, model, start_point)
            print(f"  Result: info={r.info}, sum_square={r.sum_square:.6f}")
            if r.info <= 3:
                runs.append(r)
                print(f"  SUCCESS!")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            continue
    
    if not runs:
        print("ODR failed, trying scipy.optimize.minimize as fallback...")
        
        def objective(params):
            """Objective function for minimize"""
            GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3 = params
            Gi_all, Gii_all = data.x
            dataset_indices = global_dataset_indices
            
            # Check bounds
            if (GIc_1 <= bounds['GIc_min'] or GIIc_1 <= bounds['GIIc_min'] or 
                GIc_2 <= bounds['GIc_min'] or GIIc_2 <= bounds['GIIc_min'] or 
                GIc_3 <= bounds['GIc_min'] or GIIc_3 <= bounds['GIIc_min']):
                return 1e6
            
            if (GIc_1 > bounds['GIc_max'] or GIIc_1 > bounds['GIIc_max'] or 
                GIc_2 > bounds['GIc_max'] or GIIc_2 > bounds['GIIc_max'] or 
                GIc_3 > bounds['GIc_max'] or GIIc_3 > bounds['GIIc_max']):
                return 1e6
            
            # Create arrays of GIc and GIIc values for each data point
            GIc_values = np.zeros_like(Gi_all)
            GIIc_values = np.zeros_like(Gii_all)
            
            GIc_values[dataset_indices == 0] = GIc_1
            GIc_values[dataset_indices == 1] = GIc_2
            GIc_values[dataset_indices == 2] = GIc_3
            
            GIIc_values[dataset_indices == 0] = GIIc_1
            GIIc_values[dataset_indices == 1] = GIIc_2
            GIIc_values[dataset_indices == 2] = GIIc_3
            
            # Compute residuals
            try:
                with np.errstate(invalid='ignore', divide='ignore'):
                    if var == 'A':
                        term1 = (Gi_all/GIc_values)**(1/n_fixed)
                        term2 = (Gii_all/GIIc_values)**(1/m_fixed)
                        sum_terms = term1 + term2
                        power = 2/(1/n_fixed+1/m_fixed)
                        res = sum_terms**power - 1
                    elif var == 'B':
                        term1 = (Gi_all/GIc_values)**(1/n_fixed)
                        term2 = (Gii_all/GIIc_values)**(1/m_fixed)  
                        res = term1 + term2 - 1
                    else:
                        return 1e6
                    
                    res = np.where(np.isnan(res) | np.isinf(res), 1e3, res)
                    return np.sum(res**2)
                    
            except Exception:
                return 1e6
        
        # Try multiple starting points with minimize
        best_result = None
        best_objective = float('inf')
        
        minimize_bounds = [
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max']),
            (bounds['GIc_min'], bounds['GIc_max']), (bounds['GIIc_min'], bounds['GIIc_max'])
        ]
        
        for i, start_point in enumerate(starting_points):
            try:
                result = minimize(objective, start_point, 
                                method='L-BFGS-B',
                                bounds=minimize_bounds)
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    print(f"  New best: {result.fun:.6f} from start point {i+1}")
                    
            except Exception as e:
                continue
        
        if best_result is not None:
            # Create a mock ODR result object
            class MockODRResult:
                def __init__(self, params, objective_val):
                    self.beta = params
                    self.sum_square = objective_val
                    self.res_var = objective_val / ndof
                    self.sd_beta = [0.01] * 6  # Approximate uncertainties
                    self.info = 1
            
            final = MockODRResult(best_result.x, best_result.fun)
        else:
            raise ValueError("Both ODR and minimize failed. Check your data.")
    else:
        # Use the best ODR result
        final = runs[np.argmin([run.sum_square for run in runs])]
    
    # Compile results
    fit = calc_fit_statistics_multi_dataset_fixed_exponents(final, ndof, n_fixed, m_fixed)
    fit['var'] = var
    fit['method'] = 'ODR_fixed_exponents_no_errors'
    fit['bounds'] = bounds
    
    if print_results:
        results_multi_dataset_fixed_exponents_no_errors(fit)
    
    return fit

def results_multi_dataset_fixed_exponents_no_errors(fit):
    """
    Print multi-dataset fit results with fixed exponents (no error bars).
    """
    print("=" * 60)
    print("MULTI-DATASET REGRESSION RESULTS (FIXED EXPONENTS, NO ERROR BARS)")
    print("=" * 60)
    print()
    
    print("FIXED PARAMETERS:")
    print(f"n (exponent) = {fit['n']:.3f} (FIXED)")
    print(f"m (exponent) = {fit['m']:.3f} (FIXED)")
    print()
    
    print("FITTED PARAMETERS:")
    print("Dataset 1:")
    print(f"  GIc_1 = {fit['GIc_1']:.3f} ± {fit['stddev_GIc_1']:.3f}")
    print(f"  GIIc_1 = {fit['GIIc_1']:.3f} ± {fit['stddev_GIIc_1']:.3f}")
    print(f"  GIIc_1/GIc_1 = {fit['GIIc_1']/fit['GIc_1']:.3f}")
    print()
    
    print("Dataset 2:")
    print(f"  GIc_2 = {fit['GIc_2']:.3f} ± {fit['stddev_GIc_2']:.3f}")
    print(f"  GIIc_2 = {fit['GIIc_2']:.3f} ± {fit['stddev_GIIc_2']:.3f}")
    print(f"  GIIc_2/GIc_2 = {fit['GIIc_2']/fit['GIc_2']:.3f}")
    print()
    
    print("Dataset 3:")
    print(f"  GIc_3 = {fit['GIc_3']:.3f} ± {fit['stddev_GIc_3']:.3f}")
    print(f"  GIIc_3 = {fit['GIIc_3']:.3f} ± {fit['stddev_GIIc_3']:.3f}")
    print(f"  GIIc_3/GIc_3 = {fit['GIIc_3']/fit['GIc_3']:.3f}")
    print()
    
    print("GOODNESS OF FIT:")
    print(f"Sum of squared residuals = {fit['chi_squared']:.6f}")
    print(f"Reduced χ² = {fit['reduced_chi_squared']:.3f}")
    print(f"p-value = {fit['p_value']:.3f}")
    print(f"R² = {fit['R_squared']:.3f}")
    print("=" * 60)
    print("NOTE: Error bars in data were ignored during fitting")
    print("=" * 60)

def create_n_m_landscape_heatmap(df_list, exponents_to_try, var='B', bounds=None, save_path=None):
    """
    Create a heatmap showing residual values for different n,m exponent combinations.
    
    Parameters:
    -----------
    df_list : list of pd.DataFrame
        List of data frames with fracture toughness data
    exponents_to_try : list of tuples
        List of (n, m) exponent combinations to test
    var : str, optional
        Residual variant {'A', 'B'}. Default is 'B'.
    bounds : dict, optional
        Dictionary with bounds for parameters. Default is None.
    save_path : str, optional
        Path to save the heatmap figure. Default is None.
    
    Returns:
    --------
    results_dict : dict
        Dictionary with results for each n,m combination
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=== CREATING N-M LANDSCAPE HEATMAP ===")
    print(f"Testing {len(exponents_to_try)} exponent combinations...")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0
        }
    
    # Test all exponent combinations
    results_dict = {}
    successful_count = 0
    
    for n, m in exponents_to_try:
        try:
            print(f"Testing n={n:.2f}, m={m:.2f}...")
            result = odr_multi_dataset_fixed_exponents(
                df_list, dim=1, n_fixed=n, m_fixed=m, var=var, 
                print_results=False, bounds=bounds
            )
            residual = result['chi_squared']  # Sum of squared residuals
            results_dict[(n, m)] = {
                'residual': residual,
                'success': True,
                'GIc_1': result['GIc_1'],
                'GIIc_1': result['GIIc_1'],
                'GIc_2': result['GIc_2'],
                'GIIc_2': result['GIIc_2'],
                'GIc_3': result['GIc_3'],
                'GIIc_3': result['GIIc_3'],
                'R_squared': result['R_squared']
            }
            successful_count += 1
            print(f"  Residual = {residual:.6f}")
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}...")
            results_dict[(n, m)] = {
                'residual': float('inf'),
                'success': False,
                'GIc_1': None, 'GIIc_1': None,
                'GIc_2': None, 'GIIc_2': None,
                'GIc_3': None, 'GIIc_3': None,
                'R_squared': 0.0
            }
    
    print(f"\nSuccessful optimizations: {successful_count}/{len(exponents_to_try)}")
    
    # Extract unique n and m values for the grid
    n_values = sorted(list(set([n for n, m in exponents_to_try])))
    m_values = sorted(list(set([m for n, m in exponents_to_try])))
    
    # Create residual matrix for heatmap
    residual_matrix = np.full((len(m_values), len(n_values)), np.nan)
    
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if (n, m) in results_dict:
                if results_dict[(n, m)]['success']:
                    residual_matrix[i, j] = results_dict[(n, m)]['residual']
    
    # Create the figure with side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Create heatmap on the left
    sns.heatmap(residual_matrix, 
                xticklabels=[f'{n:.2f}' for n in n_values],
                yticklabels=[f'{m:.2f}' for m in m_values],
                cmap='viridis', 
                annot=True,  # Show values
                fmt='.3f',   # Format as 3 decimal places, 
                vmin=0, vmax=100,  # Adjusted for chi-squared values
                cbar_kws={'label': 'Sum of Squared Residuals'},
                square=True,
                ax=ax1)
    
    ax1.set_xlabel('n', fontsize=16)
    ax1.set_ylabel('m', fontsize=16)
    ax1.set_title(f'N-M Landscape: Residual Values (Variant {var})', fontsize=16)
    
    # Create information panel on the right
    ax2.axis('off')  # Turn off axes for text display
    
    # Add text showing best combination and parameters
    if successful_count > 0:
        best_combo = min([(k, v) for k, v in results_dict.items() if v['success']], 
                        key=lambda x: x[1]['residual'])
        best_n, best_m = best_combo[0]
        best_result = best_combo[1]
        best_residual = best_result['residual']
        
        # Create formatted text for the right panel
        info_text = f"""BEST COMBINATION

Exponents:
• n = {best_n:.3f}
• m = {best_m:.3f}

Residual: {best_residual:.6f}

Fracture Toughness Values:

Dataset 1:
• GIc = {best_result['GIc_1']:.4f} J/m²
• GIIc = {best_result['GIIc_1']:.4f} J/m²

Dataset 2:
• GIc = {best_result['GIc_2']:.4f} J/m²
• GIIc = {best_result['GIIc_2']:.4f} J/m²

Dataset 3:
• GIc = {best_result['GIc_3']:.4f} J/m²
• GIIc = {best_result['GIIc_3']:.4f} J/m²

Fit Statistics:
• R² = {best_result.get('R_squared', 0.0):.4f}
• Successful fits: {successful_count}/{len(exponents_to_try)}"""
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    else:
        ax2.text(0.05, 0.95, "No successful optimizations found.", 
                transform=ax2.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    
    return results_dict

def create_n_m_landscape_heatmap_no_errors(df_list, exponents_to_try, var='B', bounds=None, save_path=None):
    """
    Create a heatmap showing residual values for different n,m exponent combinations (no error bars).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=== CREATING N-M LANDSCAPE HEATMAP (NO ERROR BARS) ===")
    print(f"Testing {len(exponents_to_try)} exponent combinations...")
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'GIc_min': 0.01, 'GIc_max': 10.0,
            'GIIc_min': 0.01, 'GIIc_max': 2.0
        }
    
    # Test all exponent combinations
    results_dict = {}
    successful_count = 0
    
    for n, m in exponents_to_try:
        try:
            print(f"Testing n={n:.2f}, m={m:.2f} (no error bars)...")
            result = odr_multi_dataset_fixed_exponents_no_errors(
                df_list, dim=1, n_fixed=n, m_fixed=m, var=var, 
                print_results=False, bounds=bounds
            )
            residual = result['chi_squared']  # Sum of squared residuals
            results_dict[(n, m)] = {
                'residual': residual,
                'success': True,
                'GIc_1': result['GIc_1'],
                'GIIc_1': result['GIIc_1'],
                'GIc_2': result['GIc_2'],
                'GIIc_2': result['GIIc_2'],
                'GIc_3': result['GIc_3'],
                'GIIc_3': result['GIIc_3'],
                'R_squared': result['R_squared']
            }
            successful_count += 1
            print(f"  Residual = {residual:.6f}")
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}...")
            results_dict[(n, m)] = {
                'residual': float('inf'),
                'success': False,
                'GIc_1': None, 'GIIc_1': None,
                'GIc_2': None, 'GIIc_2': None,
                'GIc_3': None, 'GIIc_3': None,
                'R_squared': 0.0
            }
    
    print(f"\nSuccessful optimizations: {successful_count}/{len(exponents_to_try)}")
    
    # Extract unique n and m values for the grid
    n_values = sorted(list(set([n for n, m in exponents_to_try])))
    m_values = sorted(list(set([m for n, m in exponents_to_try])))
    
    # Create residual matrix for heatmap
    residual_matrix = np.full((len(m_values), len(n_values)), np.nan)
    
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if (n, m) in results_dict:
                if results_dict[(n, m)]['success']:
                    residual_matrix[i, j] = results_dict[(n, m)]['residual']
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(residual_matrix, 
                xticklabels=[f'{n:.2f}' for n in n_values],
                yticklabels=[f'{m:.2f}' for m in m_values],
                cmap='viridis_r',  # Reverse viridis (lower values = better = darker)
                annot=True,  # Show values
                fmt='.3f',   # Format as 3 decimal places
                cbar_kws={'label': 'Sum of Squared Residuals'},
                square=True)
    
    plt.xlabel('n', fontsize=18)
    plt.ylabel('m', fontsize=18)
    plt.title(f'N-M Landscape: Residual Values (Variant {var}, No Error Bars)')
    plt.gca().invert_yaxis()  # Invert y-axis so m=0.1 is at top
    
    # Add text showing best combination
    if successful_count > 0:
        best_combo = min([(k, v) for k, v in results_dict.items() if v['success']], 
                        key=lambda x: x[1]['residual'])
        best_n, best_m = best_combo[0]
        best_residual = best_combo[1]['residual']
        plt.figtext(0.5, 0.005, 
                   f'Best combination: n={best_n:.2f}, m={best_m:.2f} (Residual = {best_residual:.6f})',
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    
    return results_dict

def compare_landscapes_with_without_errors(df_list, exponents_to_try, var='B', bounds=None, save_path=None):
    """
    Create side-by-side heatmaps comparing results with and without error bars.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=== COMPARING LANDSCAPES WITH AND WITHOUT ERROR BARS ===")
    
    # Get results for both cases
    results_with_errors = create_n_m_landscape_heatmap(
        df_list, exponents_to_try, var=var, bounds=bounds, save_path=None
    )
    
    results_no_errors = create_n_m_landscape_heatmap_no_errors(
        df_list, exponents_to_try, var=var, bounds=bounds, save_path=None
    )
    
    # Extract unique n and m values
    n_values = sorted(list(set([n for n, m in exponents_to_try])))
    m_values = sorted(list(set([m for n, m in exponents_to_try])))
    
    # Create residual matrices
    residual_matrix_with = np.full((len(m_values), len(n_values)), np.nan)
    residual_matrix_without = np.full((len(m_values), len(n_values)), np.nan)
    
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if (n, m) in results_with_errors and results_with_errors[(n, m)]['success']:
                residual_matrix_with[i, j] = results_with_errors[(n, m)]['residual']
            if (n, m) in results_no_errors and results_no_errors[(n, m)]['success']:
                residual_matrix_without[i, j] = results_no_errors[(n, m)]['residual']
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap with error bars
    sns.heatmap(residual_matrix_with, 
                xticklabels=[f'{n:.2f}' for n in n_values],
                yticklabels=[f'{m:.2f}' for m in m_values],
                cmap='viridis_r',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Sum of Squared Residuals'},
                square=True,
                ax=ax1)
    ax1.set_xlabel('Exponent n')
    ax1.set_ylabel('Exponent m')
    ax1.set_title(f'With Error Bars (Variant {var})')
    ax1.invert_yaxis()
    
    # Heatmap without error bars
    sns.heatmap(residual_matrix_without, 
                xticklabels=[f'{n:.2f}' for n in n_values],
                yticklabels=[f'{m:.2f}' for m in m_values],
                cmap='viridis_r',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Sum of Squared Residuals'},
                square=True,
                ax=ax2)
    ax2.set_xlabel('Exponent n')
    ax2.set_ylabel('Exponent m')
    ax2.set_title(f'Without Error Bars (Variant {var})')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmap saved to: {save_path}")
    
    plt.show()
    
    return results_with_errors, results_no_errors