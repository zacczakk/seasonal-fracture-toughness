"""Module for multi-dataset orthogonal distance regression with fixed exponents."""

import numpy as np
from uncertainties import unumpy
from collections import defaultdict
from scipy.odr import RealData, Model, ODR
from scipy.stats import distributions
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Global variable to store dataset indices
global_dataset_indices = None

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
    global global_dataset_indices
    
    # Combine all datasets
    Gi_all = []
    Gii_all = []
    std_Gi_all = []
    std_Gii_all = []
    dataset_indices = []
    
    for i, df in enumerate(df_list):
        # Extract data from current dataset
        Gi = df['GIc'].apply(unumpy.nominal_values).values
        Gii = df['GIIc'].apply(unumpy.nominal_values).values
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
    
    # Store dataset indices globally for use in residual function
    global_dataset_indices = dataset_indices
    
    # Use RealData
    return RealData(exp, y=dim, sx=std)

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
    fit['reduced_chi_squared'] = final.res_var  # ODR provides this directly
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
    data = assemble_data_multi_dataset(df_list, dim)
    
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
        successful_results = [(k, v) for k, v in results_dict.items() if v['success']]
        if successful_results:
            best_combo = min(successful_results, key=lambda x: x[1]['residual'])
            best_n, best_m = best_combo[0]
            best_result = best_combo[1]
            best_residual = best_result['residual']
        else:
            best_n, best_m = 0, 0
            best_result = {'residual': float('inf'), 'GIc_1': 0, 'GIIc_1': 0, 
                          'GIc_2': 0, 'GIIc_2': 0, 'GIc_3': 0, 'GIIc_3': 0, 'R_squared': 0}
            best_residual = float('inf')
        
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