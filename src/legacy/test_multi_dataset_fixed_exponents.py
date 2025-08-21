"""
Test script for multi-dataset fixed exponents regression.

This script creates dummy datasets and tests the regression functionality.
"""

import numpy as np
import pandas as pd
from uncertainties import ufloat
import multi_dataset_fixed_exponents as reg

def create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                        noise_level=0.05, uncertainty_level=0.1, uncertainty_type='proportional'):
    """
    Create a dummy dataset following the interaction law with realistic uncertainties.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    GIc_true : float
        True GIc value
    GIIc_true : float
        True GIIc value
    n_true : float
        True n exponent
    m_true : float
        True m exponent
    noise_level : float
        Level of noise to add (fraction of true values)
    uncertainty_level : float
        Level of uncertainty (fraction of values)
    uncertainty_type : str
        Type of uncertainty: 'proportional', 'fixed', or 'mixed'
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with GIc and GIIc columns as ufloat objects
    """
    # Generate Gi values (Mode I fracture toughness)
    Gi_values = np.linspace(0.1 * GIc_true, 0.9 * GIc_true, n_points)
    
    # Calculate corresponding Gii values using the interaction law
    # For Variant B: (Gi/GIc)^(1/n) + (Gii/GIIc)^(1/m) = 1
    # Therefore: Gii = GIIc * (1 - (Gi/GIc)^(1/n))^(1/(1/m))
    
    if n_true == 1 and m_true == 1:
        # Simplified case: Gi/GIc + Gii/GIIc = 1
        Gii_values = GIIc_true * (1 - Gi_values/GIc_true)
    else:
        # General case
        Gii_values = GIIc_true * (1 - (Gi_values/GIc_true)**(1/n_true))**(1/(1/m_true))
    
    # Add noise
    if noise_level > 0:
        Gi_noise = np.random.normal(0, noise_level * GIc_true, n_points)
        Gii_noise = np.random.normal(0, noise_level * GIIc_true, n_points)
        Gi_values += Gi_noise
        Gii_values += Gii_noise
        
        # Ensure positive values
        Gi_values = np.maximum(Gi_values, 0.01 * GIc_true)
        Gii_values = np.maximum(Gii_values, 0.01 * GIIc_true)
    
    # Create uncertainties based on type
    if uncertainty_type == 'proportional':
        # Proportional uncertainties (e.g., 10% of value)
        Gi_uncertainties = uncertainty_level * Gi_values
        Gii_uncertainties = uncertainty_level * Gii_values
        
    elif uncertainty_type == 'fixed':
        # Fixed uncertainties (e.g., ±0.05 J/m²)
        Gi_uncertainties = uncertainty_level * np.ones_like(Gi_values)
        Gii_uncertainties = uncertainty_level * np.ones_like(Gii_values)
        
    elif uncertainty_type == 'mixed':
        # Mixed: proportional + fixed component
        Gi_uncertainties = uncertainty_level * Gi_values + 0.02 * GIc_true
        Gii_uncertainties = uncertainty_level * Gii_values + 0.01 * GIIc_true
        
    elif uncertainty_type == 'realistic':
        # Realistic uncertainties based on typical experimental conditions
        # GIc typically has 5-15% uncertainty, GIIc has 8-20% uncertainty
        Gi_uncertainties = np.random.uniform(0.05, 0.15, n_points) * Gi_values
        Gii_uncertainties = np.random.uniform(0.08, 0.20, n_points) * Gii_values
        
    else:
        # Default to proportional
        Gi_uncertainties = uncertainty_level * Gi_values
        Gii_uncertainties = uncertainty_level * Gii_values
    
    # Ensure minimum uncertainties to avoid division by zero
    Gi_uncertainties = np.maximum(Gi_uncertainties, 0.001 * GIc_true)
    Gii_uncertainties = np.maximum(Gii_uncertainties, 0.001 * GIIc_true)
    
    # Create ufloat objects
    Gi_with_unc = [ufloat(gi, gi_unc) for gi, gi_unc in zip(Gi_values, Gi_uncertainties)]
    Gii_with_unc = [ufloat(gii, gii_unc) for gii, gii_unc in zip(Gii_values, Gii_uncertainties)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'GIc': Gi_with_unc,
        'GIIc': Gii_with_unc
    })
    
    return df

def test_perfect_fit():
    """Test with perfect data and correct exponents."""
    print("=" * 60)
    print("TEST 1: PERFECT FIT WITH CORRECT EXPONENTS")
    print("=" * 60)
    
    # Create datasets with n=2, m=2
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with correct exponents
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    print("\nExpected vs Fitted Values:")
    expected_values = [(0.5, 0.3), (0.6, 0.4), (0.7, 0.5)]
    for i, (exp_GIc, exp_GIIc) in enumerate(expected_values):
        fitted_GIc = results[f'GIc_{i+1}']
        fitted_GIIc = results[f'GIIc_{i+1}']
        print(f"Dataset {i+1}:")
        print(f"  GIc: expected {exp_GIc:.3f}, fitted {fitted_GIc:.3f}, diff {abs(fitted_GIc-exp_GIc):.6f}")
        print(f"  GIIc: expected {exp_GIIc:.3f}, fitted {fitted_GIIc:.3f}, diff {abs(fitted_GIIc-exp_GIIc):.6f}")
    
    return results

def test_wrong_exponents():
    """Test with data generated with different exponents than fitted."""
    print("\n" + "=" * 60)
    print("TEST 2: WRONG EXPONENTS")
    print("=" * 60)
    
    # Create datasets with n=1, m=1
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=1.0, m_true=1.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=1.0, m_true=1.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=1.0, m_true=1.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with wrong exponents (n=2, m=2)
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    print(f"\nReduced χ² with wrong exponents: {results['reduced_chi_squared']:.6f}")
    print("(Should be much higher than 1.0)")
    
    return results

def test_noisy_data():
    """Test with noisy data and correct exponents."""
    print("\n" + "=" * 60)
    print("TEST 3: NOISY DATA WITH CORRECT EXPONENTS")
    print("=" * 60)
    
    # Create datasets with n=2, m=2 and noise
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with correct exponents
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    print(f"\nReduced χ² with noisy data: {results['reduced_chi_squared']:.6f}")
    print("(Should be close to 1.0 for well-calibrated uncertainties)")
    
    return results

def test_variant_a():
    """Test with Variant A interaction law."""
    print("\n" + "=" * 60)
    print("TEST 4: VARIANT A INTERACTION LAW")
    print("=" * 60)
    
    # Create datasets with n=2, m=2
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with Variant A
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='A',
        print_results=True
    )
    
    return results

def test_different_exponents():
    """Test with different fixed exponents."""
    print("\n" + "=" * 60)
    print("TEST 5: DIFFERENT FIXED EXPONENTS")
    print("=" * 60)
    
    # Create datasets with n=1.5, m=1.5
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=1.5, m_true=1.5, 
                              noise_level=0.0, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=1.5, m_true=1.5, 
                              noise_level=0.0, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=1.5, m_true=1.5, 
                              noise_level=0.0, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with correct exponents
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=1.5,
        m_fixed=1.5,
        var='B',
        print_results=True
    )
    
    return results

def test_bounds():
    """Test with parameter bounds."""
    print("\n" + "=" * 60)
    print("TEST 6: PARAMETER BOUNDS")
    print("=" * 60)
    
    # Create datasets
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.0, uncertainty_level=0.05)
    
    df_list = [df1, df2, df3]
    
    # Run regression with tight bounds
    bounds = {
        'GIc_min': 0.4, 'GIc_max': 0.8,
        'GIIc_min': 0.2, 'GIIc_max': 0.6
    }
    
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        bounds=bounds,
        print_results=True
    )
    
    return results

def test_realistic_uncertainties():
    """Test with realistic experimental uncertainties."""
    print("\n" + "=" * 60)
    print("TEST 7: REALISTIC EXPERIMENTAL UNCERTAINTIES")
    print("=" * 60)
    
    # Create datasets with realistic uncertainties
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.03, uncertainty_type='realistic')
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.03, uncertainty_type='realistic')
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.03, uncertainty_type='realistic')
    
    df_list = [df1, df2, df3]
    
    # Print uncertainty statistics
    print("Uncertainty Statistics:")
    for i, df in enumerate(df_list):
        gi_uncertainties = [x.std_dev for x in df['GIc']]
        gii_uncertainties = [x.std_dev for x in df['GIIc']]
        gi_values = [x.nominal_value for x in df['GIc']]
        gii_values = [x.nominal_value for x in df['GIIc']]
        
        gi_rel_unc = np.array(gi_uncertainties) / np.array(gi_values) * 100
        gii_rel_unc = np.array(gii_uncertainties) / np.array(gii_values) * 100
        
        print(f"Dataset {i+1}:")
        print(f"  GIc uncertainties: {np.mean(gi_rel_unc):.1f}% ± {np.std(gi_rel_unc):.1f}%")
        print(f"  GIIc uncertainties: {np.mean(gii_rel_unc):.1f}% ± {np.std(gii_rel_unc):.1f}%")
    
    # Run regression
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    print(f"\nReduced χ² with realistic uncertainties: {results['reduced_chi_squared']:.6f}")
    print("(Should be close to 1.0 for well-calibrated uncertainties)")
    
    return results

def test_mixed_uncertainties():
    """Test with mixed uncertainty types."""
    print("\n" + "=" * 60)
    print("TEST 8: MIXED UNCERTAINTY TYPES")
    print("=" * 60)
    
    # Create datasets with different uncertainty types
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.02, uncertainty_level=0.1, uncertainty_type='proportional')
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.02, uncertainty_level=0.05, uncertainty_type='fixed')
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.02, uncertainty_level=0.08, uncertainty_type='mixed')
    
    df_list = [df1, df2, df3]
    
    # Print uncertainty information
    print("Uncertainty Types:")
    print("Dataset 1: Proportional uncertainties (10% of value)")
    print("Dataset 2: Fixed uncertainties (±0.05 J/m²)")
    print("Dataset 3: Mixed uncertainties (proportional + fixed component)")
    
    # Run regression
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    return results

def test_high_uncertainties():
    """Test with high uncertainties to see impact on fit quality."""
    print("\n" + "=" * 60)
    print("TEST 9: HIGH UNCERTAINTIES")
    print("=" * 60)
    
    # Create datasets with high uncertainties
    df1 = create_dummy_dataset(n_points=20, GIc_true=0.5, GIIc_true=0.3, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.25, uncertainty_type='proportional')
    df2 = create_dummy_dataset(n_points=20, GIc_true=0.6, GIIc_true=0.4, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.25, uncertainty_type='proportional')
    df3 = create_dummy_dataset(n_points=20, GIc_true=0.7, GIIc_true=0.5, n_true=2.0, m_true=2.0, 
                              noise_level=0.05, uncertainty_level=0.25, uncertainty_type='proportional')
    
    df_list = [df1, df2, df3]
    
    print("High uncertainties (25% of values) with 5% noise")
    
    # Run regression
    results = reg.odr_multi_dataset_fixed_exponents(
        df_list=df_list,
        n_fixed=2.0,
        m_fixed=2.0,
        var='B',
        print_results=True
    )
    
    print(f"\nReduced χ² with high uncertainties: {results['reduced_chi_squared']:.6f}")
    print("(Should be close to 1.0 if uncertainties are well-calibrated)")
    
    return results

def main():
    """Run all tests."""
    print("MULTI-DATASET FIXED EXPONENTS REGRESSION TEST SUITE")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run all tests
    test1_results = test_perfect_fit()
    test2_results = test_wrong_exponents()
    test3_results = test_noisy_data()
    test4_results = test_variant_a()
    test5_results = test_different_exponents()
    test6_results = test_bounds()
    test7_results = test_realistic_uncertainties()
    test8_results = test_mixed_uncertainties()
    test9_results = test_high_uncertainties()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Perfect fit): Reduced χ² = {test1_results['reduced_chi_squared']:.6f}")
    print(f"Test 2 (Wrong exponents): Reduced χ² = {test2_results['reduced_chi_squared']:.6f}")
    print(f"Test 3 (Noisy data): Reduced χ² = {test3_results['reduced_chi_squared']:.6f}")
    print(f"Test 4 (Variant A): Reduced χ² = {test4_results['reduced_chi_squared']:.6f}")
    print(f"Test 5 (Different exponents): Reduced χ² = {test5_results['reduced_chi_squared']:.6f}")
    print(f"Test 6 (Bounds): Reduced χ² = {test6_results['reduced_chi_squared']:.6f}")
    print(f"Test 7 (Realistic uncertainties): Reduced χ² = {test7_results['reduced_chi_squared']:.6f}")
    print(f"Test 8 (Mixed uncertainties): Reduced χ² = {test8_results['reduced_chi_squared']:.6f}")
    print(f"Test 9 (High uncertainties): Reduced χ² = {test9_results['reduced_chi_squared']:.6f}")
    
    print("\nExpected behavior:")
    print("- Test 1: Reduced χ² ≈ 0.0 (perfect fit)")
    print("- Test 2: Reduced χ² >> 1.0 (poor fit due to wrong exponents)")
    print("- Test 3: Reduced χ² ≈ 1.0 (good fit with noise)")
    print("- Test 4-6: Reduced χ² should be reasonable")
    print("- Test 7-9: Reduced χ² ≈ 1.0 (well-calibrated uncertainties)")

if __name__ == "__main__":
    main() 