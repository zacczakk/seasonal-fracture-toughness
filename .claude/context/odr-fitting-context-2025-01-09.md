# ODR Fitting Project Context
*Saved: 2025-01-09*

## Project Overview
- **Primary Goal**: Implement ODR (Orthogonal Distance Regression) fitting for fracture toughness data
- **Model**: `((Gi / GIc) ** (1 / n) + (Gii / GIIc) ** (1 / m)) - 1 = 0`
- **Package Structure**: `src/sft/` with modular design
- **Technology Stack**: Python, scipy.odr, pandas, uncertainties, numpy

## Current Implementation Status

### Completed Components
- ✅ `src/sft/data.py` - Data loading/saving utilities
- ✅ `src/sft/prepare.py` - DataFrame preparation with MultiIndex (source, series)
- ✅ `pyproject.toml` - Project configuration
- ✅ Research and planning phase completed

### In Progress
- 🔄 `src/sft/odr_fit.py` - Main ODR fitting module (Task 1 in progress)

### Pending Tasks (Todo List)
1. ✅ Create ODR fitting module (in progress)
2. ⏳ Implement residual function for fracture toughness model
3. ⏳ Create data preparation functions for ODR format conversion
4. ⏳ Implement grid search functionality for n, m parameters
5. ⏳ Create comprehensive statistics calculation functions
6. ⏳ Implement parameter validation and bounds checking
7. ⏳ Create result class for clean output organization
8. ⏳ Add testing and validation functions

## Technical Architecture

### Data Structure
- **Index**: MultiIndex with levels (source, series)
  - Sources: "manual", "video"  
  - Series: "1", "2", "3" (based on date ranges)
- **Columns**: GIc, GIIc as `uncertainties.ufloat` objects
- **Parameter Vector**: β = [GIc_1, GIIc_1, GIc_2, GIIc_2, GIc_3, GIIc_3, n, m]

### Key Design Decisions
- **Implicit ODR fitting**: Model equation = 0 approach
- **Uncertainty handling**: Use `scipy.odr.RealData` for proper error propagation
- **Modular design**: Separate data prep, fitting, statistics, and validation
- **Parameter organization**: Series-specific GIc/GIIc, shared n/m parameters

### API Design
```python
def fit_fracture_toughness_model(
    df: pd.DataFrame,
    n: Optional[float] = None,
    m: Optional[float] = None, 
    grid_search: bool = False,
    bounds: Optional[Dict] = None
) -> FractureToughnessResult
```

## Code Patterns & Conventions
- Type hints with `from __future__ import annotations`
- `pathlib.Path` for file operations
- `ufloat_fromstr()` for parsing uncertainty strings
- `unp.nominal_values()` and `unp.std_devs()` for array extraction
- MultiIndex DataFrame validation patterns

## Legacy Reference
- **Legacy implementation**: `legacy/multi_dataset_fixed_exponents.py`
- **Test suite**: `legacy/test_multi_dataset_fixed_exponents.py`
- **Key insights**: Parameter bounds, uncertainty types, convergence testing

## Next Steps
1. Continue implementation of `src/sft/odr_fit.py`
2. Implement residual function with proper parameter extraction
3. Add data preparation utilities for scipy.odr format
4. Create comprehensive test suite
5. Validate against legacy implementation results

## Agent Coordination Notes
- Used sequential thinking for implementation design
- Extensive scipy.odr documentation research completed
- TodoWrite tool for task management
- Plan approved via ExitPlanMode
- Context saved for future session continuity