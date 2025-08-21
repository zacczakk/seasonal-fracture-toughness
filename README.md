# Seasonal Fracture Toughness

Analysis package for studying seasonal variations in fracture toughness of materials. This project provides tools for data processing, regression analysis, and visualization of fracture toughness measurements.

## Features

- **Data Processing**: Load and prepare fracture toughness datasets with filtering capabilities
- **Regression Analysis**: Orthogonal Distance Regression (ODR) for robust curve fitting
- **Visualization**: Comprehensive plotting tools for fracture toughness analysis
- **Parallel Processing**: Efficient parallel computation for large datasets

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd seasonal-fracture-toughness

# Install in development mode
pip install -e .
```

## Usage

The main analysis is performed through Jupyter notebooks:

```python
from sft.data import load_df
from sft.prepare import build_fracture_toughness_df, filter_by_gc_threshold
from sft.regression import parallel_odr

# Load and prepare data
df = build_fracture_toughness_df(
    file="data/processed/df_with_fracture_toughness_final_incl_bendingstiffness_final3.pkl",
    date_ranges={
        "1": (None, "2023-02-16"),
        "2": ("2023-02-27", "2023-03-03"),
        "3": ("2023-03-06", "2023-03-08"),
    }
)

# Run regression analysis
fit_results = parallel_odr(df, source="video", max_workers=12)
```

## Project Structure

- `src/sft/`: Main package with analysis modules
- `notebooks/`: Jupyter notebooks for analysis workflows
- `data/`: Processed datasets
- `src/legacy/`: Legacy analysis scripts

## Requirements

- Python ≥ 3.9
- See `pyproject.toml` for dependencies
