from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"


def load_df(
    filename: str | Path,
    directory: str | Path = DEFAULT_PROCESSED_DIR,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a DataFrame from pickle (.pkl/.pickle/.pkl.gz) or Excel (.xlsx/.xls).

    Extra kwargs are passed to the underlying pandas loader.
    """
    p = Path(directory) / Path(filename)
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith((".pkl", ".pickle", ".pkl.gz")):
        return cast(pd.DataFrame, pd.read_pickle(p, **kwargs))
    if suffix.endswith((".xlsx", ".xls")):
        return cast(pd.DataFrame, pd.read_excel(p, **kwargs))
    raise ValueError(f"Unsupported file extension for {p}")


def save_df(
    df: pd.DataFrame,
    filename: str | Path,
    directory: str | Path = DEFAULT_PROCESSED_DIR,
    *,
    index: bool = False,
    **kwargs: Any,
) -> None:
    """Save a DataFrame to pickle (.pkl/.pickle/.pkl.gz) or Excel (.xlsx/.xls).

    Extra kwargs are passed to the underlying pandas saver.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / Path(filename)
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith((".pkl", ".pickle", ".pkl.gz")):
        df.to_pickle(p, **kwargs)
        return
    if suffix.endswith((".xlsx", ".xls")):
        df.to_excel(p, index=index, **kwargs)
        return
    raise ValueError(f"Unsupported file extension for {p}")
