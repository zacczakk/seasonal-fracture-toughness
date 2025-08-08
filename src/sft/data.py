from __future__ import annotations
import pandas as pd
from pathlib import Path

DEFAULT_PROCESSED_PATH = Path('data/processed/df_with_fracture_toughness_final_incl_bendingstiffness_final3.pkl')

def load_processed_df(path: str | Path = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    path = Path(path)
    return pd.read_pickle(path)


def save_processed_df(df: pd.DataFrame, path: str | Path = DEFAULT_PROCESSED_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
