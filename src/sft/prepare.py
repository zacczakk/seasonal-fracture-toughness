from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from uncertainties import ufloat_fromstr
from uncertainties import unumpy as unp

from .data import load_df

DateRange = Tuple[Optional[str], Optional[str]]


def _mask_range(s: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    m = s.notna()
    if start is not None:
        m &= s >= pd.Timestamp(start)
    if end is not None:
        m &= s <= pd.Timestamp(end)
    return m


def build_fracture_toughness_df(
    file: str | Path,
    *,
    date_col: str = "datetime_x",
    date_ranges: Optional[Dict[str, DateRange]] = None,
    data_cols: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with GIc and GIIc columns, indexed by (source, series).

    Parameters
    ----------
    file : str | Path
        Filename (or path) to load. Must contain the raw GIc/GIIc columns and a
        datetime column.
    date_col : str
        Name of the datetime column used to assign rows to series.
    date_ranges : dict
        Mapping series_name -> (start, end). None uses the default 3 ranges.
    data_cols : dict
        Mapping source -> (GIc_col, GIIc_col). None uses default manual/video mapping.

    Returns
    -------
    pd.DataFrame
        Frame indexed by (source, series) with two columns: GIc, GIIc.
    """

    if date_ranges is None:
        date_ranges = {
            "1": (None, "2023-02-16"),
            "2": ("2023-02-27", "2023-03-03"),
            "3": ("2023-03-06", "2023-03-08"),
        }

    if data_cols is None:
        data_cols = {
            "manual": (
                "GIc_manual_default_min_error",
                "GIIc_manual_default_min_error",
            ),
            "video": (
                "GIc_video_seriesopt_min_error",
                "GIIc_video_seriesopt_min_error",
            ),
        }

    # Load input dataframe from file
    df = load_df(file)

    date = pd.to_datetime(df[date_col], errors="coerce")

    # Label each row with a series name
    series = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (start, end) in date_ranges.items():
        series = series.mask(_mask_range(date, start, end), name)

    frames = []
    for source, (gic_col, giic_col) in data_cols.items():
        tmp = df[[gic_col, giic_col]].rename(
            {gic_col: "GIc", giic_col: "GIIc"},
            axis="columns",
        )
        # Parse to ufloat directly from strings like "v±e" or "v+/-e"
        tmp["GIc"] = tmp["GIc"].apply(ufloat_fromstr)  # type: ignore[attr-defined]
        tmp["GIIc"] = tmp["GIIc"].apply(ufloat_fromstr)  # type: ignore[attr-defined]
        tmp["source"] = source
        tmp["series"] = series
        frames.append(tmp)

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["series"])
        .set_index(["source", "series"])
        .sort_index()
    )


def filter_by_gc_threshold(
    df: pd.DataFrame, *, source: str, gc_max: float
) -> pd.DataFrame:
    """
    Filter rows for a given `source` (e.g., "manual" or "video") so that
    the nominal values of both GIc and GIIc are <= gc_max. Other sources are
    left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (source, series) dataframe with columns GIc, GIIc as ufloats.
    source : str
        Which source level to filter (e.g., "manual" or "video").
    gc_max : float
        Threshold applied to nominal values of GIc and GIIc.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if not isinstance(df.index, pd.MultiIndex) or "source" not in df.index.names:
        raise ValueError("df must be indexed by ('source', 'series')")

    src_idx = df.index.get_level_values("source")
    sel = src_idx == source

    # Start with keep-all, then filter only selected source
    keep = pd.Series(True, index=df.index)
    if sel.any():
        sub = df.loc[sel]
        gic_vals = unp.nominal_values(sub["GIc"].to_numpy())
        giic_vals = unp.nominal_values(sub["GIIc"].to_numpy())
        keep_sel = (gic_vals <= gc_max) & (giic_vals <= gc_max)
        keep.loc[sel] = keep_sel

    return df.loc[keep]
