"""Module for the visualization of experimental data and fits.

Modern plotting helpers for the refactored ODR engine.
"""

# Third-party imports
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from uncertainties import unumpy


def _extract_points(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gi = unumpy.nominal_values(df["GIc"].to_numpy())
    gii = unumpy.nominal_values(df["GIIc"].to_numpy())
    gi_err = unumpy.std_devs(df["GIc"].to_numpy())
    gii_err = unumpy.std_devs(df["GIIc"].to_numpy())
    return gi, gii, gi_err, gii_err


def _plot_series_points(ax, df_source: pd.DataFrame, colors: Dict[str, str]) -> None:
    # Support both (source, series) MultiIndex and single 'series' index or column
    if isinstance(df_source.index, pd.MultiIndex) and "series" in df_source.index.names:
        series_names = sorted(df_source.index.get_level_values("series").unique())
        for series in series_names:
            sub = df_source.xs(series, level="series")
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
            gi, gii, gi_err, gii_err = _extract_points(sub)
            ax.errorbar(
                gi,
                gii,
                xerr=gi_err,
                yerr=gii_err,
                linestyle="none",
                marker="o",
                markersize=3,
                elinewidth=0.5,
                color=colors.get(str(series), "C0"),
                alpha=0.8,
                label=f"series {series}",
            )
    else:
        # Either the index is the 'series' level or there's a 'series' column
        if df_source.index.name == "series":
            series_values = df_source.index.unique().tolist()
            for series in sorted(series_values):
                sub = df_source.loc[series]
                if isinstance(sub, pd.Series):
                    sub = sub.to_frame().T
                gi, gii, gi_err, gii_err = _extract_points(sub)
                ax.errorbar(
                    gi,
                    gii,
                    xerr=gi_err,
                    yerr=gii_err,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    elinewidth=0.5,
                    color=colors.get(str(series), "C0"),
                    alpha=0.8,
                    label=f"series {series}",
                )
        elif "series" in df_source.columns:
            series_values = sorted(pd.Series(df_source["series"]).unique())
            for series in series_values:
                sub = df_source[df_source["series"] == series]
                gi, gii, gi_err, gii_err = _extract_points(sub)
                ax.errorbar(
                    gi,
                    gii,
                    xerr=gi_err,
                    yerr=gii_err,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    elinewidth=0.5,
                    color=colors.get(str(series), "C0"),
                    alpha=0.8,
                    label=f"series {series}",
                )
        else:
            # No series information; plot all points together
            gi, gii, gi_err, gii_err = _extract_points(df_source)
            ax.errorbar(
                gi,
                gii,
                xerr=gi_err,
                yerr=gii_err,
                linestyle="none",
                marker="o",
                markersize=3,
                elinewidth=0.5,
                color="C0",
                alpha=0.8,
                label="data",
            )


def plot_side_by_side_series_odr(
    df: pd.DataFrame,
    *,
    fit_manual: dict | None = None,
    fit_video: dict | None = None,
    ci: float = 0.95,
    Gmax: float = 1.4,
    series_colors: dict[str, str] | None = None,
    style: str = "seaborn-v0_8-white",
):
    """
    Side-by-side plots (Manual vs Video) that accept outputs of sft.regression.odr (dict).

    - df must have MultiIndex ['source','series'] (or at least 'source') and columns GIc, GIIc (ufloats supported).
    - fit_* must be dicts returned by sft.regression.odr with keys 'params' and 'stddev' (length 8).
    - Confidence bands are drawn via ±z·σ on (GIc_i, GIIc_i, n, m).

    Returns
    -------
    (fig, (ax_manual, ax_video))
    """
    if series_colors is None:
        series_colors = {"1": "C0", "2": "C1", "3": "C2"}

    def _curve_vertices_legacy(GIc, GIIc, nval, mval, Gmax, grid_pts: int = 401):
        x = np.linspace(0.0, Gmax, grid_pts)
        X, Y = np.meshgrid(x, x)
        Z = np.power(X / GIc, 1.0 / nval) + np.power(Y / GIIc, 1.0 / mval) - 1.0
        fig_tmp, ax_tmp = plt.subplots()
        try:
            cs = ax_tmp.contour(X, Y, Z, levels=[0.0], linewidths=0)
            paths = cs.collections[0].get_paths()
            if not paths:
                raise RuntimeError("no contour")
            best = max(paths, key=lambda p: p.vertices.shape[0])
            verts = best.vertices
            return verts[:, 0], verts[:, 1]
        except Exception:
            Gi = np.linspace(0.0, 1.2 * GIc, 1000)
            a = np.power(Gi / GIc, 1.0 / nval)
            base = np.maximum(1.0 - a, 0.0)
            Gii = GIIc * np.power(base, mval)
            mask = np.isfinite(Gii) & (Gii >= 0)
            return Gi[mask], Gii[mask]
        finally:
            plt.close(fig_tmp)

    z = norm.ppf((1 + ci) / 2)

    plt.rcdefaults()
    with plt.style.context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
        for ax, source, title, res in (
            (axes[0], "manual", "Manual", fit_manual),
            (axes[1], "video", "Video", fit_video),
        ):
            ax.set_xlim(0, Gmax)
            ax.set_ylim(0, Gmax)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\mathcal{G}_\mathrm{I}\ (\mathrm{J/m}^2)$")
            if ax is axes[0]:
                ax.set_ylabel(r"$\mathcal{G}_\mathrm{II}\ (\mathrm{J/m}^2)$")
            ax.set_title(title)

            # Data points
            try:
                src_df = df.xs(source, level="source")
                _plot_series_points(ax, src_df, series_colors)  # uses existing helper
            except Exception:
                pass

            # Envelopes + CI from legacy ODR dict
            if isinstance(res, dict) and "params" in res and "stddev" in res:
                p = np.asarray(
                    res["params"], dtype=float
                )  # [GIc1,GIIc1,GIc2,GIIc2,GIc3,GIIc3,n,m]
                s = np.asarray(res["stddev"], dtype=float)
                # Try detect series labels from data to keep colors/labels aligned
                if (
                    isinstance(src_df.index, pd.MultiIndex)
                    and "series" in src_df.index.names
                ):
                    series_labels = sorted(
                        src_df.index.get_level_values("series").unique()
                    )
                else:
                    series_labels = ["1", "2", "3"]

                for idx, series in enumerate(series_labels[:3]):
                    GIc = p[2 * idx]
                    GIIc = p[2 * idx + 1]
                    nval = p[6]
                    mval = p[7]

                    GIc_e = float(s[2 * idx]) if s.size >= 8 else 0.0
                    GIIc_e = float(s[2 * idx + 1]) if s.size >= 8 else 0.0
                    n_e = float(s[6]) if s.size >= 8 else 0.0
                    m_e = float(s[7]) if s.size >= 8 else 0.0

                    color = series_colors.get(str(series), f"C{idx}")

                    gi_mean, gii_mean = _curve_vertices_legacy(
                        GIc, GIIc, nval, mval, Gmax
                    )
                    ax.plot(
                        gi_mean,
                        gii_mean,
                        color=color,
                        linewidth=2,
                        label=f"fit s{series}",
                    )

                    # Simple CI via ±z·σ bounds
                    GIc_lo, GIc_hi = (
                        GIc - z * GIc_e,
                        GIc + z * GIc_e,
                    )
                    GIIc_lo, GIIc_hi = (
                        GIIc - z * GIIc_e,
                        GIIc + z * GIIc_e,
                    )
                    n_lo, n_hi = nval - z * n_e, nval + z * n_e
                    m_lo, m_hi = mval - z * m_e, mval + z * m_e

                    gi_u, gii_u = _curve_vertices_legacy(
                        GIc_hi, GIIc_hi, n_lo, m_lo, Gmax
                    )
                    gi_l, gii_l = _curve_vertices_legacy(
                        GIc_lo, GIIc_lo, n_hi, m_hi, Gmax
                    )

                    if gi_u.size and gi_l.size and gi_mean.size:
                        common_gi = np.linspace(
                            0.0, min(gi_u[-1], gi_l[-1], gi_mean[-1]), 400
                        )
                        gii_u_i = np.interp(
                            common_gi, gi_u, gii_u, left=np.nan, right=np.nan
                        )
                        gii_l_i = np.interp(
                            common_gi, gi_l, gii_l, left=np.nan, right=np.nan
                        )
                        valid = np.isfinite(gii_u_i) & np.isfinite(gii_l_i)
                        if np.any(valid):
                            ax.fill_between(
                                common_gi[valid],
                                gii_l_i[valid],
                                gii_u_i[valid],
                                color=color,
                                alpha=0.15,
                                linewidth=0,
                            )

            ax.legend(frameon=False)

        # fig.tight_layout()
        return fig, (axes[0], axes[1])
