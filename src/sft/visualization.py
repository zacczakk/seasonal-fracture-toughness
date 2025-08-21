"""Module for the visualization of experimental data and fits.

Modern plotting helpers for the refactored ODR engine are added at the end of
this module. They support multi-series scenarios and side-by-side comparison of
"manual" vs "video" sources.
"""

# Third-party imports
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from uncertainties import unumpy

# Project imports
from . import regression as reg


def _extract_points(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def params(fit, curve, ci=0.95):
    """
    Compute input parameters for best fit and confidence intervals.

    A confidence band is used in statistical analysis to represent the
    uncertainty in an estimate of a curve or function based on limited
    or noisy data. Similarly, a **prediction band** is used to represent
    the uncertainty about the value of a new data-point on the curve, but
    subject to noise. Confidence and prediction bands are often used as
    part of the graphical presentation of results of a regression analysis.
    (https://en.wikipedia.org/wiki/Confidence_and_prediction_bands)

    Parameters
    ----------
    fit : dict
        Dictrionary with optimization results.
    curve : {'mean', 'lower', 'upper'}
        Specification of curve paramters to compute.
    ci : float, optional
        Confidence interval. Default is 0.95.

    Returns
    -------
    list
        Parameters (GIc, GIIc, n, m) for specified curve (mean, lower, upper).
    """
    # Convert confidence interval to percentile
    # point of the normal distribution
    pp = (1 + ci) / 2

    # Convert percentile point to number of standard deviations
    nstd = norm.ppf(pp)

    # Define multiplier
    multiplier = {"mean": 0, "lower": -nstd, "upper": +nstd}

    # Compute parameters of best fit, lower, or upper bounds
    return fit["params"] + fit["stddev"] * multiplier[curve]


def params_fixed_exponents(fit, curve, ci=0.95):
    """
    Compute input parameters for best fit and confidence intervals for fixed exponents.

    Parameters
    ----------
    fit : dict
        Dictionary with optimization results (fixed exponents).
    curve : {'mean', 'lower', 'upper'}
        Specification of curve parameters to compute.
    ci : float, optional
        Confidence interval. Default is 0.95.

    Returns
    -------
    list
        Parameters (GIc, GIIc) for specified curve (mean, lower, upper).
    """
    # Convert confidence interval to percentile point of the normal distribution
    pp = (1 + ci) / 2

    # Convert percentile point to number of standard deviations
    nstd = norm.ppf(pp)

    # Define multiplier
    multiplier = {"mean": 0, "lower": -nstd, "upper": +nstd}

    # Check if this is a constrained fit (from odr_with_fixed_exponents_constrained)
    if "optimization_success" in fit:
        # Constrained case: only GIc is free, GIIc is constrained
        GIc = fit["params"][0] + fit["stddev"][0] * multiplier[curve]
        GIIc = (0.79 / 0.56) * GIc
        return [GIc, GIIc]
    else:
        # Unconstrained case: both GIc and GIIc are free
        return fit["params"] + fit["stddev"] * multiplier[curve]


def curves(fit, ci=0.95, xymax=2.0, moderatio=False):
    """
    Add vertice lists of mean and confidence interval curves to results dict.

    Parameters
    ----------
    fit : dict
        Dictionary with optimization results.
    ci : float, optional
        Confidence interval. Default is 0.95.
    xymax : float, optional
        Maximum x and y values for contour plot. Default is 2.0.
    silent : bool, optional
        If True, suppress plot output. Default is False.
    moderatio : bool, optional
        If True, use mode ratio and total enery release rate as residual input.

    Returns
    -------
    fit : dict
        Updated results dictionary.
    """

    def moderatio_residual(beta, x, var="B", bounds=False):
        """
        Evaluate residual with mode ratio and total ERR as input.

        Parameters
        ----------
        beta : list[float]
            Model parameters (GIc, GIIc, n, m).
        x : list[float]
            Variables (Gi, Gii).
        var : {'A', 'B', 'BK', 'VA'}, optional
            Variant of interaction law. Default is 'B'.
        bound : bool, optional
            If True, enforce bounds on parameters. Default is False.

        Returns
        -------
        np.ndarray
            Residual at all points (Gi, Gii).
        """
        # Unpack inputs
        psi, G = x
        # Calculate Gi and Gii
        Gii = psi * G
        Gi = G - Gii
        # Evaluate residual
        return reg.residual(beta, [Gi, Gii], var=var, bounds=bounds)

    # Assemble high-resolution gridpoints
    inp = np.linspace(0, xymax)
    X, Y = np.meshgrid(inp, inp)

    # Initialize curves dictionary
    fit["curves"] = {"mean": [], "lower": [], "upper": []}

    # Plot curves to extract their vertices
    for curve in fit["curves"].keys():
        # Calculate residual on grid points
        if moderatio:
            Z = moderatio_residual(
                params(fit, curve, ci), [X, Y], var=fit["var"], bounds=False
            )
        else:
            Z = reg.residual(
                params(fit, curve, ci), [X, Y], var=fit["var"], bounds=False
            )
        # Plot (zero-width) contour where residual is zero
        contour = plt.contour(X, Y, Z, 0, linewidths=0)
        # Get vertices of contour line
        fit["curves"][curve] = contour.collections[1].get_paths()[0].vertices

    # Supress plot output
    plt.close()

    return fit


def axis_setup(subplots=False):
    """
    Setup figure and axes.

    Parameters
    ----------
    subplots : bool, optional
        If True, setup for subplots. Default is False.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Configured figure and axes objects.
    """
    # Font setup
    plt.rc("font", family="serif", size=11)
    plt.rc("mathtext", fontset="cm")

    # Init figure and axes
    if subplots:
        fig = plt.figure(figsize=[14, 14])
        ax = fig.add_subplot(2, 2, 3)
    else:
        fig = plt.figure(figsize=[8, 8])
        ax = plt.gca()

    # Match figure and VS code background colors. Theme colors can be found at
    # cmd+shift+P > Developer: Generate Color Theme From Current Settings
    # For background colors see "colors": {"editor.background": ...}
    # fig.set_facecolor('#282c34')
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    return fig, ax


def plot_setup(fit, ci=0.95, Gmax=1.4):
    """
    Setup plot of experimental data and fit.

    Parameters
    ----------
    fit : dict
        Dictionary with optimization results.
    ci : float, optional
        Confidence interval. Default is 0.95.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Configured figure and axes objects.
    """

    # Dict for objective function latex expressions
    eq = {"A": "", "B": "", "BK": "", "VA": ""}

    # Variant A
    eq["A"] += r"$\displaystyle\left[\left(\frac{\mathcal{G}_\mathrm{I}}"
    eq["A"] += r"{\mathcal{G}_\mathrm{Ic}}\right)^{1/n} + "
    eq["A"] += r"\left(\frac{\mathcal{G}_\mathrm{II}}{\mathcal{G}_\mathrm{IIc}}"
    eq["A"] += r"\right)^{1/m}\right]^{\frac{2}{1/n+1/m}}\!\!\! = 1$"

    # Variant B
    eq["B"] += r"$\displaystyle\left(\frac{\mathcal{G}_\mathrm{I}}"
    eq["B"] += r"{\mathcal{G}_\mathrm{Ic}}\right)^{1/n}+"
    eq["B"] += r"\left(\frac{\mathcal{G}_\mathrm{II}}"
    eq["B"] += r"{\mathcal{G}_\mathrm{IIc}}\right)^{1/m}\!\!\!=1$"

    # Variant BK
    eq["BK"] += r"$\displaystyle \mathcal{G} = \mathcal{G}_\mathrm{Ic} + "
    eq["BK"] += r"\left(\mathcal{G}_\mathrm{IIc} - \mathcal{G}_\mathrm{Ic} "
    eq["BK"] += r"\right)\left(\frac{\mathcal{G}_\mathrm{II}}"
    eq["BK"] += r"{\mathcal{G}_\mathrm{I} + \mathcal{G}_\mathrm{II}}\right)^n$"

    # Define figure and axes
    fig, ax = axis_setup()

    # Axis limits
    plt.axis([0, Gmax, 0, Gmax])
    ax.set_aspect("equal")

    # Axes labels
    plt.xlabel(
        r"$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$", fontsize=12
    )
    plt.ylabel(
        r"$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$", fontsize=12
    )

    # Plot title
    plt.title(f"Best fit with {int(ci * 100)}% confidence interval", size=12)

    # Fracture toughnesses
    g1 = r"$\mathcal{G}_\mathrm{Ic}\; = %1.2f \pm %1.2f\ \mathrm{J/m}^2$" % (
        fit["params"][0],
        fit["stddev"][0],
    )
    g2 = r"$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f\ \mathrm{J/m}^2$" % (
        fit["params"][1],
        fit["stddev"][1],
    )

    # Envelope parameters
    n = r"$n\ = %3.1f$" % fit["params"][2]
    m = r"$m = %3.1f$" % fit["params"][3]

    # Goodness of fit
    chi2 = r"$\chi_\nu^2 = %.3f$" % fit["reduced_chi_squared"]
    pval = r"$p = %.3f$" % fit["p_value"]

    # Write annotations
    plt.text(
        1.05,
        1.01,
        eq[fit["var"]],
        size=11,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        usetex=True,
    )
    plt.text(
        1.05,
        0.72,
        g1 + "\n" + g2,
        size=11,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        usetex=False,
    )
    plt.text(
        1.05,
        0.68,
        n + "\n" + m,
        size=11,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        usetex=False,
    )
    plt.text(
        1.05,
        0.55,
        chi2 + "\n" + pval,
        size=11,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        usetex=False,
    )

    return fig, ax


def plot_interactionlaw(
    df,
    fit=None,
    style="seaborn-v0_8-white",
    save_fig="test",
    title="test",
    data_color="green",
    fit_color="green",
    ci=0.95,
    Gmax=1.4,
    label="label",
    annotate=False,
):
    """
    Plot experimental data and best fit with confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Dara frame with experimental data.
    fit : dict
        Dictionary with best fit parameters and confidence intervals.
    style : {'default', 'dark_background'}, optional
        Plot style. Default is 'dark_background'.
    ci : float, optional
        Confidence intervarls. Default is 0.95.
    annotate : bool, optional
        If true, annotate data points with index. Default is False.
    """

    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        if fit:
            # Get plot data
            fit = curves(fit)
            xm, ym = fit["curves"]["mean"].T
            xu, yu = fit["curves"]["upper"].T
            xl, yl = fit["curves"]["lower"].T

            # Prepare plot
            _, ax = plot_setup(fit, ci, Gmax)

            # Get lists of confidence-interval outline coordinates
            xci = np.append(xl, xu[::-1])
            yci = np.append(yl, yu[::-1])

            # Plot best fit and confidence interval
            ax.plot(xm, ym, color=fit_color, linewidth=2)
            ax.fill(xci, yci, color=fit_color, alpha=0.15)

        else:
            # Define figure and axes
            _, ax = axis_setup()
            plt.axis([0, Gmax, 0, Gmax])
            # Axes labels
            plt.xlabel(
                r"$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ " + r"\longrightarrow$"
            )
            plt.ylabel(
                r"$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ " + r"\longrightarrow$"
            )

        # Plot fracture toughnesses with 1-sigma error bars
        ax.errorbar(
            x=df["GIc"].apply(unumpy.nominal_values),
            y=df["GIIc"].apply(unumpy.nominal_values),
            xerr=df["GIc"].apply(unumpy.std_devs),
            yerr=df["GIIc"].apply(unumpy.std_devs),
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color=data_color,
            alpha=0.7,
            label=label,
        )

        if annotate:
            # Data points and labels
            x = df["GIc"].apply(unumpy.nominal_values).values
            y = df["GIIc"].apply(unumpy.nominal_values).values
            idx = df.index.astype(str).values

            # Add index as label to each point
            for i, txt in enumerate(idx):
                ax.annotate(
                    text=txt,
                    xy=(x[i], y[i]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    color=data_color,
                    size=6,
                    alpha=0.8,
                )

    # Add styled legend
    ax.legend(
        loc="upper right",
        frameon=False,
        fancybox=False,
        shadow=False,
        fontsize=10,
        framealpha=0.8,
        facecolor="white",
        edgecolor="white",
        prop={"family": "serif", "size": 10},
    )
    plt.savefig(save_fig + title + ".jpg", dpi=600, bbox_inches="tight")


def plot_cutlengths(dfA, dfB, dfL, style="dark_background"):
    """
    Plot critical cut lengths with error bars.

    Parameters
    ----------
    dfA : pandas.DataFrame
        Dataframe with bunker 1 data.
    dfB : pandas.DataFrame
        Dataframe with bunker 2 data.
    style : str, optional
        Plot style context. Default is 'dark_background'.
    """

    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        # Setup axes
        fig, ax1 = axis_setup(subplots=True)

        # Axis limits
        ax1.axis([-70, 70, 0, 50])

        # Axes labels
        ax1.set_xlabel(r"Inclination $\varphi\, ({}^\circ)\ \longrightarrow$")
        ax1.set_ylabel(
            r"Critical cut length $a_\mathrm{c}\, (\mathrm{cm})\ \longrightarrow$"
        )

        # Plot title
        # plt.title(r'Critical cut lengths with $2\sigma$ error bars', size=9)

        # Unpack plot data for bunker 1
        x1nom = -unumpy.nominal_values(dfA.slope_incl)
        y1nom = unumpy.nominal_values(dfA.rc) / 10
        x1std = unumpy.std_devs(dfA.slope_incl)
        y1std = unumpy.std_devs(dfA.rc) / 10

        # Unpack plot data for bunker 2
        x2nom = -unumpy.nominal_values(dfB.slope_incl)
        y2nom = unumpy.nominal_values(dfB.rc) / 10
        x2std = unumpy.std_devs(dfB.slope_incl)
        y2std = unumpy.std_devs(dfB.rc) / 10

        # Unpack plot data for legacy dataset
        x3nom = -unumpy.nominal_values(dfL.slope_incl)
        y3nom = unumpy.nominal_values(dfL.rc) / 10
        x3std = unumpy.std_devs(dfL.slope_incl)
        y3std = unumpy.std_devs(dfL.rc) / 10

        # Plot bunker 1 cut lenghts with error bars
        ax1.errorbar(
            x=x1nom,
            y=y1nom,
            xerr=x1std,
            yerr=y1std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="teal",
            label="Bunker 1",
        )

        # Plot bunker 2 cut lenghts with error bars
        ax1.errorbar(
            x=x2nom,
            y=y2nom,
            xerr=x2std,
            yerr=y2std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="lightgrey",
            label="Bunker 2",
        )

        # Plot legecy cut lenghts with error bars
        ax1.errorbar(
            x=x3nom,
            y=y3nom,
            xerr=x3std,
            yerr=y3std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="orange",
            alpha=0.2,
            label="Legacy dataset",
        )

        # Show legends
        plt.legend(
            frameon=False,
            handletextpad=0,
            loc="upper left",
            fontsize=9,
            labelcolor="black",
        )

        # Add slope-angle histogram axis
        ax2 = fig.add_subplot(
            2, 2, 1, anchor="S", sharex=ax1, frameon=False, aspect=300
        )

        # Plot slope-angle histogram
        ax2.hist(
            x=[
                -dfL["slope_incl"].apply(unumpy.nominal_values),
                -dfA["slope_incl"].apply(unumpy.nominal_values),
                -dfB["slope_incl"].apply(unumpy.nominal_values),
            ],
            color=["orange", "teal", "lightgrey"],
            histtype="stepfilled",
            rwidth=0.8,
            density=True,
            bins=40,
            range=(-70, 70),
            alpha=0.7,
        )

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        # Add cut-length histogram axis
        ax3 = fig.add_subplot(
            2, 2, 4, aspect=0.005, anchor="W", frameon=False, sharey=ax1
        )

        # Plot cut-length histogram
        ax3.hist(
            x=[
                dfL["rc"].apply(unumpy.nominal_values) / 10,
                dfA["rc"].apply(unumpy.nominal_values) / 10,
                dfB["rc"].apply(unumpy.nominal_values) / 10,
            ],
            color=["orange", "teal", "lightgrey"],
            histtype="stepfilled",
            orientation="horizontal",
            density=True,
            bins=50,
            range=(0, 50),
            alpha=0.8,
        )

        # Hide axes
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.5)


def plot_modeIIfraction(dfA, dfB, dfL, style="dark_background"):
    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        # Setup axes
        fig, ax1 = axis_setup(subplots=True)

        # Axis limits
        ax1.axis([-70, 70, 0, 1.0])

        # Axes labels
        ax1.set_xlabel(r"Inclination $\varphi\, ({}^\circ)\ \longrightarrow$")
        ax1.set_ylabel(r"$\mathcal{G}_\mathrm{I\!I}/\mathcal{G}\ \longrightarrow$")

        # Plot title
        # ax1.title(
        #     'Mode II energy release rate\n' +
        #     'as fraction of total energy release rate',
        #     size=9)

        # Unpack plot data for bunker 1
        x1nom = -unumpy.nominal_values(dfA["slope_incl"])
        y1nom = unumpy.nominal_values(dfA["Gii/G"])
        x1std = unumpy.std_devs(dfA["slope_incl"])
        y1std = unumpy.std_devs(dfA["Gii/G"])

        # Unpack plot data for bunker 2
        x2nom = -unumpy.nominal_values(dfB["slope_incl"])
        y2nom = unumpy.nominal_values(dfB["Gii/G"])
        x2std = unumpy.std_devs(dfB["slope_incl"])
        y2std = unumpy.std_devs(dfB["Gii/G"])

        # Unpack plot data for legacy dataset
        x3nom = -unumpy.nominal_values(dfL["slope_incl"])
        y3nom = unumpy.nominal_values(dfL["Gii/G"])
        x3std = unumpy.std_devs(dfL["slope_incl"])
        y3std = unumpy.std_devs(dfL["Gii/G"])

        # Plot bunker 1 cut lenghts with error bars
        ax1.errorbar(
            x=x1nom,
            y=y1nom,
            xerr=x1std,
            yerr=y1std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="teal",
            label="Bunker 1",
        )

        # Plot bunker 2 cut lenghts with error bars
        ax1.errorbar(
            x=x2nom,
            y=y2nom,
            xerr=x2std,
            yerr=y2std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="lightgrey",
            label="Bunker 2",
        )

        # Plot legacy cut lenghts with error bars
        ax1.errorbar(
            x=x3nom,
            y=y3nom,
            xerr=x3std,
            yerr=y3std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="orange",
            alpha=0.2,
            label="Legacy dataset",
        )

        # Show legends
        plt.legend(
            frameon=False,
            handletextpad=0,
            loc="upper left",
            fontsize=9,
            labelcolor="black",
        )

        # Add slope-angle histogram axis
        ax2 = fig.add_subplot(
            2, 2, 1, anchor="S", sharex=ax1, frameon=False, aspect=300
        )

        # Plot slope-angle histogram
        ax2.hist(
            x=[
                -dfL["slope_incl"].apply(unumpy.nominal_values),
                -dfA["slope_incl"].apply(unumpy.nominal_values),
                -dfB["slope_incl"].apply(unumpy.nominal_values),
            ],
            color=["orange", "teal", "lightgrey"],
            histtype="stepfilled",
            density=True,
            bins=40,
            range=(-70, 70),
            alpha=0.7,
        )

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        # Add cut-length histogram axis
        ax3 = fig.add_subplot(2, 2, 4, aspect=80, anchor="W", frameon=False, sharey=ax1)

        # Plot cut-length histogram
        ax3.hist(
            x=[
                dfL["Gii/G"].apply(unumpy.nominal_values),
                dfA["Gii/G"].apply(unumpy.nominal_values),
                dfB["Gii/G"].apply(unumpy.nominal_values),
            ],
            color=["orange", "teal", "lightgrey"],
            histtype="stepfilled",
            orientation="horizontal",
            density=True,
            bins=50,
            range=(0, 1),
            alpha=0.7,
        )

        # Hide axes
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.5)


def plot_totalERR(dfA, dfB, dfL, fit=None, Gmax=2, style="dark_background"):
    """
    Plot total energy release rate vs. mode ratio.

    Parameters
    ----------
    dfA : pandas.DataFrame
        Dataframe with bunker 1 data.
    dfB : pandas.DataFrame
        Dataframe with bunker 2 data.
    dfL : pandas.DataFrame
        Dataframe with legacy data.
    fit : dict
        Dictionary with best fit parameters and confidence intervals.
    Gmax : float, optional
        Maximum energy release rate. Default is 2.
    style : str, optional
        Plot style context. Default is 'dark_background'.
    """

    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        # Get plot data for best fit
        fit = curves(fit, moderatio=True)
        xm, ym = fit["curves"]["mean"].T
        xu, yu = fit["curves"]["upper"].T
        xl, yl = fit["curves"]["lower"].T

        # Get lists of confidence-interval outline coordinates
        xci = np.append(xl, xu[::-1])
        yci = np.append(yl, yu[::-1])

        # Setup axes
        fig, ax1 = axis_setup(subplots=True)

        # Axis limits
        ax1.axis([0, 1, 0, Gmax])

        # Axes labels
        ax1.set_xlabel(r"$\mathcal{G}_\mathrm{I\!I}/\mathcal{G}\ \longrightarrow$")
        ax1.set_ylabel(r"$\mathcal{G}\ (\mathrm{J/m}^2)\ \longrightarrow$")

        # Plot best fit and confidence interval
        ax1.plot(xm, ym, color="orange", linewidth=2)
        ax1.fill(xci, yci, color="papayawhip")

        # Unpack plot data for bunker 1
        x1nom = unumpy.nominal_values(dfA["Gii/G"])
        y1nom = unumpy.nominal_values(dfA["Gc"])
        x1std = unumpy.std_devs(dfA["Gii/G"])
        y1std = unumpy.std_devs(dfA["Gc"])

        # Unpack plot data for bunker 2
        x2nom = unumpy.nominal_values(dfB["Gii/G"])
        y2nom = unumpy.nominal_values(dfB["Gc"])
        x2std = unumpy.std_devs(dfB["Gii/G"])
        y2std = unumpy.std_devs(dfB["Gc"])

        # Unpack plot data for legacy dataset
        x3nom = unumpy.nominal_values(dfL["Gii/G"])
        y3nom = unumpy.nominal_values(dfL["Gc"])
        x3std = unumpy.std_devs(dfL["Gii/G"])
        y3std = unumpy.std_devs(dfL["Gc"])

        # Plot legacy data with error bars
        ax1.errorbar(
            x=x3nom,
            y=y3nom,
            xerr=x3std,
            yerr=y3std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="pink",
            alpha=0.5,
            label="Legacy dataset",
        )

        # Plot bunker 1 data with error bars
        ax1.errorbar(
            x=x1nom,
            y=y1nom,
            xerr=x1std,
            yerr=y1std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="teal",
            label="Bunker 1",
        )

        # Plot bunker 2 data with error bars
        ax1.errorbar(
            x=x2nom,
            y=y2nom,
            xerr=x2std,
            yerr=y2std,
            linestyle="none",
            marker="o",
            markersize=3,
            elinewidth=0.5,
            color="grey",
            label="Bunker 2",
        )

        # Show legends
        # ax1.legend(frameon=False, handletextpad=0, loc='lower right',
        #           fontsize=9, labelcolor='black')

        # Add cut-length histogram axis
        ax2 = fig.add_subplot(2, 2, 4, aspect=8, anchor="W", frameon=False, sharey=ax1)

        # Plot cut-length histogram
        ax2.hist(
            x=[
                dfL["Gc"].apply(unumpy.nominal_values),
                dfA["Gc"].apply(unumpy.nominal_values),
                dfB["Gc"].apply(unumpy.nominal_values),
            ],
            color=["pink", "teal", "grey"],
            histtype="stepfilled",
            orientation="horizontal",
            density=True,
            bins=51,
            range=(0, Gmax),
            alpha=0.8,
        )

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0)


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
