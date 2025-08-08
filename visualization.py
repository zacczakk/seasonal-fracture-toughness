"""Module for the visualization of experimental data and fits."""

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

from uncertainties import unumpy
from scipy.stats import norm
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator


import matplotlib.colors as mcolors

# Project imports
import regression as reg


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
    pp = (1 + ci)/2

    # Convert percentile point to number of standard deviations
    nstd = norm.ppf(pp)

    # Define multiplier
    multiplier = {
        'mean': 0,
        'lower': -nstd,
        'upper': +nstd
    }

    # Compute parameters of best fit, lower, or upper bounds
    return fit['params'] + fit['stddev']*multiplier[curve]

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
    pp = (1 + ci)/2
    
    # Convert percentile point to number of standard deviations
    nstd = norm.ppf(pp)
    
    # Define multiplier
    multiplier = {
        'mean': 0,
        'lower': -nstd,
        'upper': +nstd
    }
    
    # Check if this is a constrained fit (from odr_with_fixed_exponents_constrained)
    if 'optimization_success' in fit:
        # Constrained case: only GIc is free, GIIc is constrained
        GIc = fit['params'][0] + fit['stddev'][0] * multiplier[curve]
        GIIc = (0.79/0.56) * GIc
        return [GIc, GIIc]
    else:
        # Unconstrained case: both GIc and GIIc are free
        return fit['params'] + fit['stddev'] * multiplier[curve]

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

    def moderatio_residual(beta, x, var='B', bounds=False):
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
        Gii = psi*G
        Gi = G - Gii
        # Evaluate residual
        return reg.residual(beta, [Gi, Gii], var=var, bounds=bounds)

    # Assemble high-resolution gridpoints
    inp = np.linspace(0, xymax)
    X, Y = np.meshgrid(inp, inp)

    # Initialize curves dictionary
    fit['curves'] = {'mean': [], 'lower': [], 'upper': []}

    # Plot curves to extract their vertices
    for curve in fit['curves'].keys():
        # Calculate residual on grid points
        if moderatio:
            Z = moderatio_residual(params(fit, curve, ci), [X, Y],
                                   var=fit['var'], bounds=False)
        else:
            Z = reg.residual(params(fit, curve, ci), [X, Y],
                             var=fit['var'], bounds=False)
        # Plot (zero-width) contour where residual is zero
        contour = plt.contour(X, Y, Z, 0, linewidths=0)
        # Get vertices of contour line
        fit['curves'][curve] = contour.collections[1].get_paths()[0].vertices

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
    plt.rc('font', family='serif', size=11)
    plt.rc('mathtext', fontset='cm')

    # Init figure and axes
    if subplots:
        fig = plt.figure(figsize=[14,14])
        ax = fig.add_subplot(2, 2, 3)
    else:
        fig = plt.figure(figsize=[8, 8])
        ax = plt.gca()

    # Match figure and VS code background colors. Theme colors can be found at
    # cmd+shift+P > Developer: Generate Color Theme From Current Settings
    # For background colors see "colors": {"editor.background": ...}
    #fig.set_facecolor('#282c34')
    fig.set_facecolor('white')
    ax.set_facecolor('white')

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
    eq = {'A': '', 'B': '', 'BK': '', 'VA': ''}

    # Variant A
    eq['A'] += r'$\displaystyle\left[\left(\frac{\mathcal{G}_\mathrm{I}}'
    eq['A'] += r'{\mathcal{G}_\mathrm{Ic}}\right)^{1/n} + '
    eq['A'] += r'\left(\frac{\mathcal{G}_\mathrm{II}}{\mathcal{G}_\mathrm{IIc}}'
    eq['A'] += r'\right)^{1/m}\right]^{\frac{2}{1/n+1/m}}\!\!\! = 1$'

    # Variant B
    eq['B'] += r'$\displaystyle\left(\frac{\mathcal{G}_\mathrm{I}}'
    eq['B'] += r'{\mathcal{G}_\mathrm{Ic}}\right)^{1/n}+'
    eq['B'] += r'\left(\frac{\mathcal{G}_\mathrm{II}}'
    eq['B'] += r'{\mathcal{G}_\mathrm{IIc}}\right)^{1/m}\!\!\!=1$'

    # Variant BK
    eq['BK'] += r'$\displaystyle \mathcal{G} = \mathcal{G}_\mathrm{Ic} + '
    eq['BK'] += r'\left(\mathcal{G}_\mathrm{IIc} - \mathcal{G}_\mathrm{Ic} '
    eq['BK'] += r'\right)\left(\frac{\mathcal{G}_\mathrm{II}}'
    eq['BK'] += r'{\mathcal{G}_\mathrm{I} + \mathcal{G}_\mathrm{II}}\right)^n$'

    # Define figure and axes
    fig, ax = axis_setup()

    # Axis limits
    plt.axis([0, Gmax, 0, Gmax])
    ax.set_aspect('equal')

    # Axes labels
    plt.xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$',fontsize=12)
    plt.ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$',fontsize=12)

    # Plot title
    plt.title(f'Best fit with {int(ci*100)}% confidence interval', size=12)

    # Fracture toughnesses
    g1 = r'$\mathcal{G}_\mathrm{Ic}\; = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (
            fit['params'][0], fit['stddev'][0])
    g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (
            fit['params'][1], fit['stddev'][1])

    # Envelope parameters
    n = r'$n\ = %3.1f$' % fit['params'][2]
    m = r'$m = %3.1f$' % fit['params'][3]

    # Goodness of fit
    chi2 = r'$\chi_\nu^2 = %.3f$' % fit['reduced_chi_squared']
    pval = r'$p = %.3f$' % fit['p_value']

    # Write annotations
    plt.text(
        1.05, 1.01, eq[fit['var']], size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', usetex=True)
    plt.text(
        1.05, .72, g1 + '\n' + g2,  size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='bottom', usetex=False)
    plt.text(
        1.05, .68, n + '\n' + m, size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', usetex=False)
    plt.text(
        1.05, .55, chi2 + '\n' + pval, size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', usetex=False)

    return fig, ax

def plot_interactionlaw(
        df, fit=None, style='seaborn-v0_8-white', save_fig = "test", title= "test", data_color="green", fit_color="green", ci=0.95,
        Gmax=1.4,label="label", annotate=False):
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
            xm, ym = fit['curves']['mean'].T
            xu, yu = fit['curves']['upper'].T
            xl, yl = fit['curves']['lower'].T

            # Prepare plot
            _, ax = plot_setup(fit, ci, Gmax)

            # Get lists of confidence-interval outline coordinates
            xci = np.append(xl, xu[::-1])
            yci = np.append(yl, yu[::-1])

            # Plot best fit and confidence interval
            ax.plot(xm, ym, color=fit_color, linewidth=2)
            ax.fill(xci, yci, color=fit_color,alpha=0.15)

        else:

            # Define figure and axes
            _, ax = axis_setup()
            plt.axis([0, Gmax, 0, Gmax])
            # Axes labels
            plt.xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ ' +
                       r'\longrightarrow$')
            plt.ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ ' +
                       r'\longrightarrow$')

        # Plot fracture toughnesses with 1-sigma error bars
        ax.errorbar(
            x=df['GIc'].apply(unumpy.nominal_values),
            y=df['GIIc'].apply(unumpy.nominal_values),
            xerr=df['GIc'].apply(unumpy.std_devs),
            yerr=df['GIIc'].apply(unumpy.std_devs),
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color=data_color, alpha=.7, label=label)

        if annotate:
            # Data points and labels
            x = df['GIc'].apply(unumpy.nominal_values).values
            y = df['GIIc'].apply(unumpy.nominal_values).values
            idx = df.index.astype(str).values

            # Add index as label to each point
            for i, txt in enumerate(idx):
                ax.annotate(
                    text=txt,
                    xy=(x[i], y[i]),
                    xytext=(2, 2),
                    textcoords='offset points',
                    color=data_color, size=6, alpha=.8)
    
    # Add styled legend
    ax.legend(loc='upper right', 
          frameon=False, 
          fancybox=False, 
          shadow=False,
          fontsize=10,
          framealpha=0.8,
          facecolor="white",
          edgecolor='white',
          prop={'family': 'serif', 'size': 10})
    plt.savefig(save_fig + title + ".jpg", dpi=600, bbox_inches='tight')


def plot_cutlengths(dfA, dfB, dfL, style='dark_background'):
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
        ax1.set_xlabel(r'Inclination $\varphi\, ({}^\circ)\ \longrightarrow$')
        ax1.set_ylabel(
            r'Critical cut length $a_\mathrm{c}\, ' +
            '(\mathrm{cm})\ \longrightarrow$')

        # Plot title
        # plt.title(r'Critical cut lengths with $2\sigma$ error bars', size=9)

        # Unpack plot data for bunker 1
        x1nom = -unumpy.nominal_values(dfA.slope_incl)
        y1nom = unumpy.nominal_values(dfA.rc)/10
        x1std = unumpy.std_devs(dfA.slope_incl)
        y1std = unumpy.std_devs(dfA.rc)/10

        # Unpack plot data for bunker 2
        x2nom = -unumpy.nominal_values(dfB.slope_incl)
        y2nom = unumpy.nominal_values(dfB.rc)/10
        x2std = unumpy.std_devs(dfB.slope_incl)
        y2std = unumpy.std_devs(dfB.rc)/10

        # Unpack plot data for legacy dataset
        x3nom = -unumpy.nominal_values(dfL.slope_incl)
        y3nom = unumpy.nominal_values(dfL.rc)/10
        x3std = unumpy.std_devs(dfL.slope_incl)
        y3std = unumpy.std_devs(dfL.rc)/10

        # Plot bunker 1 cut lenghts with error bars
        ax1.errorbar(
            x=x1nom, y=y1nom, xerr=x1std, yerr=y1std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='teal', label='Bunker 1')

        # Plot bunker 2 cut lenghts with error bars
        ax1.errorbar(
            x=x2nom, y=y2nom, xerr=x2std, yerr=y2std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='lightgrey', label='Bunker 2')

        # Plot legecy cut lenghts with error bars
        ax1.errorbar(
            x=x3nom, y=y3nom, xerr=x3std, yerr=y3std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='orange', alpha=.2,
            label='Legacy dataset')

        # Show legends
        plt.legend(frameon=False, handletextpad=0, loc='upper left',
                   fontsize=9, labelcolor='black')

        # Add slope-angle histogram axis
        ax2 = fig.add_subplot(
            2, 2, 1,
            anchor='S',
            sharex=ax1,
            frameon=False,
            aspect=300)

        # Plot slope-angle histogram
        ax2.hist(
            x=[
                -dfL['slope_incl'].apply(unumpy.nominal_values),
                -dfA['slope_incl'].apply(unumpy.nominal_values),
                -dfB['slope_incl'].apply(unumpy.nominal_values)
            ],
            color=[
                'orange',
                'teal',
                'lightgrey'
            ],
            histtype='stepfilled',
            rwidth=.8,
            density=True,
            bins=40, range=(-70, 70),
            alpha=.7)

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        # Add cut-length histogram axis
        ax3 = fig.add_subplot(
            2, 2, 4,
            aspect=.005,
            anchor='W',
            frameon=False,
            sharey=ax1)

        # Plot cut-length histogram
        ax3.hist(
            x=[
                dfL['rc'].apply(unumpy.nominal_values)/10,
                dfA['rc'].apply(unumpy.nominal_values)/10,
                dfB['rc'].apply(unumpy.nominal_values)/10
            ],
            color=[
                'orange',
                'teal',
                'lightgrey'
            ],
            histtype='stepfilled',
            orientation='horizontal',
            density=True,
            bins=50, range=(0, 50),
            alpha=.8,
            )

        # Hide axes
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=.5)


def plot_modeIIfraction(dfA, dfB, dfL, style='dark_background'):

    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):

        # Setup axes
        fig, ax1 = axis_setup(subplots=True)

        # Axis limits
        ax1.axis([-70, 70, 0, 1.0])

        # Axes labels
        ax1.set_xlabel(r'Inclination $\varphi\, ({}^\circ)\ \longrightarrow$')
        ax1.set_ylabel(r'$\mathcal{G}_\mathrm{I\!I}/\mathcal{G}\ \longrightarrow$')

        # Plot title
        # ax1.title(
        #     'Mode II energy release rate\n' +
        #     'as fraction of total energy release rate',
        #     size=9)

        # Unpack plot data for bunker 1
        x1nom = -unumpy.nominal_values(dfA['slope_incl'])
        y1nom = unumpy.nominal_values(dfA['Gii/G'])
        x1std = unumpy.std_devs(dfA['slope_incl'])
        y1std = unumpy.std_devs(dfA['Gii/G'])

        # Unpack plot data for bunker 2
        x2nom = -unumpy.nominal_values(dfB['slope_incl'])
        y2nom = unumpy.nominal_values(dfB['Gii/G'])
        x2std = unumpy.std_devs(dfB['slope_incl'])
        y2std = unumpy.std_devs(dfB['Gii/G'])

        # Unpack plot data for legacy dataset
        x3nom = -unumpy.nominal_values(dfL['slope_incl'])
        y3nom = unumpy.nominal_values(dfL['Gii/G'])
        x3std = unumpy.std_devs(dfL['slope_incl'])
        y3std = unumpy.std_devs(dfL['Gii/G'])

        # Plot bunker 1 cut lenghts with error bars
        ax1.errorbar(
            x=x1nom, y=y1nom, xerr=x1std, yerr=y1std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='teal', label='Bunker 1')

        # Plot bunker 2 cut lenghts with error bars
        ax1.errorbar(
            x=x2nom, y=y2nom, xerr=x2std, yerr=y2std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='lightgrey', label='Bunker 2')

        # Plot legacy cut lenghts with error bars
        ax1.errorbar(
            x=x3nom, y=y3nom, xerr=x3std, yerr=y3std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='orange', alpha=.2,
            label='Legacy dataset')

        # Show legends
        plt.legend(frameon=False, handletextpad=0, loc='upper left',
                   fontsize=9, labelcolor='black')

        # Add slope-angle histogram axis
        ax2 = fig.add_subplot(
            2, 2, 1,
            anchor='S',
            sharex=ax1,
            frameon=False,
            aspect=300)

        # Plot slope-angle histogram
        ax2.hist(
            x=[
                -dfL['slope_incl'].apply(unumpy.nominal_values),
                -dfA['slope_incl'].apply(unumpy.nominal_values),
                -dfB['slope_incl'].apply(unumpy.nominal_values)
            ],
            color=[
                'orange',
                'teal',
                'lightgrey'
            ],
            histtype='stepfilled',
            density=True,
            bins=40, range=(-70, 70),
            alpha=.7)

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        # Add cut-length histogram axis
        ax3 = fig.add_subplot(
            2, 2, 4,
            aspect=80,
            anchor='W',
            frameon=False,
            sharey=ax1)

        # Plot cut-length histogram
        ax3.hist(
            x=[
                dfL['Gii/G'].apply(unumpy.nominal_values),
                dfA['Gii/G'].apply(unumpy.nominal_values),
                dfB['Gii/G'].apply(unumpy.nominal_values)
            ],
            color=[
                'orange',
                'teal',
                'lightgrey'
            ],
            histtype='stepfilled',
            orientation='horizontal',
            density=True,
            bins=50, range=(0, 1),
            alpha=.7)

        # Hide axes
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=.5)


def plot_totalERR(dfA, dfB, dfL, fit=None, Gmax=2, style='dark_background'):
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
        xm, ym = fit['curves']['mean'].T
        xu, yu = fit['curves']['upper'].T
        xl, yl = fit['curves']['lower'].T

        # Get lists of confidence-interval outline coordinates
        xci = np.append(xl, xu[::-1])
        yci = np.append(yl, yu[::-1])

        # Setup axes
        fig, ax1 = axis_setup(subplots=True)

        # Axis limits
        ax1.axis([0, 1, 0, Gmax])

        # Axes labels
        ax1.set_xlabel(r'$\mathcal{G}_\mathrm{I\!I}/\mathcal{G}\ ' +
                       '\longrightarrow$')
        ax1.set_ylabel(r'$\mathcal{G}\ (\mathrm{J/m}^2)\ \longrightarrow$')

        # Plot best fit and confidence interval
        ax1.plot(xm, ym, color='orange', linewidth=2)
        ax1.fill(xci, yci, color='papayawhip')

        # Unpack plot data for bunker 1
        x1nom = unumpy.nominal_values(dfA['Gii/G'])
        y1nom = unumpy.nominal_values(dfA['Gc'])
        x1std = unumpy.std_devs(dfA['Gii/G'])
        y1std = unumpy.std_devs(dfA['Gc'])

        # Unpack plot data for bunker 2
        x2nom = unumpy.nominal_values(dfB['Gii/G'])
        y2nom = unumpy.nominal_values(dfB['Gc'])
        x2std = unumpy.std_devs(dfB['Gii/G'])
        y2std = unumpy.std_devs(dfB['Gc'])

        # Unpack plot data for legacy dataset
        x3nom = unumpy.nominal_values(dfL['Gii/G'])
        y3nom = unumpy.nominal_values(dfL['Gc'])
        x3std = unumpy.std_devs(dfL['Gii/G'])
        y3std = unumpy.std_devs(dfL['Gc'])

        # Plot legacy data with error bars
        ax1.errorbar(
            x=x3nom, y=y3nom, xerr=x3std, yerr=y3std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='pink', alpha=.5,
            label='Legacy dataset')

        # Plot bunker 1 data with error bars
        ax1.errorbar(
            x=x1nom, y=y1nom, xerr=x1std, yerr=y1std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='teal', label='Bunker 1')

        # Plot bunker 2 data with error bars
        ax1.errorbar(
            x=x2nom, y=y2nom, xerr=x2std, yerr=y2std,
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color='grey', label='Bunker 2')

        # Show legends
        #ax1.legend(frameon=False, handletextpad=0, loc='lower right',
        #           fontsize=9, labelcolor='black')

        # Add cut-length histogram axis
        ax2 = fig.add_subplot(
            2, 2, 4,
            aspect=8,
            anchor='W',
            frameon=False,
            sharey=ax1)

        # Plot cut-length histogram
        ax2.hist(
            x=[
                dfL['Gc'].apply(unumpy.nominal_values),
                dfA['Gc'].apply(unumpy.nominal_values),
                dfB['Gc'].apply(unumpy.nominal_values)
            ],
            color=[
                'pink',
                'teal',
                'grey'
            ],
            histtype='stepfilled',
            orientation='horizontal',
            density=True,
            bins=51, range=(0, Gmax),
            alpha=.8)

        # Hide axes
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0)

def plot_multiple_interactionlaws_fixed(
        data_dict, style='seaborn-v0_8-white', save_fig = "test", title= "test", ci=0.95, Gmax=1.4,
        colors=None):
    """
    Fixed version that follows the original plot_interactionlaw pattern.
    """
    
    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        
        # Setup figure and axes
        fig, ax = axis_setup()
        
        # Default colors
        if colors is None:
            colors = ['teal', 'orange', 'purple', 'red', 'blue', 'green', 'brown', 'pink']
        
        # Ensure enough colors
        while len(colors) < len(data_dict):
            colors.extend(colors)
        
        # Process fits first (like the original function does)
        fits_with_curves = {}
        for label, data in data_dict.items():
            fit = data.get('fit', None)
            if fit:
                # Process the fit curves first (like the original)
                fit_processed = curves(fit)
                fits_with_curves[label] = fit_processed
        
        # Now set up the plot with the first fit (if any) to get proper axis setup
        if fits_with_curves:
            first_fit = list(fits_with_curves.values())[0]
            _, ax = plot_setup(first_fit, ci, Gmax)
            
            # Clear the original text annotations that plot_setup added
            for text in ax.texts:
                text.remove()
        else:
            # No fits, set up basic plot
            ax.set_xlim(0, Gmax)
            ax.set_ylim(0, Gmax)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$')
            ax.set_ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$')
        
        # Plot each dataset
        for i, (label, data) in enumerate(data_dict.items()):
            df = data['df']
            fit = data.get('fit', None)
            color = colors[i % len(colors)]
            
            # Plot fit curves if available
            if fit and label in fits_with_curves:
                fit_processed = fits_with_curves[label]
                xm, ym = fit_processed['curves']['mean'].T
                xu, yu = fit_processed['curves']['upper'].T
                xl, yl = fit_processed['curves']['lower'].T
                
                # Get lists of confidence-interval outline coordinates
                xci = np.append(xl, xu[::-1])
                yci = np.append(yl, yu[::-1])
                
                # Plot best fit and confidence interval
                ax.plot(xm, ym, color=color, linewidth=2)
                ax.fill(xci, yci, color=color, alpha=0.15)
            
            # Plot experimental data with error bars
            ax.errorbar(
                x=df['GIc'].apply(unumpy.nominal_values),
                y=df['GIIc'].apply(unumpy.nominal_values),
                xerr=df['GIc'].apply(unumpy.std_devs),
                yerr=df['GIIc'].apply(unumpy.std_devs),
                linestyle='none', marker='o',
                markersize=3, elinewidth=0.5,
                color=color, alpha=0.7, label=label)
        
        # Add legend
        ax.legend(frameon=False, handletextpad=0, loc='upper right',
                 fontsize=9, labelcolor='black')
        
        # Add parameters for ALL datasets on the right side
        if fits_with_curves:
            y_pos = 0.95
            for i, (label, data) in enumerate(data_dict.items()):
                if label in fits_with_curves:
                    fit = fits_with_curves[label]
                    color = colors[i % len(colors)]
                    
                    # Show fit parameters for this dataset
                    g1 = r'$\mathcal{G}_\mathrm{Ic} = %1.2f \pm %1.2f$' % (
                        fit['params'][0], fit['stddev'][0])
                    g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f$' % (
                        fit['params'][1], fit['stddev'][1])
                    n = r'$n = %3.1f, m = %3.1f$' % (fit['params'][2], fit['params'][3])
                    chi2 = r'$\chi_\nu^2 = %.3f$' % fit['reduced_chi_squared']
                    pval = r'$p = %.3f$' % fit['p_value']
                    
                    stats_text = f'{label}:\n{g1}\n{g2}\n{n}\n{chi2}\n{pval}'
                    ax.text(1.05, y_pos, stats_text, size=12, 
                           transform=ax.transAxes, color=color,
                           horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='gray', alpha=0.9))
                    y_pos -= 0.30  # Increased spacing to accommodate p-value
        
        # Add title
        plt.title(f'Best fit with {int(ci*100)}% confidence intervals', size=11)
    plt.savefig(save_fig + title + ".jpg", dpi=600,  bbox_inches='tight')
    return fig, ax

def plot_multiple_interactionlaws_fixed_exponents(
        data_dict, style='seaborn-v0_8-white', save_fig="test", title="test", 
        ci=0.95, Gmax=1.4, colors=None):
    """
    Plot multiple interaction laws with fixed exponents for different datasets.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with format {label: {'df': df, 'fit': fit}} where fit contains fixed exponents.
    style : str, optional
        Plot style. Default is 'seaborn-v0_8-white'.
    save_fig : str, optional
        Base filename for saving figure. Default is "test".
    title : str, optional
        Title for the plot. Default is "test".
    ci : float, optional
        Confidence interval. Default is 0.95.
    Gmax : float, optional
        Maximum G value for plot limits. Default is 1.4.
    colors : list, optional
        List of colors for different datasets. Default is None (auto-generated).
    
    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects.
    """
    
    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):
        
        # Setup figure and axes
        fig, ax = axis_setup()
        
        # Default colors
        if colors is None:
            colors = ['teal', 'orange', 'purple', 'red', 'blue', 'green', 'brown', 'pink']
        
        # Ensure enough colors
        while len(colors) < len(data_dict):
            colors.extend(colors)
        
        # Process fits first (like the original function does)
        fits_with_curves = {}
        for label, data in data_dict.items():
            fit = data.get('fit', None)
            if fit:
                # Process the fit curves first using fixed exponents function
                fit_processed = curves_fixed_exponents(fit, ci, Gmax)
                fits_with_curves[label] = fit_processed
        
        # Now set up the plot with the first fit (if any) to get proper axis setup
        if fits_with_curves:
            first_fit = list(fits_with_curves.values())[0]
            _, ax = plot_setup_fixed_exponents(first_fit, ci, Gmax)
            
            # Clear the original text annotations that plot_setup_fixed_exponents added
            for text in ax.texts:
                text.remove()
        else:
            # No fits, set up basic plot
            ax.set_xlim(0, Gmax)
            ax.set_ylim(0, Gmax)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$')
            ax.set_ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$')
        
        # Plot each dataset
        for i, (label, data) in enumerate(data_dict.items()):
            df = data['df']
            fit = data.get('fit', None)
            color = colors[i % len(colors)]
            
            # Plot fit curves if available
            if fit and label in fits_with_curves:
                fit_processed = fits_with_curves[label]
                xm, ym = fit_processed['curves']['mean'].T
                xu, yu = fit_processed['curves']['upper'].T
                xl, yl = fit_processed['curves']['lower'].T
                
                # Get lists of confidence-interval outline coordinates
                xci = np.append(xl, xu[::-1])
                yci = np.append(yl, yu[::-1])
                
                # Plot best fit and confidence interval
                ax.plot(xm, ym, color=color, linewidth=2)
                ax.fill(xci, yci, color=color, alpha=0.15)
            
            # Plot experimental data with error bars
            ax.errorbar(
                x=df['GIc'].apply(unumpy.nominal_values),
                y=df['GIIc'].apply(unumpy.nominal_values),
                xerr=df['GIc'].apply(unumpy.std_devs),
                yerr=df['GIIc'].apply(unumpy.std_devs),
                linestyle='none', marker='o',
                markersize=3, elinewidth=0.5,
                color=color, alpha=0.7, label=label)
        
        # Add legend
        ax.legend(frameon=False, handletextpad=0, loc='upper right',
                 fontsize=9, labelcolor='black')
        
        # Add parameters for ALL datasets on the right side
        if fits_with_curves:
            y_pos = 0.95
            for i, (label, data) in enumerate(data_dict.items()):
                if label in fits_with_curves:
                    fit = fits_with_curves[label]
                    color = colors[i % len(colors)]
                    
                    # Show fit parameters for this dataset (fixed exponents)
                    g1 = r'$\mathcal{G}_\mathrm{Ic} = %1.2f \pm %1.2f$' % (
                        fit['params'][0], fit['stddev'][0])
                    g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f$' % (
                        fit['params'][1], fit['stddev'][1])
                    n = r'$n = %3.1f, m = %3.1f$ (fixed)' % (fit['n_fixed'], fit['m_fixed'])
                    chi2 = r'$\chi_\nu^2 = %.3f$' % fit['reduced_chi_squared']
                    pval = r'$p = %.3f$' % fit['p_value']
                    
                    stats_text = f'{label}:\n{g1}\n{g2}\n{n}\n{chi2}\n{pval}'
                    ax.text(1.05, y_pos, stats_text, size=12, 
                           transform=ax.transAxes, color=color,
                           horizontalalignment='left', verticalalignment='top', 
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='gray', alpha=0.9))
                    y_pos -= 0.30  # Increased spacing to accommodate p-value
        
        # Add title
        plt.title(f'Best fit with {int(ci*100)}% confidence intervals (fixed exponents)', size=11)
    
    plt.savefig(save_fig + title + ".jpg", dpi=600, bbox_inches='tight')
    return fig, ax
# Example usage:
"""
# Simple usage
data_dict = {
    'Bunker 1': {'df': dfA, 'fit': fitA},
    'Bunker 2': {'df': dfB, 'fit': fitB},
    'Legacy': {'df': dfL, 'fit': fitL}
}

fig, ax = plot_multiple_interactionlaws(data_dict)

# Advanced usage with custom colors and detailed annotations
data_dict_advanced = {
    'Bunker 1': {'df': dfA, 'fit': fitA, 'variant': 'B'},
    'Bunker 2': {'df': dfB, 'fit': fitB, 'variant': 'A'},
    'Legacy': {'df': dfL, 'fit': fitL, 'variant': 'BK'}
}

colors = ['teal', 'orange', 'purple']
fig, ax = plot_multiple_interactionlaws_advanced(
    data_dict_advanced, colors=colors, show_equations=True, show_stats=True
)
"""

def curves_fixed_exponents(fit, ci=0.95, xymax=2.0, moderatio=False):
    """
    Add vertice lists of mean and confidence interval curves to results dict for fixed exponents.
    """
    
    def moderatio_residual_fixed_exponents(beta, x, n_fixed, m_fixed, var='B', bounds=False):
        """
        Evaluate residual with mode ratio and total ERR as input for fixed exponents.
        """
        # Unpack inputs
        psi, G = x
        # Calculate Gi and Gii
        Gii = psi*G
        Gi = G - Gii
        # Evaluate residual using fixed exponents function
        return reg.residual_fixed_exponents(beta, [Gi, Gii], n_fixed, m_fixed, var=var, bounds=bounds)

    # Assemble high-resolution gridpoints
    inp = np.linspace(0, xymax)
    X, Y = np.meshgrid(inp, inp)

    # Initialize curves dictionary
    fit['curves'] = {'mean': [], 'lower': [], 'upper': []}

    # Get fixed exponents from fit
    n_fixed = fit['n_fixed']
    m_fixed = fit['m_fixed']

    # Plot curves to extract their vertices
    for curve in fit['curves'].keys():
        # Calculate residual on grid points using fixed exponents params function
        if moderatio:
            Z = moderatio_residual_fixed_exponents(params_fixed_exponents(fit, curve, ci), [X, Y],
                                                  n_fixed, m_fixed, var=fit['var'], bounds=False)
        else:
            Z = reg.residual_fixed_exponents(params_fixed_exponents(fit, curve, ci), [X, Y],
                                            n_fixed, m_fixed, var=fit['var'], bounds=False)
        # Plot (zero-width) contour where residual is zero
        contour = plt.contour(X, Y, Z, 0, linewidths=0)
        # Get vertices of contour line
        fit['curves'][curve] = contour.collections[1].get_paths()[0].vertices

    # Suppress plot output
    plt.close()

    return fit


def plot_interactionlaw_fixed_exponents(
        df, fit=None, style='seaborn-v0_8-white', save_fig="test", title="test", 
        data_color="green", fit_color="green", ci=0.95, Gmax=1.4, 
        label="label", annotate=False):
    """
    Plot experimental data and best fit with confidence intervals for fixed exponents.
    """
    
    # Set plot style
    plt.rcdefaults()
    with plt.style.context(style):

        if fit:
            # Get plot data using fixed exponents curves function
            fit = curves_fixed_exponents(fit, ci, Gmax)
            xm, ym = fit['curves']['mean'].T
            xu, yu = fit['curves']['upper'].T
            xl, yl = fit['curves']['lower'].T

            # Prepare plot using fixed exponents setup
            fig, ax = plot_setup_fixed_exponents(fit, ci, Gmax)

            # Get lists of confidence-interval outline coordinates
            xci = np.append(xl, xu[::-1])
            yci = np.append(yl, yu[::-1])

            # Plot best fit and confidence interval
            ax.plot(xm, ym, color=fit_color, linewidth=2)
            ax.fill(xci, yci, color=fit_color, alpha=0.15)

        else:
            # Define figure and axes
            fig, ax = axis_setup()
            plt.axis([0, Gmax, 0, Gmax])
            # Axes labels
            plt.xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ ' +
                       r'\longrightarrow$')
            plt.ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ ' +
                       r'\longrightarrow$')

        # Plot fracture toughnesses with 1-sigma error bars
        ax.errorbar(
            x=df['GIc'].apply(unumpy.nominal_values),
            y=df['GIIc'].apply(unumpy.nominal_values),
            xerr=df['GIc'].apply(unumpy.std_devs),
            yerr=df['GIIc'].apply(unumpy.std_devs),
            linestyle='none', marker='o',
            markersize=3, elinewidth=.5,
            color=data_color, alpha=.7, label=label)

        if annotate:
            # Data points and labels
            x = df['GIc'].apply(unumpy.nominal_values).values
            y = df['GIIc'].apply(unumpy.nominal_values).values
            idx = df.index.astype(str).values

            # Add index as label to each point
            for i, txt in enumerate(idx):
                ax.annotate(
                    text=txt,
                    xy=(x[i], y[i]),
                    xytext=(2, 2),
                    textcoords='offset points',
                    color=data_color, size=6, alpha=.8)
    
    # Add styled legend
    ax.legend(loc='upper right', 
          frameon=False, 
          fancybox=False, 
          shadow=False,
          fontsize=10,
          framealpha=0.8,
          facecolor="white",
          edgecolor='white',
          prop={'family': 'serif', 'size': 10})
    
    plt.savefig(save_fig + title + ".jpg", dpi=600, bbox_inches='tight')
    return fig, ax

def plot_setup_fixed_exponents(fit, ci=0.95, Gmax=1.4):
    """
    Setup plot of experimental data and fit for fixed exponents.
    
    Parameters
    ----------
    fit : dict
        Dictionary with optimization results (fixed exponents).
    ci : float, optional
        Confidence interval. Default is 0.95.
    Gmax : float, optional
        Maximum G value for plot limits. Default is 1.4.
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Configured figure and axes objects.
    """
    
    # Dict for objective function latex expressions
    eq = {'A': '', 'B': '', 'BK': '', 'VA': ''}
    
    # Variant A
    eq['A'] += r'$\displaystyle\left[\left(\frac{\mathcal{G}_\mathrm{I}}'
    eq['A'] += r'{\mathcal{G}_\mathrm{Ic}}\right)^{1/n} + '
    eq['A'] += r'\left(\frac{\mathcal{G}_\mathrm{II}}{\mathcal{G}_\mathrm{IIc}}'
    eq['A'] += r'\right)^{1/m}\right]^{\frac{2}{1/n+1/m}}\!\!\! = 1$'
    
    # Variant B
    eq['B'] += r'$\displaystyle\left(\frac{\mathcal{G}_\mathrm{I}}'
    eq['B'] += r'{\mathcal{G}_\mathrm{Ic}}\right)^{1/n}+'
    eq['B'] += r'\left(\frac{\mathcal{G}_\mathrm{II}}'
    eq['B'] += r'{\mathcal{G}_\mathrm{IIc}}\right)^{1/m}\!\!\!=1$'
    
    # Variant BK
    eq['BK'] += r'$\displaystyle \mathcal{G} = \mathcal{G}_\mathrm{Ic} + '
    eq['BK'] += r'\left(\mathcal{G}_\mathrm{IIc} - \mathcal{G}_\mathrm{Ic} '
    eq['BK'] += r'\right)\left(\frac{\mathcal{G}_\mathrm{II}}'
    eq['BK'] += r'{\mathcal{G}_\mathrm{I} + \mathcal{G}_\mathrm{II}}\right)^n$'
    
    # Define figure and axes
    fig, ax = axis_setup()
    
    # Axis limits
    plt.axis([0, Gmax, 0, Gmax])
    ax.set_aspect('equal')
    
    # Axes labels
    plt.xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$', fontsize=12)
    plt.ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$', fontsize=12)
    
    # Plot title
    plt.title(f'Best fit with {int(ci*100)}% confidence interval', size=12)
    
    # Fracture toughnesses (only GIc and GIIc)
    g1 = r'$\mathcal{G}_\mathrm{Ic}\; = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (
            fit['params'][0], fit['stddev'][0])
    g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (
            fit['params'][1], fit['stddev'][1])
    
    # Fixed envelope parameters
    n = r'$n\ = %3.1f$ (fixed)' % fit['n_fixed']
    m = r'$m = %3.1f$ (fixed)' % fit['m_fixed']
    
    # Check if this is a constrained fit and add constraint info
    if 'optimization_success' in fit:
        ratio = fit['params'][1] / fit['params'][0]
        constraint_info = r'$\mathcal{G}_\mathrm{IIc}/\mathcal{G}_\mathrm{Ic} = %1.3f$ (constrained to 0.79/0.56)' % ratio
    else:
        constraint_info = ""
    
    # Goodness of fit
    chi2 = r'$\chi_\nu^2 = %.3f$' % fit['reduced_chi_squared']
    pval = r'$p = %.3f$' % fit['p_value']
    
    # Write annotations
    plt.text(
        1.05, 1.01, eq[fit['var']], size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', usetex=True)
    plt.text(
        1.05, .72, g1 + '\n' + g2, size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='bottom', usetex=False)
    plt.text(
        1.05, .68, n + '\n' + m, size=11, transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', usetex=False)
    
    # Add constraint info if available
    if constraint_info:
        plt.text(
            1.05, .62, constraint_info, size=11, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top', usetex=False)
        plt.text(
            1.05, .55, chi2 + '\n' + pval, size=11, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top', usetex=False)
    else:
        plt.text(
            1.05, .55, chi2 + '\n' + pval, size=11, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top', usetex=False)
    
    return fig, ax


def plot_bma_envelope(GI_all, GII_all, weights, df=None, ax=None,
                      title="Bayesian Model Averaged Envelope", save_fig="test", save_title = "test",
                      params=None, best_idx=None):
    """
    Plot BMA credible region and best-fit envelope with data points overlaid.

    Parameters
    ----------
    GI_all : np.ndarray
        Array of GI values from BMA samples, shape (n_samples, n_points)
    GII_all : np.ndarray
        Array of GII values from BMA samples, shape (n_samples, n_points)
    weights : np.ndarray
        Weights associated with each sample, shape (n_samples,)
    theta : np.ndarray
        Mode-mix angle array.
    df : pd.DataFrame, optional
        DataFrame with GIc and GIIc columns to plot data points.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new one.
    title : str
        Plot title.
    params : list, optional
        List of parameter sets corresponding to GI_all and GII_all.
    best_idx : int, optional
        Index of the best-fit parameter set (min residual).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from uncertainties import unumpy

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    weights = np.asarray(weights).flatten()
    if weights.shape[0] != GI_all.shape[0]:
        raise ValueError("weights shape does not match the number of BMA samples")
    weights /= weights.sum()

    # 90% credible envelope (2D band)
    GI_low = np.percentile(GI_all, 10, axis=0)
    GI_high = np.percentile(GI_all, 90, axis=0)
    GII_low = np.percentile(GII_all, 10, axis=0)
    GII_high = np.percentile(GII_all, 90, axis=0)

    GI_band = np.concatenate([GI_low, GI_high[::-1]])
    GII_band = np.concatenate([GII_low, GII_high[::-1]])

    ax.fill(GI_band, GII_band, color='C0', alpha=0.2, label='80% credible region')

    # Plot best-fit curve (min residual)
    if best_idx is not None:
        GI_best = GI_all[best_idx]
        GII_best = GII_all[best_idx]
        ax.plot(GI_best, GII_best, '-', color='C1', linewidth=2, label='Best-fit envelope')
        if params is not None:
            GIc, GIIc, n, m = params[best_idx]
            print(f"Best-fit parameters:\n  GIc = {GIc:.3f} J/m²\n  GIIc = {GIIc:.3f} J/m²\n  n = {n:.2f}, m = {m:.2f}")

    # Plot fracture data points
    if df is not None:
        gi = unumpy.nominal_values(df["GIc"])
        gii = unumpy.nominal_values(df["GIIc"])
        ax.plot(gi, gii, 'o', color='k', ms=4, alpha=0.6, label='Fracture data')

    ax.set_xlabel("Mode I fracture toughness $G_{Ic}$ (J/m²)")
    ax.set_ylabel("Mode II fracture toughness $G_{IIc}$ (J/m²)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))
    plt.tight_layout()
    plt.savefig(save_fig + save_title + ".jpg", dpi=600,  bbox_inches='tight')
    return ax

def plot_all_fits_colored(GI_all, GII_all, residuals, ax=None, clip=40,
                          title="BMA Envelopes Colored by Residuals", save_fig="test", save_title = "test"):
    """
    Plot all BMA envelope curves with color based on residual magnitude.
    
    - Curves with lower residuals (better fits) are plotted last (on top).
    - Color grading is logarithmic to emphasize detail among low residuals.
    - Darker = better fit (low residual), Lighter = worse fit (high residual).

    Parameters
    ----------
    GI_all : np.ndarray
        Mode I values from all BMA fits, shape (n_samples, n_angles).
    GII_all : np.ndarray
        Mode II values from all BMA fits, shape (n_samples, n_angles).
    residuals : np.ndarray
        Sum of squared residuals for each fit (n_samples,).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    clip: clipped upper tail to improve contrast in percent
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot axis with the rendered envelope cloud.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    residuals = np.asarray(residuals)
    n_samples = residuals.shape[0]

    # Sort curves so best ones (lowest residuals) are plotted last (on top)
    sort_idx = np.argsort(residuals)[::-1]
    GI_all = GI_all[sort_idx]
    GII_all = GII_all[sort_idx]
    residuals = residuals[sort_idx]

    # Ensure residuals are positive and handle edge cases
    residuals = np.maximum(residuals, 1e-10)  # Avoid zeros
    
    # Log scale for residuals, clipped upper tail to improve contrast
    vmin = residuals.min()
    vmax = np.percentile(residuals, clip)
    
    # Ensure vmin and vmax are valid for LogNorm
    if vmin <= 0 or vmax <= 0 or not np.isfinite([vmin, vmax]).all():
        # Fallback to linear normalization if log fails
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    
    cmap = cm.viridis  # reversed: darkest = best

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(n_samples):
        color = cmap(norm(residuals[i]))
        ax.plot(GI_all[i], GII_all[i], color=color, alpha=0.9, linewidth=1)

    # Labels and colorbar
    ax.set_xlabel("Mode I fracture toughness $G_{Ic}$ (J/m²)")
    ax.set_ylabel("Mode II fracture toughness $G_{IIc}$ (J/m²)")
    ax.set_title(title)
    ax.grid(True)

    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    if isinstance(norm, mcolors.LogNorm):
        cbar.set_label("Sum of squared residuals (log scale)", rotation=270, labelpad=20)
    else:
        cbar.set_label("Sum of squared residuals", rotation=270, labelpad=20)
    plt.savefig(save_fig + save_title + ".jpg", dpi=600,  bbox_inches='tight')
    return ax

def plot_top_fits_colored(GI_all, GII_all, residuals, df, top_n=10, ax=None,
                          title="Top-N BMA Fits Colored by Residuals", save_fig="test", save_title = "test"):
    """
    Plot the top-N BMA envelope curves with color-graded residuals and fracture data points.

    - Curves with lower residuals (better fits) are drawn last (on top).
    - Color grading is linear within top-N and reversed: darker = better.
    - Fracture data points from df['GIc'] and df['GIIc'] are overlaid.

    Parameters
    ----------
    GI_all : np.ndarray
        Mode I values from BMA fits (n_samples, n_angles).
    GII_all : np.ndarray
        Mode II values from BMA fits (n_samples, n_angles).
    residuals : np.ndarray
        Sum of squared residuals for each fit (n_samples,).
    df : pd.DataFrame
        DataFrame with 'GIc' and 'GIIc' columns.
    top_n : int
        Number of best-fitting envelopes to plot (default: 10).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. If None, a new one is created.
    title : str
        Title of the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the final plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    residuals = np.asarray(residuals)
    sort_idx = np.argsort(residuals)
    top_idx = sort_idx[:top_n][::-1] 

    GI_top = GI_all[top_idx]
    GII_top = GII_all[top_idx]
    res_top = residuals[top_idx]

    # Normalize color map within top-N (best = dark, worst = light)
    vmin = res_top.min()
    vmax = res_top.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis  # best = dark

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(top_n):
        color = cmap(norm(res_top[i]))
        ax.plot(GI_top[i], GII_top[i], color=color, alpha=0.6, linewidth=1.5)

    # Plot fracture data points
    gi = unumpy.nominal_values(df["GIc"])
    gii = unumpy.nominal_values(df["GIIc"])
    ax.plot(gi, gii, 'o', color='k', ms=4, alpha=0.6, label='Fracture data')

    # Axis styling
    ax.set_xlabel("Mode I fracture toughness $G_{Ic}$ (J/m²)")
    ax.set_ylabel("Mode II fracture toughness $G_{IIc}$ (J/m²)")
    ax.set_title(title)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    # Add colorbar for residual grading
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"Sum of squared residuals (top {top_n})", rotation=270, labelpad=20)

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_fig + save_title + ".jpg", dpi=600,  bbox_inches='tight')
    return ax
def plot_multiple_interactionlaws_with_fit(data_dict, save_fig=None, title="", Gmax=2.0, colors=None, ci=0.95):
    """
    Plot multiple interaction laws with fit curves and confidence intervals for multi-dataset regression.
    """
    # Set plot style to match other functions
    plt.rcdefaults()
    with plt.style.context('seaborn-v0_8-white'):
        
        # Setup figure and axes using the same function as other plots
        fig, ax = axis_setup()
        
        if colors is None:
            colors = ['teal', 'orange', 'purple', 'red', 'blue', 'green', 'brown', 'pink']
        
        # Ensure enough colors
        while len(colors) < len(data_dict):
            colors.extend(colors)
        
        # Set up basic plot properties
        ax.set_xlim(0, Gmax)
        ax.set_ylim(0, Gmax)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\mathcal{G}_\mathrm{I}\, (\mathrm{J/m}^2)\ \longrightarrow$', fontsize=12)
        ax.set_ylabel(r'$\mathcal{G}_\mathrm{II}\, (\mathrm{J/m}^2)\ \longrightarrow$', fontsize=12)
        ax.set_title(f'Multi-dataset fit with {int(ci*100)}% confidence intervals', size=12)
        
        # Plot data points first
        for i, (series_name, data) in enumerate(data_dict.items()):
            df = data['df']
            color = colors[i % len(colors)]
            
            # Extract data
            Gi = df['GIc'].apply(unumpy.nominal_values).values
            Gii = df['GIIc'].apply(unumpy.nominal_values).values
            Gi_err = df['GIc'].apply(unumpy.std_devs).values
            Gii_err = df['GIIc'].apply(unumpy.std_devs).values
            
            # Plot data points with error bars
            ax.errorbar(
                x=Gi, y=Gii,
                xerr=Gi_err, yerr=Gii_err,
                linestyle='none', marker='o',
                markersize=3, elinewidth=0.5,
                color=color, alpha=0.7, label=series_name)
        
        # Plot fit curves with confidence intervals
        for i, (series_name, data) in enumerate(data_dict.items()):
            fit_params = data['fit_params']
            fit_stddev = data.get('fit_stddev', None)  # Optional standard deviations
            variant = data['variant']
            color = colors[i % len(colors)]
            
            GIc = fit_params['GIc']
            GIIc = fit_params['GIIc']
            n = fit_params['n']
            m = fit_params['m']
            
            # Generate fit curve for mean values
            G_values = np.linspace(0.01, Gmax, 1000)
            
            if variant == 'A':
                Gii_fit = GIIc * (1 - (G_values/GIc)**(1/n))**(1/(1/m))
            elif variant == 'B':
                Gii_fit = GIIc * (1 - (G_values/GIc)**(1/n))**(1/(1/m))
            else:
                continue
                
            # Only plot valid values
            valid_mask = (Gii_fit > 0) & np.isfinite(Gii_fit)
            ax.plot(G_values[valid_mask], Gii_fit[valid_mask], 
                   color=color, linewidth=2, linestyle='-')
            
            # Plot confidence intervals if standard deviations are provided
            if fit_stddev is not None:
                from scipy.stats import norm
                
                # Convert confidence interval to number of standard deviations
                pp = (1 + ci) / 2
                nstd = norm.ppf(pp)
                
                # Generate curves for upper and lower bounds
                GIc_upper = GIc + nstd * fit_stddev['GIc']
                GIc_lower = GIc - nstd * fit_stddev['GIc']
                GIIc_upper = GIIc + nstd * fit_stddev['GIIc']
                GIIc_lower = GIIc - nstd * fit_stddev['GIIc']
                
                # DEBUG: Print the confidence interval ranges
                print(f"\n=== {series_name} Confidence Intervals ===")
                print(f"GIc: {GIc:.4f} ± {nstd * fit_stddev['GIc']:.4f} = [{GIc_lower:.4f}, {GIc_upper:.4f}]")
                print(f"GIIc: {GIIc:.4f} ± {nstd * fit_stddev['GIIc']:.4f} = [{GIIc_lower:.4f}, {GIIc_upper:.4f}]")
                print(f"Standard deviations: GIc={fit_stddev['GIc']:.6f}, GIIc={fit_stddev['GIIc']:.6f}")
                
                # Upper bound curve
                if variant == 'A':
                    Gii_fit_upper = GIIc_upper * (1 - (G_values/GIc_lower)**(1/n))**(1/(1/m))
                elif variant == 'B':
                    Gii_fit_upper = GIIc_upper * (1 - (G_values/GIc_lower)**(1/n))**(1/(1/m))
                
                valid_mask_upper = (Gii_fit_upper > 0) & np.isfinite(Gii_fit_upper)
                
                # Lower bound curve
                if variant == 'A':
                    Gii_fit_lower = GIIc_lower * (1 - (G_values/GIc_upper)**(1/n))**(1/(1/m))
                elif variant == 'B':
                    Gii_fit_lower = GIIc_lower * (1 - (G_values/GIc_upper)**(1/n))**(1/(1/m))
                
                valid_mask_lower = (Gii_fit_lower > 0) & np.isfinite(Gii_fit_lower)
                
                # DEBUG: Check the curve ranges
                if np.any(valid_mask_upper):
                    print(f"Upper curve range: {Gii_fit_upper[valid_mask_upper].min():.4f} to {Gii_fit_upper[valid_mask_upper].max():.4f}")
                if np.any(valid_mask_lower):
                    print(f"Lower curve range: {Gii_fit_lower[valid_mask_lower].min():.4f} to {Gii_fit_lower[valid_mask_lower].max():.4f}")
                
                # Create confidence interval band
                if np.any(valid_mask_upper) and np.any(valid_mask_lower):
                    # Find common valid range
                    common_valid = valid_mask_upper & valid_mask_lower
                    if np.any(common_valid):
                        G_common = G_values[common_valid]
                        Gii_upper_common = Gii_fit_upper[common_valid]
                        Gii_lower_common = Gii_fit_lower[common_valid]
                        
                        # DEBUG: Check the band width
                        band_width = np.abs(Gii_upper_common - Gii_lower_common)
                        print(f"Band width range: {band_width.min():.6f} to {band_width.max():.6f}")
                        print(f"Average band width: {band_width.mean():.6f}")
                        
                        # Create confidence interval band
                        G_band = np.concatenate([G_common, G_common[::-1]])
                        Gii_band = np.concatenate([Gii_upper_common, Gii_lower_common[::-1]])
                        
                        # Plot the band with higher alpha for visibility
                        ax.fill(G_band, Gii_band, color=color, alpha=0.3, label=f'{series_name} CI')
                        print(f"Plotted confidence interval band for {series_name}")
                    else:
                        print(f"No common valid range for {series_name}")
                else:
                    print(f"No valid curves for {series_name}")
            else:
                print(f"No standard deviations provided for {series_name}")
        
        # Add legend with proper styling
        ax.legend(loc='upper right', 
                  frameon=False, 
                  fancybox=False, 
                  shadow=False,
                  fontsize=10,
                  framealpha=0.8,
                  facecolor="white",
                  edgecolor='white',
                  prop={'family': 'serif', 'size': 10})
        
        # Add equation on the right side
        eq = r'$\displaystyle\left(\frac{\mathcal{G}_\mathrm{I}}'
        eq += r'{\mathcal{G}_\mathrm{Ic}}\right)^{1/n}+'
        eq += r'\left(\frac{\mathcal{G}_\mathrm{II}}'
        eq += r'{\mathcal{G}_\mathrm{IIc}}\right)^{1/m}\!\!\!=1$'
        
        plt.text(
            1.05, 1.01, eq, size=11, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top', usetex=True)
        
        # Add fit parameters on the right side
        y_pos = 0.90
        for i, (series_name, data) in enumerate(data_dict.items()):
            fit_params = data['fit_params']
            fit_stddev = data.get('fit_stddev', None)
            color = colors[i % len(colors)]
            
            GIc = fit_params['GIc']
            GIIc = fit_params['GIIc']
            n = fit_params['n']
            m = fit_params['m']
            
            # Format parameters
            if fit_stddev is not None:
                g1 = r'$\mathcal{G}_\mathrm{Ic} = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (GIc, fit_stddev['GIc'])
                g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f \pm %1.2f\ \mathrm{J/m}^2$' % (GIIc, fit_stddev['GIIc'])
            else:
                g1 = r'$\mathcal{G}_\mathrm{Ic} = %1.2f\ \mathrm{J/m}^2$' % GIc
                g2 = r'$\mathcal{G}_\mathrm{IIc} = %1.2f\ \mathrm{J/m}^2$' % GIIc
            
            n_m = r'$n = %3.1f, m = %3.1f$' % (n, m)
            
            # Create text box for this dataset (only parameters, no individual stats)
            stats_text = f'{series_name}:\n{g1}\n{g2}\n{n_m}'
            
            ax.text(1.05, y_pos, stats_text, size=11, 
                   transform=ax.transAxes, color=color,
                   horizontalalignment='left', verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           edgecolor='gray', alpha=0.9))
            y_pos -= 0.25  # Reduced spacing since we removed individual stats
        
        # Add shared parameters and overall fit statistics at the bottom
        if data_dict:
            first_data = list(data_dict.values())[0]
            shared_n = first_data['fit_params']['n']
            shared_m = first_data['fit_params']['m']
            
            # Get overall fit statistics if available
            overall_stats = first_data.get('overall_fit_stats', None)
            
            shared_text = f'Shared exponents: n = {shared_n:.3f}, m = {shared_m:.3f}'
            if overall_stats is not None:
                overall_chi2 = overall_stats.get('reduced_chi_squared', 0)
                overall_pval = overall_stats.get('p_value', 0)
                overall_r2 = overall_stats.get('R_squared', 0)
                shared_text += f'\nOverall fit: χ²ν = {overall_chi2:.3f}, p = {overall_pval:.3f}, R² = {overall_r2:.3f}'
            
            ax.text(1.05, y_pos - 0.1, shared_text, size=11, 
                   transform=ax.transAxes, color='black',
                   horizontalalignment='left', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           edgecolor='gray', alpha=0.9))
    
    if save_fig:
        plt.savefig(save_fig + f"{title}.jpg", dpi=600, bbox_inches='tight')
    
    return fig, ax