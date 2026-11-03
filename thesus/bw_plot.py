import itertools as it

import numpy as np
import pandas as pd
from scipy.stats import t, kruskal, linregress

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn

labels = {
    'ignore': 'Ignore',
    'fixed': 'Fixed',
    'lme': 'Random',
    'gee': 'GEE',
    'meta': 'Pooled',
}
tech_round = {'ignore': 1, 'gee': 1}
icc_labels = ['<0.1', '0.1-0.4', '0.5-0.75', '>0.75']

def plot_icc_curves(iccs2_long, ax=None, scatter_kws=None, error_kws=None):
    """
    Summary of relationship between seed and real ICCsk with 95% CI error bars

    Parameters
    ----------
    icc_long: DataFrame
        A pandas dataframe mapping the the calculated ICC (fit_icc) to the 
        ICC used to seed the calculation (index). 
    ax: Axes, optional
        The axis where the data should be plotted
    scatter_kws: dict
        Paramters for the scatter plot. See matplotlib.pyplot.scatter for 
        more options
    error_kws: dict
        Paramters for the error bar plot. See matplotlib.pyplot.errorbar for
        options.

    Returns
    -------
    Axes: The matplotlib axis with the figure plotted

    """

    # Sets up plotting defaults
    default_scatter_kws = dict(marker='.', alpha=0.5)
    def_error_kws = dict(capsize=5, color='k', linestyle='', marker='o',)
    # Handles default/modified arguments
    if ax is None:
        ax = plt.axes()

    if scatter_kws is not None:
        default_scatter_kws.update(scatter_kws)

    if error_kws is not None:
        def_error_kws.update(error_kws)

    # Calculates average interval
    x_bar = iccs2_long.groupby('g_var')['fit_icc'].mean().index
    y_bar = iccs2_long.groupby('g_var')['fit_icc'].mean().values
    y_err = iccs2_long.groupby('g_var')['fit_icc'].std().values / \
        np.sqrt(iccs2_long.groupby('g_var')['fit_icc'].count() - 1).values# * \
        # t.ppf(0.975, iccs2_long.groupby('g_var')['fit_icc'].count() - 1)

    # plots of scatter of the ICC values
    ax.scatter(x=iccs2_long['g_var'], y=iccs2_long['fit_icc'], 
               **default_scatter_kws)

    # Plots the mean distribution
    ax.errorbar(x=x_bar, y=y_bar, yerr=y_err, **def_error_kws, zorder=0)

    return ax


def _corr_plot(ax, tech1, tech2, data, lims, labels=labels, scatter_kws=None,
               line_kws=None,
               labelleft=False, labelbottom=False, title=False):
    """
    Plots the correlation between two techniques for analyzing data
    """

    # Correlation scatter plot
    scatter_default = dict(marker='.', edgecolor='None', s=4)
    if scatter_kws is not None: 
        scatter_default.update(scatter_kws)
    sn.scatterplot(
        x=data.xs(tech1, level='model')['beta'],
        y=data.xs(tech2, level='model')['beta'],
        **scatter_default
    )

    # Dashed line
    line_default = dict(color='k', linestyle='--', linewidth=0.5)
    if line_kws is not None:
        line_default.update(line_kws)
    ax.plot(lims, lims, zorder=0, **line_default)

    # Sets limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Adds labels
    ax.set_xlabel((f'$\\hat{{{{\\beta}}}}$\n{{{tech1}}}').format(**labels),
                  size=10)
    ax.set_ylabel((f'{{{tech2}}}\n$\\hat{{{{\\beta}}}}$').format(**labels), 
                  size=10)
    # Tidies axis
    ax.xaxis.set_tick_params(labelsize=8, labelbottom=labelbottom, rotation=90)
    ax.xaxis.get_label().set_visible(labelbottom)

    ax.yaxis.set_tick_params(labelsize=8, labelleft=labelleft)
    ax.yaxis.get_label().set_visible(labelleft)


def _hist_plot(ax, tech1, tech2, data, lims, bin_size=0.1, hist_kws=None,
               labelpad=57, labelbottom=False, labelleft=False, labels=labels):
    """
    Plots the distribution of coeffecients as a histogram
    """
    # Plots the histogram
    hist_default = dict()
    if hist_kws is not None:
        hist_default.update(hist_kws)

    bins = np.arange(lims[0], lims[1] + bin_size, bin_size)
    sn.histplot(data=data.xs(tech1, level='model').reset_index(),
                x='beta', 
                bins=bins,
                # palette='viridis', 
               **hist_default)

    # Formats the x-axis
    ax.set_xlim(lims)
    ax.set_xlabel((f'$\\hat{{{{\\beta}}}}$\n{{{tech1}}}').format(**labels),
                  size=10)
    ax.xaxis.set_tick_params(labelsize=8, labelbottom=labelbottom, rotation=90)
    ax.xaxis.get_label().set_visible(labelbottom)

    # Formats y axis
    ax.set_ylabel(labels[tech2], size=10, labelpad=labelpad)
    ax.yaxis.set_tick_params(labelsize=8, labelleft=False, left=False)
    ax.yaxis.get_label().set_visible(labelleft)


def _plot_r2(ax, tech1, tech2, data, labelright=False, **kwargs):
    """
    Creates description for two techniques
    """
    # Sets up font defaults
    text_default = dict(size=8, ha='center', va='center')
    text_default.update(kwargs)

    # Regresses the coeffecients between the two groups
    x = data.xs(tech1, level='model')['beta']
    y = data.xs(tech2, level='model')['beta']
    m, b, r2, p, se = linregress(x, y)
    sign = {-1:'-'}.get(np.sign(b), '+')
    b = np.absolute(b)

    # Describes the relationship
    ax.text(0.5, 0.5, f'{m:1.1f}x{sign}{b:1.1f}\n$R^{{2}}$={r2:1.2f}', 
            **text_default)

    # Formats axis
    ax.xaxis.set_tick_params(left=False, labelleft=False, 
                             right=False, labelright=False)
    ax.yaxis.set_tick_params(top=False, labeltop=False, 
                             bottom=False, labelbottom=False)


def _plot_error_boxplot(ax, tech, data, palette='viridis', labels=labels,
                        tech_round=tech_round, default_round=2,
                        box_kws=None, strip_kws=None, labelbottom=False,
                        icc_labels=icc_labels, error_tech_round=tech_round):
    """
    Plots error bars as a function of the ICC grouping
    """
    # Pulls out bse data and sets up groups for plotting (all the keyword 
    # argument dictionaries!)
    model_data = data.xs(tech, level='model', axis='index').copy()
    tech_kwargs = dict(ax=ax,
                       x='icc_group',
                       y='bse',
                       data=model_data,
                       palette=palette)

    # Sets up formatting keywords
    boxplot_defaults = dict(boxprops={'facecolor': 'w'}, linewidth=1, 
                            fliersize=0)
    if box_kws is not None:
        boxplot_defaults.update(box_kws)
    strip_defaults = dict(edgecolor='None', s=1.5, jitter=0.3, alpha=0.75)
    if strip_kws is not None:
        strip_defaults.update(strip_kws)

    # Calculates the kruskal wallis p-value
    grouped = model_data.groupby('icc_group')['bse'].apply(lambda x: x.values)
    w, p = kruskal(*grouped.values)
    if p > 1e-2:
        p_str = f'p={p:1.2f}'
    elif p > 1e-3:
        p_str = f'p={p:1.3f}'
    else:
        p_str = f'p={p:1.0e}'

    # # Makes the plot!
    sn.boxplot(**tech_kwargs, **boxplot_defaults)
    sn.stripplot(**tech_kwargs, **strip_defaults)
    ax.text(ax.get_xlim()[0] + np.diff(ax.get_xlim()) * 0.05,
            ax.get_ylim()[1] + np.diff(ax.get_ylim()) * 0.05,
            p_str,
            size=8,
            ha='left', 
            va='top'
           )
    ax.set_ylim(ax.get_ylim()[0], 
                ax.get_ylim()[1] + 0.1 * np.diff(ax.get_ylim()))

    # Formats x axis
    ax.xaxis.set_tick_params(labelbottom=labelbottom, labelsize=8, rotation=90)
    ax.set_xlabel('ICC group')
    ax.set_xticklabels(icc_labels)
    ax.xaxis.get_label().set_visible(labelbottom)

    # Formats y axis
    ax.yaxis.set_tick_params(labelsize=8)
    yround = tech_round.get(tech, default_round)
    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter(f'{{x:>5.{yround}f}}')
    )
    ax.set_ylabel(f'{{{tech}}}\nBSE'.format(**labels), size=10)


def build_square_plot(fig_ll, tech_order, lims, data, labels=labels,
                      hist_labelpad=55, hist_bin_size=0.1, scatter_kws=None,
                      line_kws=None, hist_kws=None, text_kws=dict(), 
                      title_ax=False, title='', title_kws=dict()):
    """
    Makes a square plot mapping correlations and error
    """
    # Sets up parameters to order the data correctly
    num_techniques = len(tech_order)
    last = num_techniques - 1
    tech_idx = np.arange(num_techniques)
    method_combo = [(t0, t1) for t0, t1 in it.combinations(tech_order, 2)]
    pos_combo = [(t0, t1) for t0, t1 in it.combinations(tech_idx, 2)]
    ax_ypos, ax_xpos = zip(*(pos_combo))

    # Sets up gridspec for plotting
    gs_l = fig_ll.add_gridspec(num_techniques + title_ax * 1, 
                               num_techniques + 2)

    # Plots the scatter traces and scatter
    for ax_x, ax_y, (tech1, tech2) in zip(*(ax_xpos, ax_ypos, method_combo)):
        axc = fig_ll.add_subplot(gs_l[ax_x + title_ax * 1, ax_y])
        _corr_plot(ax=axc,
                   data=data,
                   tech1=tech1,
                   tech2=tech2,
                   lims=lims,
                   labelleft=(ax_y==0),
                   labelbottom=(ax_x==last),
                   scatter_kws=scatter_kws,
                   line_kws=line_kws,
                   )
        axr = fig_ll.add_subplot(gs_l[ax_y + title_ax * 1, ax_x])
        _plot_r2(ax=axr,
                 tech1=tech1,
                 tech2=tech2,
                 data=data,
                 **text_kws,
                )

    # Plots the distribution (the diagnonal)
    for ax_x, tech0 in enumerate(tech_order):
        ax = fig_ll.add_subplot(gs_l[ax_x + title_ax * 1, ax_x])
        _hist_plot(ax=ax,
                   tech1=tech0, 
                   tech2=tech0, 
                   data=data,
                   lims=lims, 
                   bin_size=hist_bin_size,
                   labelleft=(ax_x==0),
                   labelbottom=(ax_x==last),
                   labelpad=hist_labelpad,
                   hist_kws=hist_kws)

    # Plots the error
    for ax_y, tech1 in enumerate(tech_order):
        ax = fig_ll.add_subplot(gs_l[ax_y + title_ax * 1, -2:])
        _plot_error_boxplot(ax=ax,
                            tech=tech1,
                            data=data,
                            labelbottom=(ax_y==last)
                            )

    if title_ax:
        title_def_kws = dict(ha='center', va='center', size=14)
        title_def_kws.update(title_kws)
        title_ax = fig_ll.add_subplot(gs_l[0, :])
        title_ax.text(0.5, 0.5, title, **title_def_kws)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        sn.despine(ax=title_ax, left=True, right=True, top=True, bottom=True)


def _summarize_continous_variables(data, columns=['year', 'm_age', 'y']):
    """
    Summarizes continous data across cohorts and cohort types
    """
    cont = pd.concat(
        axis=0, 
        objs=[
            data.groupby(['cohort_type', 'cohort'])[columns].describe(),
            data.groupby(['cohort_type', 'spacer'])[columns].describe(),
            data.groupby(['spacer', 'spacer'])[columns].describe(),
        ]
    )
    cont = cont.unstack().unstack().dropna().unstack(1)
    cont.rename(columns={"50%": 'median', 'count': 'num_samples'}, 
                inplace=True)
    cont.set_index('num_samples', append=True, inplace=True)
    cont['mean (std)'] = cont.apply(
        lambda x: '{mean:>3.1f} ({std:>1.1f})'.format(**x.to_dict()), 
        axis=1
    )
    cont['median [min, max]'] = cont.apply(
        lambda x: '{median:>3.0f} [{min:>3.0f}, {max:3.0f}]'.format(**x.to_dict()),
        axis=1
    )
    cont_tidy = cont[['mean (std)', 'median [min, max]']].unstack(0)
    cont_tidy.columns.set_names(['value', 'group'], inplace=True)
    cont_tidy.columns = cont_tidy.columns.reorder_levels(['group', 'value'])
    cont_tidy.index = cont_tidy.index.reorder_levels(
        ['cohort_type', 'cohort', 'num_samples'])

    return cont_tidy.T

def _summarize_categorical_variables(data, var):
    """
    Summarizes categorical data across cohorts and cohort types
    """
    counts = pd.concat(
        axis=0, 
        sort=False,
        objs=[
            data.groupby(['cohort_type', 'cohort', var])['m_age'].count(),
            data.groupby(['cohort_type', 'spacer', var])['m_age'].count(),
            data.groupby(['spacer', 'spacer', var])['m_age'].count(),
            ]
          ).to_frame()
    counts.index.set_names(['cohort_type', 'cohort', 'value'], inplace=True)
    counts['group'] = var
    counts.reset_index(level='value', inplace=True)
    reference =  pd.concat(
        axis=0, 
        sort=False,
        objs=[
            data.groupby(['cohort_type', 'cohort'])['m_age'].count(),
            data.groupby(['cohort_type', 'spacer'])['m_age'].count(),
            data.groupby(['spacer', 'spacer'])['m_age'].count(),
            ]
          ).to_frame()
    counts['num_samples'] = reference.loc[counts.index]
    counts['percent'] = counts['m_age'] / counts['num_samples']

    counts['values'] = counts.apply(
        lambda x: '{m_age:>4.0f} ({percent:>6.1%})'.format(**x.to_dict()), 
        axis=1
    )

    tidy_counts = counts.copy().reset_index()
    tidy_counts.set_index(
        ['cohort_type', 'cohort', 'num_samples', 'group', 'value'],
        inplace=True
        )
    tidy_counts = tidy_counts['values'].unstack([-2, -1]).T

    return tidy_counts
