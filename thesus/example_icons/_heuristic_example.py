import numpy as np
import pandas as pd

import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn

from ._utils import hide_axes

def plot_bias_coverage_inset(fig, df, scenario=15, ref_model='lme', 
                             icc_group=3, seed_effect=-0.5):
    """
    Plots the inset to describe bias and coverage in data
    """
    sub_data = _pull_sub_data(df, scenario, ref_model, icc_group)
    
    # Calculate sthe bias and coverage dfs
    bias_, bias_df = _get_bias(sub_data)
    cov_, cov_df = _get_coverage(sub_data)

    # Sets up the axis grid for plotting
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.5, 2], width_ratios=[3, 5])
    axb0 = fig.add_subplot(gs[0, 0])
    axc0 = fig.add_subplot(gs[-1, 0])
    axb1 = fig.add_subplot(gs[0, 1], facecolor='None')
    axc1 = fig.add_subplot(gs[-1, 1], facecolor='None')

    # Tidies heuristic axies
    _tidy_heur_ax(axb0, 'Bias', bias_df['y'], bias_df['icc'], 
                  bias_df['sign_round'].apply(lambda x: f'{x:>2.0f}')
                  )
    _tidy_heur_ax(axc0, 'Coverage', cov_df['y'], cov_df['icc'], 
                  cov_df['include'])

    # Plots the insets
    _plot_bias_axis(axb0, bias_df, seed_effect)
    _plot_covearge(axc0, cov_df)

    # Adds the equations
    _annotate_wrap_bias(axb1, '$s_{i}$', width_b=2)
    _annotate_equation(axb1, 's_{i}',)
    _annotate_value(axb1, bias_, 'RdBu_r', [-0.3, 0.3])

    _annotate_wrap_bias(axc1, '$c_{i}$')
    _annotate_equation(axc1, 'c_{i}',)
    _annotate_value(axc1, cov_, 'YlOrRd_r', [0.65, 1])
    


def _pull_sub_data(df_, scenario, ref_model, icc_group):
    """
    Extracts the sub_data to be used for bias, coverage, and power
    """
    # Pulls out hte appropriate levels
    sub_data = df_.xs(scenario, level='scenario')
    sub_data = sub_data.xs(icc_group, level='icc_group')
    sub_data = sub_data.xs(ref_model, level='model').copy()
    # Drops columns we don't need
    sub_data.drop(columns=['p-value', 'converged', 'bse'], inplace=True)
    
    # Calcualtes centered values
    sub_data.reset_index(level='seed_effect', inplace=True)
    sub_data['width'] =  sub_data['ci_hi'] - sub_data['param']
    sub_data['ci_hi_c'] =  sub_data['from_ref'] + sub_data['width']
    sub_data['ci_lo_c'] =  sub_data['from_ref'] - sub_data['width']
    sub_data.set_index('seed_effect', append=True, inplace=True)

    sub_data['in_window'] = (sub_data['ci_lo_c'] > -0.15) * \
                            (sub_data['ci_hi_c'] < 0.15) * 1
    
    return sub_data


def _get_bias(sub_data, example_effect=-0.5):
    """
    Gets the bias df
    """
    # Extracts and calculatest he bias
    bias_df = sub_data.xs(example_effect, level='seed_effect').copy()
    bias_ = bias_df['sign_round'].mean()

    # Filters to the window and identifies the most extreme values
    bias_df['in_window'] = bias_df['abs_diff'] < 0.15
    bias_df = bias_df.loc[bias_df['in_window'].isin([1, True])].copy()
    
    # Pulls out the most extreme reference values
    refs =  bias_df.groupby('sign_round')['abs_diff'].describe()
    refs = refs['max'].mask(refs.index == 0, refs['min'])
    bias_df['b_ref'] = bias_df['sign_round'].replace(refs)
    bias_df['from_bref'] = np.abs(bias_df['abs_diff'] - bias_df['b_ref'])

    # Gets the bias data to show
    bias_df.sort_values(['sign_round', 'from_bref'],
                        ascending=[False, True],
                        inplace=True)
    b_cols = ['sign_round', 'icc', 'from_ref', 'from_bref']
    bias_df = bias_df.groupby('sign_round', sort=False)[b_cols].head(1)
    bias_df['y'] = 1
    bias_df['y'] = bias_df['y'].cumsum() - 0.5
    bias_df['color'] = bias_df['sign_round'].replace({1: '#c94741', 
                                                      0: '#a1a1a1', 
                                                      -1: '#3783bb'})

    return bias_, bias_df


def _get_coverage(sub_data, example_effect=-0.5):
    """
    Gets an exmaple of coverage
    """
    # Gets the overall coverage
    cov_df = sub_data.xs(example_effect, level='seed_effect').copy()
    cov = cov_df['include'].mean()

    cov_df = cov_df.loc[cov_df['in_window'].isin([1, True])].copy()

    # Gets the estimates
    refs =  cov_df.groupby('include')['abs_diff']
    refs = {0: refs.max()[0], 1: refs.min()[1]}
    cov_df['b_ref'] = cov_df['include'].replace(refs)
    cov_df['from_bref'] = np.abs(cov_df['abs_diff'] - cov_df['b_ref'])
    cov_df.sort_values(['include', 'from_bref'],
                        ascending=[False, True],
                        inplace=True)
    c_cols = ['include', 'icc', 'from_ref', 'width']
    cov_df = cov_df.groupby('include', sort=False).head(1)[c_cols]
    cov_df['color'] = cov_df['include'].replace({1: '#a1a1a1', 0: '#252525'})
    cov_df['y'] = np.arange(2)

    return cov, cov_df


def _tidy_heur_ax(ax0, yname, yticks, icc, out_):
    """Cleans up the heuristic axis"""
    ax0.xaxis.set_tick_params(labelsize=8, bottom=False, length=0)
    ax0.yaxis.set_tick_params(**hide_axes, which='major')
    ax0.yaxis.set_tick_params(**hide_axes, which='minor')
    ax0.yaxis.set_tick_params(labelsize=8, labelleft=True, 
                                which='major')
    ax0.yaxis.set_tick_params(labelsize=8, labelright=True, 
                                length=5, which='minor')
    
    ax0.set_yticks(yticks + 1e-3, minor=False)
    ax0.set_yticks(yticks - 1e-3, minor=True)
    ax0.set_yticklabels(icc.apply(lambda x: f'{x:1.2f}'), minor=False)
    ax0.set_yticklabels(out_, minor=True, ha='right')

    ax0.set_ylabel(yname, size=9)
    ax0.set_ylim(yticks.min() - 0.5, yticks.max() + 0.5)

    sn.despine(ax=ax0, left=True, right=True, top=True, bottom=True)


def _plot_bias_axis(axm1a, bias_data):
    """
    Plots the bias data, highlighting the regions of rounding
    """
    axm1a.set_xlim(-0.15, 0.15)
    xd = np.diff(axm1a.get_xlim())[0]
    yd = np.diff(axm1a.get_ylim())[0]

    # Plots the background for the bias
    background = {'#a7d0e4': [-0.3, -0.1 * np.abs(example_effect)],
                  '#f7b799': [0.3, 0.1 * np.abs(example_effect)],
                  '#f7f6f6': np.array([-0.1, 0.1]) * np.abs(example_effect),
                  }
    for cc, xx in background.items():
        axm1a.fill_between(xx, -1, 5, facecolor=cc, edgecolor='None', zorder=0)
    axm1a.plot([0, 0], [-1, 5], 'k:', linewidth=1, zorder=3)

    # Sets x labels
    axm1a.set_xticks([-0.15, 0, 0.15])
    # axm1a.set_xticklabels(['$\\hat{b}_{i}<$', '$\\beta$', '$<\\hat{b}_{i}$'])
    axm1a.set_xticklabels(['', '$\\beta$', ''])
    for l, a in zip(*(axm1a.get_xticklabels(), ['left', 'center', 'right'])):
        l.set_ha(a)

    # Plots the data and scatter
    for _, [xx, yy, cc] in bias_data[['from_ref', 'y', 'color']].iterrows():
        axm1a.plot([0, xx], [yy, yy], color=cc)
        axm1a.scatter(xx, yy, c=cc, marker='.',)


def _plot_covearge(ax0, cov_df):
    """Highlights the coverage"""
    plot_df = cov_df[['from_ref', 'y', 'color', 'width']]
    for _, [xx, yy, cc, xerr] in plot_df.iterrows():
        ax0.errorbar(x=xx, y=yy, xerr=xerr, color=cc, marker='.',
                     linewidth=1, capsize=3)
    ax0.plot([0, 0], [-1, 5], 'k:', linewidth=1, zorder=0)
    ax0.set_xlim(np.array([-0.15, 0.15]))
    ax0.set_xticks([0])
    ax0.set_xticklabels(['$\\beta$'])


def _annotate_wrap_bias(axm1b, str_, width_b=1.5):
    """Wrap sign label"""
    # Hides the x-axis
    axm1b.xaxis.set_tick_params(**hide_axes)
    axm1b.yaxis.set_tick_params(**hide_axes)
    # Sets the limits
    axm1b.set_xlim(0, 10)
    axm1b.set_ylim(-5, 5)

    # Add the arrows
    axm1b.annotate(
        str_, 
        xy=(0, 0),
        xytext=(0.8, 0),
        ha='left', va='center', size=8,
        arrowprops=dict(facecolor='black',
                         zorder=-1,
                         arrowstyle=f'-[, widthB={width_b}, lengthB=0.25',
                         connectionstyle='arc',
                         # scale=0,
                         ),
        )
    sn.despine(ax=axm1b, left=True, right=True, top=True, bottom=True)


def _annotate_equation(axm1b, str_):
    """
    Adds the bias value
    """    
    # Cleans up the axes
    axm1b.text(7.75, 0, f'$\\frac{{1}}{{N}}$$\\sum({str_})=$', 
               ha='right', va='center', size=9)
    axm1b.text(4.25, -3, 'i=1', ha='center', va='center', size=5)
    axm1b.text(4.25, 2.5, 'N', ha='center', va='center', size=5)


def _annotate_value(axm1b, b, palette, limits):
    """
    Annotates the value
    """
    # Gets the colors
    bounds = np.linspace(*limits, 10)
    colors = [mpc.to_hex(c) for c in sn.color_palette(palette, 11)]
    lb = np.hstack([-100, bounds])
    ub = np.hstack([bounds, 100])
    color = [c for l, u, c in zip(*(lb, ub, colors))
             if (l < b) & (b <= u)][0]
    axm1b.text(10, -0.25, f'{b:1.2f}', color='k', 
               ha='right', va='center', size=9)
    axm1b.fill_between([8, 10], -2, 2, color=color, edgecolor='None', 
                       zorder=0)


def _plot_bias_axis(axm1a, bias_data, example_effect=-0.5):
    """
    Plots the bias data, highlighting the regions of rounding
    """
    axm1a.set_xlim(-0.15, 0.15)
    xd = np.diff(axm1a.get_xlim())[0]
    yd = np.diff(axm1a.get_ylim())[0]

    # Plots the background for the bias
    background = {'#a7d0e4': [-0.3, -0.1 * np.abs(example_effect)],
                  '#f7b799': [0.3, 0.1 * np.abs(example_effect)],
                  '#f7f6f6': np.array([-0.1, 0.1]) * np.abs(example_effect),
                  }
    for cc, xx in background.items():
        axm1a.fill_between(xx, -1, 5, facecolor=cc, edgecolor='None', zorder=0)
    axm1a.plot([0, 0], [-1, 5], 'k:', linewidth=1, zorder=3)

    # Sets x labels
    axm1a.set_xticks([-0.15, 0, 0.15])
    # axm1a.set_xticklabels(['$\\hat{b}_{i}<$', '$\\beta$', '$<\\hat{b}_{i}$'])
    axm1a.set_xticklabels(['', '$\\beta$', ''])
    for l, a in zip(*(axm1a.get_xticklabels(), ['left', 'center', 'right'])):
        l.set_ha(a)

    # Plots the data and scatter
    for _, [xx, yy, cc] in bias_data[['from_ref', 'y', 'color']].iterrows():
        axm1a.plot([0, xx], [yy, yy], color=cc)
        axm1a.scatter(xx, yy, c=cc, marker='.',)


def _plot_covearge(ax0, cov_df):
    """Highlights the coverage"""
    plot_df = cov_df[['from_ref', 'y', 'color', 'width']]
    for _, [xx, yy, cc, xerr] in plot_df.iterrows():
        ax0.errorbar(x=xx, y=yy, xerr=xerr, color=cc, marker='.',
                     linewidth=1, capsize=3)
    ax0.plot([0, 0], [-1, 5], 'k:', linewidth=1, zorder=0)
    ax0.set_xlim(np.array([-0.15, 0.15]))
    ax0.set_xticks([0])
    ax0.set_xticklabels(['$\\beta$'])

