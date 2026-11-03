import itertools as it

import pandas as pd
import numpy as np

# Statistical tests
import statsmodels.formula.api as smf
from scipy.stats import t as t_dist
import scipy.stats
import patsy

# Displaying my data
from matplotlib import rcParams
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn

from thesus import bw_ana
from ._utils import (_summarize_kde,
                     _summarise_histogram,
                     _wrap_categorical_bars,
                     _plot_betas,
                     hide_axes,
                     _fit_models_no_filt,
                     )


def summarize_example_sim_experiment(fig, data, models, tech_order, 
                                     tidy_short, scenario_bool, seed_eff, 
                                     example_icc, rs=0, eff_mod=None):
    """
    Generates a pooled figure describing a simulation experiment describing
    the cohort composition and pooled results
    """
    [fig0, fig1, fig2] = fig.subfigures(3, 1, 
                                        height_ratios=[0.75, 3, 2], 
                                        facecolor='None')
    [figl, figr] = fig2.subfigures(1, 2, 
                                   width_ratios=[3.5, 2.5], 
                                   facecolor='None')
    summarize_simulated_cohorts(fig1, data)
    plot_pooled_betas(figr, data, models, tech_order, tidy_short)
    combine_violins(figl, data)
    summarize_simulation_parameters(fig0, scenario_bool, seed_eff, rs, 
                                    example_icc, eff_mod)
    fig0.text(0.0, 1, 'A', size=12, ha='left', va='top')
    fig1.text(0.0, 1, 'B', size=12, ha='left', va='top')
    figl.text(0.0, 1, 'C', size=12, ha='left', va='top')
    figr.text(0.0, 1, 'D', size=12, ha='left', va='top')
    

def plot_pooled_betas(fig, data,  models, tech_order, tidy_short, 
                      labelicc=True):
    """
    Plots the pooled beta under different models for hte dataset
    """
    fit_res, icc = _fit_models_no_filt(data, models, tech_order)
    axr = fig.add_subplot(1,1,1)
    _plot_betas(axr, icc, fit_res, tidy_short, labelicc)


def combine_violins(figl, data):
    """
    Combines the data into boxplots by smoking status
    """
    gsl = figl.add_gridspec(1, 3)
    axa = figl.add_subplot(gsl[0])
    axg = figl.add_subplot(gsl[1], sharey=axa, sharex=axa)
    axn = figl.add_subplot(gsl[2], sharey=axa, sharex=axa)
    plot_pooled_violin(figl.axes, data)


def build_per_cohort_fit(example_data):
    """
    Fits the per cohort data with an ols and extracts the parameters

    Parameters
    ----------
    example_data: pd.DataFrame
        Simulated data with columns smoking, race, ethncity, sex, m_age, 
        year, the simulated birthweight in y

    Returns
    -------
    pd.DataFrame
    """
    # Fits each of the cohorts if they have 2 levels of smokingd data
    example_fits = {
    (type_, cid): smf.ols(
        'y ~ smoking + C(race) + C(ethnicity) + C(sex) + m_age + year',
        data=df).fit()
        for (type_, cid), df in example_data.groupby(['cohort_type', 'cohort'])
        if (len(df['smoking'].unique()) > 1)
        }
    # Pulls out the smoking specific parameter 
    example_params = {cohort_id: _extract_cohort_fit(fit) 
                      for cohort_id, fit in example_fits.items()}
    example_params = pd.DataFrame.from_dict(example_params, orient='index')

    example_params.index.set_names(['cohort_type', 'cohort'], inplace=True)
    example_params['width'] = example_params['ci_hi'] - example_params['param']
    example_params.sort_values(['cohort_type', 'param'], 
                               inplace=True, 
                               ascending=True)

    return example_params


def plot_pooled_violin(axes, example_data, box_kws=None, violin_kws=None):
    """
    Makes a violin plot summarizing the data by cohort type
    """
    boxplot_kwargs = dict(#width=0.25,
                          # boxprops={'facecolor': 'w'},
                          #whis=0,
                          #fliersize=0, 
                          linewidth=1, 
                          palette='Greys',
                          legend=False,
                          )
    if box_kws is not None:
        boxplot_kwargs.update(box_kws)
    violin_kwargs = dict(inner=None, 
                         #palette='Greys',
                         linewidth=1,
                         offset=0,
                         zorder=10,
                         # width=1,
                         cut=0,)
    if violin_kws is not None:
        violin_kwargs.update(violin_kws)
    value_groups = [
        axes,
        [['0', '1', 0, 1], ['0', 0], ['1', 1]],
        ['All', 'General', 'NICU'],
        ]
    for i, (ax_dis, vals, title) in enumerate(zip(*(value_groups))):
        sub_data = \
            example_data.loc[example_data['cohort_type'].isin(vals)].copy()
        sn.boxplot(x='smoking', hue='smoking', y='y', data=sub_data, ax=ax_dis,
                   **boxplot_kwargs)
        # pt.half_violinplot(x='smoking', y='y', data=sub_data, ax=ax_dis,
        #                    **violin_kwargs)

        ax_dis.set_ylabel('Birth weight z-score')
        # xmin, xmax = ax_dis.get_xlim()
        # xdiff = xmax - xmin
        # ax_dis.set_xlim(np.array([xmin, xmax]) - 0.05 * xdiff)
        ax_dis.yaxis.get_label().set_visible(i == 0)
        ax_dis.yaxis.set_tick_params(labelleft=(i == 0))
        ax_dis.xaxis.get_label().set_visible(True)
        ax_dis.xaxis.set_tick_params(bottom=False, length=0)
        ax_dis.set_title(title)


def calculate_example_scale(example_params, pass_thresh=3.5, lim_scale=0.25):
    """
    Determines whether parameters need to be plotted with outliers
    """
    def _z_score(x):
        return (x - x.mean()) / x.std()

    example_params = example_params.copy().dropna()
    # Gets the z-nroamlized variance and parameters so we can exclude values
    # oustide a threshhold
    example_params['z_bse'] = (_z_score(example_params['bse']))
    example_params['z_param'] =  (_z_score(example_params['param']))
    example_params['pass_bse'] = \
        np.absolute(example_params['z_bse']) <= pass_thresh
    example_params['pass_params'] = \
        np.absolute(example_params['z_param']) <= pass_thresh

    # Figures out the minimum and maximum based ont he axis limits
    pass_bse_q = \
        np.absolute(example_params.loc[example_params['pass_bse'], 'bse'])
    pass_bse_q = pass_bse_q.quantile(0.75)
    all_pass = example_params[['pass_bse', 'pass_params']].all(axis=1)

    lim_min = min(
        example_params.loc[all_pass, 'ci_lo'].min(), 
        example_params.loc[example_params['pass_params'], 'param'].min() - 
        pass_bse_q
        )
    lim_max = max(
        example_params.loc[all_pass, 'ci_hi'].max(),
        example_params.loc[example_params['pass_params'], 'param'].max() + 
        pass_bse_q
        )

    # Rounds the values so they're pretty
    lim_min = np.round(np.floor(lim_min / lim_scale), 0) * lim_scale
    lim_max = np.round(np.ceil(lim_max / lim_scale), 0) * lim_scale
    lim_dif = lim_max - lim_min

    pass_vals = example_params['param']

    lo_z = (lim_min - pass_vals.mean()) / pass_vals.std()
    hi_z = (lim_max - pass_vals.mean()) / pass_vals.std()

    example_params['out_window'] = \
        (example_params['z_param'] < lo_z) * -1 + \
        (example_params['param'] > hi_z) * 1

    below = example_params.loc[example_params['param'] < lim_min]
    above = example_params.loc[example_params['param'] > lim_max]

    nom_scaler = 1 / (4 - (len(below) > 0) - len(above > 0))
    bse2 = np.absolute(example_params.loc[example_params['pass_bse'], 'bse'])
    bse2 = bse2.mean()

    # Handles the window below the minimum for low outliers.
    # If there are no low outliers, we don't need an axis, so we drop that
    # If tehre's one low outlier, we center it in the window. Then, we check
    # if the window overlaps the existing window. If it does, we widen the 
    # current limits, and just drop the low window.
    # If there's more than one value, then we calculate a window starting wtih
    # the minimum value and allowing a 15% bse window to pad for extending the 
    # CI. We _should_ put in checks around the limits
    lo_kwargs = (below['param'], lim_min, lim_max, nom_scaler, pass_bse_q)
    left_, lo_lims, main_lims = _check_coords_no_outliers(*lo_kwargs)
    hi_kwargs = (above['param'], *main_lims, nom_scaler, pass_bse_q)
    right_, hi_lims, main_lims = _check_coords_no_outliers(*hi_kwargs)

    return left_, right_, main_lims, lo_lims, hi_lims


def summarize_simulated_cohorts(fig, data):
    """
    A graphical cohort summary for each of the simulated data sets
    """

    # Checks the number of 
    
    gs = fig.add_gridspec(1, 9, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.5])
    
    
    # Plots the group size 
    ax0 = fig.add_subplot(gs[0], facecolor='None')
    _plot_cohort_summary(ax0, data)
    ax0.set_xlim(0, max(ax0.get_xlim()[1], 500))
    ax0.set_ylim(-1, data['x'].max() + 1)
    ax0.set_xlabel('Cohort\nSize', size=8)
    ax0.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
    ax0.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.0f}'))
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax0.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax0, left=True, right=True, top=True, bottom=False)

    # Plots the categorical data
    _wrap_categorical_bars(fig=fig, gs=gs, data=data)
    
    # Plots the categorical histograms
    _summarise_histogram(fig, gs, data)
    
    # Plots the outcome
    _summarize_kde(fig, gs, data) 

    # # Generates the forest plot
    ax = fig.add_subplot(gs[-1], sharey=ax0)
    _build_forest_plot(ax, data)



def _build_forest_plot(ax, data):
    """
    Assembles the forest plots
    """
    params = build_per_cohort_fit(data)
    lim_min = -1.6
    lim_max = 0.6
    # left_, right_, main_lims, lo_lims, hi_lims = \
    #     calculate_example_scale(params, lim_scale=0.85)
    # (lim_min, lim_max) = main_lims
    params['x'] = params.index.to_frame()['cohort'].replace(
        data.groupby('cohort')['x'].first().astype(str)).astype(float)
    params['color'] = \
        params.index.to_frame()['cohort_type'].replace({
            '0': mpc.to_hex(sn.color_palette(n_colors=3)[1]),
            '1': mpc.to_hex(sn.color_palette(n_colors=3)[0]),
            0: mpc.to_hex(sn.color_palette(n_colors=3)[1]),
            1: mpc.to_hex(sn.color_palette(n_colors=3)[0]),
        })
    params['facecolor'] = \
        params['color'].mask(params['p_value'] >= 0.05, 'w')

    for (c, fc), pdf in params.groupby(['color', 'facecolor']):
        ax.errorbar(y='x', x='param', xerr='width', 
                    data=pdf,
                    linestyle='',
                    marker='D',
                    capsize=3,
                    markersize=4,
                    color=c,
                    markerfacecolor=fc,
                    markeredgewidth=1)
    if (params['ci_lo'] < lim_min).any():
        too_lo = params.loc[(params['ci_lo'] < lim_min)].copy()
        ax.scatter(
            y=too_lo['x'],
            x=too_lo['x'] * 0 + (lim_min + (lim_max - lim_min) * 0.0225), 
            color='k',
            edgecolor='None',
            zorder=3,
            marker='<')
    if (params['ci_hi'] > lim_max).any():
        too_hi = params.loc[(params['ci_hi'] > lim_max)].copy()
        ax.scatter(
            y=too_hi['x'], 
            x=too_hi['x'] * 0 + lim_max - (lim_max - lim_min) * 0.0225, 
            color='k',
            edgecolor='None',
            zorder=3,
            marker='>')
    ylim = ax.get_ylim()
    ax.plot([0, 0], ylim, 'k-', linewidth=1)
    ax.plot([-0.5, -0.5], ylim, color='k', linestyle='dotted', linewidth=1)
    ax.set_ylim(ylim)
    ax.set_xlim(lim_min, lim_max)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
    ax.set_xlabel('$\\beta_{smoking}$', size=8)
    ax.set_title("ICS\nestimate", size=10)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)

            
def _plot_cohort_summary(ax, data, colors=None):
    """
    Summarizes the size of the estimated cohort
    """
    if colors is None:
        colors = {1 - i: mpc.to_hex(c) 
                  for i, c in enumerate(sn.color_palette(n_colors=2))}
    counts = data.groupby(['cohort', 'cohort_type'])['x'].describe()
    ax.barh(y=counts['mean'], 
            width=counts['count'], 
            left=0,
            height=0.8,
            color=counts.index.to_frame()['cohort_type'].replace(colors)
            )

def _summarize_categorical(data, cat_col, grouper='cohort', ref_='x'):
    """
    Tabulates group size and proprotion for categorical data 
    """
    counts = data.groupby([cat_col, grouper])[ref_]
    counts = counts.count().reset_index()
    counts.rename(columns={ref_: 'counts', 
                           cat_col: 'groups',
                           grouper: 'cohort'}, 
                  inplace=True)
    group_counts = data[grouper].value_counts().astype(str).to_dict()
    counts['group_size'] = counts['cohort'].replace(group_counts).astype(int)
    counts['percent'] = counts['counts'] / counts['group_size']
    counts['column'] = cat_col

    counts = counts[['column', 'groups', 'cohort', 
                     'group_size', 'counts', 'percent']]

    return counts


def _fit_kde(x):
    """
    Fits a 1D gaussian kernel density estimate with cut 0
    """
    xmin = x.min()
    xmax = x.max()
    xpred = np.linspace(xmin, xmax, 100)
    kernel = scipy.stats.gaussian_kde(x)
    den_pred = kernel(xpred)

    return np.vstack([xpred, den_pred])


def _extract_cohort_fit(fit_):
    """
    Extracts the cohort fit information
    """
    params = pd.concat(
        axis=1, 
        objs=[fit_.params, fit_.bse, fit_.conf_int(), fit_.pvalues]
    )
    params.columns = ['param', 'bse', 'ci_lo', 'ci_hi', 'p_value']
    if 'smoking' in params.index:
        return params.loc['smoking']
    else:
        return params.loc['Intercept'] * 0 + np.nan


def _check_coords_no_outliers(params, lim_min, lim_max, nom_scaler, qbse):
    """
    If there's noting in the outliers, then we skip over that
    """
    return False, (lim_min, lim_max), (lim_min, lim_max)


def _plot_effect_box_no_mod(fig_eff, seed_eff=-0.5, rs=0, eff_mod=dict()):
    """
    Plots the effect size
    """
    ax_eff = fig_eff.add_subplot(1,1,1, facecolor='None')
    ax_eff.xaxis.set_tick_params(**hide_axes, gridOn=False)
    ax_eff.yaxis.set_tick_params(**hide_axes, gridOn=False)
    ax_eff.set_xlim(0, 1)
    ax_eff.set_ylim(-0.5, 4.5)
    ax_eff.text(0.05, 3.5, 'Effect size', ha='left', va='center', size=8)
    ax_eff.text(0.05, 0.5, 'Random effect $\\sigma$', ha='left', va='center', size=8)
    ax_eff.text(0.05, 2.5, 'Effect modification', ha='left', va='center', size=8)
    ax_eff.text(0.85, 3.5, f'{seed_eff:>3.1f}sd', ha='center', va='center', size=8)
    if rs == 0:
        ax_eff.text(0.85, 0.5, '0', ha='center', va='center', size=8)
    else:
        ax_eff.text(0.85, 0.5, f'{rs:1.2f}', ha='center', va='center', size=8)
    ax_eff.text(0.85, 2.5, 'no', ha='center', va='center', size=8)
    
    ax_eff.set_position((0.05, 0.1, 0.9, 0.8))


def summarize_simulation_parameters(fig_seed, scenario_bool, seed_eff,
                                    rs, example_icc, eff_mod=None):
    """
    Summarizes the scenario parameters. Intended for a 6x0.75 figure
    """
    _, fig_scene, fig_eff, fig_icc, _ = \
        fig_seed.subfigures(1, 5, 
                            width_ratios=[1.75 / 2, 1, 1.5, 1.75, 1.75 / 2], 
                            facecolor='None')
    _build_scene_grid_vert(fig_scene, scenario_bool)
    _plot_seed_icc(fig_icc, example_icc)
    if eff_mod is None:
        _plot_effect_box_no_mod(fig_eff, seed_eff=seed_eff, rs=rs)
    else:
        _plot_effect_box_mod(fig_eff, seed_eff, rs, eff_mod)



def _plot_seed_icc(fig_icc, example_icc):
    """
    Plots the ICC on a scale
    """
    ax_icc = fig_icc.add_subplot(1,1,1, facecolor='None')
    ax_icc.xaxis.set_tick_params(**hide_axes)
    ax_icc.yaxis.set_tick_params(**hide_axes)
    ax_icc.set_xlim(-0.075, 1.075)
    ax_icc.set_ylim(0, 1.05)
    ax_icc.fill_between(x=np.linspace(0, 1, 100),
                        y1=0.55,
                        y2=np.linspace(0, 1, 100) * 0.4 + 0.55,
                        edgecolor='k',
                        facecolor='#dbdbdb',
                        linewidth=1,
                       )
    ax_icc.plot([0, 1], [0.45, 0.45], 'k-', linewidth=1,)
    ax_icc.plot(np.vstack([np.linspace(0, 1, 5)] * 2),
                np.vstack([np.ones(5) * 0.45, np.ones(5) * 0.35]),
                color='k',
                linewidth=1,
                )
    for x in np.linspace(0, 1, 3):
        ax_icc.text(x, 0.325, f'{x:1.1f}', ha='center', va='top', size=8)
    
    ax_icc.scatter(x=example_icc, y=0.35, marker='^', color='r')
    ax_icc.text(x=example_icc, y=0.2, s=f'{example_icc:1.2f}', ha='center', va='top', size=10, color='r')
    ax_icc.plot([example_icc, example_icc], np.array([0, example_icc]) * 0.4 + 0.55,
                'r-')
    ax_icc.text(-0.05, 1, 'ICC', size=8, ha='left', va='top')
    ax_icc.set_position((0.05, 0.1, 0.9, 0.8))
    sn.despine(ax=ax_icc, left=True, right=True, top=True, bottom=True)


def _build_scene_grid_vert(fig_scene, scenario, marker='s'):
    """
    Builds a vertical reference grid for plotting 
    """
    ax_scene = fig_scene.add_subplot(1,4,1, facecolor='None')
    ax_scene.set_ylim(-0.5, 3.5)
    ax_scene.set_yticks(np.arange(0, 4)[::-1])
    ax_scene.set_yticklabels(['population', 'size', 'exposure', 'outcome'])
    ax_scene.xaxis.set_tick_params(**hide_axes)
    ax_scene.scatter(x=np.ones(4) * 0, 
                     y=np.arange(0, 4)[::-1],
                     marker='s',
                     s=None,
                     facecolor='None',
                     linewidth=1,
                     edgecolor='k',
                     )
    ax_scene.scatter(x=np.ones(4) * 0, 
                     y=np.arange(0, 4)[::-1],
                     marker=marker,
                     s=None,
                     facecolor=pd.Series(scenario).replace({0: 'w', 1: 'k'}).values,
                     linewidth=0,
                     edgecolor='None',
                     )
    ax_scene.yaxis.set_tick_params(left=False, labelleft=False,
                                   right=False, labelright=True,
                                   labelsize=8, length=0,
                                   gridOn=False)
    ax_scene.set_ylabel('Scenario')
    ax_scene.set_title('vary', size=10)
    ax_scene.set_position((0.2, 0.05, 0.15, 0.7))
    sn.despine(fig_scene, left=True, right=True, top=True, bottom=True)


def _plot_effect_box_mod(fig_eff, seed_eff=-0.5, rs=0, mod_eff=0):
    """
    Plots the effect size if the there's effect modification
    """
    left_kws = dict(ha='left', va='center', size=8)
    right_kws = dict( ha='center', va='center', size=8)
    ax_eff = fig_eff.add_subplot(1,1,1, facecolor='None')
    ax_eff.xaxis.set_tick_params(**hide_axes, gridOn=False)
    ax_eff.yaxis.set_tick_params(**hide_axes, gridOn=False)
    ax_eff.set_xlim(0, 1)
    ax_eff.set_ylim(-0.5, 4.5)
    ax_eff.text(0.05, 3.5, 'Effect Modification', **left_kws)
    ax_eff.text(0.10, 2.5, 'ICS type 0', **left_kws)
    ax_eff.text(0.10, 1.5, 'ICS type 1', **left_kws)
    ax_eff.text(0.05, 0.5, 'Random effect $\\sigma$', **left_kws)

    ax_eff.text(0.85, 3.5, 'yes', **right_kws)
    ax_eff.text(0.85, 2.5, f'{seed_eff:>3.1f}sd', **right_kws)
    ax_eff.text(0.85, 1.5, f'{mod_eff:>3.1f}sd', **right_kws)
    if rs == 0:
        ax_eff.text(0.85, 0.5, '0', **right_kws)
    else:
        ax_eff.text(0.85, 0.5, f'{rs:1.2f}', **right_kws)

    ax_eff.set_position((0.05, 0.1, 0.9, 0.8))

