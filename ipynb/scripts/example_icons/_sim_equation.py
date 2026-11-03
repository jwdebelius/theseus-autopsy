import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import ticker

from ._utils import (_make_ref_table,
                            _plot_gauss_curve,
                            hide_axes
                            )


def build_sim_sum_icons(fig, sub_data, sim_colors=None, **exp_scatter_kws):
    """
    Builds a line plots showing the sub data 
    """
    # Sets up the defaults and fixed them for the data
    sim_colors_def, sim_titles, exp_scatter_kwargs = \
        _check_sim_sum_icons_defaults(sub_data, sim_colors, exp_scatter_kws)
    sub_data = exp_scatter_kwargs['data']

    # Sets up the reference background for the plots. This *only* works
    # if we dont have a constrained layout, so thats what we're working with
    gs = fig.add_gridspec(
        4, len(sub_data.columns) + 1, 
        width_ratios=np.hstack([np.ones(len(sub_data.columns) - 1), 1/3, 1]),
        height_ratios=[0.25, 2, 0.25, 2]
        )

    # Plots the strip scatter
    ### Outcome scatter
    # Plots the outcome
    ax_o = fig.add_subplot(gs[-1, -1])
    _plot_strip_scatter(ax_o, 'y', sim_colors_def, exp_scatter_kwargs, sim_titles)
    ax_o.yaxis.get_label().set_visible(True)
    ax_o.yaxis.set_label_position("right")
    ax_o.yaxis.set_tick_params(labelright=True, left=False)

    ### Exposure scatter
    ax_row = []
    for i, col_ in enumerate(sub_data.columns[:-1]):
        ax_ = fig.add_subplot(gs[-1, i], sharey=ax_o)
        _plot_strip_scatter(ax_, col_, sim_colors_def, exp_scatter_kwargs, 
                            sim_titles)
        ax_.yaxis.get_label().set_visible(i == 0)
        ax_.yaxis.set_tick_params(labelleft=(i == 0))
        ax_row.append(ax_)

    # # ### Plots the Icons
    icon_f = [_plot_covar_icon,
              _plot_exposure_icon,
              # _plot_cohort_type,
              _plot_select_ct,
              _plot_cohort_random,
              _plot_indv_error]
    for i, f_ in enumerate(icon_f):
        ax_ = fig.add_subplot(gs[1, i], facecolor='None')
        # ax_.xaxis.set_tick_params(**hide_axes)
        # ax_.yaxis.set_tick_params(**hide_axes)
        # ax_.text(0.5, 0.5, sub_data.columns[i], ha='center', va='center', size=8)
        f_(ax_, sub_data, sim_colors=sim_colors_def)
        if i == 1:
            ax_.set_xlim(fig.axes[-2].get_xlim())
            ax_.set_ylim(fig.axes[-2].get_ylim())

    # ### Sets up the equation 
    ax_eq = fig.add_subplot(gs[:, :], facecolor='None')
    ax_eq.yaxis.set_tick_params(**hide_axes)
    ax_eq.xaxis.set_tick_params(**hide_axes)
    plus_pos = [0.15, 0.306, 0.463, 0.618]
    for xx in plus_pos:
        ax_eq.text(xx, 0.44, '+', size=12, ha='center', va='center', color='k')

    ax_t = fig.add_subplot(gs[-1, -2], facecolor='None')
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(0.5, 0.5, '=', ha='center', va='center', size=15)
    ax_t.set_title('=',  pad=4)
    ax_t.xaxis.set_tick_params(**hide_axes)
    ax_t.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax_t, left=True, right=True, top=True, bottom=True)
    sn.despine(ax=ax_eq, left=True, right=True, top=True, bottom=True)

def _check_sim_sum_icons_defaults(sub_data, sim_colors, exp_scatter_kws):
    """
    Checks the formatting defaults
    """
    # Sets up the plotting colors
    sim_colors_def = {
        'fixed': '#919191',
        'exposure': '#e41a1c',
        'cohort_fixed': 'purple',
        'cohort_random': sn.color_palette(palette='Dark2', n_colors=3)[1],
        'error': sn.color_palette(palette='Dark2', n_colors=3)[0],
        'y': 'k',
        }
    if isinstance(sim_colors, dict):
        sim_colors_def.update(sim_colors)

    # Sets the axis title
    sim_titles = {'fixed': '$\\sum{\\beta_{k}w_{ik}}$',
                  'exposure': '${\\beta_{\\mathsf{exp}}}e_{i}$',
                  'cohort_fixed': '$\\beta_{\\mathsf{ct}}{ct_{i}}$',
                  'cohort_random': '$\\gamma_{i}$',
                  'error': '$\\epsilon$',
                  'y': '$z$'
                  }
    columns = ['fixed',  'exposure', 'cohort_fixed', 'cohort_random', 
               'error', 'y']

    # Sets the scatter parameters
    exp_scatter_kwargs = dict(
        data=sub_data[columns],
        jitter=0.25,
        edgecolor='None',
        s=1,
        )
    exp_scatter_kwargs.update(exp_scatter_kws)

    return sim_colors_def, sim_titles, exp_scatter_kwargs


def _plot_strip_scatter(ax_, col_, sim_colors, exp_scatter_kwargs, sim_titles):
    """
    Plots a strip plot scatter
    """
    # Scatter strip
    sn.stripplot(y=col_, ax=ax_,
                 color=sim_colors.get(col_, 'k'),
                 **exp_scatter_kwargs,
                 )
    # Formats the axis
    ax_.xaxis.set_tick_params(**hide_axes)
    ax_.xaxis.get_label().set_visible(False)
    
    ax_.yaxis.set_tick_params(left=True, 
                              right=False, 
                              labelleft=False, 
                              labelright=False, 
                              gridOn=True,
                              labelsize=8)
    ax_.set_ylabel('z-birth weight')
    ax_.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>2.0f}'))
    ax_.yaxis.get_label().set_visible(False)
    ax_.set_title(sim_titles.get(col_, col_.replace('_', '\n')), pad=4)
    

def _plot_cohort_random(ax_ce, sub_data, sim_colors, **kwargs):
    """Plots a gaussian curve with the random cohort effect"""
    _plot_gauss_curve(ax=ax_ce,
                      color=sim_colors.get('cohort_random', 'k'), 
                      ylo=-0.4, 
                      yhi=1.4,
                      sigma_eq='$\\sigma^{2} \\propto \\mathrm{ICC}$')
    # Adds the point estimate value
    cg = sub_data['cohort_random'].unique()
    ax_ce.scatter(cg, scipy.stats.norm.pdf(cg), zorder=3, 
                      color=sim_colors.get('cohort_random', 'k'))


def _plot_indv_error(ax_ce, sub_data, sim_colors, **kwargs):
    """Plots a gaussian curve wtih the individual error"""
    _plot_gauss_curve(ax=ax_ce, color=sim_colors.get('error', 'k'), 
                     ylo=-0.4, yhi=1.4)


# def _plot_cohort_type(ax_tp, sub_data, **kwargs):
#     """Illustrates the type of cohort"""
#     type_ = (sub_data['cohort_fixed'].unique()[0] != 0) * 1
#     _select_cohort_type(axtp, type_, 
#                        color=sim_colors.get('cohort_fixed', 'purple')
#                        )
#     sn.despine(ax=axtp, left=True, right=True, top=True, bottom=True)


def _plot_select_ct(ax_tp, sub_data, sim_colors, color='purple', **kwargs):
    """
    Plots a cohort type selection
    """
    type_ = (sub_data['cohort_fixed'].unique()[0] != 0) * 1
    ax_tp.set_xlim(0, 1)
    ax_tp.set_ylim(-0.5, 2)

    text_kws = dict( size=14, color=color, va='center', ha='center')
    text_kws.update(kwargs)

    ax_tp.text(0.5, 1.5, s='Gen', alpha=1-0.5*(type_ == 1), **text_kws,)
    ax_tp.text(0.5, 0.5, s='NICU', alpha=1-0.5*(type_ == 0), **text_kws,)
    ax_tp.text(0.5, 1.0 + 0.1 * (type_ == 1), s='or', 
               size=10, color='k', ha='center', va='center')

    ax_tp.fill_between(x=np.array([0.125, 0.85]), 
                       y1=0.2 + (type_ == 0), 
                       y2=0.9 + (type_ == 0), 
                       facecolor='None', 
                       edgecolor=color)

    ax_tp.yaxis.set_tick_params(**hide_axes)
    ax_tp.xaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax_tp, left=True, right=True, top=True, bottom=True)


def _plot_covar_icon(axf1, sub_data, sim_colors, **kwargs):
    """Plots covariate table"""
    _make_ref_table(axf1, palette='Greys')
    axf1.set_xlim(-0.1025, 1.1025)
    axf1.set_ylim(-1.1, 0.65)
    sn.despine(ax=axf1, left=True, right=True, top=True, bottom=True)


def _plot_exposure_icon(axf2, sub_data, sim_colors, **kwargs):
    """Plot exposure table"""
    _make_ref_table(axf2, palette='Reds', n_cols=2, offset=0.3,)
    # axf2.set_xlim(axf1.get_xlim())
    # axf2.set_ylim(axf1.get_ylim())
    sn.despine(ax=axf2, left=True, right=True, top=True, bottom=True)



