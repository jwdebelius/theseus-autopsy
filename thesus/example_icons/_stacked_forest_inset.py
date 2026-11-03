import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import ticker

from ._utils import (hide_axes, 
                     _plot_betas, 
                     _fit_models_no_filt,
                     r_tidy_short)


def _convert_sub_data_to_fit_res(params, example_data, models=None, tech_order=None,
                                 iccs=[0.01, 0.4], 
                                 color='k', eff_seed=-0.5,
                                 scenario=15,
                                 ):
    """
    Extracts example data for stacked plots
    """
    fit_res, icc = _fit_models_no_filt(example_data)
    fit_res['icc'] = icc
    # Gest the sub data out
    sub_cols = ['icc', 'param', 'bse', 'ci_lo', 'ci_hi', 'p-value', 
                'sig']
    sub_data = params.xs(0, level='iter').copy()
    sub_data = sub_data.xs(eff_seed, level='seed_effect').copy()
    sub_data = sub_data.xs(scenario, level='scenario').copy()
    sub_data.reset_index(['filter', 'adjustment'], 
                         inplace=True)
    sub_data = sub_data.loc[[0.01, 0.4], sub_cols].copy()
    sub_data.sort_values('icc', ascending=False, inplace=True)

    # cleans up the sub data
    sub_data['width'] = sub_data['ci_hi'] - sub_data['ci_lo']
    sub_data['color'] = color
    sub_data['facecolor'] = sub_data['color'].mask(sub_data['sig'] == 0, 'w')

    sub_data['y'] = \
    sub_data.index.to_frame()['model'].replace({'ignore': '5', 
                                                'risk': '4',
                                                'fixed': '3', 
                                                'gee': '2', 
                                                'lme': '1', 
                                                'meta': '0'})
    sub_data['y'] = sub_data['y'].astype(int)
    sub_data.reset_index(level=['seed_icc', 'icc_group'], 
                         inplace=True, 
                         drop=True)

    return pd.concat(axis=0, objs=[sub_data, fit_res])

def _plot_inset_res(figp, icc, fres, tidy_short=None, error_kws=None):
    """
    Plots a data inset
    """
    # Sets up the background axis
    ax0 = figp.add_subplot(1,1,1, facecolor='None')
    ax0.xaxis.set_tick_params(**hide_axes)
    ax0.yaxis.set_tick_params(**hide_axes)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.fill_between([0, 1], 0, 1, facecolor='w', alpha=1)
    ax0.set_position((0, 0, 1, 1))

    # Plots the forest plot
    axp = figp.add_subplot(1,1,1)
    _plot_betas(axp, icc, fres, tidy_short, labelicc=True, 
                error_kws=error_kws)
    axp.xaxis.set_tick_params(labelsize=8)
    axp.yaxis.set_tick_params(labelsize=8)
    axp.xaxis.get_label().set_visible(False)
    axp.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.1f}'))


def plot_forest_stack(fig, params, example_data, iccs=[0.01, 0.4], 
                      color='k', eff_size=-0.5, scenario=15,
                      error_kws=None,):
    """
    Plots the forest plot insets
    """
    print(error_kws)
    print('stacked_forest_plot')
    # Sets up the subdata
    fit_res, icc = _fit_models_no_filt(example_data)
    plot_data = _convert_sub_data_to_fit_res(params, 
                                             example_data, 
                                             iccs=iccs,
                                             color=color,
                                             eff_seed=eff_size, 
                                             scenario=scenario)

    # Sets up the sub figures
    [fig0, _], [_, _] = \
        fig.subfigures(2, 2, width_ratios=[8, 2], height_ratios=[8, 2], 
                       facecolor='None')
    [_, _, _], [_, fig1, _], [_, _, _] = \
        fig.subfigures(3, 3, width_ratios=[1, 8, 1], height_ratios=[1, 8, 1], 
                      facecolor='None')
    [_, _], [_, fig2] = \
        fig.subfigures(2, 2, width_ratios=[2, 8], height_ratios=[2, 8], 
                       facecolor='None')
    sfigs = [fig0, fig1, fig2]

    hard_pos = (0.5491328732252023, 0.23968557027243687,               
                0.34872463322964153, 0.6188331243260576)
    # Plots stacked forest plots 
    for figp, (icc, fres) in zip(*(sfigs, plot_data.groupby('icc', sort=False))):
        _plot_inset_res(figp, icc, fres, error_kws=error_kws)
        figp.axes[-1].set_position(hard_pos)

