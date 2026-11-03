import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn

from ._utils import (hide_axes,
                     _plot_icc_wedge)

from ._stacked_forest_inset import plot_forest_stack 
from ._evil_tree_highlight import highlight_evil_tree
from ._cohort_plots import _build_scene_grid_vert
from ._heuristic_example import plot_bias_coverage_inset


def highlight_heuristics(fig, df, example_data, scenario, 
                         example_effect):
    """
    Builds a heuristic summary plot
    """
    [[figsct, figexpt, figheut], [figsc, figexpi, figheu]] = \
    fig.subfigures(2, 3, 
                   height_ratios=[0.1, 1.25], 
                   width_ratios=[1, 1.25, 6 - (1 + 1.25)],
                   facecolor='None')
    figsca, figsci = figsc.subfigures(2, 1, height_ratios=[0.75, 0.5])
    figheuet, figheueq = figheu.subfigures(1, 2, width_ratios=[1.25, 2.5])

    _build_scene_grid_vert(figsca, [1, 1, 1, 1])
    _sub_icc_wedge(figsci)
    plot_forest_stack(figexpi, df, example_data, 0.15, 
                  error_kws=dict(markersize=3, linewidth=0.75))
    highlight_evil_tree(figheuet, df.xs(scenario, level='scenario'), 
                        example_data, example_effect=example_effect)
    plot_bias_coverage_inset(figheueq, df, scenario, 
                             seed_effect=example_effect)

    _add_sub_title(figsct, 'Scenario')
    _add_sub_title(figexpt, 'Experiments')
    _add_sub_title(figheut, 'Example Heuristics')


def _add_sub_title(fig, title, color='#dbdbdb', size=8):
    """
    Adds title figure
    """
    axct = fig.add_subplot(1,1,1, facecolor=color)
    axct.set_position((0.05, 0.05, 0.9, 0.9))
    axct.xaxis.set_tick_params(**hide_axes)
    axct.yaxis.set_tick_params(**hide_axes)
    sn.despine(fig, left=True, right=True, top=True, bottom=True)
    axct.text(0.5, 0.5, title, ha='center', va='center', size=size)
    

def _sub_icc_wedge(figsci):
    """Makes little ICC wedge"""
    ax_sci = figsci.add_subplot(1,1,1, facecolor='None')
    ax_sci.xaxis.set_tick_params(**hide_axes)
    ax_sci.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax_sci, left=True, right=True, top=True, bottom=True)
    _plot_icc_wedge(ax_sci)
    ax_sci.text(ax_sci.get_xlim()[0], ax_sci.get_ylim()[1], 'ICC range',
                size=8, ha='left', va='top')