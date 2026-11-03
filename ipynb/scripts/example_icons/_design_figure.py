import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from ._vary_param_icons import (show_icc_range,
                                       _plot_effect_sizes,
                                       )
from ._vary_pop_dist import (show_vary_population,
                                    show_vary_size,
                                    show_vary_on_exposure,
                                    show_vary_on_outcome,
                                    )
from scripts._exp_summary import show_experiment
from scripts.example_icons._utils import hide_axes

def show_design_figure(fig, example_data00, example_data01, 
                       example_data02, example_data03, example_data04, 
                       example_data05, goal_iccs, smoking_effect, 
                       models, tech_order, tidy_short, 
                       cid='cohort_03',  seed_icc=0.15, seed_effect=-0.5):
    """
    Highlights the design figure
    """
    # Splits dat out into subfigures
    figsc, figprm, figexp = \
        fig.subfigures(3, 1, height_ratios=[1.65, 1, 1.5], 
                       facecolor='None')

    # Plots the scenario cartoons
    figvp, figvs, figve, figvo = figsc.subfigures(1, 4)
    show_vary_population(figvp, example_data00, example_data01)
    show_vary_size(figvs, example_data00, example_data02)
    show_vary_on_exposure(figve, example_data00, example_data03, 
                          smoking_effect)
    show_vary_on_outcome(figvo, example_data00, example_data04)

    # Plots ICC and beta schemas
    fig_icc, fig_eff, _ = \
        figprm.subfigures(1, 3, width_ratios=[2.5, 2.5, 1], 
                          facecolor='None')
    show_icc_range(fig_icc, goal_iccs, line_kws=None)
    _plot_effect_sizes(fig_eff, mod_sd=0.15)

    # Plots the example expeirment
    show_experiment(figexp, example_data05, cid, models, tech_order, tidy_short)
    figexp2 = figexp.subfigures(1,1, facecolor='None')
    axexp2 = figexp2.add_subplot(1,1,1, facecolor='None')
    axexp2.xaxis.set_tick_params(**hide_axes)
    axexp2.yaxis.set_tick_params(**hide_axes)
    axexp2.set_position((0.0, 0.01, 1.0, 0.98))
