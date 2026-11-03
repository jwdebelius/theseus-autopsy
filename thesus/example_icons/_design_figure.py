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
from ._exp_summary import show_experiment
from ._experiment_stripe import highlight_heuristics
from ._utils import (hide_axes,
                     r_tidy_short,
                     r_tech_order,
                     r_models,
                     )


def show_design_figure(fig, example_data00, example_data01, 
                       example_data02, example_data03, example_data04, 
                       example_data05, heur_df, goal_iccs, 
                       smoking_effect,
                       models=r_models, tech_order=r_tech_order, 
                       tidy_short=r_tidy_short, 
                       cid='cohort_03',  seed_icc=0.15, seed_effect=-0.5):
    """
    Highlights the design figure
    """
    # Splits dat out into subfigures
    figsct, figsc, figprm, figexp, figo = \
        fig.subfigures(5, 1,
                       height_ratios=[0.15, 1.65, 1, 1.5, 1.25],
                       # hspace=0,
                       # wspace=0,
                       facecolor='None')
    
    # Plots the scenario cartoons    
    figsct.text(0.5, 0.5, 'Study design parameters to vary in each scenario',
            size=10, ha='center', va='center')
    figsct.set_facecolor('#dbdbdb')
    figvp, figvs, figve, figvo = figsc.subfigures(1, 4)
    show_vary_population(figvp, example_data00, example_data01)
    show_vary_size(figvs, example_data00, example_data02)
    show_vary_on_exposure(figve, example_data00, example_data03, 
                          smoking_effect)
    show_vary_on_outcome(figvo, example_data00, example_data04)


    # Plots ICC and beta schemas
    fig_icc, fig_eff = \
        figprm.subfigures(1, 2, width_ratios=[3.5, 2.5], 
                              facecolor='None')
    show_icc_range(fig_icc, goal_iccs, line_kws=None)
    _plot_effect_sizes(fig_eff, mod_sd=0.15)

    # Plots the example experiment
    show_experiment(figexp, example_data05, cid, models, tech_order, tidy_short)
    figexp2 = figexp.subfigures(1,1, facecolor='None')
    axexp2 = figexp2.add_subplot(1,1,1, facecolor='None')
    axexp2.xaxis.set_tick_params(**hide_axes)
    axexp2.yaxis.set_tick_params(**hide_axes)
    axexp2.set_position((0.0, 0.01, 1.0, 0.98))


    # Summarizes the heuristics for the experiment
    highlight_heuristics(figo, heur_df, example_data05, 15, 
                         seed_effect)

    # Adds labels
    figvp.text(0.025, 1-0.025, 'A', size=12, ha='left', va='top')
    figvs.text(0.025, 1-0.025, 'B', size=12, ha='left', va='top')
    figve.text(0.025, 1-0.025, 'C', size=12, ha='left', va='top')
    figvo.text(0.025, 1-0.025, 'D', size=12, ha='left', va='top')
    figprm.text(0, 1-0.025, 'E', size=12, ha='left', va='top')
    fig_eff.text(0, 1-0.025, 'F', size=12, ha='left', va='top')
    figexp.text(0, 1-0.025, 'G', size=12, ha='left', va='top')
    figo.subfigs[3].text(0, 1, 'H', size=12, ha='left', va='top')
    figo.subfigs[-1].text(0.05, 1, 'I', size=12, ha='left', va='top')




    