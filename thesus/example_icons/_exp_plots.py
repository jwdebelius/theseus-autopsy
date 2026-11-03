r"""
This script will take simulated example data and generate plots showing the indivudal cohort effects and the results of the combined and/or filtered
cohorts
"""

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


from ._sim_process_pop import (build_sim_process_ref,
                                                    build_table_plot,
                                                    )
from ._sim_process_exp import (illustrate_exposure_res,
                                                    select_exposure,
                                                    )
from ._sim_equation import build_sim_sum_icons
from ._utils import hide_axes
from ._plot_example_cohort import plot_example_cohort
from ._plot_pooled_res import plot_pooled_res

from ._cohort_plots import (_fit_models_no_filt,
                                   summarize_simulation_parameters,
                                   _plot_betas,
                                   )

# import scripts.bw_ana as bw_ana


def plot_simulate_ics(figa, example_data, sim_values, scenario_bool, 
                      cohort_id, models, tech_order, smoking_effect,
                      seed_effect=-0.5, rs=0, example_icc=0.15, 
                      eff_mod=None, 
                      ):
    """
    Plots the example plot for the data and sub data
    """
    # Gets the appropriate data set
    fit_res, icc_, sub_data, cohort_data = \
        _extract_sim_cohort(example_data, sim_values, models, tech_order, 
                            cohort_id)

    # Sets up the figures for simulation
    fig_seed, fig_ind, fig_dep, fig_c = \
        figa.subfigures(4, 1, height_ratios=[0.75, 2.5, 2.25, 1.5], 
                        facecolor='None')

    # Describes the simulation parameters
    summarize_simulation_parameters(fig_seed, scenario_bool, seed_effect, rs, 
                                    example_icc, eff_mod)

    # Plots the ICS specific simulation
    fig_dem, fig_exp1 = fig_ind.subfigures(1, 2, width_ratios=[3.5, 2.5], 
                                           facecolor='None')    
    # Shows the simulation of the example data based
    figdl, fig_dm = fig_dem.subfigures(2, 1, height_ratios=[1.5, 1],
                                       facecolor='None')
    build_sim_process_ref(figdl, cohort_data)
    build_table_plot(fig_dm, cohort_data, n_head=3)

    # Shows the exposure data
    fig_exp, fig_rand = fig_exp1.subfigures(2, 1, height_ratios=[1.5, 1])
    select_exposure(fig_exp, cohort_data, smoking_effect)
    illustrate_exposure_res(fig_rand, cohort_data)

    # Shows the summation
    build_sim_sum_icons(fig_dep, sub_data)

    # Puts wrapper around the iteration
    figs, figw, _ = figa.subfigures(3, 1, height_ratios=[0.75, 4.75, 1.4])
    _wrap_ics_for_loop(figw)

    # Shows the experiment data pool
    fig_cl, fig_cr = fig_c.subfigures(1, 2, width_ratios=[4, 2], facecolor='None')
    plot_example_cohort(fig_cl, example_data, cohort_id)
    plot_pooled_res(fig_cr, fit_res=fit_res, icc=icc_)
    fig_cr.axes[1].set_position((0.35, 0.625, 0.55, 0.275))
    fig_cr.axes[2].set_position((0.35, 0.3, 0.55, 0.275))
    
    # Adds letters labeling the data
    fig_seed.text(0.1, 1, 'A', size=12, ha='left', va='top')
    fig_dem.text(0.01, 0.95, 'B', size=12, ha='left', va='top')
    fig_exp1.text(0, 0.95, 'C', size=12, ha='left', va='top')
    fig_dep.text(0.01, 1, 'D', size=12, ha='left', va='top')
    fig_cl.text(0.01, 1, 'E', size=12, ha='left', va='top')
    fig_cr.text(0.0825, 1, 'F', size=12, ha='left', va='top')


def _extract_sim_cohort(example_data, sim_values, models, tech_order, cid):
    """
    Extracts the fit, sim_components, and cohort id
    """
    fit_res, icc_ = _fit_models_no_filt(example_data, models, tech_order)
    sub_data = sim_values.loc[example_data['cohort'] == cid].copy()
    sub_data.drop(columns=['cohort_slope'], inplace=True)
    cohort_data = example_data.loc[example_data['cohort'] == cid]

    return fit_res, icc_, sub_data, cohort_data

    
def _wrap_ics_for_loop(figw):
    """
    Creates a box around the ICS data
    """
    axw = figw.add_subplot(1,1,1, facecolor='None')
    axw.xaxis.set_tick_params(**hide_axes)
    axw.yaxis.set_tick_params(**hide_axes)
    axw.set_position((0, 0.025, 1, 0.975))
    axw.set_xlim(0, 1)
    axw.set_ylim(0, 1)
    axw.set_yticks([0, 0.9825])
    axw.set_xticks([0, 1])
    axw.text(0.03, 1, 'For the $i$th ICS:', size=12, va='top', ha='left')
    axw.plot([0, 0.02], [0.9825, 0.9825], 'k-', linewidth=1)
    axw.plot([1, 0.25], [0.9825, 0.9825], 'k-', linewidth=1)
    sn.despine(ax=axw, left=False, right=False, bottom=False, top=True, trim=True)