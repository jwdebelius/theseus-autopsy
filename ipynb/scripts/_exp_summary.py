import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from scripts.example_icons._plot_example_cohort import (_extract_sub_data, 
                                                        _plot_inset)
from scripts.example_icons._plot_pooled_res import plot_pooled_res
from scripts._exp_plots import _extract_sim_cohort
from scripts._cohort_plots import(_build_scene_grid_vert,
                                  plot_pooled_betas)
from scripts.example_icons._utils import hide_axes

def show_experiment(fig, data, cid, models, tech_order, tidy_short, 
                    scenario=[1, 1, 1, 1], seed_icc=-0.15, 
                    seed_effect=-0.5):
    """
    Plots an experimetn description
    """
    figp, figa, figc, figb, fige = \
        fig.subfigures(1, 5, 
                       width_ratios=[1, 0.25, 2.75, 0.25, 1.5], 
                           facecolor='None')
    _plot_big_arrow(figa)
    _plot_big_arrow(figb)
    fig_scen, fig_icc = figp.subfigures(2, 1, height_ratios=[3, 2])
    _build_scene_grid_vert(fig_scen, scenario)
    _wrap_sim_count(fig_icc, seed_icc=seed_icc, seed_beta=seed_effect)

    
    plot_pooled_betas(fige, data, models, tech_order, 
                      tidy_short, labelicc=True)
    fige.axes[0].set_position((0.35, 0.3, 0.5, 0.575))
    fige.axes[0].yaxis.set_tick_params(labelsize=8)
    fige.axes[0].xaxis.set_tick_params(labelsize=8)
    fige.axes[0].xaxis.get_label().set_size(8)
    
    sub_data = _extract_sub_data(data, 'cohort_03', num_total=7)
    _plot_inset(figc, sub_data, highlight=False)


def _wrap_sim_count(fig_icc, seed_icc=0.15, seed_beta=-0.5):
    """
    Wraps the seed ICC and effect size
    """
    ax_icc = fig_icc.add_subplot(1,1,1, facecolor='None')
    ax_icc.xaxis.set_tick_params(**hide_axes)
    ax_icc.yaxis.set_tick_params(**hide_axes)
    ax_icc.set_ylim(0, 2.1)
    ax_icc.set_xlim(0, 1)
    ax_icc.text(0.05, 1.5, 'Seed ICC', ha='left', va='center', size=8)
    ax_icc.text(0.05, 0.5, 'Seed $\\beta_{e}$', ha='left', va='center', size=8)
    ax_icc.text(0.95, 1.5, f'{seed_icc:1.2f}',
                ha='right', va='center', size=8)
    ax_icc.text(0.95, 0.5, f'{seed_beta:1.2f}sd', 
                ha='right', va='center', size=8)


def _plot_big_arrow(fig):
    """Plots a big arrow"""
    ax = fig.add_subplot(1, 1, 1, facecolor='None')
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 3)
    
    ax.fill_between(x=np.array([0, 1, 1, 3]), 
                    y1=np.array([-1, -1, -3, 0]), 
                    y2=np.array([1, 1, 3, 0]), 
                    facecolor='#9f9f9f')
    # ax.set_position((0, 0, 1, 1))
    sn.despine(fig, left=True, right=True, top=True,bottom=True)