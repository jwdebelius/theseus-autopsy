import itertools as it

import pandas as pd
import numpy as np

# Displaying my data
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sn

import scripts.bw_ana as bw_ana
from ._utils import (_summarize_kde,
                     _summarise_histogram,
                     _wrap_categorical_bars,
                     hide_axes,
                    )


def plot_example_cohort(fig, example_data, ref_ds, highlight=True, 
                        hide_xlab=True):
    """
    Wraps the example cohort display for a the example simulation
    """
    sub_data = _extract_sub_data(example_data, ref_ds)
    _plot_inset(fig, sub_data, highlight, hide_xlab)


def _extract_sub_data(full_data, ref_ds, num_total=7):
    """
    Determines the reference dataset and reference data
    """
    num_keep = (num_total - 3) / 2
    num_levels = np.linspace(-num_keep, num_keep, int(num_total - 2))
    cr = full_data.groupby('cohort')['x'].first().sort_values().to_frame()
    cr['x2'] = np.arange(len(cr))
    cr['xr'] = cr['x2'] - cr.loc[ref_ds, 'x2']
    levels = np.hstack([cr['xr'].min(), cr['xr'].max(), 
                        num_levels])
    exp_ds = cr.index[cr['xr'].isin(levels)]
        
    sub_data = \
        full_data.loc[full_data['cohort'].isin(exp_ds)].copy()
    sub_data.sort_values('x', ascending=True, inplace=True)
    sub_data.reset_index(drop=True, inplace=True)
    sub_data['xo'] = sub_data['x'].copy()
    sub_data['x'] = (~sub_data['cohort'].duplicated(keep='first')) * 1
    sub_data['x'] = sub_data['x'].cumsum()
    sub_data['x'] = sub_data['x'] + \
        (sub_data['x'] > sub_data['x'].min()) + \
        (sub_data['x'] == sub_data['x'].max())
    sub_data['x'] = sub_data['x'] - sub_data['x'].min()
    
    return sub_data


def _plot_inset(fig, sub_data, highlight=True, hide_xlab=True):
    """
    Highlights a simulated dataset in a dataset
    """
    gs = fig.add_gridspec(1, 7)
    ax_r = fig.add_subplot(gs[0], facecolor='None')
    ax_r.xaxis.set_tick_params(**hide_axes)
    ax_r.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax_r, left=True, right=True, top=True, bottom=True)

    # Plots the categorical plots
    _wrap_categorical_bars(fig, gs, sub_data, 
                                     cat_pos={'race': 0,
                                              'ethnicity': 1,
                                              'sex': 2,
                                              'smoking': 5,}
                                     )
    # plots the histogram and kdes
    _summarise_histogram(fig, gs, sub_data, off=3)
    _summarize_kde(fig, gs, sub_data, off=6)
    fig.axes[-1].xaxis.set_major_formatter(StrMethodFormatter('{x:1.0f}'))

    xx_min = sub_data.groupby('cohort')['x'].mean().min()
    xx_max = sub_data.groupby('cohort')['x'].mean().max()
    xx_bar = sub_data.groupby('cohort')['x'].mean().mean()
    
    # Tidies the axes
    for i, ax_ in enumerate(fig.axes[1:]):
        ax_.text(np.mean(ax_.get_xlim()), y=xx_min + 1.2, s='...', 
                 ha='center', va='center', size=10, zorder=8)
        ax_.text(np.mean(ax_.get_xlim()), y=xx_max - 0.8, s='...', 
                 ha='center', va='center', size=10, zorder=8)
        
        ax_.set_title(['Race', 'Eth.', 'Sex', 'Smoke', 'M.age', 'Year', 
                       'BW Z', '', ''][i],
                       size=8)
        
        ax_.set_ylim(xx_min - 0.75, xx_max + 0.75)
        ax_.set_facecolor('None')
        ax_.xaxis.set_tick_params(labelsize=8,
                                  labelbottom=(~hide_xlab) & (i > 3))
        ax_.xaxis.get_label().set_visible(~hide_xlab)
        ax_.set_position((0.0625 + i * 0.11 + i * 0.02,
                          0.35 - 0.2 * (hide_xlab),
                          0.1,
                          0.5 + 0.2 * (hide_xlab)))
    
    # Adds overlay axis to highlight the simulated cohort
    axo = fig.add_subplot(gs[:, :], facecolor='None')
    axo.set_position((0.0625, 0.35 - 0.2 * hide_xlab,
                      (i + 1) * 0.11 + i * 0.025,
                      0.5 + 0.2 * hide_xlab))
    axo.xaxis.set_tick_params(**hide_axes)
    axo.yaxis.set_tick_params(**hide_axes)
    axo.set_ylim(ax_.get_ylim())
    axo.set_xlim(0, 1)
    if highlight:
        axo.fill_between(x=[-0.5, 1.5], y1=xx_min - 1, y2=xx_bar - 0.5, 
                         alpha=2/3, facecolor='w', edgecolor='None')
        axo.fill_between(x=[-0.5, 1.5], y1=xx_bar + 0.5, y2=xx_max + 1,
                         alpha=2/3, facecolor='w', edgecolor='None')
        axo.fill_between([0.005, 0.995],  xx_bar - 0.6, xx_bar + 0.6, facecolor='None',
                         edgecolor='k', linewidth=1.25)
        axo.set_yticks([xx_bar])
        axo.set_yticklabels(['$\\Rightarrow$'], size=8, weight='bold')
        axo.yaxis.set_tick_params(labelleft=True)
        axo.set_position((0.0625, 0.35 - (0.2 * hide_xlab),
                          0.11 * (i + 1) + 0.02 * i,
                          0.5 + 0.2 * hide_xlab))
    sn.despine(ax=axo, left=True, right=True, top=True, bottom=True)

