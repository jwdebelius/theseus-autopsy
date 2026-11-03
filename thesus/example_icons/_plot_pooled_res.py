import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from ._utils import (hide_axes,
                            _plot_betas,
                            _build_exp_brace,
                            )

def plot_pooled_res(fig_cr, fit_res, icc):
    """
    Plots a figure with the pooled results, splitting for space because 
    we're cool like that
    """
    gs = fig_cr.add_gridspec(2, 2, width_ratios=[1, 4])
    
    # Plots the brackets
    ax_br = fig_cr.add_subplot(gs[:, 0], facecolor='None')
    ax_br.xaxis.set_tick_params(**hide_axes)
    ax_br.yaxis.set_tick_params(**hide_axes)
    _build_exp_brace(ax_br)
    ax_br.set_xlim(-0.45, 0.15)
    ax_br.set_ylim(-1.25, 1.25)
    ax_br.set_position((0, 0, 0.1, 1))
    sn.despine(ax=ax_br, left=True, right=True, top=True, bottom=True)
    
    # # Adds the forest plots
    ax_mod1 = fig_cr.add_subplot(gs[0, 1])
    ax_mod2 = fig_cr.add_subplot(gs[1, 1], sharex=ax_mod1)
    tidy_short2 = {'gee': "GEE", 'lme': "RE", 'meta': 'MA', 'risk': 'ICS type'}
    for i, ax_ in enumerate([ax_mod1, ax_mod2]):
        _plot_betas(ax_, icc, fit_res, tidy_short2, labelicc=(i == 0))
        sn.despine(ax=ax_, 
                   left=False, 
                   right=False, 
                   top=(i == 1), 
                   bottom=(i == 0)
                   )
        ax_.xaxis.set_tick_params(bottom=(i == 1), 
                                  labelbottom=(i == 1),
                                  top=False, 
                                  labeltop=False,
                                  labelsize=8,
                                  )
        ax_.xaxis.get_label().set_visible(i == 1)

    ax_mod1.set_ylim(3.5, 5.5)
    ax_mod2.set_ylim(-0.5, 1.5)
    
    d = 0.03
    kwargs = dict(color='k', clip_on=False, linewidth=0.75, zorder=3)
    ax_mod1.plot((0+d, 0-d), (0-2*d, 0+2*d), **kwargs, 
                 transform=ax_mod1.transAxes)
    ax_mod1.plot((1+d, 1-d), (0-2*d, 0+2*d), **kwargs, 
                 transform=ax_mod1.transAxes)
    ax_mod2.plot((0+d, 0-d), (1-2*d, 1+2*d), **kwargs, 
                 transform=ax_mod2.transAxes)
    ax_mod2.plot((1+d, 1-d), (1-2*d, 1+2*d), **kwargs, 
                 transform=ax_mod2.transAxes)
