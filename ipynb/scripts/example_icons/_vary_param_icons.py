import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import ticker

from scripts.example_icons._utils import (hide_axes,
                                          _plot_icc_wedge,
                                          )


def show_icc_range(fig_icc, goal_iccs, line_kws=None):
    """
    Plots an ICC scale with all the ICC values
    """
    # Sets up the axis
    ax_icc = fig_icc.add_subplot(1,1,1, facecolor='None')
    ax_icc.set_position((0.05, 0.05, 0.9, 0.9))
    ax_icc.xaxis.set_tick_params(**hide_axes)
    ax_icc.yaxis.set_tick_params(**hide_axes)
    ax_icc.set_xlim(-0.1, 1.1)
    ax_icc.set_ylim(0, 1.1)

    # Generates associated scaling parameters
    xdiff = np.diff(ax_icc.get_xlim())[0]
    ydiff = np.diff(ax_icc.get_ylim())[0]
    scale = ydiff / xdiff
    
    line_kwargs = dict(linewidth=0.5, color='k')
    if isinstance(line_kws, dict):
        line_kwargs.update(line_kws)

    # Plots the ICC wedge
    _plot_icc_wedge(ax_icc, offset=0)

    # Plots and labels the goal ICCs
    for i, icc in enumerate(goal_iccs):
        x, y = _calc_icc_theta(i, icc, 0.3, scale)
        ax_icc.plot(x, y, **line_kwargs)
        if icc == 0:
            ax_icc.text(x.min(), y.max(), '0', 
                        ha='right', va='center', size=7)
        elif icc == 1e-3:
            ax_icc.text(icc, y.max() + 0.025,
                        f'{icc:1.3f}', 
                        rotation=90, ha='center', va='bottom', size=7)
        elif icc < 0.05:
            ax_icc.text(x.max() + 0.06, 
                        y.max() + 0.06 * np.tan(45),
                        f'{icc:1.3f}', size=7, ha='center', va='center', 
                        rotation=37.5)
        else:
            ax_icc.text(x.max() + 0.04, 
                        y.max() + 0.04 * np.tan(45) + 0.025,
                        f'{icc:1.2f}', size=7, ha='center', va='center', 
                        rotation=37.5)
    ax_icc.text(x=np.mean(ax_icc.get_xlim()), 
                y=ax_icc.get_ylim()[1] - np.diff(ax_icc.get_ylim())[0] * 0.05,
                s='Seed ICC', 
                size=10, 
                ha='center', 
                va='top')


def _calc_icc_theta(i, icc, yoff, scale):
    """
    Calculates the x and y line to plot the icc
    """
    t45 = np.arctan(45) / scale
    
    if icc == 0:
        x = np.array([icc, icc, -0.025])
        y = np.array([0, 1, 1]) * 0.15 + \
            np.array([0, 0, 1]) * (0.025 * t45) + \
            yoff
    elif icc == 1e-3:
        x = np.array([icc, icc, icc])
        y = np.array([0, 0.5, 0.5]) + yoff
    elif icc < 0.05:
        x =  np.array([0, 0, 0.025 * (1 + (i < 3))])
        y = np.array([0, 1, 1]) * 0.125 + \
            np.array([0, 1, 1]) * ((5.5 - i) * 0.1) + \
            np.array([0, 0, 1]) * x * t45 + \
            yoff
        x = x + icc
    elif icc < 0.2:
        x = np.array([icc, icc])
        y = np.array([0, 0.4 * icc]) + \
            np.array([0, 1]) * (8 - i) * 0.06 + \
            yoff
    else:
        x = np.array([icc, icc])
        y = np.array([0, 0.4 * icc + 0.025]) + yoff

    return x, y


def _plot_effect_sizes(fig, mod_sd=0.15):
    """
    Makes a reference plot of the effect size distributions
    """
    # Sets up the effect sizes
    effect_colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78']
    effect_yy = np.arange(4) + np.floor(np.arange(4) / 2) * 0.5
    effect_xx = np.array([-0.5, -0.5, -0.1, -0.1])

    # Sets up the kdes for the cases where we'll vary the distribution
    kde_xx = np.vstack([np.linspace(-3, 3, 121)] * 4)
    kde_yy = scipy.stats.norm.pdf(kde_xx)

    kde_xx = np.absolute(np.atleast_2d(effect_xx) * mod_sd).T * kde_xx + \
        np.atleast_2d(effect_xx).T
    kde_yy = kde_yy / kde_yy.max() * 0.75 + 0.25 / 2 + \
        np.atleast_2d(effect_yy).T - 0.5

    # Effect modification: CT = 1 -> b = 0
    effect_yy_mod = effect_xx * np.array([0, 1, 0, 1])

    # Sets up the axes
    ax0 = fig.add_subplot(1, 3, 1, position=(0.20, 0.275, 0.225, 0.5))
    ax1 = fig.add_subplot(1, 3, 2, sharex=ax0, sharey=ax0,
                          position=(0.475, 0.275, 0.225, 0.5))
    ax2 = fig.add_subplot(1, 3, 3, sharex=ax0, sharey=ax0,
                          position=(0.75, 0.275, 0.225, 0.5))
    
    for ax in fig.axes: 
        ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
        ax.yaxis.set_tick_params(**hide_axes)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.plot([ 0  ,  0  ], [-1, 5], 'k-',  linewidth=1, zorder=0)
        ax.plot([-0.1, -0.1], [-1, 5], 'k:',  linewidth=1, zorder=0)
        ax.plot([-0.5, -0.5], [-1, 5], 'k-.', linewidth=1, zorder=0)
        ax.set_yticks(effect_yy)
        ax.set_yticklabels(['NICU', 'Gen', 'NICU', 'Gen'])
        ax.set_ylim(effect_yy.min() -0.5, effect_yy.max() + 0.5)
        ax.set_xlim(-0.75, 0.1)
        # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.1f}'))
        ax.set_yticks([0.5, 2.5 + 0.5], minor=True)
        ax.set_yticklabels(['-0.5', '-0.1'], minor=True, rotation=90, 
                           ha='center', va='center', size=10)
        ax.yaxis.set_tick_params(which='minor', length=24, left=False, 
                                 labelleft=False)
        ax.set_xlabel('  $\\beta_{e}$', size=8, va='bottom', ha='left')
        sn.despine(ax=ax, left=True,  right=True, top=True, bottom=False)

    # Plots the data without modification
    ax0.scatter(x=effect_xx, y=effect_yy, c=effect_colors, s=15)
    ax0.yaxis.set_tick_params(labelleft=True)
    ax0.yaxis.set_tick_params(labelleft=True, which='minor')
    ax0.set_title('Base effect', size=8)
    
    for xx, yy, c in zip(*(kde_xx, kde_yy, effect_colors)):
        ax1.plot(xx, yy, color=c, linewidth=1)
    ax1.set_title('ICS effect', size=8)
    
    ax2.scatter(x=effect_xx * np.array([0, 1, 0, 1]),
                y=effect_yy,
                c=effect_colors, 
                s=15)
    ax2.set_title('Effect mod.', size=8)
            