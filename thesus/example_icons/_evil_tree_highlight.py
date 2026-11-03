import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sn

import thesus.heuristic_plots as hp
from ._utils import (_fit_models_no_filt,
                     r_tidy_short)

def highlight_evil_tree(fig, scene_df, example_data, model0='lme', 
                        model1='meta', example_effect=-0.5, tidy_short=None):
    """
    Highlights an inset on the evil tree plot
    """
    if tidy_short is None:
        tidy_short = r_tidy_short
    # Fits the example data
    fit_res, icc_ = _fit_models_no_filt(example_data)
    fit_res['from_ref'] = fit_res['param'] - example_effect

    # Sets up the axes
    gs = fig.add_gridspec(4, 4, wspace=0.0)
    axa0 = fig.add_subplot(gs[0:2, 0:2], facecolor='w')
    axa1 = fig.add_subplot(gs[0:2, 2:4], facecolor='w')

    # Plots the reference evil trees
    error_dict1 = error_kws=dict(linestyle='', alpha=1/3, linewidth=0.25, 
                                 zorder=0)
    for axa, model in zip(*([axa0, axa1], [model0, model1])):
        df_ =  scene_df.xs(model, level='model').copy()
        param = fit_res.loc[model, 'from_ref']
        width = fit_res.loc[model, 'width']
        cover = np.sign((param - width) * (param + width))
        color = {1: '#b2182b', -1: '#4d4d4d'}[cover]
        hp.plot_scary_tree_inset(axa, df_, error_kws=error_dict1)
        axa.errorbar(icc_, param, width, color=color, marker='', 
                     linewidth=1, zorder=1000)
        axa.set_title(tidy_short.get(model, model.title()), size=8, pad=-3)
        

    # Sets up the main inset
    axr0 = fig.add_subplot(gs[2:4, 1:3], facecolor='w')
    axr0.set_position((0.25, 0.125, 0.5, 0.5))
    df0 = scene_df.xs(model0, level='model').copy()
    hp.plot_scary_tree_inset(axr0, df0, 
                             error_kws=dict(linestyle='', alpha=0.5, 
                                            linewidth=0.25, zorder=0))
    
    # Plots the reference distribution in the inset
    xlim = np.array([-0.075, 0.075]) + icc_ + 0.025
    ylim = np.array([-0.2, 0.2]) + 0.1
    axr0.set_xlim(xlim)
    axr0.set_ylim(ylim)
    param0 = fit_res.loc[model0, 'from_ref']
    width0 = fit_res.loc[model0, 'width']
    cover0 = np.sign((param0 - width0) * (param0 + width0))
    color0 = {1: '#b2182b', -1: '#4d4d4d'}[cover0]
    _plot_inset(axr0, icc_, param0, width0, widthb=1.1, color=color0)

    x0 = icc_
    y0 = param0 - width0
    xs = 0.15 * np.diff(axr0.get_xlim())[0] / 2
    ys = 0.15 * np.diff(axr0.get_ylim())[0] / 2

    axa0.fill_between(xlim, ylim[0], ylim[1], facecolor='None', zorder=1000, 
                      linewidth=0.5)    


def _plot_inset(ax_, icc_, param, width, widthb=2.5, color='#4d4d4d'):
    """
    Plots the inset highlighting the parameter
    """
    # Plots the estimate
    ax_.errorbar(x=icc_, y=param, yerr=width, color=color, marker='.', 
                 linewidth=1)

    # Gets the xlim for scaling
    xscale = np.diff(ax_.get_xlim())[0]
    yscale = np.diff(ax_.get_ylim())[0]
    [x0, x1] = ax_.get_xlim()
    [y0, y1] = ax_.get_ylim()

    # Adds bracket annotation
    ax_.annotate('95\\%CI', 
                 xy=(icc_ + 0.075 * xscale, param),
                 xytext=(icc_ + 0.125 * xscale, param),
                 ha='left', va='center', size=8,
                 arrowprops=dict(facecolor='black',
                                 zorder=-1,
                                 arrowstyle=f'-[, widthB={widthb}, lengthB=0.15',
                                 connectionstyle='arc',
                                 # scale=0,
                                 ),
                 zorder=100,
                 )
    # Adds diff annotation
    ax_.text(x0 + 0.025 * xscale,
             y1 - 0.025 * yscale,
             '$\\beta - \\hat{b}$',
             ha='left', 
             va='top',
             size=8)
    ax_.annotate(xytext=(x0 + 0.075 * xscale, y1 - 0.175 * yscale),
                 xy=((icc_ - 0 * xscale), (param + yscale * 0)),
                 # edgecolor='None',
                 text='',
                  arrowprops=dict(facecolor='black',#width=1, 
                                 zorder=-1,
                                 arrowstyle='->',
                                 connectionstyle='arc3,rad=0.25',
                                 ),
              )


                 