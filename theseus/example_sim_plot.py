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
import ptitprince as pt

import scripts.bw_ana as bw_ana


def _make_ref_table(ax_dist, palette='BuGn', linewidth=1, n_cols=5, offset=0, 
                    cover_color='w', yoff=0.35):
    """
    A helper function to make a pretty reference table icon
    """
    ax_dist.set_ylim(-1, 1)

    y1 = np.round(np.arange(0, 0.2*(n_cols) + 0.1, 0.05), 3)
    x2 = ((y1 * 10) != np.floor(y1 * 10)) * 1

    colors = sn.color_palette(palette, 9)

    facecolor=colors[3]
    edgecolor = colors[-2]
    # edgecolor='k',
    highlight = colors[5]

    # Plots the main table layout
    ax_dist.barh(left=np.arange(0, 0.2 * n_cols, 0.2) + offset,
                 width=0.2,
                 y=0.25/2 + yoff,
                 height=0.25,
                 facecolor=highlight,
                 edgecolor=edgecolor,
                 linewidth=linewidth
                 )
    for i in np.arange(0.2, -0.65, -0.2):
        ax_dist.barh(left=np.arange(0, 0.2 * n_cols, 0.2) + offset,
                 width=0.2,
                 y=i + 0.15 / 2,
                 height=0.2,
                 facecolor=facecolor,
                 edgecolor=edgecolor,
                 linewidth=linewidth
                 )
    # ax_dist.fill_between(x=y1 + offset, y1=-x2 * 0.05 +yoff-0.85, y2=yoff-1,
    #                      color=cover_color,
    #                      linewidth=2*linewidth)

    ax_dist.xaxis.set_tick_params(bottom=False, labelbottom=False)
    ax_dist.yaxis.set_tick_params(left=False, labelleft=False)

    
    
