import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def _plot_inset(ax_, icc_, param, width, widthb=2.5, color='#4d4d4d'):
    """
    Plots the inset highlighting the parameter
    """
    # Plots the estimate
    ax_.errorbar(x=icc_, y=param, yerr=width, color=color, marker='.', linewidth=1)

    # Gets the xlim for scaling
    xscale = np.diff(ax_.get_xlim())[0]
    yscale = np.diff(ax_.get_ylim())[0]

    # Adds bracket annotation
    ax_.annotate('95\\%CI', 
                 xy=(icc_ + 0.1 * xscale, param),
                 xytext=(icc_ + 0.15 * xscale, param),
                 ha='left', va='center', size=8,
                 arrowprops=dict(facecolor='black',
                                 zorder=-1,
                                 arrowstyle=f'-[, widthB={widthb}, lengthB=0.15',
                                 connectionstyle='arc3',
                                 # scale=0,
                                 ),
                 zorder=100,
                 )
    # Adds diff annotation
    ax_.annotate('$\\beta - \\hat{b}$', size=8, ha='left', va='bottom',
                 xytext=(ax_.get_xlim()[0] + 0.025 * xscale, ax_.get_ylim()[0] + yscale * 0.025),
                 xy=(icc_ - 0.05 * xscale, param),
                 arrowprops=dict(facecolor='black',#width=1, 
                                 zorder=-1,
                                 arrowstyle='->',
                                 connectionstyle='arc',
                                 ),
                 zorder=100)
                 