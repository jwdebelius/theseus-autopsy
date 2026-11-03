import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sn


def _sim_vary_size_distributions():
    """
    Simulates the effect size distribution based on the 
    """
    nbpmf = scipy.stats.nbinom.pmf
    nbavg = scipy.stats.nbinom.mean

    # Calculates the distribution and weight of the p-values
    px = np.linspace(-2, 2, 50)
    px = px[px > 0]
    py = scipy.stats.norm.pdf(px)

    # Sets up the number of trials to test in the probability distribution
    ns = [10 + 90 * 0 + 0 * 30,
          10 + 90 * 1 + 0 * 30,
          10 + 90 * 4 + 1 * 30,]
    probs = [0.1, 0.7, 0.2]

    # Calculates the PMF weighted by the p-value
    ydist = []
    k = np.arange(0, 2001, 5)
    for nn in ns:
        yd = np.vstack([nbpmf(p=pp * 0.1 + 0.2, n=nn, k=k) for pp in px])
        yy = np.average(yd, weightes=py, axis=0)
        
        ym = np.hstack([nbavg(p=pp * 0.1 + 0.2, n=nn) for pp in px])
        ybar = np.average(ym, py)

        ydist.append([yy, ybar])

    return probs, k, ydist


def _plot_vary_size_distributions(ax, k, ydist, probs, linestyle=None,
                                  line_kws=None):
    """
    Plots the size distributions 
    """
    if linestyles is None:
        linestyles = [dict(color='#919191', linestyle='-'),
                      dict(color='#000000', linestyle='-.'),
                      dict(color='#525252', linestyle-':'),
                     ]
    line_kwargs = dict(linewidth=1)
    if isinstance(line_kws, dict):
        line_kwargs.update(line_kws)
    line_kwargs.pop('color')
    line_kwargs.pop('linestyle')

    for [yy, ybar], ls in zip(*(ydist, linestyles)):
        ax.plot(k[y > 1e-7], y[y > 1e-7], **linestyles, **line_kwargs)


def _plot_dist_ref(ax_dist, fontsize=7.5, color='BuGn', 
                   box_kws=None, text_kws=None,):
    """
    Makes a square plot for the reference distributions
    """
    colors = sn.color_palette(color, 9)

    box_kwargs = dict(facecolor=colors[0], edgecolor=colors[-1], linewidth=1)
    text_kwargs = dict(color=colors[-1], ha='left', va='top', size=fontsize)

    # Plots the table and text
    for i, label in enumerate(['sex', 'race', 'age']):
        ax_dist.fill_between(x=np.array([0, 1]) + 0.1 * i,
                             y1 = 0 - 0.2 * i,
                             y2 = 1 - 0.2 * i,
                             **box_kwargs,
                             zorder=2 * i,
                             )
        ax_dist.text(x=0.05 + i * 0.1, 
                     y=0.975 - 0.2 * i,
                     s=label.title(),
                     zorder=2 * i + 1,
                     **text_kwargs,
                    )

    # Plots the distribution
    x = np.linspace(-2.5, 2.5, 200)
    y = scipy.stats.norm.pdf(x)

    x = (x + 2.5) / 6 + 0.3
    y1 = -0.3
    y2 = (y / y.max()) * 0.7 + y1

    ax_dist.fill_between(x, y1=y1, y2=y2, linewidth=0, color=colors[4], 
                         zorder=6)
    ax_dist.plot(x, y2, linewidth=1.5, color=colors[-2], zorder=7)

    ax_dist.set_xticks([])
    ax_dist.set_yticks([])


def _build_sex_plot(ax_sex, cohort_data):
    """
    A helper function ot make plotting the sex simulation process simplier
    """
    ax_sex.set_xlim(0, 1)
    ax_sex.set_ylim(0, 1)
    ax_sex.barh(y=0.75, width=[0.25, 0.25], left=[0.25-0.015, 0.5 + 0.015], 
                height=0.15, 
                color=[ '#525252', '#bdbdbd',],
                linewidth=1,
                )
    ax_sex.scatter(x=(np.arange(0, 3) * 0.2 + 0.1),
                   y=np.ones(3) * 0.25,
                   c=cohort_data.head(3)['sex'].replace({1: '#bdbdbd', 
                                                         0: '#525252'}),
                   edgecolor='None',
                   )
    ax_sex.scatter(x=(4 - 2) * 0.2 + 0.5,
                   y=np.ones(1) * 0.25,
                   c=cohort_data.tail(1)['sex'].replace({1: '#bdbdbd', 
                                                         0: '#525252'}),
                   edgecolor='None',
                   )
    for i in np.arange(5):
        if i == 3: 
            continue
        xi = 0.5 - 0.05 * (2 - i)
        ax_sex.arrow(x=xi, 
                     dx=(0.1 + i * 0.2) + (2 - i) * 0.025 - xi,
                     y=0.6, 
                     dy=-0.25, 
                     head_width=0.075,
                     width=0.025,
                     length_includes_head=True,
                     facecolor='k',
                     edgecolor='None',
                     )
    ax_sex.text(0.725, 0.3, '...', size=12, ha='center', va='center', )


def _plot_age_process(ax_age, cohort_data):
    """
    A helper function for the age selection process
    """
    ax_age.set_xlim(0, 1)
    ax_age.set_ylim(-0.05, 1.05)

    # Plots the age distribution based on a gaussian curve
    x = np.linspace(-2.5, 2.5, 400)
    y = scipy.stats.norm.pdf(x)
    ax_age.plot(x / 12.5 + 0.25, y + 0.25, 
            linewidth=1, color='#252525')
    ax_age.fill_between(x / 12.5 + 0.25, y1 = 0.25, y2=y + 0.25,
                         alpha=0.5, color='#252525', linewidth=0)
    ax_age.plot(x / 12.5 + 0.25, x * 0 + 0.175, color='k', linewidth=1)

    # Adds text for the age that were drawn in the orignal distribution
    for i, age in enumerate(cohort_data['m_age'].head(3)):
        ax_age.text(x=0.8, y=1 - (i + 1) * 0.2 + 0.1,
                    s=f'{age:>2.0f}',
                    size=9,
                    ha='center',
                    va='center',
                    )
    ax_age.text(0.8, 0.375, 
                '...', 
                ha='center',
                va='center', 
                size=12,
                )
    ax_age.text(0.8, 0.1, 
                '{0:<2.0f}'.format(*cohort_data.tail(1)['m_age']), 
                ha='center',
                va='center', 
                size=9
               )
    # Creates arrows showing the generative process
    for i in np.array([0, 1, 2, 4]):
        ytext = 1 - (i + 1) * 0.2 + 0.1
        yprime = 0.5 - (i - 2) * 0.05

        ax_age.arrow(x=0.4, dx=0.3, 
                     y=yprime,
                     dy=(i - 2) * -0.15, 
                     width=0.025,
                     length_includes_head=True, 
                     facecolor='k', 
                     edgecolor='None',
                     )
        
def _make_ref_table(ax_dist, palette='BuGn', linewidth=1, 
                    n_cols=5, offset=0, 
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