import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sn

from ._utils import (_make_ref_table,
                            hide_axes
                            )


def build_sim_process_ref(figul, cohort_data):
    """
    Makes a figure showing the process for building the confounder matrix
    """
    gs = figul.add_gridspec(4, 8)
    ax_dist = figul.add_subplot(gs[:, 0:2], aspect=1, facecolor='w')
    ax_tabl = figul.add_subplot(gs[:, -2:], facecolor='w')
    ax_ar1 = figul.add_subplot(gs[:, 2], facecolor='w')
    ax_ar2 = figul.add_subplot(gs[:, -3], facecolor='w')
    ax_age = figul.add_subplot(gs[2:4, 3:5], facecolor='w')
    ax_sex = figul.add_subplot(gs[0:2, 3:5], facecolor='w')
    ax_spl = figul.add_subplot(gs[:, 3:5], facecolor='w') 

    # Sets the position for hte final figure
    ax_dist.set_position((0, 0, 0.25, 1))
    ax_tabl.set_position((0.75, 0, 0.25, 1))
    ax_ar1.set_position((0.25, 0, 0.125, 1))
    ax_ar2.set_position((0.6125, 0, 0.125, 1))
    ax_age.set_position((0.375, 0, 0.25, 0.5))
    ax_sex.set_position((0.375, 0.5, 0.25, 0.5))
    ax_spl.set_position((0.375, 0, 0.25, 1))

    for ax in figul.axes:
        ax.xaxis.set_tick_params(**hide_axes)
        ax.yaxis.set_tick_params(**hide_axes)
        ax.set_facecolor('None')


    _plot_dist_ref(ax_dist, color='Greys')
    _make_ref_table(ax_tabl, palette='Greys')
    ax_tabl.set_xlim(-0.1125, 1.1125)
    ax_tabl.set_ylim(-1.45, 1.45)

    _build_sex_plot(ax_sex, cohort_data)
    _plot_age_process(ax_age, cohort_data)
    ax_spl.text(0.5, 0.5, '...', ha='center', va='center', size=15)
    ax_spl.set_xlim(0, 1)
    ax_spl.set_ylim(0, 1)

    arrow_kws = dict(edgecolor='None', width=0.025, 
                     length_includes_head=True, 
                     facecolor='k')
    ax_ar1.arrow(x=0.1, dx=0.8, y=0.55, dy=+0.2, **arrow_kws)
    ax_ar1.arrow(x=0.1, dx=0.8, y=0.45, dy=-0.2, **arrow_kws)
    ax_ar1.set_ylim([0, 1])
    ax_ar2.arrow(x=0.1, dx=0.8, y=0.75, dy=-0.2, **arrow_kws)
    ax_ar2.arrow(x=0.1, dx=0.8, y=0.25, dy=+0.2, **arrow_kws)
    ax_ar2.set_ylim([0, 1])

    sn.despine(figul, left=True, right=True, top=True, bottom=True)


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

def build_table_plot(fig, cohort_data, n_head=3):
    """
    Builds a table showing the first few lines fo the data
    """
    # Sets up the axis
    ax = fig.add_subplot(1,1,1, facecolor='None')
    ax.set_position((0.075, 0.025, 0.85, 0.9))
    ax.yaxis.set_tick_params(left=False, labelleft=False, 
                             right=False, labelright=False)
    ax.xaxis.set_tick_params(left=False, labelleft=False, 
                             right=False, labelright=False)
    sn.despine(left=True, right=True, top=True, bottom=True)

    # Pulls out the data we're intereseted in as a merged object including
     # a row of elipses to space things
    cols_ = ['sex', 'race', 'ethnicity', 'year', 'm_age']
    sub_table1 = cohort_data[cols_].head(n_head)
    sub_table2 = cohort_data[cols_].tail(1)
    sub_tables = cohort_data[cols_].head(1)
    sub_tables.replace(
        {v: '...' for v in np.unique(sub_tables.values.flatten())},
        inplace=True
    )
    merged_table = pd.concat(axis=0, objs=[sub_table1, sub_tables, sub_table2])
    merged_table.index = \
        np.hstack([np.arange(n_head) + 1, '...', len(cohort_data)])

    # Casts the table to latex and does some latex clena up because the versions
    # dont quite line up between what  pandas puts out and what my system
    # at least is set up to handle
    merged_lt = merged_table.to_latex(column_format='| c | c | c | c | c | c |',)
    relabels = {'\\toprule': '\\hline',
                '\\midrule': '\\hline',
                '\\bottomrule': '\\hline',
                '\\\\\n': '\\\\\n \\hline\n'}
    for ori, new in relabels.items():
        merged_lt = merged_lt.replace(ori, new)
    merged_lt = merged_lt.replace('\\hline\n\\hline', '\\hline')
    # plots the text
    _ = ax.text(x=0.5, 
                y=0.5, 
                s=merged_lt.replace('\n', ' '), 
                ha='center', 
                va='center', 
                size=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.fill_between([0.075, 0.925], y1=0.0, y2=1, color='w')
    ax.fill_between([0.075, 0.925], y1=0.825, y2=1, color='#d1d1d1')
