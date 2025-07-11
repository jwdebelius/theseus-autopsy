import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn
import ptitprince as pt


def _plot_dist_ref(ax_dist, fontsize=7.5, color='BuGn', box_kws=None, 
                  text_kws=None,):
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
                 y=0.15/2 + yoff,
                 height=0.15,
                 facecolor=highlight,
                 edgecolor=edgecolor,
                 linewidth=linewidth
                 )
    for i in np.arange(0.2, -0.65, -0.15):
        ax_dist.barh(left=np.arange(0, 0.2 * n_cols, 0.2) + offset,
                 width=0.2,
                 y=i + 0.15 / 2,
                 height=0.15,
                 facecolor=facecolor,
                 edgecolor=edgecolor,
                 linewidth=linewidth
                 )
    ax_dist.fill_between(x=y1 + offset, y1=-x2 * 0.05 +yoff-0.85, y2=yoff-1,
                         color=cover_color,
                         linewidth=2*linewidth)

    ax_dist.xaxis.set_tick_params(bottom=False, labelbottom=False)
    ax_dist.yaxis.set_tick_params(left=False, labelleft=False)


def _plot_gauss_curve(ax, color, ylo=-0.35, yhi=1.45, textoff=0.05,
                     sigma_eq='$\sigma^{2}=1$', text_kws=None, plot_ticks=True, 
                     tick_off=0.1, hide=True):
    """
    Plots a guassian curve 
    """
    # Sets up some defaults
    text_kwargs = dict(ha='center', va='bottom', size=9)
    if text_kws is not None:
        text_kwargs.update(text_kws)

    # Gets and plots the gaussian curve
    x = np.linspace(-3.5, 3.5, 100)
    y = scipy.stats.norm.pdf(x)
    ax.plot(x, y, color=color, zorder=2)
    ax.fill_between(x, y, color=color, 
                    alpha=0.25, zorder=0, edgecolor='None',)

    # Plots the reference means and std lines
    ax.plot([0, 0], [0, y.max()], color='k', linewidth=0.75, zorder=1)
    ax.plot([-1, 1], [scipy.stats.norm.pdf(1)] * 2, 
            color='k', linewidth=0.75, zorder=1)

    # Adjusts the y limits to plot the data in new shiny ways
    ylim = ax.get_ylim()
    print(ylim)
    ax.set_ylim(ylim[0] + ylo * np.diff(ylim), ylim[0] + np.diff(ylim) * yhi)

    # Annotates the data with the sigma values
    ax.text(x=0, 
            y=y.max() + np.diff(ylim) * textoff, 
            s=f'$\mu = 0$\n{sigma_eq}', 
            **text_kwargs,
            )

    # Plots reference spans with an offset
    if plot_ticks:
        xmin, xmax = ax.get_xlim()
        xticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 2)
        ax.plot(xticks, xticks * 0 - tick_off * np.diff(ylim),
                   color='k', linestyle='-', marker='|', linewidth=0.75,
                   )
        ax.set_xlim(xmin, xmax)

    if hide:
        ax.xaxis.set_tick_params(left=False, labelleft=False, right=False, 
                                 labelright=False, length=0, labelsize=0)
        ax.yaxis.set_tick_params(left=False, labelleft=False, right=False, 
                                 labelright=False, length=0, labelsize=0)
        sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)    


def _select_cohort_type(axct, type_=0, color='purple', **kwargs):
    """
    Plots a cohort type selection
    """
    axct.set_xlim(0, 1)
    axct.set_ylim(-0.5, 2)

    text_kws = dict( size=14, color=color, va='center', ha='center')
    text_kws.update(kwargs)

    axct.text(0.5, 1.5, s='Gen', alpha=1-0.5*(type_ == 1),
              **text_kws,
             )
    axct.text(0.5, 0.5, s='NICU', alpha=1-0.5*(type_ == 0),
              **text_kws,
             )
    axct.text(0.5, 1.0 + 0.1 * (type_ == 1), s='or', size=10, 
              color='k', ha='center', va='center')

    axct.fill_between(np.array([0.125, 0.85]), y1=0.2 + (type_ == 0), 
                      y2=0.9 + (type_ == 0), 
                      facecolor='None', edgecolor=color)
    axct.yaxis.set_tick_params(left=False, labelleft=False, right=False, 
                               labelright=False)
    axct.xaxis.set_tick_params(left=False, labelleft=False, right=False, 
                               labelright=False)

def build_sim_sum_icons(fig, sub_data, sim_colors=None, **exp_scatter_kws):
    """
    Builds a line plots showing the sub data 
    """
    # Sets up the defaults and fixed them for the data
    sim_colors_def = {
        'fixed': '#919191',
        'exposure': '#e41a1c',
        'cohort_fixed': 'purple',
        'cohort_random': sn.color_palette()[1],
        'error': sn.color_palette()[0],
        }
    sim_colors_def.update(sim_colors)
    sim_colors = sim_colors_def

    sim_titles = {'fixed': '$\sum{\\beta_{k}w_{ik}}$',
              'exposure': '${\\beta_{\\mathsf{exp}}}e_{i}$',
              'cohort_fixed': '$\\beta_{\\mathsf{ct}}{ct_{i}}$',
              'cohort_random': '$\\gamma_{i}$',
              'error': '$\\epsilon$'
              }
    exp_scatter_kwargs = dict(
        data=sub_data,
        jitter=0.25,
        edgecolor='None',
        s=1,
        )
    exp_scatter_kwargs.update(exp_scatter_kws)

    # Sets up the reference background for the plots. This *only* works
    # if we dont have a constrained layout, so thats what we're working with
    gs = fig.add_gridspec(4, len(sub_data.columns) * 2 + 1)

    # Reference axis for the data
    ax_hide = dict(left=False, right=False, labelleft=False, labelright=False)
    ax_eq = fig.add_subplot(gs[:, :], facecolor='None')
    ax_eq.yaxis.set_tick_params(**ax_hide)
    ax_eq.xaxis.set_tick_params(**ax_hide)

    ### Outcome scatter
    # Plots the outcome
    sn.stripplot(y='y', ax=ax_o, color='k', **exp_scatter_kwargs,)
    # Cleans up the axes
    ax_o.set_ylabel('z-birth-weight')
    ax_o.yaxis.set_label_position("right")
    ax_o.yaxis.set_tick_params(left=False, right=True, labelright=True, 
                               labelleft=False)
    ax_o.xaxis.set_tick_params(bottom=False, labelbottom=False, labelsize=0)
    ax_o.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>2.0f}'))
    ax_o.grid(True, axis='y')
    ax_o.set_title('z', pad=4)

    ### Exposure scatter
    for i, col_ in enumerate(sub_data.columns[:-1]):
        # Makes the axis
        ax_ = fig.add_subplot(gs[2:4, 2*i:2*(i+1)], sharey=ax_o)
        # Plots the data
        sn.stripplot(y=col_, ax=ax_,
                     color=sim_colors.get(col_, 'k'),
                     **exp_scatter_kwargs,
                     )
        # Formats the axis
        ax_.xaxis.set_tick_params(bottom=False, labelbottom=False)
        ax_.set_ylabel('z-birth weight')
        ax_.yaxis.get_label().set_visible(i == 0)
        ax_.xaxis.get_label().set_visible(False)
        ax_.yaxis.set_tick_params(left=True, labelleft=(i == 0))
        ax_r.append(ax_)
        ax_.set_title(sim_titles.get(col_, col_.replace('_', '\n')), pad=4)
        ax_.grid(True, axis='y')

    # ### Plots the Icons
    # Cohort Random effect 
    ax_ce = fig.add_subplot(gs[0:2, 6:8], facecolor='None')
    _plot_gauss_curve(ax=ax_ce,
                      color=sim_colors.get('cohort_fixed', 'k'), 
                      ylo=-0.4, 
                      yhi=1.4,
                      sigma_eq='$\sigma^{2} \propto \mathrm{ICC}$')
    # Adds the point estimate value
    cg = sub_data['cohort_fixed'].unique()
    ax_ce.scatter(cg, scipy.stats.norm.pdf(cg), zorder=3, 
                      color=sim_colors.get('cohort_fixed', 'k'))

    # Plots the individual error term
    ax_ce = fig.add_subplot(gs[0:2, 8:10], facecolor='None')
    plot_gauss_curve(ax=ax_ce, color=sim_colors.get('error', 'k'), 
                     ylo=-0.4, yhi=1.4)

    # Cohort fixed effect
    axtp = fig.add_subplot(gs[0:2, 4:6], facecolor='None')
    type_ = (sub_data['cohort_fixed'].unique()[0] != 0) * 1
    select_cohort_type(axtp, type_, 
                       color=sim_colors.get('cohort_fixed', 'purple')
                       )
    sn.despine(ax=axtp, left=True, right=True, top=True, bottom=True)

    # Demographic/exposure icon
    axf1 = fig.add_subplot(gs[0:2, 0:2], facecolor='None',)
    make_ref_table(axf1, palette='Greys')
    axf1.set_xlim(-0.1025, 1.1025)
    axf1.set_ylim(-1.1, 0.65)
    sn.despine(ax=axf1, left=True, right=True, top=True, bottom=True)

    axf2 = fig.add_subplot(gs[0:2, 2:4], facecolor='None')
    make_ref_table(axf2, palette='Reds', n_cols=1, offset=0.4,)
    axf2.set_xlim(axf1.get_xlim())
    axf2.set_ylim(axf1.get_ylim())
    sn.despine(ax=axf2, left=True, right=True, top=True, bottom=True)

    # Adds equation labels
    ax_eq.text(0.15, 0.55, '+', size=12, ha='center', va='center')
    ax_eq.text(0.306, 0.55, '+', size=12, ha='center', va='center', color='k')
    ax_eq.text(0.463, 0.55, '+', size=12, ha='center', va='center', color='k')
    ax_eq.text(0.618, 0.55, '+', size=12, ha='center', va='center', color='k')