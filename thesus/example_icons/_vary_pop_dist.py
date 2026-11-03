import itertools as it

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sn

from ._utils import (_plot_dist_ref,
                                          _plot_categorical_by_cohort,
                                          _plot_cont_histogram_by_cohort,
                                          hide_axes,
                                          _calc_exp_prop_pop, 
                                          _build_mod_cs,
                                          _plot_cont_kde_by_cohort,
                                          _plot_icc_wedge,
                                          )



def show_vary_on_outcome(fig, example_data0, example_data1, arrow_kws=None):
    """
    Example selection on outcome
    """

    dist_kwargs, arrow_kwargs = \
        _set_dist_arrow_kwargs(None, arrow_kws)
    
    sub_data0 = _relocate_sub_data(example_data0)
    sub_data4 = _relocate_sub_data(example_data1, [21, 13, 12, 0])
    
    
    gs = fig.add_gridspec(11, 5, width_ratios=[1, 1, 3, 1, 3])
    
    # Sets up the label axes
    ax0_ = fig.add_subplot(gs[:, 0])
    ax10 = fig.add_subplot(gs[0:5, 1])
    ax11 = fig.add_subplot(gs[6:11, 1])
    _label_fig(ax0_, ax10, ax11, 'Select on outcome')
    
    # Uniform outcome
    axo = fig.add_subplot(gs[2, 2])
    _label_cohort_type(axo, 0)
    axa0 = fig.add_subplot(gs[0:5, 3], facecolor='None')
    _size_arrow(axa0, arrow_kwargs)
    
    for y, cdf in sub_data4.groupby('x'):
        axs = fig.add_subplot(gs[6 + (4 - y), 2], facecolor='None')
        axa1 = fig.add_subplot(gs[6 + (4 - y), 3], facecolor='None')
        axo = fig.add_subplot(gs[6 + (4- y), 4], facecolor='None')
        _label_cohort_type(axs, cdf['y'].mean())
        _plot_ha_arrow(axa1, arrow_kwargs)
        _plot_cont_kde_by_cohort(axo, data=cdf, column='y')
        axo.set_xlim(-5, 2.5)
        axo.xaxis.set_tick_params(**hide_axes)
        axo.yaxis.set_tick_params(**hide_axes)
        sn.despine(ax=axo, left=True, right=True, top=True, bottom=True)
    
    for y, cdf in sub_data0.groupby('x'):
        axo = fig.add_subplot(gs[(4- y), 4], facecolor='None')
        _plot_cont_kde_by_cohort(axo, data=cdf, column='y')
        axo.set_xlim(-5, 2.5)
        axo.xaxis.set_tick_params(**hide_axes)
        axo.yaxis.set_tick_params(**hide_axes)
        sn.despine(ax=axo, left=True, right=True, top=True, bottom=True)
    
    ax00 = fig.add_subplot(gs[0:5, 2:], facecolor='None')
    ax00.xaxis.set_tick_params(**hide_axes)
    ax00.yaxis.set_tick_params(**hide_axes)
    ax00.set_position((0.325, 0.515, 0.65, 0.375))
    
    ax11 = fig.add_subplot(gs[6:11, 2:], facecolor='None')
    ax11.xaxis.set_tick_params(**hide_axes)
    ax11.yaxis.set_tick_params(**hide_axes)
    ax11.set_position((0.325, 0.10, 0.65, 0.375))
    
    for xx, yy in it.product([-1, -3], [9, 3]):
        if (xx == -3) & (yy == 3):
            continue
        ax = fig.add_subplot(gs[yy, xx], facecolor=None)
        _add_ellipsus_ax(ax)



def show_vary_on_exposure(fig, example_data0, example_data1, smoking_effect, 
                          arrow_kws=None):
    """
    Example comparing selection on exposure
    """

    dist_kwargs, arrow_kwargs = \
        _set_dist_arrow_kwargs(None, arrow_kws)

    sub_data3 = _relocate_sub_data(example_data1)
    sub_data0 = _relocate_sub_data(example_data0)


    gs = fig.add_gridspec(11, 7, width_ratios=[1, 1, 2, 1, 2.5, 1, 2])
    
    ax0_ = fig.add_subplot(gs[:, 0])
    ax10 = fig.add_subplot(gs[0:5, 1])
    ax11 = fig.add_subplot(gs[6:11, 1])
    _label_fig(ax0_, ax10, ax11, 'Select on Expsoure')
    
    # Single rate
    ax0l = fig.add_subplot(gs[2, 2])
    _plot_rate(ax0l, 0.1)
    axa0 = fig.add_subplot(gs[0:5, 3], facecolor='None')
    _size_arrow(axa0, arrow_kwargs)
    for y, cdf in sub_data0.groupby('x'):
        ax_p = fig.add_subplot(gs[4 - y, 4], facecolor='None')
        axa = fig.add_subplot(gs[4-y, 5])
        axd = fig.add_subplot(gs[4 - y, 6], facecolor='None')
        _plot_pop_dist(ax_p, cdf, smoking_effect)
        _plot_ha_arrow(axa, arrow_kwargs)
        cdf['x'] = 0
        _wrap_exp_distributions(axd, cdf)
    
    for y, cdf in sub_data3.groupby('x'):
        axs = fig.add_subplot(gs[6 + (4 - y), 2], facecolor='None')
        axa1 = fig.add_subplot(gs[6 + (4 - y), 3], facecolor='None')
        axa2 = fig.add_subplot(gs[6 + (4 - y), 5], facecolor='None')
        axd = fig.add_subplot(gs[6 + (4 - y), 6], facecolor='None')
        s_ = _plot_rate(axs, cdf['smoking'].mean())
        _plot_ha_arrow(axa1, arrow_kwargs)
        _plot_ha_arrow(axa2, arrow_kwargs)
        if s_ == 'Pop':
            axp = fig.add_subplot(gs[6 + (4 - y), 4], facecolor='None')
            _plot_pop_dist(axp, cdf, smoking_effect)
        else:
            axp = fig.add_subplot(gs[6 + (4 - y), 4], facecolor='None')
            _plot_mod_dist(axp, s_)
        cdf['x'] = 0
        _wrap_exp_distributions(axd, cdf)

    for xx, yy in it.product([-1, -3], [9, 3]):
        ax = fig.add_subplot(gs[yy, xx], facecolor=None)
        _add_ellipsus_ax(ax)
    
    ax00 = fig.add_subplot(gs[0:5, 2:], facecolor='None')
    ax00.xaxis.set_tick_params(**hide_axes)
    ax00.yaxis.set_tick_params(**hide_axes)
    ax00.set_position((0.275, 0.515, 0.65, 0.375))
    
    ax11 = fig.add_subplot(gs[6:11, 2:], facecolor='None')
    ax11.xaxis.set_tick_params(**hide_axes)
    ax11.yaxis.set_tick_params(**hide_axes)
    ax11.set_position((0.275, 0.10, 0.65, 0.375))


def show_vary_population(fig, example_data0, example_data1, 
                         arrow_kws=None,
                         dist_kws=None):
    """
    Shows an example of a static population vs varied population
    """
    dist_kwargs, arrow_kwargs = \
        _set_dist_arrow_kwargs(dist_kws, arrow_kws)
    
    # Sets up the axes
    gs = fig.add_gridspec(11, 7, width_ratios=[1, 1, 1, 2, 2, 1, 2])
    
    # Sets up the label axes
    ax0_ = fig.add_subplot(gs[:, 0])
    ax10 = fig.add_subplot(gs[0:5, 1])
    ax11 = fig.add_subplot(gs[6:11, 1])
    _label_fig(ax0_, ax10, ax11, 'Population')

    # ### NO VARIATION ###
    # Looks at the top row (no variation)
    sub_data0 = _relocate_sub_data(example_data0)
    
    ax0a = fig.add_subplot(gs[0:5, 3])
    ax0r = fig.add_subplot(gs[0:5, 4], sharey=ax0a)
    ax0e = fig.add_subplot(gs[0:5, 5], sharey=ax0a)
    ax0y = fig.add_subplot(gs[0:5, 6], sharey=ax0a)
    
    _wrap_pop_distributions(ax0r, ax0e, ax0y, sub_data0)
    
    # Plots the arrows
    _size_arrow(ax0a, arrow_kwargs)

    # Plots the distribution reference
    axd0 = fig.add_subplot(gs[2, 2], facecolor='None')
    _plot_dist_ref(axd0, **dist_kwargs)
    sn.despine(ax=axd0, left=True, right=True, top=True, bottom=True)
    
    # ### WITH VARIATION ###
    sub_data1 = _relocate_sub_data(example_data1)

    ax1a = fig.add_subplot(gs[6:11, 3])
    ax1r = fig.add_subplot(gs[6:11, 4], sharey=ax1a)
    ax1e = fig.add_subplot(gs[6:11, 5], sharey=ax1a)
    ax1y = fig.add_subplot(gs[6:11, 6], sharey=ax1a)

    _wrap_pop_distributions(ax1r, ax1e, ax1y, sub_data1)
    # Wraps the arrows
    for y in sub_data1['x'].unique():
        ax1a.arrow(x=0.1, y=y, dx=0.9, dy=0, **arrow_kwargs)

    # Plots the distribution plots
    for yy in sub_data1['x'].unique():
        ax1i = fig.add_subplot(gs[6 + (4 - int(yy)), 2], facecolor='None')
        _plot_dist_ref(ax1i, **dist_kwargs)
        sn.despine(ax=ax1i, left=True, right=True, top=True, bottom=True)

    # Cleans up the axes
    for ax_ in [ax0a, ax0r, ax0e, ax0y, ax1a, ax1r, ax1e, ax1y]:
        ax_.xaxis.set_tick_params(**hide_axes, gridOn=False)
        ax_.yaxis.set_tick_params(**hide_axes, gridOn=False)
        ax_.set_yticks(np.arange(0, 5))
        ax_.set_ylim(-0.5 - 0.25 / 2, 4.5  + 0.25 / 2)
        ax_.set_facecolor('None')
        sn.despine(ax=ax_, left=True, right=True, top=True, bottom=True)

    ax00 = fig.add_subplot(gs[0:5, 2:], facecolor='None')
    ax00.xaxis.set_tick_params(**hide_axes)
    ax00.yaxis.set_tick_params(**hide_axes)
    ax00.set_position((0.295, 0.515, 0.65, 0.375))

    ax11 = fig.add_subplot(gs[6:11, 2:], facecolor='None')
    ax11.xaxis.set_tick_params(**hide_axes)
    ax11.yaxis.set_tick_params(**hide_axes)
    ax11.set_position((0.295, 0.10, 0.65, 0.375))


def show_vary_size(fig, example_data0, example_data1, arrow_kws=None):
    """
    Shows an example of a static vs varied cohort sizes
    """
    # Gets size data sets
    sub_data0 = _relocate_sub_data(example_data0)
    sub_data2 = _relocate_sub_data(example_data1)

    # Sets up the defaults
    
    gs = fig.add_gridspec(11, 7, width_ratios=[1, 1, 2, 1, 2, 1, 2.5])
    kk, probs, ydists = _sim_effect_distribution()
    dist_kwargs, arrow_kwargs = _set_dist_arrow_kwargs()

    # Sets up the label axes
    ax0_ = fig.add_subplot(gs[:, 0])
    ax10 = fig.add_subplot(gs[0:5, 1])
    ax11 = fig.add_subplot(gs[6:11, 1])
    _label_fig(ax0_, ax10, ax11, 'Size')


    kk, probs, ydists = _sim_effect_distribution()
    dist_kwargs, arrow_kwargs = _set_dist_arrow_kwargs()


    # Vary sizes
    for (y, _), s in sub_data2.groupby(['x', 'cohort'])['y'].count().items():
        axd = fig.add_subplot(gs[6 + (4 - y), 2])
        axa0 = fig.add_subplot(gs[6 + (4 - y), 3])
        axa1 = fig.add_subplot(gs[6 + (4 - y), 5])
        axe = fig.add_subplot(gs[6 + (4 - y), 4])
        axo = fig.add_subplot(gs[6 + (4 - y), -1])
        _plot_vary_size_dists(axd, kk, ydists, probs)
        _plot_vary_size_dists(axe, kk, ydists, probs)
        _plot_size_dist(axe, s, kk, ydists, probs)
        _plot_ha_arrow(axa0, arrow_kwargs)
        _plot_ha_arrow(axa1, arrow_kwargs)
        _add_size_text(axo, s)
    
    # Uniform sizes
    ax_no0 = fig.add_subplot(gs[2, 2])
    _plot_uniform_size_dists(ax_no0, kk, ydists, probs)
    axa0 = fig.add_subplot(gs[0:5, 3], facecolor='None')
    _size_arrow(axa0, arrow_kwargs)
    
    #Uniform rows
    for (y, _), s in sub_data0.groupby(['x', 'cohort'])['y'].count().items():
        axa1 = fig.add_subplot(gs[(4 - y), 5])
        axe = fig.add_subplot(gs[(4 - y), 4])
        axo = fig.add_subplot(gs[(4 - y), -1])
        _plot_uniform_size_dists(axe, kk, ydists, probs)
        _plot_ha_arrow(axa1, arrow_kwargs)
        _add_size_text(axo, s)

    for xx, yy in it.product([-1, -3], [9, 3]):
        ax = fig.add_subplot(gs[yy, xx], facecolor='None')
        _add_ellipsus_ax(ax)

    ax00 = fig.add_subplot(gs[0:5, 2:], facecolor='None')
    ax00.xaxis.set_tick_params(**hide_axes)
    ax00.yaxis.set_tick_params(**hide_axes)
    ax00.set_position((0.275, 0.515, 0.65, 0.375))
    
    ax11 = fig.add_subplot(gs[6:11, 2:], facecolor='None')
    ax11.xaxis.set_tick_params(**hide_axes)
    ax11.yaxis.set_tick_params(**hide_axes)
    ax11.set_position((0.275, 0.10, 0.65, 0.375))


def _sim_effect_distribution():
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
    probs = [0.5, 0.7, 0.2]

    # Calculates the PMF weighted by the p-value
    ydist = []
    k = np.arange(0, 2001, 5)
    for nn in ns:
        yd = np.vstack([nbpmf(p=pp * 0.1 + 0.2, n=nn, k=k) for pp in px])
        yy = np.average(yd, weights=py, axis=0)
        
        ym = np.hstack([nbavg(p=pp * 0.1 + 0.2, n=nn) for pp in px])
        # ybar = np.average(ym, weights=py)

        ydist.append(yy)

    return k, probs, ydist


def _plot_vary_dists(ax, kk, ydist, probs, line_kws=None):
    """
    Plots size distributions
    """
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_facecolor('None')
    kk_diff = kk.max() - kk.min()
    ax.set_xlim(kk.min() - 0.05 * kk_diff, 
                kk.max() + 0.05 * kk_diff)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)

    
    line_kwargs = dict(linestyle='-', 
                      linewidth=0.5, 
                      color='#919191')
    if isinstance(line_kws, dict):
        line_kwargs.update(line_kws)

    for i, yy in enumerate(ydist):
        ax.plot(kk[yy > 1e-6], yy[yy > 1e-6] / yy.max() * probs[i],
                **line_kwargs)


def _plot_uniform_size_dists(ax, kk, ydists, probs, line_kws=None):
    """Plots uniform size distributions"""
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_facecolor('None')
    kk_diff = kk.max() - kk.min()
    ax.set_xlim(kk.min() - 0.05 * kk_diff, 
                kk.max() + 0.05 * kk_diff)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)

    ax.set_ylim(0, 1)
    ax.plot([250, 250], [0, 1], 'k-', linewidth=1)


def _plot_size_dist(ax, s, kk, ydist, probs):
    """
    Highlights the distribution used
    """
    i = (s > 100) * 1 + (s > 800) * 1
    yr = ydist[i]
    pr = probs[i]
    kk = kk[yr > 1e-6]
    yr = yr[yr > 1e-6] / yr.max() * pr

    ax.fill_between(x=kk, y1=np.zeros(kk.shape), y2=yr,
                    facecolor='#525252',
                    linewidth=0,
                    edgecolor='None')
    ax.plot(kk, yr, 'k-', linewidth=0.75)


def _add_size_text(axo, s):
    """Labels the simulated size"""
    axo.set_xlim(0, 1)
    axo.set_ylim(0, 1)
    axo.text(1, 0.5, f'{s:1,.0f}', ha='right', va='center', size=7)
    axo.xaxis.set_tick_params(**hide_axes)
    axo.yaxis.set_tick_params(**hide_axes)
    axo.set_facecolor('None')
    sn.despine(ax=axo, left=True, right=True, top=True, bottom=True)


def _plot_ha_arrow(axa, arrow_kwargs):
    """Plots horizontal arrow"""
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.arrow(x=0.0, y=0.5, dx=1, dy=0, **arrow_kwargs)
    axa.xaxis.set_tick_params(**hide_axes)
    axa.yaxis.set_tick_params(**hide_axes)
    axa.set_facecolor('None')
    sn.despine(ax=axa, left=True, right=True, top=True, bottom=True)

    
def _add_ellipsus_ax(ax_):
    """
    Adds ellipses to a single axis
    """
    ax_.set_xlim(0, 1)
    ax_.set_ylim(0, 1)
    ax_.text(0.5, 0.75, '...', ha='center', va='center', size=12)
    ax_.xaxis.set_tick_params(**hide_axes)
    ax_.yaxis.set_tick_params(**hide_axes)
    ax_.set_facecolor('None')
    sn.despine(ax=ax_, left=True, right=True, top=True, bottom=True)


def _size_arrow(axa0, arrow_kwargs):
    """Plots arrow expansion"""
    arrow_pos0 = [dict(x=0.1, y=3, dx=0.85, dy=1),
                  dict(x=0.05, y=2.5, dx=0.90, dy=0.5),
                  dict(x=0.0, y=2, dx=1, dy=0),
                  dict(x=0.1, y=1, dx=0.85, dy=-1),
                  ]
    axa0.xaxis.set_tick_params(**hide_axes)
    axa0.yaxis.set_tick_params(**hide_axes, gridOn=False)
    axa0.set_xlim(0, 1)
    axa0.set_ylim(-0.5, 4.5)
    axa0.set_facecolor('None')
    for ap in arrow_pos0:
        axa0.arrow(**ap, **arrow_kwargs)
    sn.despine(ax=axa0, left=True, right=True, top=True, bottom=True)


def _plot_vary_size_dists(ax, kk, ydist, probs, line_kws=None):
    """
    Plots size distributions
    """
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_facecolor('None')
    kk_diff = kk.max() - kk.min()
    ax.set_xlim(kk.min() - 0.05 * kk_diff, 
                kk.max() + 0.05 * kk_diff)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)

    
    line_kwargs = dict(linestyle='-', 
                      linewidth=0.5, 
                      color='#919191')
    if isinstance(line_kws, dict):
        line_kwargs.update(line_kws)

    for i, yy in enumerate(ydist):
        ax.plot(kk[yy > 1e-6], yy[yy > 1e-6] / yy.max() * probs[i],
                **line_kwargs)


def _wrap_exp_distributions(axe, sub_data):
    """
    Wraps the column plotting for vary pop icons
    """
    exp_colors = {0: '#fcae91', 1: '#cb181d'}
    # Plots the race axis
    _plot_categorical_by_cohort(axe, sub_data, 'smoking', exp_colors)
    axe.set_xlim(-0.05, 1.05)

    # Adds text to the distributions
    for ax_ in [axe]:
        ax_.xaxis.set_tick_params(**hide_axes)
        # ax_.set_ylim(-0.75, 4.75)


def _plot_mod_dist(ax, s_):
    [off, on] = np.array(s_.split(':')).astype(int)
    if off == on:
        [off, on] = 0.5
    ax.barh(y=0.5, 
            left=np.array([0, on]) * 0.02 / 0.1  +  0.3,
            width=np.array([on, off]) * 0.02 / 0.1, 
            height=0.075, 
            edgecolor='k',
            linewidth=0.5, 
            color=['#a50f15', '#fee0d2']
            )
    ax.set_xlim(0.1115, 0.5185)
    ax.set_ylim(0.3375, 0.67875)
    # ax.set_ylim(0, 1)
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)
    ax.set_facecolor('None')


def _plot_pop_dist(ax, coh_data, smoking_effect, p_base=0.1):
    """
    Plots 
    """
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_facecolor('None')

    
    coh_data = pd.concat([coh_data.head(3), coh_data.tail(1)])
    p_mod = _calc_exp_prop_pop(coh_data, smoking_effect)
    p_color = _build_mod_cs(p_mod)

    # Plots the cohort modification
    sub_color = np.hstack([p_color.head(3), 'None', p_color.tail(1)])
    color_y = np.linspace(-0.13, 0.13, 5) + 0.5
    step = np.round(np.diff(color_y)[0], 4)
    for c, y in zip(*(sub_color, color_y[::-1])):
        ax.bar(x=[0.155], bottom=[y - step / 2], height=step, 
                    color=c, 
                    width=0.05, 
                    edgecolor='k', 
                    linestyle={'None': ':'}.get(c, '-'),
                    linewidth=0.5,
                    )
    ax.plot(np.array([[0.23, 0.23], [0.27, 0.27]]), 
                 np.array([[0.475, 0.525], [0.525, 0.475]]),
             color='k', linewidth=1,)

    ax.barh(y=0.5, 
            left=np.array([0, 0.02 * p_base / 0.1]) + 0.3,  
            width=np.array([p_base, 1 - p_base]) * 0.02 / 0.1, 
            height=0.075, 
            edgecolor='k',
            linewidth=0.5, 
            color=['#a50f15', '#fee0d2']
            )
    sn.despine(ax=ax, left=True, right=True, bottom=True, top=True)


def _plot_rate(ax, mean):
    """
    Plots the rate of smoking
    """
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    ax.set_facecolor('None')
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if mean == 0:
        s_ = '1:0'
    elif mean == 1:
        s_ = '0:1'
    elif (mean >= 0.25):
        s_ = '1:1'
    else:
        s_ = 'Pop'

    ax.text(0.5, 0.5, s_, ha='center', va='center', size=7)

    return s_


def _label_cohort_type(ax, ybar):
    """
    Labels the cohort type based on the mean
    """
    f_ = lambda x: 'NICU' if (x < -1.75) else 'Gen'
    str_ = f_(ybar)
    color = {'NICU': sn.color_palette()[0], 
             'Gen': sn.color_palette()[1],
             }
    ax.text(0.5, 0.5, str_, ha='center', va='center', size=8,
            color=color.get(str_, 'k')
            )
    ax.xaxis.set_tick_params(**hide_axes)
    ax.yaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax, left=True, right=True, bottom=True)
    ax.set_facecolor('None')


def _label_fig(ax0_, ax10, ax11, property_):
    """
    Labels inset figure for icon plotting
    """
    ax0_.text(0.5, 0.5, property_,  size=8, 
              ha='center', va='center', rotation=90)
    for ax_ in [ax10, ax11]:
        ax_.scatter([0.5], [0.5], 
                    marker='s', 
                    facecolor='None', 
                    edgecolor='k', 
                    linewidth=1)
    ax11.scatter([0.5], [0.5], 
                 marker='s', 
                 facecolor='k', 
                 edgecolor='None',
                 linewidth=0)
    for ax_ in [ax0_, ax10, ax11]:
        ax_.xaxis.set_tick_params(**hide_axes)
        ax_.yaxis.set_tick_params(**hide_axes)
        ax_.set_facecolor('None')
        sn.despine(ax=ax_, left=True, right=True, top=True, bottom=True)
    ax0_.set_position((0.05, 0.1, 0.1, 0.8))
    ax10.set_position((0.15, 0.525, 0.1, 0.9 - 0.525))
    ax11.set_position((0.15, 0.100, 0.1, 0.9 - 0.525))


def _set_dist_arrow_kwargs(dist_kws=None, arrow_kws=None):
    """
    Helper function for plotting arrows and distributions
    """
         # Sets up th e defaults
    dist_kwargs = dict(fontsize=0, 
                       color='Grays', 
                       box_kws=dict(linewidth=0.25),
                       dist_trace_kws=dict(linewidth=0.25),
                       text=False,
                       )
    arrow_kwargs = dict(linewidth=0, 
                        length_includes_head=True, 
                        width=0.15, 
                        head_length=0.25, 
                        color='k')

    if isinstance(dist_kws, dict):
        dist_kwargs.update(dist_kws)
    if isinstance(arrow_kws, dict):
        arrow_kwargs.update(arrrow_kws)

    return dist_kwargs, arrow_kwargs


def _relocate_sub_data(example_data, x=[19, 18, 17, 0]):
    """
    Extracts a subset of the data to plot
    """
    # Extracts the desired rows
    sub_data = example_data.loc[example_data['x'].isin(x)].copy()

    # Sorts them
    sub_data.sort_values('x', ascending=True, inplace=True)
    sub_data['x0'] = sub_data['x']
    # Assigns a new scaling to be a 5 row plot
    sub_data['x'] = (~sub_data['x0'].duplicated(keep='first')) * 1
    sub_data['x'] = sub_data['x'].cumsum() - 1
    sub_data['x'] = sub_data['x'] + (sub_data['x'] > 0) * 1
    
    return sub_data


def _wrap_pop_distributions(axr, axe, axy, sub_data):
    """
    Wraps the column plotting for vary pop icons
    """
    race_colors = {0: '#66c2a5',
                   1: '#fc8d62', 
                   2: '#8da0cb', 
                   3: '#e78ac3'}
    # Plots the race axis
    _plot_categorical_by_cohort(axr, sub_data, 'race', race_colors)
    axr.set_xlim(-0.05, 1.05)
    # Plots the year
    _plot_cont_histogram_by_cohort(axy, sub_data, 'year')
    axy.set_xlim(-1, 21)

    # Adds text to the distributions
    for ax_ in [axr, axy]:
        ax_.text(np.mean(ax_.get_xlim()), 1.2, '...', 
                 ha='center', va='center', size=9)
        ax_.xaxis.set_tick_params(**hide_axes)
        ax_.set_ylim(-0.75, 4.75)

    axe.set_xlim(0, 1)
    for y in sub_data['x'].unique():
        axe.text(0.5, y + 0.2, '...', ha='center', va='center', size=7)
