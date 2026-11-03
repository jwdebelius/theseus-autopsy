import itertools as it

import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.formula.api as smf
import patsy

import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn

import thesus.bw_ana as bw_ana

hide_axes = dict(left=False,  labelleft=False, 
                 right=False, labelright=False,
                 length=0,
                 labelsize=0)

r_models = [bw_ana.ignore_cohort, 
            bw_ana.fixed_type,
            bw_ana.fixed_cohort, 
            bw_ana.gee_cohort, 
            bw_ana.lme_cohort, 
            bw_ana.meta_cohort,
            ]
r_tech_order = ['ignore',  'risk', 'fixed', 'gee', 'lme', 'meta',]
r_tidy_short = {'gee': 'GEE', 
              'lme': 'RE', 
              'meta': 'MA',
              'risk': 'ICS Type',
              }


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

    ax_dist.xaxis.set_tick_params(bottom=False, labelbottom=False)
    ax_dist.yaxis.set_tick_params(left=False, labelleft=False)


def _plot_gauss_curve(ax, color, ylo=-0.35, yhi=1.45, textoff=0.05, 
                      sigma_eq='$\\sigma^{2}=1$', text_kws=None, plot_ticks=True, 
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
    ydiff = np.diff(ylim)[0]
    ax.set_ylim(ylim[0] + ylo * ydiff,
                ylim[0] + ydiff * yhi)

    # Annotates the data with the sigma values
    ax.text(x=0, 
            y=y.max() + ydiff * textoff, 
            s=f'$\\mu = 0$\n{sigma_eq}', 
            **text_kwargs,
            )

    # Plots reference spans with an offset
    if plot_ticks:
        xmin, xmax = ax.get_xlim()
        xticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 2)
        ax.plot(xticks, xticks * 0 - tick_off * ydiff,
                   color='k', linestyle='-', marker='|', linewidth=0.75,
                   )
        ax.set_xlim(xmin, xmax)

    if hide:
        ax.xaxis.set_tick_params(**hide_axes)
        ax.yaxis.set_tick_params(**hide_axes)
        sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)    


def _wrap_categorical_bars(fig, gs, data, cat_pos=None, palettes=None, 
                           ref_ax=None):
    """
    Plots the categprocal data by cohort site
    """
    colors = {'race': {0: '#66c2a5',
                       1: '#fc8d62', 
                       2: '#8da0cb', 
                       3: '#e78ac3'}, 
              'ethnicity': {0: '#9e9ac8', 
                              1: '#54278f'},
              'sex': {0: '#74c476', 
                        1: '#006d2c'},
              'smoking': {0: '#fcae91', 
                            1: '#cb181d'},
              }
    if isinstance(palettes, dict):
        colors.update(palettes)
    if cat_pos is None:
        cat_pos = {'race': 1, 'ethnicity': 2, 'sex': 3, 'smoking': 6}
    for col_, axi in cat_pos.items():
        ax = fig.add_subplot(gs[axi], sharey=ref_ax)
        _plot_categorical_by_cohort(ax, data=data, column=col_, 
                                    colors=colors[col_])
        ax.set_xlim(0, 1)
        ax.set_title(col_.title(), size=10)


def _summarise_histogram(fig, gs, data, off=4, con_pos=None):
    """
    Plots cohort-level histograms
    """
    if con_pos is None:
        con_pos = {'m_age': [17, 43], 'year': [0, 20]}

    # Plots the categorical histograms
    for axi, (col, xlim) in enumerate(con_pos.items()):
        ax = fig.add_subplot(gs[axi + off], sharey=fig.axes[0])
        _plot_cont_histogram_by_cohort(ax, data, col, xlim=xlim)
        ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8,
                                 top=False, labeltop=False)
        ax.yaxis.set_tick_params(**hide_axes)
        if col == 'm_age':
            ax.set_xticks(np.linspace(20, 40, 3))
            ax.set_xlabel('age (years)', size=8)
            ax.set_title('Maternal\nAge', size=10)
        elif col == 'year':
            ax.set_xticks(np.linspace(2, 18, 3))
            ax.set_xlabel('Year', size=8)
            ax.set_title('Calendar\nyear', size=10)
        ax.xaxis.set_major_formatter(
            ticker.StrMethodFormatter('{x:1.0f}'))


def _summarize_kde(fig, gs, data, off=7, cont_cols=None):
    """
    Plots cohort-level kdes
    """
    if cont_cols is None:
        cont_cols = ['y']
    for axi, col in enumerate(cont_cols):
        ax = fig.add_subplot(gs[axi + off], sharey=fig.axes[0])
        _plot_cont_kde_by_cohort(ax, data, 'y')
        ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
        ax.yaxis.set_tick_params(**hide_axes)
        sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)
        if col == 'y':
            ax.set_xlabel('z-score', size=8)
            ax.set_title('Outcome', size=10)
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.1f}'))


def _build_exp_brace(ax_r):
    """
    Builds an exposure icon
    """
    ax_r.set_ylim(-1.45, 1.45)
    ax_r.set_xlim(-0.45, 0.65)
    # # ax_r.set_xlim(0, 1)
    brace_x = np.array([0, 1, 1, 2, 1, 1, 0]) * 0.125 * np.diff(ax_r.get_xlim()) + \
        ax_r.get_xlim()[0] + 0.1 * np.diff(ax_r.get_xlim())
    brace_y = np.array([0.04, 0.05, 0.49, 0.5, 0.51, 0.95, 0.96]) - 0.5
    brace_y = brace_y * np.diff(ax_r.get_ylim()) * 0.8 + \
        np.mean(ax_r.get_ylim())
    ax_r.plot(brace_x, brace_y, color='k', linewidth=1,)


def _plot_betas(axr, icc, fit_res, tidy_short, labelicc=True, 
                error_kws=None):
    """
    Plots the fit
    """
    # print('betas')
    # print(error_kws)

    error_kwargs = dict(marker='o',
                        markersize=5,
                        capsize=3,
                        linestyle='')
    if isinstance(error_kws, dict):
        error_kwargs.update(error_kws)
    if tidy_short is None:
        tidy_short = r_tidy_short
    for (c, fc), df in fit_res.groupby(['color', 'facecolor']):
        axr.errorbar(y='y', x='param', xerr='width', 
                     data=df,
                     color=c,
                     markerfacecolor=fc,
                     **error_kwargs,
                     )
    axr.plot([0, 0], [-1, 6], 'k-', linewidth=1)
    axr.plot([-0.5, -0.5], [-1, 6], 'k:', linewidth=1)
    axr.set_yticks(fit_res['y'])
    axr.set_yticklabels([tidy_short.get(i, i.title()) for i in fit_res.index])
    axr.yaxis.set_tick_params(left=False, labelleft=True, labelsize=10, 
                              length=0)
    axr.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.2f}'))
    axr.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=10)
    axr.set_xlabel('$\\hat{\\beta}_{smoking}$', size=12)
    axr.set_ylim(-0.5, 5.5)
    if labelicc:
        axr.text(x=0 - float(np.diff(axr.get_xlim())[0]) * 0.025, 
                 y=axr.get_ylim()[1],
                 s=f'ICC={icc:1.2f}',
                 size=8, 
                 ha='right', 
                 va='bottom',
                 # va='top'
                 )
    sn.despine(ax=axr, left=True, right=True, top=True, bottom=False)


def _fit_kde(x):
    """
    Fits a 1D gaussian kernel density estimate with cut 0
    """
    xmin = x.min()
    xmax = x.max()
    xpred = np.linspace(xmin, xmax, 100)
    kernel = scipy.stats.gaussian_kde(x)
    den_pred = kernel(xpred)

    return np.vstack([xpred, den_pred])


def _plot_cont_histogram_by_cohort(ax, data, column, color='#525252', 
                                   xlim=None):
    """
    Plots a histogram of continous data
    """
    # Counts the values in bins
    counts0 = _summarize_categorical(data, column)
    counts1 = counts0.pivot(index='cohort', columns='groups', values='counts')
    counts1.fillna(0, inplace=True)
    counts1.sort_index(axis='columns', ascending=True, inplace=True)
    counts2 = counts1.div(counts1.max(axis=1), axis=0) * 0.8
    
    y = data.groupby('cohort')['x'].mean()[counts2.index]

    for coh, y_ in y.items():
        ax.bar(x=counts2.columns.astype(int),
               height=counts2.loc[coh],
               bottom=y_ - 0.5 + 0.1,
               width=1,
               linewidth=0,
               color=color,
          )
    if xlim is not None:
        ax.set_xlim(*xlim)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=False)
    ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
    ax.yaxis.set_tick_params(**hide_axes)


def _plot_cont_kde_by_cohort(ax, data, column, palette=None, ref_color='#252525', 
                             shade=True, linewidth=1):
    """
    Plots a kde plot of continous data
    """
    if palette is None:
        palette = {0: mpc.to_hex(sn.color_palette()[1]),
                   1: mpc.to_hex(sn.color_palette()[0]),
                   }
    trace_vals = data.groupby('cohort')[column].apply(_fit_kde).to_dict()
    for (type_, cohort), y in \
            data.groupby(['cohort_type', 'cohort'])['x'].mean().items():
        color = palette.get(type_, ref_color)
        [xx, yy] = trace_vals[cohort]
        yy = yy / yy.max() * 0.75
        y0 = xx * 0 + y - 0.5 + 0.125
        if shade:
            ax.fill_between(x=xx,  y1=y0, y2=yy + y0,
                            alpha=0.5,
                            edgecolor='None',
                            facecolor=color)
        ax.plot(xx, yy + y0, linewidth=linewidth, 
                color=color)


def _plot_categorical_by_cohort(ax_, data, column, colors):
    """
    Plots a stacked barchart showing protion of groups in a histogram
    """
    count_summary = _summarize_categorical(data, column)
    rel_summary = count_summary.pivot(index='cohort',
                                      columns='groups', 
                                      values='percent').fillna(0)
    rel_summary.sort_index(axis='columns', inplace=True)
    left_ = rel_summary.cumsum(axis=1) - rel_summary
    
    y = data.groupby('cohort')['x'].mean()[rel_summary.index]

    for c, v in rel_summary.items():
        ax_.barh(y=y, left=left_[c], width=v, color=colors[c])
    ax_.yaxis.set_tick_params(**hide_axes)
    ax_.xaxis.set_tick_params(**hide_axes)
    sn.despine(ax=ax_, left=True, right=True, top=True, bottom=True)


def _summarize_categorical(data, cat_col, grouper='cohort', ref_='x'):
    """
    Tabulates group size and proprotion for categorical data 
    """
    counts = data.groupby([cat_col, grouper])[ref_]
    counts = counts.count().reset_index()
    counts.rename(columns={ref_: 'counts', 
                           cat_col: 'groups',
                           grouper: 'cohort'}, 
                  inplace=True)
    group_counts = data[grouper].value_counts().astype(str).to_dict()
    counts['group_size'] = counts['cohort'].replace(group_counts).astype(int)
    counts['percent'] = counts['counts'] / counts['group_size']
    counts['column'] = cat_col

    counts = counts[['column', 'groups', 'cohort', 
                     'group_size', 'counts', 'percent']]

    return counts


def _plot_dist_ref(ax_dist, fontsize=7.5, color='BuGn', box_kws=None, text=True,
                   text_kws=None, dist_trace_kws=None):
    """
    Makes a square plot for the reference distributions
    """
    colors = sn.color_palette(color, 9)

    box_kwargs = dict(facecolor=colors[0], edgecolor=colors[-1], linewidth=1)
    if isinstance(box_kws, dict):
        box_kwargs.update(box_kws)
    text_kwargs = dict(color=colors[-1], ha='left', va='top', size=fontsize)
    if isinstance(text_kws, dict):
        text_kwargs.update(text_kws)

    # Plots the table and text
    for i, label in enumerate(['sex', 'race', 'age']):
        ax_dist.fill_between(x=np.array([0, 1]) + 0.1 * i,
                             y1 = 0 - 0.2 * i,
                             y2 = 1 - 0.2 * i,
                             **box_kwargs,
                             zorder=2 * i,
                             )
        if text:
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
    dist_kwargs = dict(linewidth=1.5)
    if isinstance(dist_trace_kws, dict):
        dist_kwargs.update(dist_trace_kws)

    ax_dist.fill_between(x, y1=y1, y2=y2, linewidth=0, color=colors[4], 
                         zorder=6)
    ax_dist.plot(x, y2, color=colors[-2], zorder=7, 
                 **dist_kwargs)

    ax_dist.set_xticks([])
    ax_dist.set_yticks([])


def _calc_exp_prop_pop(cohort_data, smoking_effect):
    """
    Calculates the p-value modifier for cohort_data
    """
    def extract_var(x):
        x = x.split('[')[0]
        return x
    
    formula = '+'.join(np.unique([
        extract_var(x) for x in smoking_effect.index
    ]))
    expanded = patsy.dmatrix(data=cohort_data, formula_like=formula)
    expanded = pd.DataFrame(data=expanded, index=cohort_data.index,
                            columns=expanded.design_info.column_names)
    keep_cols = [i for i in smoking_effect.index if i in expanded.columns]

    p_mod = (expanded[keep_cols] * smoking_effect[keep_cols]).sum(axis=1)

    return p_mod


def _build_mod_cs(p_mod, palette='PuOr_r'):
    """
    Converts prob modifiers into colors
    """
    p_mod.index.set_names('id', inplace=True)
    p_abs = np.absolute(p_mod)
    thresh = p_abs.quantile([0.8, 0.4, 0.2, 0.2, 0.4, 0.8])
    thresh = thresh * np.array([-1, -1, -1, 1, 1, 1])
    level = pd.concat(axis=0, objs=[p_mod > t for t in thresh])
    level = level.groupby('id').sum()
    colors = pd.Series({
        i: mpc.to_hex(c) 
        for i, c in enumerate(sn.color_palette(palette, n_colors=7))
        })
    p_colors = level.replace(colors)

    return p_colors


def _plot_icc_wedge(ax, offset=0, slope=0.4):
    """Plots a reference wedge for the ICC"""
    x = np.linspace(0, 1, 100)
    y1 = np.zeros(x.shape)
    y2 = x * slope

    # Plots 
    ax.fill_between(x=x, 
                        y1=y1 + offset + 0.3, 
                        y2=y2 + offset + 0.3,
                        edgecolor='k',
                        facecolor='#dbdbdb',
                        linewidth=1,
                        )
            
    # Plots axis in panel
    ax.plot([0, 1], np.array([0.25, 0.25]) + offset, 'k-', linewidth=1,)
    ax.plot(np.vstack([np.linspace(0, 1, 5)] * 2),
                np.vstack([np.ones(5) * 0.15, np.ones(5) * 0.25]) + offset,
                color='k',
                linewidth=1,
                )
    for x in np.linspace(0, 1, 3):
        ax.text(x, offset, f'{x:1.1f}', ha='center', va='bottom', size=8)


def _fit_models_no_filt(data, models=None, tech_order=None):
    """
    Fits unfiltered data to generate plottable results
    """
    if models is None:
        models = r_models
    if tech_order is None:
        tech_order = r_tech_order

    crude_lme = smf.mixedlm('y ~ 1', 
                            groups='cohort', 
                            data=data.loc[data['cohort_type'].isin([0, '0'])],
                            ).fit()
    crude_icc = bw_ana.icc_from_lme(crude_lme)
    adj_eq = 'm_age + year + C(race) + C(ethnicity) + C(sex)'
    fit_res = {tech: fit_(crude_icc, data, adj_eq)[2]
               for tech, fit_ in zip(*(tech_order, models))}
    fit_res = pd.DataFrame.from_dict(fit_res, orient='index')
    fit_res.index.set_names('model', inplace=True)
    fit_res['width'] = fit_res['ci_hi'] - fit_res['param']
    fit_res['color'] = 'k' #'#1b9e77'
    fit_res['sig'] = \
        ((fit_res['p-value'] < 0.05) & fit_res['p-value'].notna()) | \
        ((np.sign(fit_res['ci_lo']) == np.sign(fit_res['ci_hi'])) & 
         fit_res['p-value'].isna())
    fit_res['facecolor'] = \
        fit_res['color'].mask(~fit_res['sig'], 'w')
    fit_res['y'] = np.arange(len(fit_res))[::-1]

    return fit_res, crude_icc
