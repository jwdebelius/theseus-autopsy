import pandas as pd
import numpy as np
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sn

import patsy


from ._utils import (_make_ref_table,
                            hide_axes,
                            _build_exp_brace
                            )


def illustrate_exposure_res(fig_rand, cohort_data):
    """
    Illustrates the expsoures simulation  with real data
    """
    # Adds axes and sets their positions, assuming the figure is 2.5" wide 
    # and 1" tall
    ax_r1 = fig_rand.add_subplot(1, 2, 1)
    ax_r2 = fig_rand.add_subplot(1, 2, 2)

    ax_r1.set_position((0.025, 0.025, 0.45, 0.9))
    ax_r2.set_position((0.675, 0.325, 0.3, 0.625))

    # Gets the latex table and plots it
    exp_lt = _build_exposure_latex_table(cohort_data)
    _plot_exposure_table(ax_r1, exp_lt)

    # Plots the proprotion of samples
    _plot_exp_prop(ax_r2, cohort_data)


def select_exposure(fig2, cohort_data, smoking_effect, overlay=True):
    """
    Builds a figure showing the exposure selection process for cohorts with 
    a population-based seed or selection critierion
    """
    gs2 = fig2.add_gridspec(2, 4)
    ax_prob2 = fig2.add_subplot(gs2[1, 0:-1], facecolor='w')
    ax_demo2 = fig2.add_subplot(gs2[1, 0], facecolor='None')
    ax_prob1 = fig2.add_subplot(gs2[0, 0:-1], facecolor='None')
    ax_demo1 = fig2.add_subplot(gs2[0, 0], facecolor='None')

    ax_r = fig2.add_subplot(gs2[0:2, -1], facecolor='None')

    for ax in fig2.axes:
        ax.xaxis.set_tick_params(**hide_axes)
        ax.yaxis.set_tick_params(**hide_axes)

    ax_prob1.set_position((0, 0.525, 0.75, 0.425))
    ax_demo1.set_position((0, 0.525, 0.25, 0.425))
    ax_prob2.set_position((0, 0.025, 0.75, 0.425))
    ax_demo2.set_position((0, 0.025, 0.25, 0.425))
    ax_r.set_position((0.75, 0, 0.25, 1))

    _build_demo_prob_plot(ax_prob1, ax_demo1, cohort_data, smoking_effect)

    _build_selection_prob_plot(ax_prob2, ax_demo2, overlay=overlay)
    _make_ref_table(ax_r, linewidth=1, palette='Reds', 
                              offset=0.05, n_cols=2)
    _build_exp_brace(ax_r)

    ax_prob1.text(np.mean(ax_prob1.get_xlim()), 
                  ax_prob1.get_ylim()[0],
                  'OR',
                  ha='center', 
                  va='top',
                  size=12
                 )

    sn.despine(fig2, left=True, right=True, top=True, bottom=True)


def _build_demo_prob_plot(ax_prob, ax_demo, cohort_data, smoking_effect, 
                          p_base=0.1, 
                          palette="PuOr_r"):
    """
    Builds a plot of effect modification based on the plotted data
    """
    # Pulls out the color effects needed for plotting the data
    p_mod = _calc_exp_prop_pop(cohort_data, smoking_effect)
    p_color = _build_mod_cs(p_mod)

    # Sets up the plotting by level
    sub_p = (np.hstack([p_mod.head(3), p_mod.tail(1)]) + 1) * 0.1
    exp_v = pd.concat(axis=0, objs=[cohort_data['smoking'].head(3), 
                                    cohort_data['smoking'].tail(1)])
    exp_y = np.array([0.8, 0.6, 0.4, 0]) + 0.1

    # Gets the emographic axis and plots that...
    _make_ref_table(ax_demo, linewidth=0.5, palette='Greys', 
                              offset=0.05)
    ax_demo.set_xlim(-0.05, 1.45)
    ax_demo.set_ylim(-1.25, 1.25)

    # Sets up the probability axis  with guide points... to be removed later
    # ax_prob.set_xticks(np.linspace(0, 1, 10))
    ax_prob.xaxis.set_tick_params(labelbottom=False, bottom=False)
    ax_prob.set_xlim(-0.42, 1.0)
    ax_prob.set_ylim(0, 1)
    ax_prob.yaxis.set_tick_params(left=False, labelleft=False)

    # Plots the cohort modification
    sub_color = np.hstack([p_color.head(3), 'None', p_color.tail(1)])
    color_y = np.linspace(-0.13, 0.13, 5) + 0.5
    step = np.round(np.diff(color_y)[0], 4)
    for c, y in zip(*(sub_color, color_y[::-1])):
        ax_prob.bar(x=[0.155], bottom=[y - step / 2], height=step, 
                    color=c, 
                    width=0.05, 
                    edgecolor='k', 
                    linestyle={'None': ':'}.get(c, '-'),
                    linewidth=0.5,
                    )
    ax_prob.plot(np.array([[0.23, 0.23], [0.27, 0.27]]), 
                 np.array([[0.475, 0.525], [0.525, 0.475]]),
             color='k', linewidth=1,)

    ax_prob.barh(y=0.5, 
                 left=np.array([0, 0.02 * p_base / 0.1]) + 0.3,  
                 width=np.array([p_base, 1 - p_base]) * 0.02 / 0.1, 
                 height=0.075, 
                 edgecolor='k',
                 linewidth=0.5, 
                 color=['#a50f15', '#fee0d2']
                 )

    # Plots the individual probabiolity and exposure
    ax_prob.scatter(0.9 * np.ones(4), exp_y, 
                c=exp_v.replace({0: '#fee0d2', 1: '#a50f15'}),
                s=24,
                edgecolor='k',
                linewidth=0.5,
                )
    for i, (y, p) in enumerate(zip(*(exp_y, sub_p  * 0.0125 / 0.1 ))):
        ax_prob.barh(y=y, 
                     left=np.array([0, p]) + 0.675, 
                     width=np.array([p, 0.125-p]), 
                     height=0.05, edgecolor='k',linewidth=0.25, 
                     color=['#a50f15', '#fee0d2'])
        ax_prob.arrow(x=0.825, dx=0.025, y=y, dy=0, length_includes_head=True, 
                      linewidth=0, width=0.025, head_length=0.025, color='k')
    # Plotted elipses
    ax_prob.plot(np.array([0, 0.15]) + 0.75, [0.3, 0.3], ':', color='k')

    # Adds remaining arrows
    for i in range(5):
        ax_prob.arrow(x=0.55, 
                      dx=0.65-0.55, 
                      y=0.5 + (2 - i) * 0.05, 
                      dy=(0.9 - i * 0.2) - ((2 - i) * 0.05 + 0.5),
                      linewidth=0,
                      color='k',
                      length_includes_head=True,
                      width=0.015,
                      )
    ax_prob.arrow(x=0, dx=0.1, y=0.5, dy=0, 
                  length_includes_head=True, 
                  color='k',
                  width=0.1,
                  linewidth=0,
                  head_length=0.075,
                  head_width=0.2,
                  )

def _build_selection_prob_plot(ax_prob, ax_demo, p_select=0.5, overlay=True):
    """
    Simulates data assuming a selection rate for the outcome
    """
    # Sets up the axis
    ax_demo.set_xlim(-0.05, 1.15)
    ax_demo.set_ylim(-1.5, 1.5)

    # Sets up the probability axis  with guide points... to be removed later
    # ax_prob.set_xticks(np.linspace(0, 1, 10))
    ax_prob.xaxis.set_tick_params(labelbottom=False, bottom=False)
    ax_prob.set_xlim(-0.42, 1.0)
    ax_prob.set_ylim(0, 1)
    ax_prob.yaxis.set_tick_params(left=False, labelleft=False)

    # Plots the probability bar 
    ax_prob.barh(y=0.5, 
                 left=np.array([0, p_select]) / 0.1 * 0.02 + 0.3, 
                 width=np.array([p_select, 1 - p_select]) / 0.1 * 0.02,
                 height=0.075, 
                 edgecolor='k',
                 linewidth=0.5, 
                 color=['#a50f15', '#fee0d2'])

    # Estimates and outcome rate and plots that
    exp_v = pd.Series(np.random.binomial(n=1, p=p_select, size=4))
    ax_prob.scatter(
        y=np.array([0.9, 0.7, 0.5, 0.1]),
        x=0.9 * np.ones(4),
        c=exp_v.replace({0: '#fee0d2', 1: '#a50f15'}),
        s=24,
        edgecolor='k',
        linewidth=0.5,
    )

    # Adds in arrows to individual values and elispes
    for i in range(5):
        y1 = (i - 2) * 0.2 + 0.5
        y2 = (i - 2) * 0.05 + 0.5
        ax_prob.arrow(x=0.6, dx=0.75-0.55, 
                       y=y2, dy=y1 - y2,
                       linewidth=0,
                       color='k',
                       length_includes_head=True,
                       width=0.015,
                       )
    ax_prob.plot(np.array([0, 0.15]) + 0.85, [0.3, 0.3], ':', color='k')

    # Adds overlay if desired
    if overlay:
        ax_c = ax_prob.get_facecolor()
        fig_c = mpc.to_hex(ax_prob.figure.get_facecolor())
        if (ax_c[-1] == 0) & (fig_c[-1] == 0):
            over_color = 'w'
        elif (ax_c[-1] == 0):
            over_color = mpc.to_hex(fig_c)
        else:
            over_color = mpc.to_hex(ax_c[:-1])

        ax_prob.fill_between(ax_prob.get_xlim(), *ax_prob.get_ylim(), 
                             color=over_color, 
                             alpha=0.75,
                             zorder=4,)

def extract_var(x):
    x = x.split('[')[0]
    return x
    

def _calc_exp_prop_pop(cohort_data, smoking_effect):
    """
    Calculates the p-value modifier for cohort_data
    """
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


def illustrate_exposure_res(fig_rand, cohort_data):
    """
    Illustrates the expsoures simulation  with real data
    """
    # Adds axes and sets their positions, assuming the figure is 2.5" wide 
    # and 1" tall
    ax_r1 = fig_rand.add_subplot(1, 2, 1)
    ax_r2 = fig_rand.add_subplot(1, 2, 2)

    ax_r1.set_position((0.025, 0.025, 0.45, 0.9))
    ax_r2.set_position((0.675, 0.325, 0.3, 0.625))

    # Gets the latex table and plots it
    exp_lt = _build_exposure_latex_table(cohort_data)
    _plot_exposure_table(ax_r1, exp_lt)

    # Plots the proprotion of samples
    _plot_exp_prop(ax_r2, cohort_data)


def _build_exposure_latex_table(cohort_data):
    """
    Constructs a latex formatted table for the exposure groups
    """
    exp_table = pd.concat(
    axis=0, 
    objs=[cohort_data[['smoking']].reset_index(drop=True).head(3), 
          cohort_data[['smoking']].tail(1).replace({0: np.nan, 1: np.nan}), 
          cohort_data[['smoking']].reset_index(drop=True).tail(1)]
    )
    exp_table.index = exp_table.index + np.array([1, 1, 1, 0, 1])
    exp_table.rename({exp_table.index[-2]: '...'}, inplace=True)
    exp_table = exp_table['smoking'].apply(
        lambda x: '...' if pd.isnull(x) else f'{x:1.0f}')
    exp_table.index.set_names('PregID', inplace=True)
    exp_table = exp_table.reset_index()
    exp_table2 = exp_table.copy()

    exp_lt = exp_table.to_latex(index=False, column_format='| c | c |')
    for rule_ in ['\\toprule', '\\bottomrule', '\\midrule']:
        exp_lt = exp_lt.replace(rule_, '\\hline')
    exp_lt = exp_lt.replace('\\\\\n', '\\\\\n\\hline\n')
    exp_lt = exp_lt.replace('\\hline\n\\hline', '\\hline')

    return exp_lt


def _plot_exposure_table(ax_r1, exp_lt):
    """
    Plots the exposure latex table plus a background
    """
    # Sets up the axes
    ax_r1.set_xlim(0, 1)
    ax_r1.set_ylim(0, 1)
    ax_r1.xaxis.set_tick_params(left=False, labelleft=False, length=0)
    ax_r1.yaxis.set_tick_params(left=False, labelleft=False, length=0)
    # Adds the table
    ax_r1.text(0.5, 0.5, exp_lt.replace('\n', ' '), ha='center', va='center', 
               size=8.5)
    # Adds table background
    ax_r1.fill_between([0, 1], y1=0, y2=1, color='#ffdbdb')
    ax_r1.fill_between([0, 1], y1=0.825, y2=1, color='#ff9191')

    # Despite
    sn.despine(ax=ax_r1, left=True, right=True, top=True, bottom=True)
    return exp_lt


def _plot_exp_prop(ax_r2, cohort_data):
    """
    Plots the porportion of exposed and unexposed individuals in a study
    """
    percent = \
        cohort_data['smoking'].value_counts().sort_index() / len(cohort_data)
    ax_r2.bar(x=np.array([0.5, 1.5]),
              bottom=0,
              height=percent,
              color=['#fee0d2', '#a50f15'],
              edgecolor='k',
              linewidth=1,
              width=0.75,
              )
    ax_r2.set_xlabel('Smoking', size=8)
    ax_r2.set_ylabel('\\% Part.', size=8, labelpad=0)

    ax_r2.set_xlim(-0.1, 2.1)
    ax_r2.set_xticks([0.5, 1.5])
    ax_r2.set_xticklabels(['$\\texttt{0}$', '$\\texttt{1}$'])
    ax_r2.xaxis.set_tick_params(bottom=False, labelbottom=True, 
                                labelsize=8, length=0)
    ax_r2.set_ylim(0, 1)
    ax_r2.set_yticks(np.linspace(0, 1, 5))
    ax_r2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.0%}'))
    ax_r2.yaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
