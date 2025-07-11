r"""
This script will take simulated example data and generate plots showing the 
indivudal cohort effects and the results of the combined and/or filtered
cohorts

Per-Cohort Displays

* 

Pooled Displays
* 

Either
* built_table1
* 

"""

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


def built_table1(example_data):
    """
    Summarizes the example data into a table 1 and its latex friendly cousin

    Parameters
    ----------
    example_data: DataFrame
        The simulated data frame with columns for the maternal age (m_age),
        ethnicity (ethnicity), race (race), sex (sex), year (year), exposure
        (smoking), outcome (y) and cohort id (cohort).

    Returns
    -------
    pd.DataFrame
        A summarized table 1
    """
    
    # Handles the continous statistics by summarizing the data, making tidy
    # strings, and then making a long form version that will play nicely with 
    # the categocial data, since the continous data has one groupingt level
    # (continous covaraite) and the cateogricla data has two (group and 
    # group_value)
    cont_stats = example_data.groupby('smoking')[['m_age', 'year']].describe()
    cont_stats = cont_stats.unstack().unstack(-2)

    cont_stats.index.set_names(['group', 'smoking'], inplace=True)
    cont_stats.rename(columns={'25%': 'q1', '50%': 'q2',  '75%': 'q3', 
                               'count': 'sample_size'}, 
                      inplace=True)
    cont_stats['mean (std)'] = cont_stats.apply(
        lambda x: '{mean:>4.1f} ({std:>4.1f})'.format(**x), axis=1)
    cont_stats['median [25%, 75%]'] = cont_stats.apply(
        lambda x: '{q2:2.0f} [{q1:2.0f}, {q3:2.0f}]'.format(**x), axis=1)
    cont_stats = \
        cont_stats.reset_index().melt(
        id_vars=['smoking', 'group', 'sample_size'],
        value_vars=['mean (std)', 'median [25%, 75%]'],
        var_name='group_value',
        value_name='tidy',
        )
    # Pulls out the categorical data by counting the number of samples ine ach
    # group. We then to some dataframe clean up, to make it pretty and easier 
    # to handle. We calculate hte percentage, tidy the whole thing into a 
    # nice package so we have a long form set of data to matcht he continous
    # results
    categorical = pd.DataFrame.from_dict(
        data={
            col_: example_data.groupby(['smoking', col_])['y'].count()
            for col_ in ['race', 'ethnicity', 'sex']
            }
        )
    categorical.index.set_names(['smoking', 'group_value'], inplace=True)
    categorical.columns.set_names('group', inplace=True)
    categorical = \
        categorical.unstack([-1, -2]).dropna()
    categorical = categorical.reset_index(level=['group', 'group_value'])
    categorical.rename(columns={0: 'count'}, inplace=True)
    categorical['sample_size'] = example_data['smoking'].value_counts()
    categorical['percent'] = categorical['count'] / categorical['sample_size']
    categorical['tidy'] = categorical.apply(
        lambda x: '{count:>4.0f} ({percent:>5.1%})'.format(**x), 
        axis=1)

    # We merge the continous cand categorical data into the long form table,
    # do some more clean up to make it extra pretty (group value pivots, 
    # mostly) 
    example_table1 = pd.concat(
        axis=0, 
        objs=[cont_stats, categorical.reset_index()]
        )
    example_table1.drop(columns=['count', 'percent'], inplace=True)
    example_table1['sample_size'] = \
        example_table1['sample_size'].apply(lambda x: f'(n={x:1.0f})')
    example_table1 = example_table1.pivot(index=['group', 'group_value'],
                                          columns=['smoking', 'sample_size'],
                                          values='tidy',
                                          )
    example_table1.drop(
        index=[('m_age', 'median [25%, 75%]'), ('year', 'mean (std)')],
        inplace=True
        )
    example_table1 = \
        example_table1.loc[['m_age', 'sex', 'race', 'ethnicity', 'year']]

    return example_table1


def _extract_cohort_fit(fit_):
    """
    Extracts the cohort fit information
    """
    params = pd.concat(
        axis=1, 
        objs=[fit_.params, fit_.bse, fit_.conf_int(), fit_.pvalues]
    )
    params.columns = ['param', 'bse', 'ci_lo', 'ci_hi', 'p_value']
    if 'smoking' in params.index:
        return params.loc['smoking']
    else:
        return params.loc['Intercept'] * 0 + np.nan


def build_per_cohort_fit(example_data):
    """
    Fits the per cohort data with an ols and extracts the parameters
    """
    # Fits each of the cohorts if they have 2 levels of smokingd data
    example_fits = {
    (type_, cid): smf.ols(
        'y ~ smoking + C(race) + C(ethnicity) + C(sex) + m_age + year',
        data=df).fit()
        for (type_, cid), df in example_data.groupby(['cohort_type', 'cohort'])
        if (len(df['smoking'].unique()) > 1)
        }
    # Pulls out the smoking specific parameter 
    example_params = {cohort_id: _extract_cohort_fit(fit) 
                      for cohort_id, fit in example_fits.items()}
    example_params = pd.DataFrame.from_dict(example_params, orient='index')

    example_params.index.set_names(['cohort_type', 'cohort'], inplace=True)
    example_params['width'] = example_params['ci_hi'] - example_params['param']
    example_params.sort_values(['cohort_type', 'param'], 
                               inplace=True, 
                               ascending=True)

    return example_params


def extract_example_data_summary(example_data):
    """
    A helper function to summarize the outcome
    """
    example_bw = \
        example_data.groupby(['cohort_type', 'cohort'])[['y']].describe()
    example_bw = example_bw.reset_index().melt(
        id_vars=['cohort_type', 'cohort'],
        var_name=['covariate', 'statistic'],
        )
    example_bw = example_bw.pivot(
        index=['cohort_type', 'cohort', 'covariate'],
        columns='statistic',
        values='value',
    )
    example_bw['sem'] = example_bw['std'] / np.sqrt(example_bw['count'] - 1)
    example_bw['t'] = t_dist.ppf(0.975, example_bw['count'] - 1)
    example_bw['ci'] = example_bw['sem'] * example_bw['t']
    example_bw.sort_values(['cohort_type', 'mean'], 
                           ascending=True,
                           inplace=True)
    example_bw.reset_index(level='covariate', inplace=True, drop=True)

    return example_bw


def summarize_example_smoking(example_data):
    """
    Checks the example data summary
    """
    smoke_counts = \
        example_data.groupby(['cohort_type', 'cohort', 'smoking'])['y'].count()
    smoke_counts = smoke_counts.unstack(-1)
    smoke_counts.fillna(0, inplace=True)
    smoke_perc = smoke_counts.div(smoke_counts.sum(axis=1).values, axis=0)

    return smoke_perc.add_prefix('smoke_')


def tabulate_example_data(example_data, levels=['cohort_type', 'param', 'mean'], 
                          palette=None):
    """
    Combines the example data and calcultes the x position
    """
    example_summary = pd.concat(
        axis=1, 
        objs=[build_per_cohort_fit(example_data), 
              extract_example_data_summary(example_data), 
              summarize_example_smoking(example_data)]
    )
    example_summary.sort_values(levels, ascending=True, inplace=True)
    example_summary.reset_index(level='cohort_type', inplace=True)

    example_summary['x_pos'] = 1
    example_summary['x_pos'] = \
        example_summary['x_pos'].cumsum() + \
        example_summary['cohort_type'] * 2 - 1

    palettes = {i: mpc.to_hex(c) 
                for i, c in  enumerate(sn.color_palette(palette, n_colors=2))}
    example_summary['color'] = example_summary['cohort_type'].replace(palettes)

    return example_summary


def _coords_no_outliers(params, lim_min, lim_max, nom_scaler, qbse):
    """
    If there's noting in the outliers, then we skip over that
    """
    return False, (lim_min, lim_max), (lim_min, lim_max)


def _coords_one_outlier(params, lim_min, lim_max, nom_scaler, qbse):
    """
    If there's one outlier, we center it in the window so it's pretty
    """
    lo_inter = nom_scaler * (lim_max - lim_min)
    lo_mean = params.mean()
    lo_min, lo_max = lo_mean + np.array([-0.5, 0.5]) * lo_inter

    if lo_max > lim_min:
        lim_min = lo_min
        return False, (lim_min, lim_max), (lim_min, lim_max)
    elif lo_min < lim_max:
        lim_max = lo_max
        return False, (lim_min, lim_max), (lim_min, lim_max)
    else:
        return True, (lo_min, lo_max), (lim_min, lim_max)


def _coords_more_outliers(params, lim_min, lim_max, nom_scaler, qbse):
    """
    There's more than one outlier. Ahhh!!! 
    """
    lo_inter = nom_scaler * (lim_max - lim_min)
    lo_min, lo_max = params.min() + np.array([-0.5, 0.5]) * lo_inter

    # Checks that all the values are within hte parameters window
    if np.any((params < lo_min) | (params > lo_max)):
        raise ValueError('There is data outside the window')
    else:
        return True, (lo_min, lo_max), (lim_min, lim_max)


def _z_score(x):
    return (x - x.mean()) / x.std()


def calculate_example_scale(example_params, pass_thresh=2, lim_scale=0.25):
    """
    Determines whether parameters need to be plotted with outliers
    """
    def _z_score(x):
        return (x - x.mean()) / x.std()

    example_params = example_params.copy().dropna()
    # Gets the z-nroamlized variance and parameters so we can exclude values
    # oustide a threshhold
    example_params['z_bse'] = (_z_score(example_params['bse']))
    example_params['z_param'] =  (_z_score(example_params['param']))
    example_params['pass_bse'] = \
        np.absolute(example_params['z_bse']) <= pass_thresh
    example_params['pass_params'] = \
        np.absolute(example_params['z_param']) <= pass_thresh

    # Figures out the minimum and maximum based ont he axis limits
    pass_bse_q = \
        np.absolute(example_params.loc[example_params['pass_bse'], 'bse'])
    pass_bse_q = pass_bse_q.quantile(0.75)
    all_pass = example_params[['pass_bse', 'pass_params']].all(axis=1)

    lim_min = min(
        example_params.loc[all_pass, 'ci_lo'].min(), 
        example_params.loc[example_params['pass_params'], 'param'].min() - 
        pass_bse_q
        )
    lim_max = max(
        example_params.loc[all_pass, 'ci_hi'].max(),
        example_params.loc[example_params['pass_params'], 'param'].max() + 
        pass_bse_q
        )

    # Rounds the values so they're pretty
    lim_min = np.round(np.floor(lim_min / lim_scale), 0) * lim_scale
    lim_max = np.round(np.ceil(lim_max / lim_scale), 0) * lim_scale
    lim_dif = lim_max - lim_min

    pass_vals = example_params['param']

    lo_z = (lim_min - pass_vals.mean()) / pass_vals.std()
    hi_z = (lim_max - pass_vals.mean()) / pass_vals.std()

    example_params['out_window'] = \
        (example_params['z_param'] < lo_z) * -1 + \
        (example_params['param'] > hi_z) * 1

    below = example_params.loc[example_params['param'] < lim_min]
    above = example_params.loc[example_params['param'] > lim_max]

    nom_scaler = 1 / (4 - (len(below) > 0) - len(above > 0))
    bse2 = np.absolute(example_params.loc[example_params['pass_bse'], 'bse'])
    bse2 = bse2.mean()

    # Handles the window below the minimum for low outliers.
    # If there are no low outliers, we don't need an axis, so we drop that
    # If tehre's one low outlier, we center it in the window. Then, we check
    # if the window overlaps the existing window. If it does, we widen the 
    # current limits, and just drop the low window.
    # If there's more than one value, then we calculate a window starting wtih
    # the minimum value and allowing a 15% bse window to pad for extending the 
    # CI. We _should_ put in checks around the limits
    lo_kwargs = (below['param'], lim_min, lim_max, nom_scaler, pass_bse_q)

    if len(below) == 0:
        # print('low, 0')
        left_, lo_lims, main_lims = _coords_no_outliers(*lo_kwargs)
    elif len(below) == 1:
        # print('low, 1')
        left_, lo_lims, main_lims = _coords_one_outlier(*lo_kwargs)
    else:
        # print('low, lots')
        left_, lo_lims, main_lims = _coords_more_outliers(*lo_kwargs)

    hi_kwargs = (above['param'], *main_lims, nom_scaler, pass_bse_q)
    if len(above) == 0:
        # print('high, 0')
        right_, hi_lims, main_lims = _coords_no_outliers(*hi_kwargs)
    elif len(above) == 1:
        # print('high, 1')
        right_, hi_lims, main_lims = _coords_one_outlier(*hi_kwargs)
    else:
        # print('high, lots')
        right_, hi_lims, main_lims = _coords_more_outliers(*hi_kwargs)

    if left_ & ((main_lims[0] != lim_min) | (main_lims[1] != lim_max)) & \
            (len(below) == 1):
        left_, lo_lims, main_lims = \
            _coords_no_outliers(below['params'], *main_lims, nom_scaler, 
                                pass_bse_q)
    elif left_ & ((main_lims[0] != lim_min) | (main_lims[1] != lim_max)):
        left_, lo_lims, main_lims = \
            _coords_more_outliers(below['params'], lim_min, lim_max, nom_scaler,
                                  pass_bse_q)

    return left_, right_, main_lims, lo_lims, hi_lims


def plot_perc(ax0, example_summary):
    """
    Plots the percentage of smokers with appropriate colors
    """
    ax0.barh(
        y=example_summary['x_pos'],
        width=example_summary['smoke_1'],
        color=example_summary['color'],
        zorder=3,
    )

    # Formats the axis
    ax0.set_ylim(example_summary['x_pos'].max() + 1, 
                 example_summary['x_pos'].min() - 1) 
    ax0.grid(True, axis='y')
    ax0.set_xticks(np.linspace(0, 1, 3))
    ylim = ax0.get_ylim()
    ax0.plot([0.1, 0.1], ylim, 'k-', zorder=4, linewidth=1)
    ax0.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>3.0%}'))
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(left=True, labelleft=False, length=0, labelsize=0)
    ax0.set_xlabel('\% Smokers')
    ax0.xaxis.get_label().set_visible(True)


def plot_weight_dist(example_data, ax1, colors, pad_color=None, box_kws=None, 
                     violin_kws=None):
    """
    Plots the distribution of brith weights
    """

    if pad_color is None:
        pad_color = colors[-1]

    box_kwargs = dict(fliersize=0,
                      width=0.4,
                      linewidth=0.75,
                      boxprops={'facecolor': 'w'},
                      whis=0,
                      )
    if box_kws is not None:
        box_kwargs.update(box_kws)

    violin_kwargs = dict(inner=None,
                         linewidth=0.25,
                         offset=0,
                         zorder=10,
                         width=1.5,
                         cut=0,
                         )
    if violin_kws is not None:
        violin_kwargs.update(violin_kws)

    data_kwargs = dict(y='x', x='y', data=example_data,
                       order=np.arange(0, example_data['x'].max() + 1),
                       palette=np.hstack([colors, [pad_color] * 5]),
                       ax=ax1,
                       orient='h',
                       )

    sn.boxplot(**data_kwargs, **box_kwargs)
    pt.half_violinplot(**data_kwargs, **violin_kwargs)


    ax1.grid(axis='y')
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.set_xlabel('Birth weight z-score')
    ax1.yaxis.set_tick_params(left=False, labelleft=False)


def plot_bars(ax2, example_summary, lim_min, lim_max, effect, ax2l=None,
              ax2r=None, left=False, right=False, lo_min=np.nan, lo_max=np.nan, 
              hi_min=np.nan, hi_max=np.nan,  d=0.1, xscale=0.1, yscale=1,
              **kwargs):
    """
    Builds the forest plot
    """
    if left & pd.isnull([ax2l, lo_min, lo_max]).any():
        raise ValueError('There should be information associated with a left '
                         'axis, but a value is missing')
    elif right & pd.isnull([ax2r, hi_min, hi_max]).any():
        raise ValueError('There should be information associated with a right '
                         'axis, but a value is missing')

    error_kwargs = dict(marker='D', capsize=3, markeredgewidth=2,)
    error_kwargs.update(kwargs)

    # Sets up the left and right axis as needed
    if left:
        axl = ax2l
        left_lim = lo_min
    else:
        axl = ax2
        left_lim = lim_min

    if right:
        axr = ax2r
        right_lim = hi_max
    else:
        axr = ax2
        right_lim = lim_max

    # Plots the error bar
    for cohort_, df in example_summary.iterrows():
        cohort_kwargs = dict(
            y='x_pos', x='param', xerr='width', data=df,
            color=df['color'],
            markeredgecolor=df['color'],
            markerfacecolor={True: df['color']}.get(df['p_value'] < 0.05, 'w'),
            **error_kwargs,
            )
        # Makes the forest plot
        ax2.errorbar(**cohort_kwargs)
        if left:
            ax2l.errorbar(**cohort_kwargs)
        if right:
            ax2l.errorbar(**cohort_kwargs)

        # Adds the arrow to the 
        if df['ci_lo'] < left_lim:
            axl.scatter(y=df['x_pos'], x=left_lim + (lim_max - lim_min) * 0.0225, 
                         color='k',
                         edgecolor='None',
                         zorder=3,
                         marker='<')
        if df['ci_hi'] > right_lim:
             ax2.scatter(y=df['x_pos'], x=right_lim - (lim_max - lim_min) * 0.0225, 
                         color='k',
                         edgecolor='None',
                         zorder=3,
                         marker='>')

    # Formats the axes
    lims = pd.Series([lim_min, lim_max, lo_min, lo_max, hi_min, hi_max]).dropna()
    lims = np.atleast_2d(lims.values).reshape(int(len(lims) / 2), 2)

    yfix = np.mean([lo_max - lo_min, hi_max - hi_min]) / (lim_max - lim_min)


    if left & right:
        line_pos = [
            [[np.array([-1, 1]) * d * yscale * yfix + 0, 
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale * yfix + 0, 
              np.array([-1, 1]) * d * xscale + 1],
             [np.array([-1, 1]) * d * yscale * yfix + 1, 
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale * yfix + 1, 
              np.array([-1, 1]) * d * xscale + 1],
             ],
            [[np.array([-1, 1]) * d * yscale + 1, 
              np.array([-1, 1]) * d * xscale + 1],
             [np.array([-1, 1]) * d * yscale + 1, 
              np.array([-1, 1]) * d * xscale + 0],],
            [[np.array([-1, 1]) * d * yscale + 0, 
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale + 0, 
              np.array([-1, 1]) * d * xscale + 1],]
        ]

    elif left:
        line_pos = [
            [[np.array([-1, 1]) * d * yscale / 3 + 0,  
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale / 3 + 0, 
              np.array([-1, 1]) * d * xscale + 1],], 
            [[np.array([-1, 1]) * d * yscale + 1, 
              np.array([-1, 1]) * d * xscale + 1],
             [np.array([-1, 1]) * d * yscale + 1, 
              np.array([-1, 1]) * d * xscale + 0],],
            ]
    elif right:
        line_pos = [
            [[np.array([-1, 1]) * d * yscale * yfix + 1, 
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale * yfix + 1, 
              np.array([-1, 1]) * d * xscale + 1],
             ],
            [[np.array([-1, 1]) * d * yscale + 0, 
              np.array([-1, 1]) * d * xscale + 0],
             [np.array([-1, 1]) * d * yscale + 0, 
              np.array([-1, 1]) * d * xscale + 1],]
            ]
    else:
        line_pos = []

    for i, (ax, lims) in enumerate(zip(*([ax2, ax2l, ax2r], lims))):
        ax.set_xlim(*lims)
        ax2.grid(axis='y')
        ax2.yaxis.set_tick_params(left=False, labelleft=False)
        ax2.plot([effect] * 2, ax2.get_ylim(), 'k:', linewidth=0.5, zorder=1)
        ax2.plot([0] * 2, ax2.get_ylim(), 'k-', linewidth=1, zorder=1)

        if len(line_pos) > 0:
            kwargs = dict(transform=ax.transAxes, color='k',
                          clip_on=False, linewidth=2/3,)
            for l in line_pos[i]:
                ax.plot(*l, **kwargs)
        sn.despine(ax=ax, top=False, bottom=False, 
                   left=(left & (i == 0)) | (right & (i == 2)),
                   right=(left & (i == 1)) | (right & (i == 0)),
                   )


def built_filter_fits(example_data, filters, models, tech_order):
    """
    Filters the data and provides estimates 
    """
    example_filtered = {f_name: f_(example_data) 
                        for f_name, f_ in filters.items()}
    filter_model_combo = \
        it.product(example_filtered.items(), zip(*(tech_order, models)))
    adj_eq = 'm_age + year + C(race) + C(ethnicity) + C(sex)'
    fit_res = {
        (m_name, f_name): model_f(0.15, data, adj_eq)[2]
        for (f_name, data), (m_name, model_f) in filter_model_combo
        }
    fit_res = pd.DataFrame.from_dict(fit_res, orient='index')
    fit_res.index.set_names(['model', 'filter'], inplace=True)

    return fit_res


def tabulate_res(fit_res, filter_order, tech_order, palette='Dark2'):
    """
    Summaries the example fits intos omething plottable
    """
    idx_ = fit_res.index.to_frame()
    idx_.replace(
        to_replace={
            'model': {id_: i * (len(filter_order) + 2.5) 
                      for i, id_ in enumerate(tech_order)},
            'filter': {id_: i for i, id_ in enumerate(filter_order)},
        },
        inplace=True,
        )
    fit_res['y'] = idx_.sum(axis=1)

    fit_res['width'] = fit_res['ci_hi'] - fit_res['param']
    fit_res['significant'] =  \
        np.sign(fit_res[['ci_lo', 'ci_hi']]).product(axis=1) == 1
    fit_res['marker'] = idx_['filter'] 
    n_filter = len(filter_order)
    colors = {
        v: mpc.to_hex(c) 
        for v, c in enumerate(sn.color_palette(palette, n_colors=n_filter))
        }
    fit_res['color'] = idx_['filter']

    markers = {0: 'o', 1: 'D', 2: 's', 3: 'X', 4: '^', 5: 'v'}
    fit_res.replace({'marker': markers,
                     'color': colors
                    },
                    inplace=True)
    fit_res.loc[('meta', 'none'), 'color'] = 'None'
    fit_res.loc[('meta', 'small'), 'color'] = 'None'

    fit_res.sort_values(['y'], ascending=True, inplace=True)
    return fit_res


def table1_to_latex(example_table_1):
    """
    Makes a table 1 that can be plotted
    """
    exp_t1_tex  = example_table_1.rename(
        columns={'0': 'No', 0: 'No', 1: 'Yes', '1': 'Yes'})
    exp_t1_tex = exp_t1_tex.to_latex()
    for rule_ in ['\\toprule', '\\midrule', '\\bottomrule']:
        exp_t1_tex = exp_t1_tex.replace(rule_, '\\hline')
    while '  ' in exp_t1_tex:
        exp_t1_tex = exp_t1_tex.replace('  ', ' ')
    replacements = {
        # 'smoking': '', 
        'sample\_size': '', 
        'm\_age & mean (std)': '\\multicolumn{2}{l}{mat. age, mean (std)}',
        'year & median [25\\%, 75\\%]': '\\multicolumn{2}{l}{birth year} ',
        '\n\\hline\n\\end': ('\\multicolumn{2}{r}{median [25\\%, 75\\%]} '
                             '& & \\\\\n\\hline\n\\end'),
        'group & group\\_value & & \\\\\n': '',
        'llll': 'llcc',
    }
    for ori, rep in replacements.items():
        exp_t1_tex = exp_t1_tex.replace(ori, rep)

    return exp_t1_tex


def plot_filtered_fits(fit_res, ax, effect, filter_order, tidy_short=dict(), 
                       filter_tidy=dict()):
    """
    Makes a forest plot of the grouped estimates by model
    """
    levels = fit_res.groupby(['color', 'marker', 'significant'], sort=False)
    for (color, marker, fill), df in levels:
        ax.errorbar(y='y', x='param', xerr='width', 
                    data=df,
                    marker=marker,
                    color=color,
                    linestyle='',
                    markeredgecolor=color,
                    markerfacecolor={True: color, False:'None'}[fill],
                    capsize=4,
                    markersize=5,
                    capthick=1.5,
                    linewidth=1.5,
                    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_ylim(ylim[::-1])
    ylim = ax.get_ylim()
    ydiff = np.diff(ylim)

    ax.plot([effect] * 2, ax.get_ylim(), linestyle=':', zorder=0, 
            linewidth=0.75, color='#525252')
    ax.plot([0] * 2, ax.get_ylim(), linestyle='-', zorder=0, linewidth=0.75, 
            color='k')
    yticks = fit_res.groupby('model')['y'].mean()
    ax.set_yticks(yticks.values)
    ax.set_yticklabels([tidy_short.get(v, v.title()) for v in yticks.index])

    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    xdiff = np.diff(xlim)

    ax.fill_between(x=np.array([0.475, 0.975]) * np.diff(xlim) + xlim[0],
                y1=ylim[0] + np.diff(ylim) * (0.125),
                y2=ylim[0] + np.diff(ylim) * 0.5,
                facecolor='w',
                alpha=0.75,
                edgecolor='k',
                # linewidth=2,
                zorder=4,
                )
    ax.text(np.array([0.475, 0.975]).mean() * np.diff(xlim) + xlim[0],
            ylim[0] + 0.425 * ydiff,
            s='Filter',
            ha='center',
            zorder=5,
            # fontweight='bold',
            )
    markers = fit_res.groupby('filter', sort=False)['marker'].first()
    colors = fit_res.groupby('filter', sort=False)['color'].first()
    for i, label in enumerate(filter_order):
        ax.text(x=xlim[0] + xdiff * 0.625, 
                y=ylim[0] + (0.40 - i * 0.05) * ydiff,
                # s=label,
                s=filter_tidy.get(label, label).replace('≥', '$\\geq$'),
                size=9,
                va='center',
                zorder=5,
                )
        ax.errorbar(x=xlim[0] + xdiff * 0.55,
                    y=ylim[0] + (0.40 - i * 0.05) * ydiff,
                    xerr=xdiff * 0.04,
                    marker=markers[label],
                    markeredgecolor=colors[label],
                    color=colors[label],
                    zorder=5,
                    capsize=5,
                    capthick=1.25,
                   )
    ax.yaxis.set_tick_params(left=False, length=0)
    ax.set_ylabel('Model')
    ax.set_xlabel('$\hat{{\\beta}}_{\\textsf{smoking}}$')


def _plot_table_1(ax, example_table, fontsize=8):
    """
    Formats table 1 as a latex table for plotting
    """
    exp_t1_tex = table1_to_latex(example_table).replace('\n', ' ')
    # Plots the table
    ax.xaxis.set_tick_params(left=False, labelleft=False, right=False, 
                             labelright=False)
    ax.yaxis.set_tick_params(left=False, labelleft=False, right=False, 
                             labelright=False)
    sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)
    ax.text(x=0.5, 
              y=0.5, 
              s=exp_t1_tex,
              ha='center', 
              va='center',
              size=fontsize,
              )


def plot_pooled_violin(axes, example_data, box_kws=None, violin_kws=None):
    """
    Makes a violin plot summarizing the data by cohort type
    """
    boxplot_kwargs = dict(width=0.25,
                          boxprops={'facecolor': 'w'},
                          whis=0,
                          fliersize=0, 
                          linewidth=1, 
                          palette='Greys',)
    if box_kws is not None:
        boxplot_kwargs.update(box_kws)
    violin_kwargs = dict(inner=None, 
                         palette='Greys',
                         linewidth=1,
                         offset=0,
                         zorder=10,
                         width=1,
                         cut=0,)
    if violin_kws is not None:
        violin_kwargs.update(violin_kws)
    value_groups = [
        axes,
        [['0', '1', 0, 1], ['0', 0], ['1', 1]],
        ['All', 'General', 'NICU'],
        ]
    for i, (ax_dis, vals, title) in enumerate(zip(*(value_groups))):
        sub_data = \
            example_data.loc[example_data['cohort_type'].isin(vals)].copy()
        sn.boxplot(x='smoking', y='y', data=sub_data, ax=ax_dis,
                   **boxplot_kwargs)
        pt.half_violinplot(x='smoking', y='y', data=sub_data, ax=ax_dis,
                           **violin_kwargs)

        ax_dis.set_ylabel('Birth weight z-score')
        xmin, xmax = ax_dis.get_xlim()
        xdiff = xmax - xmin
        ax_dis.set_xlim(np.array([xmin, xmax]) - 0.05 * xdiff)
        ax_dis.yaxis.get_label().set_visible(i == 0)
        ax_dis.yaxis.set_tick_params(labelleft=(i == 0))
        ax_dis.xaxis.get_label().set_visible(True)
        ax_dis.xaxis.set_tick_params(bottom=False, length=0)
        ax_dis.set_title(title)


def _plot_dist_ref(ax_dist, fontsize=7.5, color='BuGn', box_kws=None, text_kws=None,):
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
        ax.xaxis.set_tick_params(bottom=False, labelbottom=False, 
                                 top=False, labeltop=False)
        ax.yaxis.set_tick_params(bottom=False, labelbottom=False, 
                                 top=False, labeltop=False)
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

    arrow_kws = dict(edgecolor='None', width=0.025, length_includes_head=True, 
                     facecolor='k')
    ax_ar1.arrow(x=0.1, dx=0.8, y=0.55, dy=+0.2, **arrow_kws)
    ax_ar1.arrow(x=0.1, dx=0.8, y=0.45, dy=-0.2, **arrow_kws)
    ax_ar1.set_ylim([0, 1])
    ax_ar2.arrow(x=0.1, dx=0.8, y=0.75, dy=-0.2, **arrow_kws)
    ax_ar2.arrow(x=0.1, dx=0.8, y=0.25, dy=+0.2, **arrow_kws)
    ax_ar2.set_ylim([0, 1])

    sn.despine(figul, left=True, right=True, top=True, bottom=True)


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


def _build_table_plot(fig, cohort_data, n_head=3):
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

    # Cleans up the merged_table labels so we can get  unispaced results when
    # the table gets "plotted"
    merged_table = \
        merged_table.unstack().apply(lambda x: f'\\\\texttt{{{x}}}').unstack().T
    merged_table.rename(
        columns={x: f'\\\\texttt{{{x}}}' for x in merged_table.columns},
        index={x: f'\\\\texttt{{{x}}}' for x in merged_table.index},
        inplace=True
    )

    # Casts the table to latex and does some latex clena up because the versions
    # dont quite line up between what  pandas puts out and what my system
    # at least is set up to handle
    merged_lt = merged_table.to_latex(column_format='| c | c | c | c | c | c |')
    # Convers the pandas altex header to actual latex
    merged_lt = merged_lt.replace('\\textbackslash \\textbackslash texttt', 
                                  '\\texttt')
    merged_lt = merged_lt.replace('\\{', '{').replace('\\}', '}')
    # adds spacer lines between each row
    for v in np.hstack([sub_table1['m_age'].unique(), '...']):
        merged_lt = merged_lt.replace(f'\\texttt{{{v}}} \\\\\n', 
                                      f'\\texttt{{{v}}} \\\\\n \\hline\n')
    # Adds an index label on the same level ast he rest of the columns
    merged_lt = merged_lt.replace('&  \\texttt{sex}', 
                                  '\\texttt{PregID} & \\texttt{sex}')
    # Cleans up the row splits... probably a latex engine issues
    for rule_ in ['\\toprule', '\\midrule', '\\bottomrule']:
        merged_lt = merged_lt.replace(rule_, '\\hline')

    merged_lt = merged_lt.replace('\\hline\n\\hline', '\\hline\n')

    # plots the text
    _ = ax.text(x=0.5, 
                y=0.5, 
                s=merged_lt.replace('\n', ' '), 
                ha='center', 
                va='center', 
                size=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.fill_between([0, 1], y1=0.0, y2=1, color='w')
    ax.fill_between([0, 1], y1=0.825, y2=1, color='#d1d1d1')


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
    if sim_colors is not None:
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
    ax_o = fig.add_subplot(gs[2:4, -2:])
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
    ax_r = []
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
                      color=sim_colors.get('cohort_random', 'k'), 
                      ylo=-0.4, 
                      yhi=1.4,
                      sigma_eq='$\sigma^{2} \propto \mathrm{ICC}$')
    # Adds the point estimate value
    cg = sub_data['cohort_random'].unique()
    ax_ce.scatter(cg, scipy.stats.norm.pdf(cg), zorder=3, 
                      color=sim_colors.get('cohort_random', 'k'))

    # Plots the individual error term
    ax_ce = fig.add_subplot(gs[0:2, 8:10], facecolor='None')
    _plot_gauss_curve(ax=ax_ce, color=sim_colors.get('error', 'k'), 
                     ylo=-0.4, yhi=1.4)

    # Cohort fixed effect
    axtp = fig.add_subplot(gs[0:2, 4:6], facecolor='None')
    type_ = (sub_data['cohort_fixed'].unique()[0] != 0) * 1
    _select_cohort_type(axtp, type_, 
                       color=sim_colors.get('cohort_fixed', 'purple')
                       )
    sn.despine(ax=axtp, left=True, right=True, top=True, bottom=True)

    # Demographic/exposure icon
    axf1 = fig.add_subplot(gs[0:2, 0:2], facecolor='None',)
    _make_ref_table(axf1, palette='Greys')
    axf1.set_xlim(-0.1025, 1.1025)
    axf1.set_ylim(-1.1, 0.65)
    sn.despine(ax=axf1, left=True, right=True, top=True, bottom=True)

    axf2 = fig.add_subplot(gs[0:2, 2:4], facecolor='None')
    _make_ref_table(axf2, palette='Reds', n_cols=2, offset=0.3,)
    axf2.set_xlim(axf1.get_xlim())
    axf2.set_ylim(axf1.get_ylim())
    sn.despine(ax=axf2, left=True, right=True, top=True, bottom=True)

    # Adds equation labels
    ax_eq.text(0.15, 0.55, '+', size=12, ha='center', va='center')
    ax_eq.text(0.306, 0.55, '+', size=12, ha='center', va='center', color='k')
    ax_eq.text(0.463, 0.55, '+', size=12, ha='center', va='center', color='k')
    ax_eq.text(0.618, 0.55, '+', size=12, ha='center', va='center', color='k')

    ax_t = fig.add_subplot(gs[2:4, -3], facecolor='None')
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(0.5, 0.5, '=', ha='center', va='center', size=15)
    ax_t.set_title('=',  pad=4)
    ax_t.xaxis.set_tick_params(left=False, labelleft=False, 
                               right=False, labelright=False)
    ax_t.yaxis.set_tick_params(left=False, labelleft=False, 
                               right=False, labelright=False)
    sn.despine(ax=ax_t, left=True, right=True, top=True, bottom=True)
    sn.despine(ax=ax_eq, left=True, right=True, top=True, bottom=True)


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
            over_color = '#ffffff'
        elif (ax_c[-1] == 0):
            over_color = mpc.to_hex(fig_c)
        else:
            over_color = mpc.to_hex(ax_c[:-1])

        ax_prob.fill_between(ax_prob.get_xlim(), *ax_prob.get_ylim(), 
                             color=over_color, 
                             alpha=0.75,
                             zorder=4,)


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


def _select_exposure(fig2, cohort_data, smoking_effect, overlay=True):
    """
    Builds a figure showing the exposure selection process for cohorts with 
    a population-based seed or selection critierion
    """
    gs2 = fig2.add_gridspec(2, 4)
    ax_prob2 = fig2.add_subplot(gs2[1, 0:-1], facecolor='None')
    ax_demo2 = fig2.add_subplot(gs2[1, 0], facecolor='None')
    ax_prob1 = fig2.add_subplot(gs2[0, 0:-1], facecolor='None')
    ax_demo1 = fig2.add_subplot(gs2[0, 0], facecolor='None')

    ax_r = fig2.add_subplot(gs2[0:2, -1], facecolor='None')

    for ax in fig2.axes:
        ax.xaxis.set_tick_params(left=False, labelleft=False, 
                                 right=False, labelright=False)
        ax.yaxis.set_tick_params(left=False, labelleft=False, 
                                 right=False, labelright=False)

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
    exp_table.mask(exp_table.notna(), 
                   exp_table.dropna().astype(int).astype(str),
                   inplace=True)
    exp_table.replace({np.nan: '...'}, inplace=True)
    exp_table.index.set_names('PregID', inplace=True)
    exp_table.reset_index(inplace=True)
    exp_table2 = exp_table.copy()

    exp_table = \
        exp_table.unstack().apply(lambda x: f'\\\\texttt{{{x}}}').unstack().T
    exp_table.rename(
        columns={x: f'\\\\texttt{{{x}}}' for x in exp_table.columns},
        inplace=True,)

    exp_lt = exp_table.to_latex(index=False, column_format='| c | c |')
    exp_lt = exp_lt.replace('\\textbackslash \\textbackslash texttt', 
                            '\\texttt')
    exp_lt = exp_lt.replace('\\{', '{').replace('\\}', '}')
    for v in exp_table2['smoking'].unique():
        exp_lt = exp_lt.replace(f'\\texttt{{{v}}} \\\\\n', 
                        f'\\texttt{{{v}}} \\\\\n \\hline\n')
    for rule_ in ['\\toprule', '\\midrule']:
        exp_lt = exp_lt.replace(rule_, '\\hline')
    exp_lt = exp_lt.replace('\\bottomrule\n', '')

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


def build_split_figure(fig_c, example_summary, example_data, example_params, 
                       clutch, s1=1/100, xy_ratio=4.5 / 1.5, d=0.04):
    """
    Builds a figure selecting an example cohort
    """
    s = s1 * xy_ratio

    # ### AXES ###
    # Sets up the gridspec and axes
    gs = fig_c.add_gridspec(6, 11)

    ax1a = fig_c.add_subplot(gs[0, 3:7])
    ax0a = fig_c.add_subplot(gs[0, 1:3], sharey=ax1a)
    ax2a = fig_c.add_subplot(gs[0, 7:], sharey=ax1a)

    ax1b = fig_c.add_subplot(gs[1:-2, 3:7],sharex=ax1a)
    ax0b = fig_c.add_subplot(gs[1:-2, 1:3], sharey=ax1b, sharex=ax0a)
    ax2b = fig_c.add_subplot(gs[1:-2, 7:], sharey=ax1b, sharex=ax2a)

    ax1c = fig_c.add_subplot(gs[-2, 3:7],sharex=ax1a)
    ax0c = fig_c.add_subplot(gs[-2, 1:3], sharey=ax1c, sharex=ax0a)
    ax2c = fig_c.add_subplot(gs[-2, 7:], sharey=ax1c, sharex=ax2a)

    # Positions the axes to maximize the figure space
    # ax3a.set_position((0, 1-1/8 - 1 * s1, 1/11 - 2 * s1, 1/8))
    ax0a.set_position((2/12 - 4 * s1, 1 - 1/9 - s, 2/12, 1/9))
    ax1a.set_position((4/12 - 3 * s1, 1 - 1/9 - s, 4/12, 1/9))
    ax2a.set_position((8/12 - 2 * s1, 1 - 1/9 - s, 4/12, 1/9))

    ax0b.set_position((2/12 - 4 * s1, 1 - 4/9 - 2 * s, 2/12, 3/9))
    ax1b.set_position((4/12 - 3 * s1, 1 - 4/9 - 2 * s, 4/12, 3/9))
    ax2b.set_position((8/12 - 2 * s1, 1 - 4/9 - 2 * s, 4/12, 3/9))

    ax0c.set_position((2/12 - 4 * s1, 1 - 5/9 - 3 * s, 2/12, 1/9))
    ax1c.set_position((4/12 - 3 * s1, 1 - 5/9 - 3 * s, 4/12, 1/9))
    ax2c.set_position((8/12 - 2 * s1, 1 - 5/9 - 3 * s, 4/12, 1/9))


    left, right, main_lims, (lo_min, lo_max), (hi_min, hi_max) = \
        calculate_example_scale(example_params, lim_scale=0.85)

    # Sets up the sets of axes for plotting
    ax_sets = [[ax1a, ax0a, ax2a], [ax1b, ax0b, ax2b], [ax1c, ax0c, ax2c]]

    # Plots the full data set on each set of axes so we have the option to modify
    for i, [ax1, ax0, ax2] in enumerate(ax_sets):
        plot_weight_dist(ax1=ax1, example_data=example_data, 
                                   colors=example_summary['color'], 
                                   violin_kws={'width': 1})
        plot_perc(ax0, example_summary)
        plot_bars(ax2=ax2, example_summary=example_summary, 
                            lim_min=main_lims[0], lim_max=main_lims[1], left=left, 
                            right=right, effect=-0.5, markersize=4,
                            )
        ax1.set_yticks(example_summary['x_pos'].unique())    
        ax0.yaxis.set_tick_params(labelleft=True, labelsize=8)
        ax0.set_yticklabels([f'({x:>4.0f})' for x in example_summary['count']])
        ax2.set_xlabel('$\\hat{\\beta}_{smoking}$', size=10, labelpad=0)
        ax0.set_xticks(np.array([0, 0.5, 1]))
        ax1.yaxis.get_label().set_visible(False)

        for ax_ in [ax0, ax1, ax2]:
            sn.despine(ax=ax_, left=False, right=False, top=(i > 0), 
                       bottom=(i < 2))
            ax_.xaxis.set_tick_params(bottom=(i == 2), labelbottom=(i == 2),
                                      top=False, labeltop=False, labelsize=8)
            ax_.xaxis.get_label().set_visible(i == 2)
            ax_.xaxis.get_label().set_fontsize(9)

    # Adds an overlay to make the selected data stand out. 
    for axes in ax_sets:
        for ax in axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.fill_between(xlim, y1=ylim[1], y2=clutch - 0.5, facecolor='w', 
                            alpha=0.35, zorder=20, edgecolor='None',)
            ax.fill_between(xlim, y1=ylim[0], y2=clutch + 0.5, facecolor='w', 
                            alpha=0.35, zorder=20, edgecolor='None',)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Sets the subset limits on the data
    ax1a.set_ylim(np.array([0.5, -0.5]) + example_summary['x_pos'].min())
    ax1b.set_ylim(np.array([1.5, -1.5]) + clutch)
    ax1c.set_ylim(np.array([0.5, -0.5]) + example_summary['x_pos'].max())

    # Adds overlap selecting the key features
    axr = fig_c.add_subplot(gs[:, :], facecolor='None')
    axr.set_position((2/12 - 4 * s1, 1 - 5/9 - 3 * s, 10 / 12 + 2 * s1, 5/9 + 2 * s))

    axr.xaxis.set_tick_params(left=False, labelleft=False,
                              right=False, labelright=False,
                              labelsize=0, length=0)
    axr.yaxis.set_tick_params(left=False, labelleft=False, 
                              right=False, labelright=False,
                              labelsize=0, length=0)
    axr.set_xlim(-0.001, 1.001)
    axr.set_ylim(0, 1)
    axr.fill_between([0.001, 1], y1=0.4, y2=0.6, facecolor='None', 
                     edgecolor='k', linewidth=1)
    sn.despine(ax=axr, left=True, right=True, top=True, bottom=True)

    # Adds breaks
     # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(color='k', clip_on=False, linewidth=0.75, zorder=21)
    ax0a.plot((0+d, 0-d), (0 - d * 3, 0 + d * 3), **kwargs, 
              transform=ax0a.transAxes)
    ax0b.plot((0+d, 0-d), (1 - d, 1 + d), **kwargs, transform=ax0b.transAxes)
    ax0b.plot((0+d, 0-d), (0 - d, 0 + d), **kwargs, transform=ax0b.transAxes)
    ax0c.plot((0+d, 0-d), (1 - d * 3, 1 + d * 3), **kwargs, 
              transform=ax0c.transAxes)

    # # # xscale = np.diff(ax0a.get_xlim()) / np.diff(ax2a.get_xlim())
    # # # # xscale = xscale[0]
    xscale = 1/ 2

    ax2a.plot((1+d * xscale, 1-d * xscale), (0 - d * 3, 0 + d * 3), **kwargs, 
              transform=ax2a.transAxes)
    ax2b.plot((1+d * xscale, 1-d * xscale), (1 - d, 1 + d), **kwargs, 
              transform=ax2b.transAxes)
    ax2b.plot((1+d * xscale, 1-d * xscale), (0 - d, 0 + d), **kwargs, 
              transform=ax2b.transAxes)
    ax2c.plot((1+d * xscale, 1-d * xscale), (1 - d * 3, 1 + d * 3), **kwargs, 
              transform=ax2c.transAxes)


def pooled_res_plot(fig_cr, fit_res, example_data):
    """
    A truncated fit plot
    """

    gs = fig_cr.add_gridspec(5, 7)

    # Plots the brace from the per-cohort simulation
    ax_br = fig_cr.add_subplot(gs[:, 0], facecolor='None')
    ax_br.set_xlim(0, 1)
    ax_br.set_ylim(0, 1)
    ax_br.yaxis.set_tick_params(left=False, labelleft=False, right=False, 
                                labelright=False, length=0, labelsize=0)
    ax_br.xaxis.set_tick_params(left=False, labelleft=False, right=False, 
                                labelright=False, length=0, labelsize=0)
    _build_exp_brace(ax_br)
    ax_br.set_xlim(-0.45, 0.15)
    ax_br.set_ylim(-1.25, 1.25)
    ax_br.set_position((0, 0, 0.1, 1))

    # Plots the fitted result
    ax_mod1 = fig_cr.add_subplot(gs[2:4, 2:])
    ax_mod2 = fig_cr.add_subplot(gs[0:2, 2:])

    fit_res = fit_res.xs('none', level='filter').copy()
    fit_res['y'] = np.linspace(0, 1, 6)

    lme_ = smf.mixedlm(
        'y ~ 1', 
        data=example_data.loc[example_data['cohort_type'] == 0],
        groups=example_data.loc[example_data['cohort_type'] == 0, 'cohort']
        ).fit()
    icc = bw_ana.icc_from_lme(lme_)


    for ax_mod in ax_mod1, ax_mod2: 
        ax_mod.errorbar(y='y', x='param', xerr='width',
                        data=fit_res.loc[fit_res['significant']],
                        marker=fit_res.iloc[0]['marker'],
                        color=fit_res.iloc[0]['color'],
                        markerfacecolor=fit_res.iloc[0]['color'],
                        markeredgecolor=fit_res.iloc[0]['color'],
                        capsize=3,
                        capthick=1,
                        linestyle='',
                        )
        ax_mod.errorbar(y='y', x='param', xerr='width',
                        data=fit_res.loc[~fit_res['significant']],
                        marker=fit_res.iloc[0]['marker'],
                        color=fit_res.iloc[0]['color'],
                        markerfacecolor='w',
                        markeredgecolor=fit_res.iloc[0]['color'],
                        capsize=3,
                        capthick=1,
                        linestyle='',
                        )
        ax_mod.plot([-0.5, -0.5], [-0.2, 1.2], 'k:', linewidth=0.75)
        ax_mod.plot([0, 0], [-0.2, 1.2], 'k-', linewidth=0.75)
        ax_mod.set_xlim(-0.75, 0.1)
        ax_mod.set_yticks(np.linspace(0, 1, 6))
        ax_mod.set_yticklabels(['Ignore', 'Cohort\nType', 'Fixed', 'GEE', 'LME', 
                                'Meta-\nAnalysis'])
        ax_mod.yaxis.set_tick_params(left=False, labelsize=8, length=0)
        ax_mod.xaxis.set_tick_params(labelsize=8,)
        ax_mod.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.1f}'))

    ax_mod2.set_ylim(0.3, -0.1, )
    ax_mod1.set_ylim(1.1, 1.1 + np.diff(ax_mod2.get_ylim()))

    ax_mod1.set_xlabel('pooled $\hat{\\beta}_{smoking}$', labelpad=-2)
    ax_mod2.set_title(f'.ICC={icc:1.2f}', size=8, loc='right', pad=-8)

    sn.despine(ax=ax_mod2, top=False, bottom=True, left=False, right=False)
    sn.despine(ax=ax_mod1, top=True, bottom=False, left=False, right=False)

    ax_mod2.xaxis.set_tick_params(bottom=False, labelbottom=False, top=False,
                                  labeltop=False,)

    d = 0.03
    kwargs = dict(color='k', clip_on=False, linewidth=0.75, zorder=3)
    ax_mod1.plot((0+d, 0-d), (1-2*d, 1+2*d), **kwargs, transform=ax_mod1.transAxes)
    ax_mod1.plot((1+d, 1-d), (1-2*d, 1+2*d), **kwargs, transform=ax_mod1.transAxes)
    ax_mod2.plot((0+d, 0-d), (0-2*d, 0+2*d), **kwargs, transform=ax_mod2.transAxes)
    ax_mod2.plot((1+d, 1-d), (0-2*d, 0+2*d), **kwargs, transform=ax_mod2.transAxes)

    sn.despine(ax=ax_br, left=True, right=True, top=True, bottom=True)

    x, y, dx, dy = ax_mod2.get_position().bounds
    ax_mod2.set_position((x, y + 1 / 50, dx, dy))