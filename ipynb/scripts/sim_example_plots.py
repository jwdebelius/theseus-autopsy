import copy
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

class SimulationSummary():
    _defalts = dict(
        y_sort_levels=['cohort_type', 'param', 'mean'],
        y_sort_ascend=[False, True, True],
        type_palette ={
            1 - i: mpc.to_hex(c) 
            for i, c in enumerate(sn.color_palette(n_colors=2))
            },
        tech_order=['ignore',  'risk', 'fixed', 'gee', 'lme', 'meta',],
        filter_order=['none', 'small', 'only', 'general',  'all', 'other'],
        filter_tidy = {'small': 'n≥30', 'only': 'mixed exp'},
        formula='y ~ smoking + C(race) + C(ethnicity) + C(sex) + m_age + year',
        )

    def __init__(self, data, filters, models, **kwargs):
        """
        
        """
        # Starts with default details
        self.data = data
        self.filters = filters
        self.models = models

        # Handles keyword arguments
        kws = copy.copy(self._defalts)
        kws.update(kwargs)

        for k, v in kws.items():
            setattr(self, k, v)

        # Tabulates the example data for plotting
        self.tabulate_results()

        # Filts the filtered data
        # self.build_filter_fits()

    def tabulate_results(self):
        """
        Tabulates the results of the simulations
        """
        summary = pd.concat(
            axis=1, 
            objs=[self._build_per_cohort_fit(), 
                  self._extract_example_data_summary(), 
                  self._summarize_example_smoking()]
                  )
        summary.sort_values(self.y_sort_levels, 
                            ascending=self.y_sort_ascend, 
                            inplace=True)
        summary.reset_index(level='cohort_type', inplace=True)

        summary['x_pos'] = 1
        summary['x_pos'] = \
            summary['x_pos'].cumsum() + \
            summary['cohort_type'] * 2 - 1
        # summary['color'] = \

        self.summary = summary

        self._update_data_pos()

    def build_filter_fits(self):
        """
        Filters the data and provides estimates 

        Parameters
        ----------
        example_data: DataFame
        filters: dict
        models: 
        tech_order: np.array

        Returns
        -------
        DataFrame
        """
        example_filtered = {f_name: f_(self.data) 
                            for f_name, f_ in self.filters.items()}
        filter_model_combo = \
            it.product(example_filtered.items(), 
                       zip(*(self.tech_order, self.models)))
        adj_eq = self.formula.split('~')[-1]
        fit_res = {
            (m_name, f_name): model_f(0.15, data, adj_eq)[2]
            for (f_name, data), (m_name, model_f) in filter_model_combo
            }
        fit_res = pd.DataFrame.from_dict(fit_res, orient='index')
        fit_res.index.set_names(['model', 'filter'], inplace=True)

        self.filtered_fits = _tabulate_filter_fits(fit_res)

    def _tabulate_filter_fits(fit_res):
        """
        Summaries the example fits intos omething plottable
        """
        idx_ = fit_res.index.to_frame()
        idx_ = idx_.replace(
            to_replace={
                'model': {id_: str(i * (len(self.filter_order) + 2.5))
                          for i, id_ in enumerate(self.tech_order)},
                'filter': {id_: str(i) 
                           for i, id_ in enumerate(self.filter_order)},
            },
            ).astype(float)
        fit_res['y'] = idx_.sum(axis=1)

        fit_res['width'] = fit_res['ci_hi'] - fit_res['param']

        fit_res['significant'] =  \
            np.sign(fit_res[['ci_lo', 'ci_hi']]).product(axis=1) == 1
        fit_res['marker'] = idx_['filter'] 
        
        n_filter = len(filter_order)
        colors = {
            v: mpc.to_hex(c) 
            for v, c in enumerate(sn.color_palette(n_colors=n_filter))
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

    def _update_data_pos(self):
        # Adds the y position for the cohort the data
        y_pos = self.summary['x_pos'].astype(str)
        self.data['x'] = self.data['cohort'].replace(y_pos).astype(float)

    def _build_per_cohort_fit(self):
        """
        Fits the per cohort data with an ols and extracts the parameters

        Returns
        -------
        pd.DataFrame
        """
        # Fits each of the cohorts if they have 2 levels of smokingd data
        grouper = self.data.groupby(['cohort_type', 'cohort'])
        example_fits = {
            (type_, cid): smf.ols(self.formula, data=df).fit()
                for (type_, cid), df in grouper
                if (len(df['smoking'].unique()) > 1)
                }

        # Pulls out the smoking specific parameter 
        example_params = {cohort_id: self._extract_cohort_fit(fit) 
                          for cohort_id, fit in example_fits.items()}
        example_params = pd.DataFrame.from_dict(example_params, orient='index')

        example_params.index.set_names(['cohort_type', 'cohort'], 
                                       inplace=True)
        example_params['width'] = \
            example_params['ci_hi'] - example_params['param']
        example_params.sort_values(['cohort_type', 'param'], 
                                   inplace=True, 
                                   ascending=True)
        return example_params


    @staticmethod
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

    def _extract_example_data_summary(self):
        """
        A helper function to summarize the outcome
        """
        example_bw = \
            self.data.groupby(['cohort_type', 'cohort'])['y'].describe()
        example_bw.reset_index(inplace=True)
        example_bw = example_bw.melt(
            id_vars=['cohort_type', 'cohort'],
            var_name='statistic',
            )
        example_bw['covariate'] = 'y'
        example_bw = example_bw.pivot(
            index=['cohort_type', 'cohort', 'covariate'],
            columns='statistic',
            values='value',
        )
        example_bw['sem'] = \
            example_bw['std'] / np.sqrt(example_bw['count'] - 1)
        example_bw['t'] = t_dist.ppf(0.975, example_bw['count'] - 1)
        example_bw['ci'] = example_bw['sem'] * example_bw['t']
        example_bw.sort_values(['cohort_type', 'mean'], 
                               ascending=True,
                               inplace=True)
        example_bw.reset_index(level='covariate', inplace=True, drop=True)

        return example_bw

    def _summarize_example_smoking(self):
        """
        Checks the example data summary for the proprtion of smokers
        """
        smoke_counts = \
            self.data.groupby(['cohort_type', 'cohort', 'smoking'])['y']
        smoke_counts = smoke_counts.count()
        smoke_counts = smoke_counts.unstack(-1)
        smoke_counts.fillna(0, inplace=True)
        smoke_perc = smoke_counts.div(smoke_counts.sum(axis=1).values, axis=0)

        return smoke_counts


class ExampleScale():
    def __init__(self, params, pass_thresh=2, lim_scale=0.85):
        self.params = params
        self.pass_thresh = pass_thresh
        self.lim_scale = lim_scale

        #left_, right_, main_lims, lo_lims, hi_lims = self._calculate_scale()

        #return left_, right_, main_lims, lo_lims, hi_lims

    def get_scale(self):
        """
        Determine sthe way the data should be sclaed.
        """
        left_, right_, main_lims, lo_lims, hi_lims = self._calculate_scale()
        return left_, right_, main_lims, lo_lims, hi_lims

    def _calculate_scale(self):
        """
        Determines the way data should be scaled
        """
        def _z_score(x):
            return (x - x.mean()) / x.std()


        example_params = self.params.copy().dropna()

        # Gets the z-nroamlized variance and parameters so we can exclude 
        # values oustide a threshhold
        example_params['z_bse'] = (_z_score(example_params['bse']))
        example_params['z_param'] =  (_z_score(example_params['param']))
        example_params['pass_bse'] = \
            np.absolute(example_params['z_bse']) <= self.pass_thresh
        example_params['pass_params'] = \
            np.absolute(example_params['z_param']) <= self.pass_thresh

        # Estimates an axis limit based on the data
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
        lim_min = \
            np.round(np.floor(lim_min / self.lim_scale), 0) * self.lim_scale
        lim_max = \
            np.round(np.ceil(lim_max / self.lim_scale), 0) * self.lim_scale
        lim_dif = lim_max - lim_min

        # Identifies values outside the inclusion window    
        pass_vals = example_params['param']

        lo_z = (lim_min - pass_vals.mean()) / pass_vals.std()
        hi_z = (lim_max - pass_vals.mean()) / pass_vals.std()

        example_params['out_window'] = \
            (example_params['z_param'] < lo_z) * -1 + \
            (example_params['param'] > hi_z) * 1

        below = example_params.loc[example_params['param'] < lim_min]
        above = example_params.loc[example_params['param'] > lim_max]

        nom_scaler = 1 / (4 - (len(below) > 0) - len(above > 0))
        bse2 = \
            np.absolute(example_params.loc[example_params['pass_bse'], 'bse'])
        bse2 = bse2.mean()

        # Handles the window below the minimum for low outliers.
        # If there are no low outliers, we don't need an axis, so we drop 
        # that.
        # 
        # If there's one low outlier, we center it in the window. Then, we 
        # check if the window overlaps the existing window. If it does, we 
        # widen the current limits, and just drop the low window.
        # 
        # If there's more than one value, then we calculate a window starting
        #  wtih the minimum value and allowing a 15% bse window to pad for
        # extending the  CI. We _should_ put in checks around the limits
        lo_kwargs = (below, (lim_min, lim_max), nom_scaler, pass_bse_q)
        left_, lo_lims, main_lims = self._check_params(*lo_kwargs)
        hi_kwargs = (above, main_lims, nom_scaler, pass_bse_q)
        right_, hi_lims, main_lims = self._check_params(*hi_kwargs)

        if left_ & ((main_lims[0] != lim_min) | (main_lims[1] != lim_max)) & \
                (len(below) == 1):
            left_, lo_lims, main_lims = \
                self._coords_no_outliers(below['params'], *main_lims, nom_scaler, 
                                         pass_bse_q)
        elif left_ & ((main_lims[0] != lim_min) | (main_lims[1] != lim_max)):
            left_, lo_lims, main_lims = \
                self._coords_more_outliers(below['params'], lim_min, lim_max, 
                                           nom_scaler, pass_bse_q)

        return left_, right_, main_lims, lo_lims, hi_lims

    def _check_params(self, check_param, main_lims, nom_scaler, pass_bse_q):
        """
        Wraps identifying the axes
        """
        args = (check_param['param'], main_lims[0], main_lims[1], nom_scaler, 
                pass_bse_q)
        if len(check_param) == 0:
            dir_, dir_lims, main_lims = self._coords_no_outliers(*args)
        elif len(check_param) == 1:
            dir_, dir_lims, main_lims = self._coords_one_outlier(*args)
        else:
            dir_, dir_lims, main_lims = self._coords_more_outliers(*args)

        return dir_, dir_lims, main_lims

    @staticmethod
    def _coords_no_outliers(params, lim_min, lim_max, nom_scaler, qbse):
        """
        If there's noting in the outliers, then we skip over that
        """
        return False, (lim_min, lim_max), (lim_min, lim_max)

    @staticmethod
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

    @staticmethod
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


def plot_per_cohort_data(sim_summary, effect, fig=None, palette=None, pass_thresh=2, lim_scale=0.85):
    """
    Summarizes the per-cohort data
    """
    # Calculates the scale for the forest plot
    es = ExampleScale(sim_summary._build_per_cohort_fit(),
                          pass_thresh, lim_scale)
    left_, right_, main_lims, lo_lims, hi_lims = es.get_scale()
    
    if fig is None:
        fig = plt.figure(dpi=150, 
                         constrained_layout=False,
                         figsize=(6, 6),
                         facecolor='w',
                         )
    gs = fig.add_gridspec(4, 11)
    ax00 = fig.add_subplot(gs[1:3])
    ax01 = fig.add_subplot(gs[3:7], sharey=ax00)
    if not (left_ | right_):
        forest_axes = dict(ax2=fig.add_subplot(gs[7:], sharey=ax01),
                           ax2l=None,
                           axr2=None,)
    elif left_ & right_:
        forest_axes = dict(ax2=fig.add_subplot(gs[8:-1], sharey=ax01),
                            ax2l=fig.add_subplot(gs[7:], sharey=ax01),
                            axr2=fig.add_subplot(gs[-1], sharey=ax01),
                            )
    elif left_:
        forest_axes = dict(ax2=fig.add_subplot(gs[8:-1], sharey=ax01),
                           ax2l=fig.add_subplot(gs[7:], sharey=ax01),
                           ax2r=None,
                           )
    elif right_:
        forest_axes = dict(ax2=fig.add_subplot(gs[7:-1], sharey=ax01),
                            ax2l=None,
                            axr2=fig.add_subplot(gs[-1], sharey=ax01),
                            )
    
    # Plots the weight distribution
    plot_weight_dist(ax1=ax01,
                     example_data=sim_summary.data, 
                     palette=sim_summary.type_palette,
                     violin_kws={'width': 1.25})
    # Plots the percentage of smokers
    plot_perc(ax00, sim_summary.summary)

    # Makes the forest plot
    xp_plots.plot_bars(**forest_axes,
                       example_summary=sim_summary.summary, 
                       lim_min=main_lims[0], 
                       lim_max=main_lims[1], 
                       left=left, 
                       right=right, 
                       effect=effect)
    
    # Adds the size labels
    _ = ax00.set_yticks(ss.summary['x_pos'])
    ax00.set_yticklabels([f'({x:>4.0f})' 
                          for x in sim_summary.summary['count']])
    ax00.yaxis.set_tick_params(labelleft=True, labelsize=9)
    ax00.set_xticks(np.linspace(0, 1, 3))

    for ax_ in fig.axes[1:]:
        ax_.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.1f}'))

    ax00.set_ylabel('Cohort', size=12)
    ax01.yaxis.get_label().set_visible(False)
    _ = forest_axes['ax2'].set_xlabel('$\\hat{{\\beta}}_{\\textsf{smoking}}$')

    return fig


def plot_cohort_demo_summary(sim_summary, fig=None):
    """
    A graphical cohort summary for each of the simulated data sets
    """
    if fig is None:
        fig = plt.figure(dpi=150, constrained_layout=True, figsize=(6, 4))

    gs = fig.add_gridspec(1, 8)
    palettes = {'race': {0: '#66c2a5',
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
    cat_pos = {'race': 1, 'ethnicity': 2, 'sex': 3, 'smoking': 6}
    con_pos = {'m_age': [17, 43], 'year': [0, 20]}
    # Plots the group size 
    ax0 = fig.add_subplot(gs[0])
    _plot_cohort_summary(ax0, sim_summary.data)
    ax0.set_xlim(0, max(ax0.get_xlim()[1], 500))
    ax0.set_ylim(-1, sim_summary.data['x'].max() + 1)
    ax0.set_xlabel('Cohort\nSize', size=8)
    ax0.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)

    # Plots the categorical data
    for col_, axi in cat_pos.items():
        ax = fig.add_subplot(gs[axi], sharey=ax0)
        _plot_categorical_by_cohort(ax, sim_summary.data, col_, 
                                    colors=palettes[col_])
        ax.set_xlim(0, 1)
        ax.xaxis.set_tick_params(bottom=False, top=False, labelbottom=False,
                                 labeltop=False, labelsize=0, length=0)
    # Plots the categorical histograms
    for axi, (col, xlim) in enumerate(con_pos.items()):
        ax = fig.add_subplot(gs[axi + 4], sharey=ax0)
        _plot_cont_histogram_by_cohort(ax, sim_summary.data, col, xlim=xlim)
        ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8,
                                 top=False, labeltop=False)
        if axi == 0:
            ax.set_xticks(np.linspace(20, 40, 3))
            ax.set_xlabel('age (years)', size=8)
        elif axi == 1:
            ax.set_xticks(np.linspace(2, 18, 3))
            ax.set_xlabel('Year', size=8)
    
    # Plots the outcome
    ax = fig.add_subplot(gs[-1], sharey=ax0)
    _plot_cont_kde_by_cohort(ax, sim_summary.data, 'y', 
                             palette=sim_summary.type_palette)
    ax.xaxis.set_tick_params(bottom=True, labelbottom=True, labelsize=8)
    ax.set_xlabel('z-score', size=8)

    # Titles the axes
    fig.axes[0].set_title('Cohort size', size=10)
    fig.axes[1].set_title('Race', size=10)
    fig.axes[2].set_title('Ethnicity', size=10)
    fig.axes[3].set_title('Sex', size=10)
    fig.axes[4].set_title('Smoking', size=10)
    fig.axes[5].set_title('Maternal\nAge', size=10)
    fig.axes[6].set_title('Calendar year', size=10)
    fig.axes[-1].set_title('Outcome', size=10)

    for ax in fig.axes:
        ax.yaxis.set_tick_params(left=False, labelleft=False,
                                 right=False, labelright=False,
                                 labelsize=0, length=0)

    sn.despine(left=True, right=True, top=True, bottom=False)
    sn.despine(ax=fig.axes[1], left=True, right=True, top=True, bottom=True)
    sn.despine(ax=fig.axes[2], left=True, right=True, top=True, bottom=True)
    sn.despine(ax=fig.axes[3], left=True, right=True, top=True, bottom=True)
    sn.despine(ax=fig.axes[4], left=True, right=True, top=True, bottom=True)

    return fig


def plot_simulation_summary(sim_summary, effect, fig=None, ):
    """
    Plots a summary of the simulation
    """
    if fig is None:
        fig = plt.figure(dpi=150, constrained_layout=True, facecolor='w')
    figr, figl = fig06c.subfigures(1, 2,)
    figr0, figr1 = figr.subfigures(2, 1)

    # Filtered forest plot
    ax = figl.add_subplot(1,1,1)
    plot_filtered_fits(sim_summary.filtered_fits, 
                       ax=ax, 
                       effect=effect,
                       tidy_short=tidy_short, 
                       filter_order=filter_order,
                       filter_tidy=filter_tidy)

    # Table 1
    axr0 = figr0.add_subplot(1,1,1, facecolor='None',)
    plot_table_1(axr0, exp_plots.build_table1(sim_summary.data),)

    # Distribution of weight by cohort type and exposure group
    gsr1 = figr1.add_gridspec(1, 3)
    violin_axes = [figr1.add_subplot(gsr1[0])]
    violin_axes.append(figr1.add_subplot(gsr1[1], sharey=violin_axes[0]))
    violin_axes.append(figr1.add_subplot(gsr1[2], sharey=violin_axes[0]))
    plot_pooled_violin(violin_axes, sim_summary.data)

    # Label axes
    axr0.text(x=axr0.get_xlim()[0] - np.diff(axr0.get_xlim()) * 0.1,
              y=axr0.get_ylim()[1] - np.diff(axr0.get_ylim()) * 0.1, 
              s='A', 
              size=12)
    figr1.axes[0].text(
        x=figr1.axes[0].get_xlim()[0] - np.diff(figr1.axes[0].get_xlim()) * 0.70,
        y=figr1.axes[0].get_ylim()[1] + np.diff(figr1.axes[0].get_ylim()) * 0.05,
        s='B', 
        size=12
        )
    _ = ax.text(x=ax.get_xlim()[0] - np.diff(ax.get_xlim()) * 0.25,
                y=ax.get_ylim()[1] - np.diff(ax.get_ylim()) * 0.075, 
                s='C',
                size=12)

    return fig


def plot_weight_dist(example_data, ax1, palette, box_kws=None, violin_kws=None):
    """
    Plots the distribution of birth weights
    """
    box_kwargs = dict(fliersize=0,
                      width=0.4,
                      linewidth=0.75,
                      #boxprops={'facecolor': 'w'},
                      # whis=0,
                      )
    if box_kws is not None:
        box_kwargs.update(box_kws)

    violin_kwargs = dict(inner=None,
                         linewidth=0.25,
                         offset=0,
                         zorder=10,
                         width=1. ,
                         cut=0,
                         )
    if violin_kws is not None:
        violin_kwargs.update(violin_kws)

    data_kwargs = dict(y='x', x='y', hue='cohort_type',
                       data=example_data,
                       order=np.arange(0, example_data['x'].max() + 1),
                       palette=palette,
                       ax=ax1,
                       orient='h',
                       legend=False,
                       )

    sn.boxplot(**data_kwargs, **box_kwargs)

    ax1.grid(axis='y')
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.set_xlabel('Birth weight z-score')
    ax1.yaxis.set_tick_params(left=False, labelleft=False)


def plot_bars(ax2, example_summary, lim_min, lim_max, effect, ax2l=None, ax2r=None, left=False, right=False, lo_min=np.nan, lo_max=np.nan, hi_min=np.nan, hi_max=np.nan,  d=0.1, xscale=0.1, yscale=1, **kwargs):
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
            axl.scatter(y=df['x_pos'], 
                        x=left_lim + (lim_max - lim_min) * 0.0225, 
                        color='k',
                        edgecolor='None',
                        zorder=3,
                        marker='<')
        if df['ci_hi'] > right_lim:
             ax2.scatter(y=df['x_pos'], 
                         x=right_lim - (lim_max - lim_min) * 0.0225, 
                         color='k',
                         edgecolor='None',
                         zorder=3,
                         marker='>')

    # Formats the axes
    lims = pd.Series([lim_min, lim_max, lo_min, lo_max, hi_min, hi_max]
                     ).dropna()
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


def plot_perc(ax0, example_summary):
    """
    Plots the percentage of smokers with appropriate colors
    """
    print(example_summary.head())
    ax0.barh(
        y=example_summary['x_pos'],
        width=example_summary['smoke'],
        color=example_summary['color'],
        zorder=3,
    )

    # Formats the axis
    ax0.set_ylim(example_summary['x_pos'].max() + 1, 
                 example_summary['x_pos'].min() - 1) 
    ax0.grid(True, axis='y')
    ax0.set_xticks(np.linspace(0, 1, 3))
    ax0.set_xlim(0, 1.05)
    ylim = ax0.get_ylim()
    ax0.plot([0.1, 0.1], ylim, 'k-', zorder=4, linewidth=1)
    ax0.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:>3.0%}'))
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(left=True, labelleft=False, length=0, labelsize=0)
    ax0.set_xlabel('\\% Smokers')
    ax0.xaxis.get_label().set_visible(True)


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


def _plot_cont_kde_by_cohort(ax, data, column, palette=None, ref_color='#252525', shade=True, linewidth=1):
    """
    Plots a kde plot of continous data
    """
    if palette is None:
        palette = dict()
    trace_vals = data.groupby('cohort')[column].apply(_fit_kde).to_dict()
    groups = data.groupby(['cohort_type', 'cohort'])['x'].mean()
    for (type_, cohort), y in groups.items():
        color = palette.get(type_, ref_color)
        [xx, yy] = trace_vals[cohort]
        yy = yy / yy.max() * 0.75
        y0 = xx * 0 + y - 0.5 + 0.125
        if shade:
            ax.fill_between(x=xx,  y1=y0, y2=yy + y0,
                            alpha=0.5,
                             edgecolor='None',
                             facecolor=color)
        ax.plot(xx, yy + y0, 
                linewidth=linewidth, 
                color=color)


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


def _plot_cohort_summary(ax, data, colors=None):
    """
    Summarizes the size of the estimated cohort
    """
    if colors is None:
        colors = {1 - i: mpc.to_hex(c) 
                  for i, c in enumerate(sn.color_palette(n_colors=2))}
    counts = data.groupby(['cohort', 'cohort_type'])['x'].describe()
    ax.barh(y=counts['mean'], 
            width=counts['count'], 
            left=0,
            height=0.8,
            color=counts.index.to_frame()['cohort_type'].replace(colors)
            )


def _plot_categorical_by_cohort(ax, data, column, colors):
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
        ax.barh(y=y, left=left_[c], width=v, color=colors[c])


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