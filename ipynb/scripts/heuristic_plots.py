import numpy as np
import pandas as pd
import scipy

from matplotlib import ticker
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import seaborn as sn

from scripts._heuristics_summary import TabulateHeuristics

grid_axes = dict(bottom=True, 
                 labelbottom=False,
                 top=True, 
                 labeltop=False,
                 tickdir='in',
                 labelsize=8)

hide_axes = dict(left=False, labelleft=False,
                 right=False, labelright=False,
                 labelsize=0, length=0)

code_bools = {0:  [0, 0, 0, 0],
              1:  [1, 0, 0, 0],
              2:  [0, 1, 0, 0],
              3:  [1, 1, 0, 0],
              4:  [0, 0, 1, 0],
              5:  [1, 0, 1, 0],
              6:  [0, 1, 1, 0],
              7:  [1, 1, 1, 0],
              8:  [0, 0, 0, 1],
              9:  [1, 0, 0, 1],
              10: [0, 1, 0, 1],
              11: [1, 1, 0, 1],
              12: [0, 0, 1, 1],
              13: [1, 0, 1, 1],
              14: [0, 1, 1, 1],
              15: [1, 1, 1, 1]
              }
              

def plot_ref_scatter(fig, gs, scenarios, check_mark='x', box_size=None, 
                     boxlinewidth=1,  ax_labels=None, check_color='k', 
                     offset=0, labeltop=True):
    """
    Creates a panel highlighting what varies in a dataaset
    """
    for i, (bool_, _) in enumerate(scenarios):
        # Axis
        ax = fig.add_subplot(gs[i + offset, 0], facecolor='None',)
    
        num_factors = len(bool_)
        x_ref = np.linspace(0, 1, num_factors)
        # Generates empty check boxes in each position
        ax.scatter(x=x_ref,
                   y=x_ref * 0,
                   marker='s', 
                   s=box_size, 
                   facecolor='None', 
                   linewidth=boxlinewidth, 
                   edgecolor='k')

        # Figures out the boxes to check and checks them
        x = np.array(bool_) * (x_ref + 1)
        x = x[x > 0] - 1
        ax.scatter(x=x,
                   y=x * 0,
                   s=box_size,
                   marker=check_mark, 
                   color=check_color,
                   )
    
        # Tidies the axis
        ax.set_xlim(-0.15, 1.15)
        ax.set_xticks(np.linspace(0, 1, num_factors))
        ax.set_xticklabels(['Population', 'Size', 
                            'Exposure', 'Outcome'], 
                           rotation=90, size=9)
        ax.xaxis.set_tick_params(bottom=False, labeltop=(i == 0) & labeltop,
                                 labelbottom=False)
        ax.yaxis.set_tick_params(left=False, labelleft=False)
        sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)


def plot_ref_scatter_mod(fig, gs, scenario_codes, check_mark='x', 
                     box_size=None, 
                     boxlinewidth=1,  ax_labels=None, check_color='k', 
                     offset=0, labeltop=True, show_mod=False):
    """
    Creates a panel highlighting what varies in a dataaset allowing 
    notes of effect modification
    """
    codes = np.array(scenario_codes)
    codes_s = np.floor(codes)
    codes_m = codes - np.floor(codes)

    for i, code in enumerate(scenario_codes):
        ax = fig.add_subplot(gs[i + offset, 0], facecolor='None',)
        x_ref = np.arange(4)
        y_ref = np.zeros(x_ref.shape)
        ax.scatter(x=x_ref,
                   y=x_ref * 0,
                   marker='s', 
                   s=box_size, 
                   facecolor='None', 
                   linewidth=boxlinewidth, 
                   edgecolor='k')
            
        code_b = np.floor(code).astype(int)
        x = np.array(code_bools[code_b]) * (x_ref + 1)
        x = x[x > 0] - 1
        ax.scatter(x=x,
                   y=x * 0,
                   s=box_size,
                   marker=check_mark, 
                   color=check_color,
                   )
        if show_mod:
            mod_symbol = {0: '$$\\O$$', 
                          0.25: '\\propto \\sigma^{2}$',
                          0.5: '$\\propto ct$'
                          }
            mod = code - code_b
            ax.text(x=4, y=0, s=mod_symbol[mod], 
                    ha='center', 
                    va='center', 
                    size=8)
            ax.set_xticks(np.arange(4 + show_mod))
            ax.set_xticklabels(['Population', 'Size', 
                                'Exposure', 'Outcome', 
                                'Eff. Mod.'], 
                                rotation=90, size=9)
            ax.set_xlim(-0.5, 5)
        else:
            ax.set_xticks(x_ref)
            ax.set_xticklabels(['Population', 'Size', 
                                'Exposure', 'Outcome'], 
                                rotation=90, size=9)
            ax.set_xlim(-0.5, 3.5)
        ax.xaxis.set_tick_params(bottom=False, labeltop=(i == 0) & labeltop,
                                 labelbottom=False)
        ax.yaxis.set_tick_params(left=False, labelleft=False)
        sn.despine(ax=ax, left=True, right=True, top=True, bottom=True)
    

def plot_scary_tree_inset(ax_, df_, x='icc', error_kws=None):
    """
    
    """
    error_kwargs = dict(linewidth=0.5, capsize=0, alpha=0.75)
    if isinstance(error_kws, dict):
        error_kwargs.update(error_kws)

    # Calculates parameters for hte data set
    df_['width'] = df_['ci_hi'] - df_['param']
    df_['outside_lo'] = df_[['from_ref', 'width']].sum(axis=1) < -1.5
    df_['outside_hi'] = (df_['from_ref'] - df_['width']) > 1.5
    df_['inside'] = ~df_[['outside_lo', 'outside_hi']].any(axis=1)
    # Plots the results
    ax_.errorbar(x=x, y='from_ref',  yerr='width', 
                data=df_.loc[df_[['include', 'inside']].all(axis=1)],
                color='#919191',
                **error_kwargs)
    ax_.errorbar(x=x, y='from_ref',  yerr='width', 
                data=df_.loc[~df_['include'] & df_['inside']],
                color='red',
                **error_kwargs)
    yscale = np.diff(ax_.get_ylim()) * 0.1

    # If there are outliers, adds a rugplot for hte outpliers
    if df_['outside_lo'].any():
        x_lo = df_.loc[df_['outside_lo'], [x, x]].copy()
        y_lo = np.vstack([np.ones(len(x_lo)) * ax_.get_ylim()[0],
                          np.ones(len(x_lo)) * ax_.get_ylim()[0] + yscale])
        ax_.plot(x_lo.T, y_lo, color='#99000d', linewidth=0.5)
    if df_['outside_hi'].any():
        x_hi = df_.loc[df_['outside_hi'], [x, x]].copy()
        y_hi = np.vstack([np.ones(len(x_hi)) * ax_.get_ylim()[1],
                          np.ones(len(x_hi)) * ax_.get_ylim()[1] - yscale])
        ax_.plot(x_hi.T, y_hi, color='#99000d', linewidth=0.5)

    _tidy_axes(ax_)
    # Cleans up the axis
    ax_.set_ylim(-1.5, 1.5)
    ax_.set_yticks(np.linspace(-1, 1, 3))
    ax_.grid(visible=True, axis='y', which='major')


def plot_corr_scatter(ax_, sub_data, y, x='icc', ylim=None,
                      scatter_kws=None):
    """
    Plots scatter between ICC and outcome
    """
    scatter_kwargs = dict(alpha=0.75, 
                       legend=False,
                       edgecolor='None',
                       marker='.',
                       s=5,
                       color='#525252',
                       )
    if isinstance(scatter_kws, dict):
        scatter_kwargs.update(scatter_kws)
    sn.scatterplot(x=x, y=y, ax=ax_, data=sub_data, **scatter_kwargs)
    yflag = (ylim is not None)
    if ylim is None:
        ylim = ax_.get_ylim()
    ax_.set_ylim(ylim)
    yscale = np.diff(ax_.get_ylim())[0] * 0.1
    if (sub_data[y] > ylim[1]).any() & yflag:
        x_hi = sub_data.loc[sub_data[y] > ylim[1], [x, x]].copy()
        y_hi = np.vstack([
            np.ones(len(x_hi)) * ax_.get_ylim()[1],
            np.ones(len(x_hi)) * ax_.get_ylim()[1] - yscale
            ])
        ax_.plot(x_hi.T, y_hi, color='#525252', linewidth=0.5)
    if (sub_data[y] < ylim[0]).any() & yflag:
        x_lo = sub_data.loc[sub_data[y] < ylim[0], [x, x]].copy()
        y_lo = np.vstack([
            np.ones(len(x_lo)) * ax_.get_ylim()[0],
            np.ones(len(x_lo)) * ax_.get_ylim()[0] + yscale
            ])
        ax_.plot(x_lo.T, y_lo, color='#525252', linewidth=0.5)

    # Cleans up axes
    _tidy_axes(ax_)
    ax_.set_facecolor('None')


def _tidy_axes(ax_):
    """
    Collapses axis clean up code
    """
    # Cleans up x axis
    ax_.set_xlim(-0.05, 1.05)
    ax_.set_xticks(np.linspace(0, 1, 3))
    ax_.set_xticklabels(['0', '0.5', '1'])
    ax_.xaxis.set_tick_params(**grid_axes)
    ax_.xaxis.get_label().set_visible(False)

    # Cleans up yaxis
    ax_.yaxis.set_tick_params(**grid_axes)
    ax_.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:1.0f}'))
    ax_.yaxis.get_label().set_visible(False)


def ylabel_plot(axl, text, pad=30):
    """
    Adds a ylabel to grid plots
    """
    ax_l.set_ylabel(text, size=12, rotation=-90, va='bottom', labelpad=pad)
    ax_l.yaxis.set_label_position('right')
    ax_l.set_xlabel('ICC', size=12, labelpad=18)
    ax_l.xaxis.set_tick_params(bottom=False, labelbottom=False, top=False, 
                               labeltop=False, length=0, labelsize=0)
    ax_l.yaxis.set_tick_params(bottom=False, labelbottom=False, top=False, 
                               labeltop=False, length=0, labelsize=0)
    sn.despine(ax=ax_l, left=True, right=True, top=True, bottom=True)


class HeuristicsByICCFigure():
    """
    
    """
    _hm_kwargs = [dict(vmin=-3, vmax=-1, cmap='PuRd'),
                  dict(vmin=0.65, vmax=1, cmap='YlOrRd_r'),
                  dict(vmin=-0.3, vmax=0.3, cmap='RdBu_r'),
                  dict(vmin=0, vmax=1.0, cmap='BuPu_r'),
                  dict(vmin=0, vmax=1.0, cmap='BuPu_r'),
                  ]
    _analysis_levels = ['All uniform', 
                       'Vary population', 
                       'Vary size',
                       'Select on exposure',
                       'Select on outcome', 
                       'Vary all',
                       ]
    _scenarios = [[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 1, 1, 1]
                  ]
    _scenario_labels = ['Population', 'Size', 'Exposure', 'Outcome']
    _titles = ['log$_{10}$(MSE)', 
               'Coverage', 
               'Bias', 
               'Power\n(0.1sd)', 
               'Power\n(0.5sd)']
    
    def __init__(self, sim_eff, dpi=150, analysis_levels=None, titles=None, 
                 color_kwargs=None, fig_dims=(6, 5), scenarios=None,
                 scene_step_label=None, check_mark='s', check_color='k',
                 boxlinewidth=1, boxsize=None,
                 ):
        # Sets up the ordered arguments
        self.sim_eff = sim_eff
        self.colors = {None: self._hm_kwargs}.get(color_kwargs, color_kwargs)
        self.anal_lvls = {None: self._analysis_levels}.get(
            analysis_levels, analysis_levels)
        self.scenarios = {None: self._scenarios}.get(scenarios, scenarios)
        self.scene_bool_label = {None: self._scenario_labels}.get(
            scene_step_label, scene_step_label)
        self.titles = {None: self._titles}.get(titles, titles)
        self.check_mark = check_mark
        self.check_color = check_color
        self.boxlinewidth = 1
        self.box_size = boxsize

        num_eff = np.min([len(self.sim_eff), 
                          len(self.colors), 
                          len(self.titles)])
        num_anal = len(self.anal_lvls)

        # Sets up the figure and grid
        self.fig = plt.figure(dpi=dpi, constrained_layout=True, 
                               figsize=fig_dims)
        self.gs = self.fig.add_gridspec(
            2 + num_anal, 2 + num_eff,
            width_ratios=np.hstack([1, 0.5, np.ones((num_eff,))]),
            height_ratios=np.hstack([0.5, 0.5, np.ones((num_anal,))])
            )
        ax_r = None
        for i, level in enumerate(self.anal_lvls):
            ax_ms, ax_row = self._plot_scenario_hm(i, level, ax_ms=ax_r)
            ax_r = ax_ms
            # We'll add the colorbars on the first row
            if i == 0:
                self._plot_upper_labels(ax_row)
        self._plot_ref_scatter()
        

    def _plot_upper_labels(self, ax_row):
        """
        Adds the colorbars and labels to match the upper row of the figure
        """
        iter_pack = zip(*(ax_row, self.colors, self.titles))
        for j, (ax_r, colors, title) in enumerate(iter_pack):
            ax_lb = self.fig.add_subplot(self.gs[1, j + 2], 
                                         facecolor='None', 
                                         sharex=ax_r)
            ax_cb = self.fig.add_subplot(self.gs[0, j + 2], facecolor='None')
            self._tidy_label_ax(ax_lb)
            self._plot_cbar_h(ax_cb, title, colors)

    
    @staticmethod
    def _plot_cbar_h(ax_cb, title, kwargs):
        """
        Plots the colormap bar
        """
        ax_cb.set_ylim(0, 1)
        ax_cb.set_xlim(-0.2, 1.2)
        ax_cb.set_ylim(0, 1)
        ax_cb.set_xlim(-0.2, 1.2)
        ax_cb.xaxis.set_tick_params(**hide_axes)
        ax_cb.yaxis.set_tick_params(**hide_axes)
        xx = np.linspace(0, 1, 13)[:12]
        xp = np.linspace(0, 1, 3)
        xl = np.linspace(kwargs['vmin'], kwargs['vmax'], 3)
        palette = sn.color_palette(kwargs['cmap'], n_colors=12)
        palette = [mpc.to_hex(c) for c in palette]
        ax_cb.barh(left=xx, width=1 / 12, y=0.25, height=0.2, color=palette)
        ax_cb.scatter(xp, 
                      y=np.ones(xp.shape) * (0.25 + 0.1), 
                      marker='|', 
                      color='k')
        
        for x_, l_ in zip(*(xp, xl)):
            ax_cb.text(x_, 0.25 + 0.2, l_, ha='center', va='bottom', size=8)
    
        ax_cb.set_title(title)
        sn.despine(ax=ax_cb, left=True, right=True, top=True, bottom=True)


    @staticmethod
    def _tidy_label_ax(ax_xl):
        """
        Creates a label axis
        """
        ax_xl.yaxis.set_tick_params(**hide_axes)
        ax_xl.xaxis.set_tick_params(bottom=True, labelbottom=True, 
                                     top=False, labeltop=False,             
                                     labelsize=8, length=-3)
        for l in ax_xl.get_xticklabels():
            l.set_rotation(90)
            l.set_va('bottom')
        sn.despine(ax=ax_xl, left=True, right=True, top=True, bottom=True)

    @staticmethod
    def plot_counted(ax_ic, idx, threshhold=10):
        """
        Plots the number of simulations in a group
        """
        # Sets up barplot arguments
        size_ = idx['size']
        ylab_ = idx['icc_group'].values
        color = (size_ >= threshhold).replace({False: '#dbdbdb', 
                                               True: '#252525'})
        y = np.arange(len(idx)) + 0.5
        ax_ic.set_yticks(y)
        ax_ic.set_yticklabels(idx['icc_group'].values)
        # Plots number of samples in band
        ax_ic.barh(y, width=size_, color=color, edgecolor='#252525', 
                   linewidth=0.5)
        ax_ic.set_xticks(np.linspace(0, 200, 3))
        ax_ic.set_xticks([threshhold], minor=True)
        ax_ic.xaxis.set_tick_params(**hide_axes, gridOn=True, which='minor')
        ax_ic.xaxis.set_tick_params(bottom=True, labelbottom=False, 
                                    labelsize=8, 
                                    which='major')
        ax_ic.yaxis.set_tick_params(**hide_axes)
        sn.despine(ax=ax_ic, left=False, bottom=False, right=True, top=True)
        ax_ic.set_yticks(y)
        ax_ic.set_yticklabels(idx['icc_group'].values)
        ax_ic.yaxis.set_tick_params(labelleft=True, labelsize=8)
    
    
    def _plot_scenario_hm(self, i, scenario, ax_ms=None):
        """
        Plots a row of scenarios based on the data
        """
        # Adds the axes
        # ax_ms = fig_.add_subplot(gs[0 + 2, 3], facecolor='#dfdfdf')
        ax_ms = self.fig.add_subplot(self.gs[i + 2, 2], 
                                     facecolor='#dfdfdf', 
                                     sharex=ax_ms, 
                                     sharey=ax_ms)
        ax_ic = self.fig.add_subplot(self.gs[i + 2, 1], 
                                     sharey=ax_ms)
        ax_ic.yaxis.set_tick_params(**hide_axes)
        ax_kws = dict(facecolor='#dfdfdf', sharey=ax_ms, sharex=ax_ms)
        ax_cv = self.fig.add_subplot(self.gs[i + 2, 3], **ax_kws)
        ax_bs = self.fig.add_subplot(self.gs[i + 2, 4], **ax_kws)
        ax_ps = self.fig.add_subplot(self.gs[i + 2, 5], **ax_kws)
        ax_pl = self.fig.add_subplot(self.gs[i + 2, 6], **ax_kws)
        hm_axes = [ax_ms, ax_cv, ax_bs, ax_ps, ax_pl]
    
        # Plots the heatmaps
        zipper = zip(*(hm_axes, self.sim_eff, self.colors))
        for j, (ax_hm, eff, kwargs) in enumerate(zipper):
            ax_hm.xaxis.set_tick_params(**hide_axes)
            ax_hm.yaxis.set_tick_params(**hide_axes)
            data_ = eff.xs(scenario, level='scenario')
            sn.heatmap(ax=ax_hm, data=data_, **kwargs, 
                       linewidth=0.5,
                       linecolor='k',
                       cbar=False,
                       )                   
            ax_hm.yaxis.get_label().set_visible(False)
            ax_hm.xaxis.get_label().set_visible(False)
            ax_hm.set_xticks(np.arange(6) + 0.5)
            ax_hm.set_xticklabels(['Ignore', 'CT', 'Fixed', 'GEE', 
                                   'RE', 'MA'], 
                                  rotation=90)
            ax_hm.xaxis.set_tick_params(labelsize=8, labeltop=False)

        # Counts the number of experiments in each group
        data_ = self.sim_eff[0].xs(scenario, level='scenario')
        self.plot_counted(ax_ic, data_.index.to_frame())
        
        return ax_ms, hm_axes

    def _plot_ref_scatter(self):
        """
        Adds a reference scatter highlighting what's been varied across 
        datasets
        """
        ax_lb = self.fig.add_subplot(self.gs[0:2, 0], facecolor="None")
        ax_lb.yaxis.set_tick_params(**hide_axes)
        ax_lb.xaxis.set_tick_params(**hide_axes)
        ax_lb.set_xticks(np.linspace(0, 1, len(self.scene_bool_label)))
        ax_lb.set_xlim(-0.15, 1.15)
        ax_lb.set_xticklabels(self.scene_bool_label, rotation=90, size=8)
        self._tidy_label_ax(ax_lb)
        
        for i, bool_ in enumerate(self.scenarios):
            ax_sc = self.fig.add_subplot(self.gs[i + 2, 0], 
                                         sharex=ax_lb, 
                                         facecolor='None')
            ax_sc.xaxis.set_tick_params(**hide_axes)
            ax_sc.yaxis.set_tick_params(**hide_axes)
            num_factors = len(bool_)
            x_ref = np.linspace(0, 1, num_factors)
            # Generates empty check boxes in each position
            ax_sc.scatter(x=x_ref,
                       y=x_ref * 0,
                       marker='s', 
                       s=self.box_size, 
                       facecolor='None', 
                       linewidth=self.boxlinewidth, 
                       edgecolor='k')
    
            # Figures out the boxes to check and checks them
            x = np.array(bool_) * (x_ref + 1)
            x = x[x > 0] - 1
            ax_sc.scatter(x=x,
                          y=x * 0,
                          s=self.box_size,
                          marker=self.check_mark, 
                          color=self.check_color,
                          )
            sn.despine(ax=ax_sc, left=True, right=True, top=True, bottom=True)
            