import numpy as np
import pandas as pd
import scipy


class TabulateHeuristics():
    _tech_order = ['ignore',  'risk', 'fixed', 'gee', 'lme', 'meta']
    _tech_short = {'gee': 'GEE', 
                   'ignore': 'Ignore',
                   'lme': 'RE', 
                   'meta': 'MA',
                   'risk': 'CT',
                   }
    _scenario_names = {0: 'All uniform',
                       1: 'Vary population',
                       2: 'Vary size',
                       4: 'Select on exposure',
                       8: 'Select on outcome',
                       15: 'Vary all'
                       }
    _scenario_code = [0, 1, 2, 4, 8, 15]
    _icc_thresh =  [0.05, 0.1, 0.4, 0.7]
    _effects = [-0.1, -0.5, -9]
    _summary_f = {'size': lambda x: x['from_ref'].count(),
                  'mse': lambda x: np.square(x['from_ref']).mean(),
                  'bias': lambda x: x['sign_round'].mean(),
                  'coverage': lambda x: x['include'].mean(),
                  'power': lambda x: x['sig'].mean(),
                  }
 
    """
    """
    def __init__(self, extracted_params, seed_eff=None, icc_thresh=None, 
                 summary_f=None):
        """
        """
        # Gets the data
        self.df = extracted_params.copy()

        # Sets up the arguments
        self.icc_thresh = {None: self._icc_thresh}.get(icc_thresh, icc_thresh)
        self._prep_df(self.icc_thresh)
        
        self.scenarios = self.df.index.to_frame()['scenario'].unique()
        self.tech_order = self.df.index.to_frame()['model'].unique()
        self.icc_groups = self.df.index.to_frame()['icc_group'].unique()
        self.seed_effects = self.df.index.to_frame()['seed_effect'].unique()
        self.summary_f = {None: self._summary_f}.get(summary_f, summary_f)
        
        self.sim_stats = None

    
    def tabulate_sim_stats(self):
        """
        Tabulates simulations statistics
        """
        # Summaries and groups the data to make apply tghe function easier
        icc_g = self.df.groupby(['scenario', 'model', 'icc_group'])
        icc_eff_g = \
            self.df.groupby(['scenario', 'model', 'icc_group', 'seed_effect'])
        eff_g = self.df.groupby(['scenario', 'model', 'seed_effect'])
        all_g = self.df.groupby(['scenario', 'model'])
    
        stats_g = [icc_g, icc_eff_g, eff_g, all_g]

        # Summarizes the statiscs and adds a dummy level if no qualifying
        # trials are identified
        sim_stats = pd.concat([
            pd.DataFrame({s: df_.apply(f_) 
                          for s, f_ in self.summary_f.items()}).reset_index()
            for df_ in stats_g])
        sim_stats = pd.concat([sim_stats, 
                               self._build_sim_ref()[sim_stats.columns]])
        sim_stats.replace({'icc_group': {np.nan: -1},
                           'seed_effect': {np.nan: -9}},
                          inplace=True)
        for c in sim_stats.columns:
            if c in {'model'}:
                continue
            sim_stats[c] = sim_stats[c].astype(float)
        
        sim_stats.sort_values([
            'scenario', 'model', 'icc_group', 'seed_effect', 'size'],
            inplace=True,
            ascending=[True, True, True, True, False]
            )
        sim_stats.drop_duplicates(
            ['scenario', 'model', 'icc_group', 'seed_effect'],
            keep='first',
            inplace=True)
        sim_stats.reset_index(drop=True, inplace=True)
        
        # Pivot madness to re-orient the data and mask values less than 10
        sim_stats = sim_stats.melt(
            id_vars=['scenario', 'model', 'icc_group', 'seed_effect', 'size']
            )
        sim_stats.set_index(['scenario', 'icc_group', 'seed_effect'],
                            inplace=True)
        sim_stats['size'] = \
            sim_stats.groupby(sim_stats.index.names)['size'].max()
        sim_stats.reset_index(inplace=True)
        sim_stats['value'] = sim_stats['value'].mask(sim_stats['size'] < 10)
        sim_stats['model'] = pd.Categorical(sim_stats['model'], ordered=True, 
                                            categories=self._tech_order)
        self.sim_stats = \
            sim_stats.pivot(
                index=['variable', 'seed_effect', 'scenario', 'icc_group',
                       'size'],
                columns='model',
                values='value'
                )
        # Cleans up labeling
        self.sim_stats.sort_index(
            level=['variable', 'seed_effect', 'scenario', 'icc_group'],
             ascending=[True, False, True, True],
            inplace=True)

    
    def summarize_simulations(self):
        """
        Summarizes the simulations at all ICC levels into a pretty(?) table
        """
        self.tabulate_sim_stats()
        # Pulls out the overall summary for all effects
        stats_all = self.sim_stats.xs(-1, level='icc_group').copy()
        stats_all = stats_all.xs(-9, level='seed_effect').copy()
        stats_all.reset_index('size', drop=True, inplace=True)
        stats_all = stats_all.round(3)
        stats_all = stats_all.loc[['mse', 'bias', 'coverage']]

        # Because power is specific to the effect size, we pull that out
        # seperately
        power = self.sim_stats.xs(-1, level='icc_group').copy()
        power = power.xs('power', level='variable')
        power.reset_index(level='size', inplace=True, drop=True)
        power.drop(index=[-9], inplace=True, level='seed_effect')
        power.rename(index={-0.1: '-0.1',
                            -0.5: '-0.5',},
                     level='seed_effect',
                     inplace=True)
        power = power.round(3)

        # Retunrs the combination
        sim_stats_c = pd.concat([stats_all, power])
        sim_stats_c.index.set_names(['variable', 'scenario'], 
                                    inplace=True)
        sim_stats_c.reset_index(inplace=True)
        sim_stats_cr = sim_stats_c.copy().drop_duplicates('variable')
        sim_stats_cr.set_index(['variable', 'scenario'], inplace=True)
        sim_stats_cr = sim_stats_cr * np.nan
        sim_stats_cr.reset_index(inplace=True)
        sim_stats_cr['scenario'] = -1
        sim_stats2 = pd.concat(axis=0, objs=[sim_stats_cr, sim_stats_c])
        sim_stats2['variable'] = \
            pd.Categorical(sim_stats2['variable'],
                           ['mse', 'bias', 'coverage', '-0.1', '-0.5'])
        sim_stats2.set_index(['variable', 'scenario'], 
                             inplace=True)
        sim_stats2.sort_index(ascending=True, inplace=True)
        sim_stats2.rename(index={'mse': 'MSE',
                                 'bias': 'Bias',
                                 'coverage': 'Coverage',
                                 # '-0.1': 'Power (-0.1sd)',
                                 # '-0.5': 'Power (-0.5sd)',
                                 },
                          level='variable',
                          inplace=True)
        sim_stats2.rename(index=self._scenario_names,
                          level='scenario',
                          inplace=True)
        sim_stats2.rename(index={-1: ''}, 
                          columns=self._tech_short, 
                          inplace=True)
        sim_stats2 = sim_stats2.map(
            lambda x: '' if pd.isnull(x) else f'{x:1.3f}')

        self._tidy_sim_stats()

        return sim_stats2
        
    def _tidy_sim_stats(self, scenario_names=None, model_names=None):
        """
        
        """
        icc_tn = np.vstack([np.hstack([0, self.icc_thresh]),
                            np.hstack([self.icc_thresh, 1]),
                            ])
        icc_name = {i: f'[{lo_:1.2f}, {hi_:1.2f})'
                    for i, (lo_, hi_) in enumerate(icc_tn.T)}
        icc_name[-1] = 'All'
        
        if scenario_names is None:
            scenario_names = self._scenario_names
        if model_names is None:
            model_names = self._tech_short
        self.sim_stats.rename(index=icc_name, level='icc_group', inplace=True)
        self.sim_stats.rename(index=scenario_names, level='scenario', inplace=True)
        # self.sim_stats.rename(index=model_names, level='model', inplace=True)

    def _build_sim_ref(self):
        """
        Builds a simulation reference
        """
        ref_nans = np.ones(len(self.summary_f)) * np.nan
        ref_heur = pd.DataFrame(
            data=np.vstack([np.hstack([sc, mod, icc, se, ref_nans])
                            for sc in self.scenarios
                            for mod in self.tech_order
                            for icc in np.hstack([self.icc_groups, -1])
                            for se in np.hstack([self.seed_effects, -9])]),
            columns=np.hstack(['scenario', 'model', 'icc_group', 
                               'seed_effect', 
                               list(self.summary_f.keys())])
            )
        ref_heur['size'] = 0
        return ref_heur

    def _prep_df(self, icc_thresh=None):
        """
        Prepares the df for analysis
        """
        if 'seed_effect' in self.df.index.names:
            self.df.reset_index(level='seed_effect', inplace=True)
        
        # Calculates accuracy as relationship with seed value
        self.df['from_ref'] =  self.df['param'] - self.df['seed_effect']
        self.df['abs_diff'] = np.absolute(self.df['from_ref'])
        self.df['sign_diff'] = np.sign(self.df['from_ref'])
        self.df['sign_round'] = \
            self.df['sign_diff'].mask(self.df['abs_diff'] < 5e-2, 0)

        # Checks coverage
        self.df['include'] = \
            ((self.df['ci_lo'] <= self.df['seed_effect']) & \
             (self.df['ci_hi'] >= self.df['seed_effect'])) * 1

        # Checks significance
        self.df['sig'] = \
            ((self.df['p-value'] < 0.05) |  
             (np.sign(self.df['ci_lo']) == np.sign(self.df['ci_hi']))) * 1

        # Calculates ICC group
        if icc_thresh is None:
            icc_thresh = self._icc_thresh
        if 'icc' in self.df.index.names:
            self.df.reset_index(level='icc', inplace=True)
        icc_values = self.df['icc'].unique()
        icc_mat = np.ones((len(icc_values), len(icc_thresh))) * \
            np.atleast_2d(icc_values).T
        thresh_mat =  np.ones((len(icc_values), len(icc_thresh))) * \
            np.atleast_2d(icc_thresh)
        icc_bin = pd.DataFrame(
            data=icc_mat >= thresh_mat,
            index=icc_values)
        icc_bin = icc_bin.sum(axis=1)
        self.df['icc_group'] = self.df['icc'].replace(icc_bin)

        self.df.set_index(['seed_effect', 'icc_group'], 
                          append=True, 
                          inplace=True)

    def summarize_sim_eff_by_icc(self):
        """
        Provides a list of the simulated effect by ICC group
        """
        # Preps data slices
        if self.sim_stats is None:
            self.tabulate_sim_stats()
        self._tidy_sim_stats()
        all_eff = self.sim_stats.xs(-9, level='seed_effect').copy()
        all_eff.drop(['All'], level='icc_group', inplace=True)
        pwr_eff = self.sim_stats.drop(-9, level='seed_effect').copy()
        pwr_eff.drop(['All'], level='icc_group', inplace=True)

        sim_eff = [
            np.log10(all_eff.xs('mse', level='variable')),
            all_eff.xs('coverage', level='variable'),
            all_eff.xs('bias', level='variable'),
            pwr_eff.xs('power', level='variable').xs(-0.1, level='seed_effect'),
            pwr_eff.xs('power', level='variable').xs(-0.5, level='seed_effect'),
        ]
        return sim_eff   


