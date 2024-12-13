import numpy as np
import pandas as pd
import scipy

import statsmodels.formula.api as smf
import statsmodels.api as sms
import statsmodels.stats.meta_analysis as sma
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import scripts.bw_sim


def icc_from_lme(lme_fit):
    """
    Estimates an Interclass Correlation from the fit model
    """
    var_covar = lme_fit.cov_params()
    icc = var_covar.loc['Group Var', 'Group Var'] / np.trace(var_covar)
    return icc


def convert_icc_to_g_var_no_slope(icc1):
    """
    Estimates the required g_var value to simulate based on a seed ICC

    Directly based on the sim.icc function from the R multilevel package
    https://rdrr.io/cran/multilevel/man/sim.icc.html
    """
    g_var = np.sqrt((1/(1 - icc1)) - 1)
    return g_var


def cast_to_icc_long(fit_iccs):
    """
    A convenience function to extract ICC information from simulations
    """
    iccs2 = pd.DataFrame({g_var: fit['iccs2'] 
                          for g_var, fit in fit_iccs.items()})
    iccs2.columns.set_names('g_var', inplace=True)
    iccs2_long = iccs2.unstack().reset_index(level='g_var')
    iccs2_long.rename(columns={0: 'fit_icc'}, inplace=True)
    iccs2_long.dropna(inplace=True)

    return iccs2_long


def ignore_cohort(icc, data, adj):
    """
    A simple function to fit an OLS model ignoring the cohort effect
    """
    fit = smf.ols(f'y ~ C(smoking) + {adj}', data=data).fit()
    params = pd.Series(
        np.hstack([fit.params.loc['C(smoking)[T.1]'], 
                  fit.bse.loc['C(smoking)[T.1]'],
                  fit.conf_int().loc['C(smoking)[T.1]'],
                  fit.pvalues.loc['C(smoking)[T.1]'],
                  ]),
        index=['param', 'bse', 'ci_lo', 'ci_hi', 'p-value'],
    )
    return icc, fit, params, True, 'ignore'


def fixed_cohort(icc, data, adj):
    """
    A simple function to fit an OLS model treating cohort as a fixed effect
    """
    fit = smf.ols(f'y ~ C(smoking) + cohort + {adj}', data=data).fit()
    params = pd.Series(
        np.hstack([fit.params.loc['C(smoking)[T.1]'], 
                  fit.bse.loc['C(smoking)[T.1]'],
                  fit.conf_int().loc['C(smoking)[T.1]'],
                  fit.pvalues.loc['C(smoking)[T.1]'],
                  ]),
        index=['param', 'bse', 'ci_lo', 'ci_hi', 'p-value'],
    )
    return icc, fit, params, True, 'fixed'


def gee_cohort(icc, data, adj):
    """
    A simple function to fit a GEE nested by cohort
    """
    fit = smf.gee(f'y ~ C(smoking) + {adj}', 
                  groups=data['cohort'], 
                  cov_struct=sms.cov_struct.Exchangeable(),
                  data=data).fit()
    params = pd.Series(
        np.hstack([fit.params.loc['C(smoking)[T.1]'], 
                  fit.bse.loc['C(smoking)[T.1]'],
                  fit.conf_int().loc['C(smoking)[T.1]'],
                  fit.pvalues.loc['C(smoking)[T.1]'],
                  ]),
        index=['param', 'bse', 'ci_lo', 'ci_hi', 'p-value'],
    )
    return icc, fit, params, fit.converged, 'gee'


def lme_cohort(icc, data, adj):
    """
    A simple function to fit an LME with cohort as a random intercept
    """
    fit = smf.mixedlm(f'y ~ C(smoking) + {adj}', groups=data['cohort'], 
                      data=data).fit()
    params = pd.Series(
        np.hstack([fit.params.loc['C(smoking)[T.1]'], 
                  fit.bse.loc['C(smoking)[T.1]'],
                  fit.conf_int().loc['C(smoking)[T.1]'],
                  fit.pvalues.loc['C(smoking)[T.1]'],
                  ]),
        index=['param', 'bse', 'ci_lo', 'ci_hi', 'p-value'],
    )
    return icc, fit, params, fit.converged, 'lme'


def lme_group(icc, data, adj, risk_='cohort_type'):
    """
    A simple function to fit an LME with cohort as a random intercept
    """
    fit = smf.mixedlm(f'y ~ C(smoking) + {adj}', groups=data[risk_], 
                      data=data).fit()
    params = pd.Series(
        np.hstack([fit.params.loc['C(smoking)[T.1]'], 
                  fit.bse.loc['C(smoking)[T.1]'],
                  fit.conf_int().loc['C(smoking)[T.1]'],
                  fit.pvalues.loc['C(smoking)[T.1]'],
                  ]),
        index=['param', 'bse', 'ci_lo', 'ci_hi', 'p-value'],
    )
    return icc, fit, params, fit.converged, 'risk'


def meta_cohort(icc, data, adj):
    """
    A simple function build a meta analysis across cohorts with an effect
    """
    cohorts = np.sort(data['cohort'].unique())
    fits = {c: smf.ols(f'y ~ C(smoking) + {adj}', data=df).fit()
            for c, df in data.groupby('cohort')
            }
    fits = {c: fit for c, fit in fits.items() 
            if ('C(smoking)[T.1]' in fit.params.index)}
    m = pd.Series({
        c: fit.params.loc['C(smoking)[T.1]'] for c, fit in fits.items() 
    })
    err = pd.Series({
        c: fit.bse.loc['C(smoking)[T.1]'] for c, fit in fits.items() 
    })

    meta_res = sma.combine_effects(effect=m.values,
                                   variance=err.values,
                                   row_names=m.index,
                                   )
    params = meta_res.summary_frame().loc['fixed effect'].copy()
    params.rename(index={'eff': 'param', 
                         'sd_eff': 'bse', 
                         'ci_low': 'ci_lo', 
                         'ci_upp': 'ci_hi'},
                  inplace=True
                 )
    params['p-value'] = np.nan
    params.drop(index=['w_fe', 'w_re'], inplace=True)
    return icc, (fits, meta_res), params, (len(fits) == len(cohorts)), 'meta'


models = [ignore_cohort, fixed_cohort, gee_cohort, lme_cohort, meta_cohort, lme_group]
icc_thresh = [0.1, 0.4, 0.75]

def extract_cohort_parameters(data, models=models, adj_eq='1', 
                              icc_thresh=icc_thresh):
    """
    Fits simulated data to the model and tabulates the results
    """
    # Fits *all* the models and data sets, which return the ICC, fit, or 
    # meta analysis results object, parameters, whether the model converged,
    # and the type of fit performed
    model_res = [
        model(icc, df, adj_eq) 
        for icc, df in data.items()
        for model in models
    ]
    # Determines if the fits converged
    model_converged = pd.Series({
        (model, icc): converged 
        for icc, fit, params, converged, model in model_res
    })

    # Extracts the parameters for the model and adds incormation about 
    # convergence into the model. Then, we clean up the data frame because
    # that just makes life easier.

    param_res = pd.DataFrame({
        (model, icc): params
        for icc, fit, params, converged, model in model_res
    }).T
    param_res['converged'] = model_converged * 1
    # Cleans up labeling
    param_res.index.set_names(['model', 'icc'], inplace=True)
    param_res.rename(columns={'param': 'beta'}, inplace=True)

    # Sets up a threshholded ICC value based ont he specified list of ICC 
    # cutoffs
    param_res['icc_group'] = pd.concat(
        axis=0,
        objs=[(param_res.index.to_frame()['icc'] > q) * 1 
              for q in icc_thresh]
        ).groupby(['model', 'icc']).sum()

    return model_res, param_res