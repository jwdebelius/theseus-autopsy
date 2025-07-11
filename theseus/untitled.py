import numpy as np
import pandas as pd

import patsy
import statsmodels.formula.api as smf

from scripts import bw_ana


def sim_uniform(min_, max_, size, cut_lo=0, cut_hi=20):
    """
    Simulates a unifrom distribution with integer values
    """
    dist = np.random.uniform(min_, max_, size=size)
    dist = np.round(dist, 0)
    dist[dist < cut_lo] = cut_lo
    dist[dist > cut_hi] = cut_hi

    return dist.astype(int)

def sim_normal(mean, std, size, cut_lo=18, cut_hi=42):
    """
    Simulates a normal distribution
    """
    dist = np.round(np.random.normal(size=size) * std, 0)
    dist = dist + mean
    dist[dist < cut_lo] = cut_lo
    dist[dist > cut_hi] = cut_hi

    return dist.astype(int)

def simulate_exposure(demo_dmat, cohort_ids, baseline=0.1, demo_mod=None, 
                      cohort_mod=None):
    """
    Simulates the probability a mother is a smoker/indivdual is exposed
    
    Parameters
    ----------
    demo_dmat: pd.DataFrame
        The expanded design matrix with each cateogrical column represneted as 
        a boolean level and each continous column represented.
    cohort: pd.Series
        The cohort assignment corresponding to each row in the design matrix
    baseline: float, (0, 1)
        The baseline probability a mother is a smoker
    demo_mod: dict
        The way in which demographics modify the probabiliy of being a smoker.
        The probability modifier is calculated as (1+sum(demo_mod)) and
        multipled by the baseline p. 
    cohort_mod: dict
        A baseline value for a cohort (i.e. smoking as part of the inclusion
        critiera. This p-value will superceed any modification of hte baseline
        p-value
    """

    # Gets the demographic modified p-value
    if demo_mod is not None:
        demo_mod = pd.Series(demo_mod)
        overlap = [c for c in demo_dmat.columns if c in demo_mod.index]
        overlap = [i for i in demo_mod.index if i in overlap]

        mod = 1 + (demo_dmat[overlap] * demo_mod[overlap]).sum(axis=1)

    else:

        mod = pd.Series(np.ones(len(demo_dmat), ), index=demo_dmat.index)
    smoke_p = mod * baseline

    # Handles the cohort-specific p-values, if they're incldued
    if cohort_mod is not None:
        for cohort, cohort_p in cohort_mod.items():
            smoke_p[cohort_ids == cohort] = cohort_p
    smoke_p[smoke_p < 0] = 0
    smoke_p[smoke_p > 1] = 1

    exposure = pd.Series(np.random.binomial(p=smoke_p.values, n=1),
                         index=demo_dmat.index,
                         name='smoker'
                        )

    return exposure

