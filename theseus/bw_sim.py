import numpy as np
import pandas as pd

import patsy
import statsmodels.formula.api as smf

from scripts import bw_ana


def construct_condensed_predictors(cohort_seeds):
    """
    Simulates a cohort based on a side and seed value

    Parameters
    ----------
    seeds: dict
        The seeds dictionary maps the cohort name (key) to the sample size 
        and a dictionary of cohort-specific term probabilities for each of
        the functions used to simulate data

    Returns
    -------
    DataFrame
        A dataframe of indepdent predictor variables for each cohort based on 
        the provided seed values
    """
    cohort_data = []
    for i, (size, seed) in cohort_seeds.items():
        data = pd.DataFrame({col: f_(size=size) for col, f_ in seed.items()})
        data['cohort'] = i
        cohort_data.append(data)

    cohort_data = pd.concat(axis=0, objs=cohort_data).reset_index(drop=True)
    cohort_data['cohort'] = cohort_data['cohort'].astype(str)

    return cohort_data


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


def simulate_cohort_effects(predictors, cohort_ids, iccs):
    """
    Simulates the random cohort effect based on a covariance matrix

    Parameters
    ----------
    predictors: 1D list like
        The list of expanded predictor variables names (column names from a 
        patsy design matrix)
    cohort_ids: 1D list-like
        The names of the cohorts effects are being simulated for
    iccs: float, ndarray
        The interclass correlation target for the simulated data

    Returns
    -------
    DataFrame
        A per-cohort matrix of the random effects per cohort and variable
    """
    # Gets the number of cohorts and number of cohort ids 
    num_cohorts = len(cohort_ids)
    num_var = len(predictors)

    # Generates the random variation for the data. We get a cross
    # variation matrix, and draw the slope from a covariance
    # distribution; although there's only covariation in the intercept
    # (i.e. there's a cohort effect but no slope)
    g = np.zeros((num_var, num_var), dtype=float)
    if isinstance(iccs, float):
        g[0][0] = bw_ana.convert_icc_to_g_var_no_slope(icc)
    else:
        for i, v in enumerate(iccs):
            g[i][i] = v

    # Simulates the variation
    cohort_random = pd.DataFrame(
        data=np.random.multivariate_normal(mean=np.zeros(num_var),
                                             cov=g,
                                            size=num_cohorts),
        index=cohort_ids,
        columns=predictors,
    )
    return cohort_random


def assemble_converged(target, fit):
    """
    A convenience function to avoid horribly branching logic
    """
    if fit.converged:
        icc = bw_ana.icc_from_lme(fit)
        target['iccs'].append(icc)
        if (icc > 1) | np.any(np.trace(fit.cov_params()) < 0):
            target['iccs2'].append(np.nan)
        else:
            target['iccs2'].append(icc)
    else:
        target['iccs'].append(np.nan)
        target['iccs2'].append(np.nan)

    return target



def simulate_log_birthweight_gest_age(ga_days, 
                                      offset:int=None, 
                                      mu_params:dict=None, 
                                      delta_sigma_params:dict=None, 
                                      e_sigma_params:dict=None):
    """
    Simulates the log10(birthweight) based on gestational age

    This function is based on the model (and by default parameters) in
    Nicolaides et al [1]. They built a model where they defined a
    relationship such that their birth weight (BW) wehre y = log10(BW), and
    y = mu + delta + e.
    They fit a data using a markov-Chain Monte Carlo simulations with
    uninformed priors, they found and determined
    * The relationship between the mean, $\mu$ and gestational age is cubic.
    * The delta term is a random effect common to the two methods of 
      fitting weight.
        * It follows a normal distribution with a mean of 0 and standard 
          deviaton sigma_{d}
        * The relationship between \sigma_{d} and gestational age is linear
    * e is a random effect specific to the birth weight measurement
        * It follows a normal distribution with mean 0 and standard
          deviation \sigma_{e}
        * \sigma_{e} does not vary with gestational age.

    Parameters
    ----------
    ga_days: 1D-ndarray
        The gestation age (in days) of the inviduals
    offset: int
        The number of days to use to shift the mean-based model.
    mu_params: dict
        A dictionary of the coeffecients for the mean brithweight by 
        gestational age based on the model fit by Nicolaides et al. Should 
        contain 4 terms corresponding to the cubic model used to fit the mean 
        relationship.
    delta_params: dict
        A dictionary of terms for the variance in the random effect by 
        gestational age. Should have an intercept and slope term.
    e_params: dict
        A dictionary of terms for variance in the second random effect based
        on gestation age from Nicolaides et al. Should contain one term

    Returns
    -------
    1d-ndarray
        The simulated log10 birthweight based on gestational age

    References
    ----------
    [1] Nicolaides, K.H.; Wright, D.; Syngelaki, A.; Wright, A.; and R. 
        Akolekar. (2018). "Fetal Medicine Foundation fetal and neonatal
        population weight charts." Ultrasound in Obstectrics and Gynocology. 
        52:44-51. doi: 10.1002/uog.19073
    """

    # Sets the defaults based on the data fit by Nicolaides et al.
    if offset is None:
        offset = 199
    if mu_params is None:
        mu_params = {0: [3.0893, 0.0004372],
                     1: [0.008350, 0.000006459],
                     2: [-0.00002965, 0.0000001882],
                     3: [-0.00000006062, 0.000000002281],
                     }
    if delta_sigma_params is None:
        delta_sigma_params = {0: [0.02464, 0.000448117],
                              1: [0.00005640, 0.000002172],
                              }
    if e_sigma_params is None:
        e_sigma_params = {0: (0.03363, 0.0002915)}

    # Calculates the days with the adjusted offset.
    days_update = ga_days - offset

    # Gets the number of samples to simulate
    size=len(days_update)

    # Simulates the mean term as a function of gestational age, introducing
    # individual error into each term
    mu = np.vstack([
        np.power(days_update, i) * np.random.normal(*params, size)
        for i, params in mu_params.items()
    ]).sum(axis=0)

    # Simulates the error term for the first (delta) random effect. This is 
    # random for the entire cohort(?)
    sigma_delta = np.vstack([
        np.power(ga_days, i) * np.random.normal(*params, 1)
        for i, params in delta_sigma_params.items()
    ]).sum(axis=0)
    delta = np.random.normal(0, sigma_delta)

    # Simulates the error term, e
    sigma_e = np.vstack([
        np.power(ga_days, i) * np.random.normal(*params, 1)
        for i, params in e_sigma_params.items()
    ]).sum(axis=0)
    e = np.random.normal(0, sigma_e)

    return np.power(10, mu + delta + e)


def simulate_cohort(cohort_seeds: dict,
                    betas: pd.Series,
                    icc: float,
                    smoking_effect: float,
                    indv_var: float=1,
                    slope_var: float=0,
                    g_age_col: str=None,
                    baseline_exposure_p: float=0.1,
                    exposure_demo_mod: dict=None,
                    exposure_cohort_mod: dict=None,
                    cohort_types: dict=None,
                    type_effect: dict=None,
                    type_slope: dict=None,
                    formula: str='1 + C(sex) + C(race) + C(ethnicity) + m_age + year',
                    rescale: bool=False,
                    ):
    """
    Simulates a cohort based on seed parameters

    Parameters
    ----------
    cohort_seeds: dict
        A nested set of dictionaries that map the cohort name (key) to the
        simulated parameters for each cohort. For each cohort, a cohort size
        should be supplied, along with a dictionary that maps the field to a 
        function that generates a random variable of the supplied size. See 
        `sim_uniform` as an example of this type of function
    betas: pd.Series
        The fixed effects for predicting the outcome
    icc: float
        The interclass correlation coeffecient used to seed hte cohort effects
    smoking_effect: float
        The effect of smoking on the outcome.
    indv_var: float, optional
        The variance in the simulation, if there is no column in the simulation
        named for g_age_col
    g_age_col: str, optional
        The column containing gestational age in weeks information. If this 
        column name is provided and present in the data, then it will be given
        priority for variance over the individual var (`indv_var`) covariate.
    baseline_exposure_p: float, optional
        The population-level baseline prevelance for the exposure. If no other
        modifications are added, this is the p-value that will be used for all
        individuals.
    exposure_demo_mod: Series, optional
        The way in which demographics modify the probabiliy of being a smoker.
        The probability modifier is calculated as (1+sum(demo_mod)) and
        multipled by the baseline p. 
    exposure_cohort_mod: dict, optional
        A baseline value for a cohort (i.e. smoking as part of the inclusion
        critiera.) This p-value will superceed any modification of hte baseline
        p-value.
    cohort_effect: dict, optional
        An intercept based on the cohort type
    cohort_slope: dict, optional
        A modifier for the relationship between the exposure and outcome based
        on the cohort type. This should *only* be specific for cohort types 
        where the expsoure should be modified. Any cohort type not specified
        here will use the effect given by the `effect` covariate
    formula: str, optional
        The formula used to expand demogrpahic data; the names in this formula
        should match the ones in the cohort_seeds 
    rescale: bool, optional
        Whether the simulated data should be rescaled to the z-score

    Returns
    -------
    DataFrame
        The simulated cohorts based on the seed. The demographic predictors are
        simulated based on the provided seeds; the outcome is simulated as
        either a population-based level given by the baseline_expousre_p and
        exposure_demo_mod variables, ofr the exposure_cohort_mod; and the
        outcome (`y`) as the demogrpahic effects, exposure effect, and cohort
        effects.

    Also See
    --------
    bw_sim.simulate_log_birthweight_gest_age
    bw_sim.simulate_gestational_age

    """

    # DEMOGRAPHICS
    # Simulates the cohort based on the seed demographic data
    data = construct_condensed_predictors(cohort_seeds)

    # Pulls out the error function, if 
    if ((g_age_col) is not None) and (g_age_col in data.columns):
        g_age_days = data[g_age_col] * 7 + \
            np.random.choice(np.arange(7) - 4)
        y_base = simulate_log_birthweight_gest_age(g_age_days).round(0)
        random_scale = y_base[data[g_age_col] >= 35].std()
    else:
        y_base = pd.Series(np.random.normal(size=len(data)) * indv_var,
                           index=data.index)
        random_scale = indv_var

    if cohort_types is None:
        cohort_types = {c_id: 0 for c_id in data['cohort'].unique()}
    data['cohort_type'] = data['cohort'].replace(cohort_types).astype(int)


    # EXPOSURE
    # Simulates the exposure using a seed 
    demo_dmat = patsy.dmatrix(
        formula,
        data=data
    )
    demo_dmat = pd.DataFrame(data=np.asarray(demo_dmat),
                             index=data.index,
                             columns=demo_dmat.design_info.column_names
                             )
    data['smoking'] = simulate_exposure(demo_dmat=demo_dmat,
                                        cohort_ids=data['cohort'],
                                        baseline=baseline_exposure_p,
                                        demo_mod=exposure_demo_mod,
                                        cohort_mod=exposure_cohort_mod
                                        )
    num_var = len(demo_dmat.columns) + 1

    # INDVIDUAL EFFECTS
    # The sum of the demographic predictors, the effect fromn the smoking, and
    # cohort effects with a variance drawn from a normal distribution.
    fixed = (demo_dmat * betas).sum(axis=1)

    # Expsoure
    if type_slope is None:
        type_slope = dict()
    exposure = (data['smoking'] == 1) * \
        data['cohort_type'].apply(lambda x: type_slope.get(x, smoking_effect))

    # COHORT EFFECTS
    if type_effect is None:
        type_effect = {v: 0 for v in data['cohort_type'].unique()}
    cohort_fixed = data['cohort_type'].replace(type_effect).astype(float)

    # Gets the cohrot variation based on the ICC\
    # The cohort effect is normally distributed with mean 0 and variance g based
    # on the ICC. 
    cohort_ids = data['cohort'].unique()
    cohort_random = simulate_cohort_effects(
        np.hstack([demo_dmat.columns, 'C(smoking)[T.1]']),
        cohort_ids,
        iccs=np.hstack([icc, np.zeros(len(demo_dmat.columns) - 1), slope_var]),
        )
    cohort_random = cohort_random * random_scale
    cohort_intercept = \
        cohort_random.loc[data['cohort'], demo_dmat.columns[0]]
    cohort_intercept = cohort_intercept.reset_index(drop=True)
    cohort_slope = \
        cohort_random.loc[data['cohort'],  'C(smoking)[T.1]'].values * \
        data['smoking']

    # OUTCOME
    sim_vals = pd.DataFrame.from_dict(orient='columns', data=dict(
        fixed=fixed, 
        exposure=exposure,
        cohort_fixed=cohort_fixed,
        cohort_random=cohort_intercept,
        cohort_slope=cohort_slope,
        error=y_base,
    ))
    sim_vals['y'] = sim_vals.sum(axis=1)
    data['y'] = sim_vals['y']

    if rescale:
        general = data.loc[data['cohort_type'] == 0]
        if exposure_cohort_mod is not None:
            exp = list(exposure_cohort_mod.keys())
            general = general.loc[~general['cohort'].isin(exp), 'y']
        else:
            general = general['y']
        data['y'] = (data['y'] - general.mean()) / general.std()
    if ((g_age_col) is not None) and (g_age_col in data.columns):
        data = data.loc[data['y'] >= 500].copy()
        sim_vals = sim_vals.loc[data.index]
    else:
        data = data.loc[data['y'] >= -5].copy()
        sim_vals = sim_vals.loc[data.index]

    return data, sim_vals



def g_age_reference_model_seeds(model='Georgia', n=None, p=None, p_late=None,
                          baseline=40, lims=[None, None]):
    """
    Provides the reference distribution based on the model

    Parameters
    ----------
    model: {'georgia', "american", "combined", "echo_term", "echo_nicu", None}
        A set of preset values for n, p, and p_late based on previously fit
        data.
        "Georgia" draws on data from Vatskjold et al [1], looking at
        pregnancies from Georgia (the country) between X and Y, although the 
        data was modified to remove pregnancies after 44 weeks.
        "America" uses data fit based on Nicolaides et al [2] with no
        modifications.
        "Combined" combines the data from the two populations, although 
        pregancies are not allowed to continue past 44 weeks.
        "echo_term" and "echo_general" were fit from data provided by Kristen
        McArthur for non-NICU cohorts selected for EC0610; "echo_nicu" is
        based on NICU cohorts selected under the same conditions.
    n : float, optional
        The number of failures (weeks), fit from the negative binomial 
        regression of the number of weeks from 40 weeks. 
    p : float, optional
        The probability of a birth outside of the 40 weeks term measurement
    p_late : dict, optional
        The probability of a late term birth for each week outside of term.
        This value should be 0 for term births
    baseline: float, optional
        the number of weeks that are considered "term" - i.e. the baseline
        time 0 for the model.
    lims: list
        The lower and upper limit for gestational ages

    Returns
    -------
    Dict
        A dictionary summmarizing the n, p, p_late, baseline, and lims based
        on the models.

    Also See
    --------
    simulate_gestational_age

    References
    ----------
    [1] Vaktskjold, A.; Talykova, L.V.; Chashchin, V.P.; Odland, J.O.; and E.
        Nieboer. (2007). "Small-for-gestational-age newborns of feamle 
        refinery workers exposed to nickel." Int. J, Occup Med Environ 
        Health. 20:327. doi:10.2478/v10001-007-0034-0.
    [2] Nicolaides, K.H.; Wright, D.; Syngelaki, A.; Wright, A.; and R. 
        Akolekar. (2018). "Fetal Medicine Foundation fetal and neonatal
        population weight charts." Ultrasound in Obstectrics and Gynocology.
        52:44-51. doi: 10.1002/uog.19073

    """
    gestational_age_models = dict(
        georgia=dict(
            n=0.58626,
            p=0.40645,
            baseline=40,
            p_late={1: 0.40366,  2: 0.27518,  3: 0.13175,  4: 0.04091,},
            lims=[0, 12],
            ),
        american=dict(
            n=3.07943,
            p=0.71145,
            baseline=40,
            p_late={1: 0.44765,  2: 0.18848,  3: 0.02127,},
            lims=[0, 16],
            ),
        combined=dict(
            n=2.30674,
            p=0.64875,
            baseline=40,
            p_late={1: 0.44296,  2: 0.20334,  3: 0.03818,  4: 0.00709,},
            lims=[0, 16],
        ),
        echo_general=dict(
            n=2.15515,
            p=0.64875,
            baseline=39,
            p_late={1: 0.59478,  2: 0.54876,  3: 0.19164,  4: 0.12555},
            lims=[0, 14],
            ),
        echo_nicu=dict(
            n=31.02069,
            p=0.94259, 
            baseline=27,
            p_late={1: 0.49565, 2: 0.52261, 3: 0.20155, 4: 0.41818, 
                    5: 0.50000},
            lims=[0, 5],
            ),
        echo_term=dict(
            n=14780,
            p=0.99992,
            baseline=39,
            p_late={1: 0.59478, 2: 0.54876, 3: 0.27226, 4: 0.50000},
            lims=[0, 4],

        ),
    )

    args_model = dict(p=p, n=n, p_late=p_late, 
                      baseline=baseline, lims=lims)
    model_ = gestational_age_models.get(model.lower(), args_model)

    return model_


def simulate_gestational_age(size,  n=None, p=None, p_late=None,
                             baseline=40, lims=[None, None],):
    """
    Simulates the gestational age of an infant, based on parameters

    Here, we model gestational age of infants in a two step process. First, we 
    determine the number of weeks from baseline (i.e. term, or generally 40
    weeks) using a negative binomial model, where "failure" is considered birth
    after n weeks. Then, we determine the probability that a birth is premature
    or late from baseline based on the population-based probability that a birth
    that many weeks out is late.

    Parameters
    ----------
    size: int
        The number of births to simulate
    n : float, optional
        The number of failures (weeks), fit from the negative binomial 
        regression of the number of weeks from 40 weeks. 
    p : float, optional
        The probability of a birth outside of the 40 weeks term measurement
    p_late : dict, optional
        The probability of a late term birth for each week outside of term.
        This value should be 0 for term births
    baseline: float, optional
        the number of weeks that are considered "term" - i.e. the baseline
        time 0 for the model.

    Returns
    -------
    1D-ndarray
        The simulated gestational age for the infant.

    Also See
    --------
    reference_model_seeds
    """
    min_, max_ = lims

    # Checks that the data is defined
    if pd.isnull([n, p, p_late, baseline, min_, max_]).any():
        raise ValueError(
            'Values must be supplied for the p, n, p_late, and baseline.\n'
            'Please check your parameters and try again.') 

    # Determines the number of weeks from 40
    weeks_from_40 = pd.DataFrame(
        np.atleast_2d(np.random.negative_binomial(n=n, p=p, size=size)).T,
        columns=['from_40']
    )
    weeks_from_40.mask(weeks_from_40 <= min_, min_, inplace=True)
    weeks_from_40.mask(weeks_from_40 >= max_, max_, inplace=True)

    # Decides if the infant is late
    late_check = weeks_from_40['from_40'].isin(p_late.keys())
    weeks_from_40['prob_late'] = 0
    weeks_from_40.loc[late_check, 'prob_late'] = \
        weeks_from_40.loc[late_check, 'from_40'].replace(p_late)
    weeks_from_40['is_late'] = \
        np.random.binomial(n=1, p=weeks_from_40['prob_late'])
    # Calculates offset from 40 weeks
    weeks_from_40['offset'] = \
        (-1 + 2 * weeks_from_40['is_late']) * weeks_from_40['from_40']

    # Gets the gestational age
    weeks_from_40['gest_age'] = weeks_from_40['offset'] + baseline

    return weeks_from_40['gest_age'].values