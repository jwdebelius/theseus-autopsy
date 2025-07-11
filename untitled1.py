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

import scripts.bw_ana as bw_ana


class SimulationSchematic:

    def __init__(self, sim_summary, cohort):
        """
        """

