# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:24:23 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats
from invwishart import invwishartrand

def simple_gaussian(dims = 1, observations_range = range(10,101,10), num_datasets = 10, cov_var_const = 4):
    ds = {}
    for obs in observations_range:
        ds[obs] = []
        for n in range(num_datasets):
            m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*15)
            K = invwishartrand(dims + 2, np.eye(dims) * cov_var_const)
            ds[obs].append({"params":(m, K),
                            "obs":stats.multivariate_normal.rvs(m, K, size=obs)})
    return ds
            