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
from distributions import invwishart_rv

def simple_gaussian(dims = 1, observations_range = range(10,101,10), num_datasets = 10, cov_var_const = 4):
    ds = {}
    for nobs in observations_range:
        ds[nobs] = []
        for n in range(num_datasets):
            m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*15)
            K = invwishart_rv(np.eye(dims) * cov_var_const, dims + 2)
            ds[nobs].append({"params":(m, K),
                            "obs":stats.multivariate_normal.rvs(m, K, size=nobs).reshape((nobs, dims))})
    return ds
            