# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:28:21 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from numpy.linalg import inv
from scipy.misc import logsumexp

import numpy as np
import scipy.stats as stats
import cPickle as pickle

import time

from slice_sampling import slice_sample_all_components

from sobol.sobol_seq import i4_sobol, i4_sobol_generate
import halton
import synthdata
from plotting import plot_var_bias_mse
from evidence import analytic_postparam_logevidence_mvnorm, evidence_from_importance_weights
from invwishart import norm_invwishart, invwishart_logpdf, invwishart_rv, invwishart


def multivariate_normal_fit(samples):
    mu = samples.mean(0)
    diff = samples - mu
    return (mu, diff.T.dot(diff))

def sample_params(num_samples, D, K_pr, nu_pr, mu_pr, kappa_pr):
    rval_K = []
    rval_mu = []
    K = invwishart_rv(K_pr, nu_pr)
    mu = stats.multivariate_normal(mu_pr, K / kappa_pr).rvs()
    def llhood():
        return stats.multivariate_normal.logpdf(D, mu, K).sum()
    for i in range(num_samples):        
        slice_sample_all_components(K, llhood, invwishart(K_pr, nu_pr))
        slice_sample_all_components(mu, llhood, stats.multivariate_normal(mu_pr, K / kappa_pr))
        print("Posterior sample", i)
        rval_K.append(K.copy())
        rval_mu.append(mu.copy())
    return (rval_mu, rval_K)

## Data generation ##
dims = 2

datasets = synthdata.simple_gaussian(dims = dims,
                                     observations_range = range(1000,1001,10),
                                     num_datasets = 1)

## MODEL Likelihood 

# mu_li ~ N(mu_pr, sd_pr)
sd_li = 1.


## MODEL  prior ##

mu_pr = np.zeros(dims)
nu_pr = 5
K_pr = np.eye(dims) * 10
kappa_pr = 5
pr = stats.multivariate_normal(mu_pr, K_pr)

## Number of posterior samples to draw ##
num_post_samples = 100


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 1000
lowdisc_seq = i4_sobol_generate(1, num_imp_samples + 2, 2).flat[:]
lowdisc_seq_halt = halton.sequence(num_imp_samples, 3).flat[:]

estimator_names = ["qis(sobol)","is","priorIs"] #,"qis(halton)"
est = {}
res = {}
num_evid_samp = np.logspace(1,3,15, base=10).astype(int)

for obs_size in datasets:
    est[obs_size] = {"an":[]}
    for estim in estimator_names:
        est[obs_size][estim] = []
    for ds in datasets[obs_size]:
        D = ds["obs"]
        
        ## Sample from and fit gaussians to the posteriors ##
        (samp_mu, samp_K) = sample_params(num_post_samples, D,
                                          K_pr, nu_pr,
                                          mu_pr, kappa_pr)
        samp_concat = np.array([np.hstack((samp_mu[i].flat, samp_K[i].flat))
                                   for i in range(len(samp_mu))])
        param_fit = multivariate_normal_fit(samp_concat)
        fit = stats.multivariate_normal(param_fit[0], param_fit[1])
        
        ## Analytic evidence
        ((mu_post, prec_post, kappa_post, nu_post),
          evid) = analytic_postparam_logevidence_mvnorm(D,
                                                        mu_pr,
                                                        inv(K_pr),
                                                        kappa_pr,
                                                        nu_pr)
        est[obs_size]["an"].append(evid)
        samp_m = samp_concat.mean(0).flat
        m_mu = samp_m[:dims]
        m_K = samp_m[dims:].reshape((dims, dims))
        
        print("================\n",
             "Analytic",
              mu_post, inv(prec_post),
              "\nfitted", m_mu, m_K)
        