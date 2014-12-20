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

from modsel.mc import mcmc

from sobol.sobol_seq import i4_sobol, i4_sobol_generate
import halton
import synthdata
from plotting import plot_var_bias_mse
from evidence import importance_weights, analytic_postparam_logevidence_mvnorm_known_K_li, evidence_from_importance_weights
from modsel.distributions import norm_invwishart, invwishart_logpdf, invwishart_rv, invwishart
from modsel.distributions import mvnorm

import modsel.mixture
#from scipy.stats import norm as mvnorm

import estimator_statistics as eststat

np.random.seed(1)

dims = 2

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 10000

#norm_const = exp(log_norm_const)
log_norm_const = -900


class dummy_prior(object):
    def logpdf(self, x):
        return np.zeros(1)

def mm_dens_(x):
    x = np.atleast_2d(x)
    mvn = stats.multivariate_normal(np.zeros(dims),5*np.eye(dims))
    return log(x[:,0])*2+mvn.logpdf(x)

def mm_dens_(x):
    # Some multimodal density
    x = np.atleast_2d(x)
    uvn = stats.norm(0,1)
    mvn = stats.multivariate_normal(np.zeros(dims),5*np.eye(dims))
    rval = uvn.logcdf(np.sin(x.sum(1)))+ mvn.logpdf(x)
    return log_norm_const+rval

def gauss_dens(x):
    return log_norm_const+stats.multivariate_normal.logpdf(x,
                                                           np.ones(dims)*(-10),
                                                           np.ones((dims, dims))*3)


mm_dens = gauss_dens

def sample_params(num_samples):
    (s, tr) = mcmc.sample(num_samples, -2*np.ones(dims), mcmc.ComponentWiseSliceSamplingKernel(mm_dens))

    return s





## MODEL  prior ##

mu_pr = np.zeros(dims)
nu_pr = 5
K_pr = np.eye(dims) * 100
kappa_pr = 5
pr = mvnorm(mu_pr, K_pr)



lowdisc_seq_sob = i4_sobol_generate(dims+1, num_imp_samples , 2).T


est = {}
num_est_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)

for num_obs in [0]:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["qis(sobol)","is","priorIs"]:
        est[num_obs][estim] = []
    for ds in [0, 1]:
        
        ## Sample from function and fit gaussians to the posteriors ##
        samp = sample_params(num_post_samples)
        
        fit = modsel.mixture.GMM(2, dims, samples = samp)
       # assert()
        est[num_obs]["GroundTruth"].append(log_norm_const)
        
        
        
        # draw quasi importance samples using the pointwise percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf
        logpost_unnorm = mm_dens
        
        qis_samples = fit.ppf(lowdisc_seq_sob).reshape((num_imp_samples, dims))
        (qis_sob_w, qis_sob_w_norm) = importance_weights(mm_dens, fit, qis_samples)
        est[num_obs]["qis(sobol)"].append(evidence_from_importance_weights(qis_sob_w, num_est_samp))
        
        ## draw standard importance samples
        
        
        (is_w, is_w_norm) = importance_weights(mm_dens, fit, fit.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["is"].append(evidence_from_importance_weights(is_w, num_est_samp))
        
        
        ## draw importance samples from the prior
        (prior_is_w, prior_is_w_norm) = importance_weights(mm_dens, pr,
                                               pr.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_est_samp))

    for key in est[num_obs]:
        est[num_obs][key] = np.array(est[num_obs][key])
        
    


res = eststat.logstatistics(est)
        
print(res, est)

res_file_name = ("Multimodal_" + str(dims)+"d_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_est_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_est_samp), "MV-Normal", num_post_samples, num_imp_samples, dims, outfname = "results/"+res_file_name+".pdf")