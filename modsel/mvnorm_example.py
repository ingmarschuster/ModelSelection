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



from sobol.sobol_seq import i4_sobol, i4_sobol_generate
import halton
import synthdata
from plotting import plot_var_bias_mse
from evidence import importance_weights, analytic_postparam_logevidence_mvnorm_known_K_li, evidence_from_importance_weights
from distributions import norm_invwishart, invwishart_logpdf, invwishart_rv, invwishart
from distributions import mvnorm

from mc import mcmc

import estimator_statistics as eststat


def sample_params_known_K_li(num_samples, D, prior, K_li):
    def lpost(x):
        return mvnorm(x, K_li).logpdf(D).sum() + prior.logpdf(x)
        
    (samp, trace) = mcmc.sample(num_samples, prior.rvs(), mcmc.ComponentWiseSliceSamplingKernel(lpost))
    return samp


## Dimension of Gaussian ##
dims = 3
num_obs = 100

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 10000

num_datasets = 20

if False:
    dims = 1
    num_obs=10
    num_post_samples = 100
    num_imp_samples=1000
    num_datasets=2


## Data generation ##
datasets = synthdata.simple_gaussian(dims = dims,
                                     observations_range = range(num_obs, num_obs + 1,10),
                                     num_datasets = num_datasets)

#assert()
## MODEL Likelihood 

# mu_li ~ N(mu_pr, sd_pr)
K_li = np.eye(dims) #+ np.ones((dims,dims))


## MODEL  prior ##

mu_pr = np.zeros(dims)
nu_pr = 5
K_pr = np.eye(dims) * 100
kappa_pr = 5
pr = mvnorm(mu_pr, K_pr)


lowdisc_seq_sob = i4_sobol_generate(dims, num_imp_samples , 2).T


est = {}
num_est_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)

for num_obs in datasets:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["qis(sobol)","is","priorIs"]:
        est[num_obs][estim] = []
    for ds in datasets[num_obs]:
        D = ds["obs"]
        ## Analytic evidence
        ((mu_post, K_post, Ki_post),
         evid) = analytic_postparam_logevidence_mvnorm_known_K_li(D, mu_pr, K_pr, K_li)
        print(evid, mu_post, K_post)
        #continue
        est[num_obs]["GroundTruth"].append(evid)
        
        ## Sample from and fit gaussians to the posteriors ##
        samp = sample_params_known_K_li(num_post_samples, D,
                                           mvnorm(mu_pr, K_pr),
                                           K_li).reshape((num_post_samples, dims))
                        
        param_fit = mvnorm.fit(samp)
        #print(ds["params"], param_fit)
        fit = mvnorm(param_fit[0], param_fit[1])
        

        
        
        
        # draw quasi importance samples using the pointwise percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf
        def logpost_unnorm(posterior_samples):
            return np.array([mvnorm(mean, K_li).logpdf(D).sum()
                                               for mean in posterior_samples]) + pr.logpdf(posterior_samples)
        
        qis_samples = fit.ppf(lowdisc_seq_sob).reshape((num_imp_samples, dims))
        (qis_sob_w, qis_sob_w_norm) = importance_weights(logpost_unnorm, fit,
                                                         qis_samples)
        est[num_obs]["qis(sobol)"].append(evidence_from_importance_weights(qis_sob_w, num_est_samp))
        
        ## draw standard importance samples
        
        
        (is_w, is_w_norm) = importance_weights(logpost_unnorm, fit,
                                               fit.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["is"].append(evidence_from_importance_weights(is_w, num_est_samp))
        
        
        ## draw importance samples from the prior
        (prior_is_w, prior_is_w_norm) = importance_weights(logpost_unnorm, pr,
                                               pr.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_est_samp))

    for key in est[num_obs]:
        est[num_obs][key] = np.array(est[num_obs][key])
        
    


res = eststat.logstatistics(est)
        



res_file_name = ("MV-Normal_" + str(dims)+"d_"
                 + str(num_obs) + "_Observations_"
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_est_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_est_samp), "MV-Normal", num_post_samples, num_imp_samples, dims, outfname = "results/"+res_file_name+".pdf")