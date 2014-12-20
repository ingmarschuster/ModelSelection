# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:14:34 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats
import cPickle as pickle

import time


from sobol.sobol_seq import i4_sobol, i4_sobol_generate
import halton
import synthdata
from plotting import plot_var_bias_mse
from evidence import importance_weights, analytic_logevidence_scalar_gaussian, evidence_from_importance_weights

import estimator_statistics as eststat

from mc import mcmc


def sample_params(num_samples, D, mu_pr, sd_pr, sd_li): 
    prior = stats.norm(mu_pr, sd_pr)
    
    def lpost(x):
        return stats.norm.logpdf(D, x, sd_li).sum() + prior.logpdf(x)
        
    (samp, trace) = mcmc.sample(num_samples, prior.rvs((1,)), mcmc.ComponentWiseSliceSamplingKernel(lpost))

    return samp



## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 1000

num_datasets = 50

## Data generation ##

datasets = synthdata.simple_gaussian(dims = 1,
                                     observations_range = range(10,11,10),
                                     num_datasets = num_datasets)

## MODEL Likelihood 

# mu_li ~ N(mu_pr, sd_pr)
sd_li = 1.


## MODEL  prior ##

mu_pr = 0.
sd_pr = 10.
pr = stats.norm(mu_pr, sd_pr)

lowdisc_seq_sob = i4_sobol_generate(1, num_imp_samples + 2, 2).flat[:]
lowdisc_seq_halt = halton.sequence(num_imp_samples, 3).flat[:]

estimator_names = ["qis(sobol)","is","priorIs"] #,"qis(halton)"
est = {}
res = {}
num_est_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)


for num_obs in datasets:
    est[num_obs] = {"an":[]}
    for estim in estimator_names:
        est[num_obs][estim] = []
    for ds in datasets[num_obs]:
        D = ds["obs"]        
        

        ## Sample from and fit gaussians to the posteriors ##
        samp_post = sample_params(num_post_samples, D, mu_pr, sd_pr, sd_li)
        param_fit = stats.t.fit(samp_post)
        fit = stats.t(param_fit[0], param_fit[1])
        
        
        ## Analytic evidence
        
        est[num_obs]["an"].append(analytic_logevidence_scalar_gaussian(D, mu_pr, sd_pr, sd_li))
        
        # draw quasi importance samples using the percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf 
        def logpost_unnorm(imp_samp): 
            return np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                           for mean in imp_samp]) + pr.logpdf(imp_samp)
        
        (qis_w, qis_w_norm) = importance_weights(logpost_unnorm, fit,
                                                 fit.ppf(lowdisc_seq_sob))
        est[num_obs]["qis(sobol)"].append(evidence_from_importance_weights(qis_w, num_est_samp))

        if False:
            (hqis_w, hqis_w_norm) = importance_weights(logpost_unnorm, fit,
                                                     fit.ppf(lowdisc_seq_halt))
            est[num_obs]["qis(halton)"].append(evidence_from_importance_weights(hqis_w, num_est_samp))

        ## draw standard importance samples        
        (is_w, is_w_norm) = importance_weights(logpost_unnorm, fit,
                                               fit.rvs(num_imp_samples))
        est[num_obs]["is"].append(evidence_from_importance_weights(is_w, num_est_samp))
        
        ## draw importance samples from the prior       
        (prior_is_w, prior_is_w_norm) = importance_weights(logpost_unnorm, pr,
                                               pr.rvs(num_imp_samples))
        est[num_obs]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_est_samp))
        
    for key in est[num_obs]:
        est[num_obs][key] = np.array(est[num_obs][key])
        
    res[num_obs] = {"bias^2":{}, "var": {}, "mse":{}}
    
    # now calculate bias, variance and mse of estimators when compared
    # to analytic evidence
    for estim in estimator_names:
        estimate = est[num_obs][estim]
        analytic = est[num_obs]["an"].reshape((len(est[num_obs]["an"]), 1))
        
        res[num_obs]["bias^2"][estim] = eststat.logbias2exp(analytic, estimate, axis = 0).flat[:]
        res[num_obs]["mse"][estim] =  eststat.logmseexp(analytic, estimate, axis = 0).flat[:]
        res[num_obs]["var"][estim] =  eststat.logvarexp(estimate, axis = 0).flat[:]


res_file_name = ("Scalar_Normal_" 
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_est_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_est_samp), "Scalar Normal", num_post_samples, num_imp_samples, 1, outfname = "results/"+res_file_name+".pdf")