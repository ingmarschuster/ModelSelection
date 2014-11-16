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

from slice_sampling import slice_sample_all_components

from sobol.sobol_seq import i4_sobol, i4_sobol_generate
import halton
import synthdata
from plotting import plot_var_bias_mse
from evidence import importance_weights, analytic_logevidence_scalar_gaussian, evidence_from_importance_weights

import estimator_statistics as eststat


def sample_params(num_samples, D, mu_pr, sd_pr, sd_li):    
    rval = []
    prior = stats.norm(mu_pr, sd_pr)
    theta = prior.rvs((1,))
    for i in range(num_samples):
        print("Posterior sample", i)
        slice_sample_all_components(theta, lambda: stats.norm.logpdf(D, theta, sd_li).sum(), prior)
        rval.append(theta.copy())
    return np.array(rval)

"""
def importance_weights(D, sd_li, prior, proposal_dist, imp_samp):
    w = ((prior.logpdf(imp_samp) # log prior of samples
           + np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                           for mean in imp_samp])) # log likelihood of samples
            - proposal_dist.logpdf(imp_samp) # log pdf of proposal distribution
         )
    w_norm = w - logsumexp(w)
    return (w, w_norm)"""




## Number of posterior samples to draw ##
num_post_samples = 500


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 1000

num_datasets = 5

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
num_evid_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)


for obs_size in datasets:
    est[obs_size] = {"an":[]}
    for estim in estimator_names:
        est[obs_size][estim] = []
    for ds in datasets[obs_size]:
        D = ds["obs"]        
        

        ## Sample from and fit gaussians to the posteriors ##
        samp_post = sample_params(num_post_samples, D, mu_pr, sd_pr, sd_li)
        param_fit = stats.t.fit(samp_post)
        fit = stats.t(param_fit[0], param_fit[1])
        
        
        ## Analytic evidence
        
        est[obs_size]["an"].append(analytic_logevidence_scalar_gaussian(D, mu_pr, sd_pr, sd_li))
        
        # draw quasi importance samples using the percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf 
        def llhood_func(imp_samp): 
            return np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                           for mean in imp_samp])
        
        (qis_w, qis_w_norm) = importance_weights(D, llhood_func, pr, fit,
                                                 fit.ppf(lowdisc_seq_sob))
        est[obs_size]["qis(sobol)"].append(evidence_from_importance_weights(qis_w, num_evid_samp))

        if False:
            (hqis_w, hqis_w_norm) = importance_weights(D, llhood_func, pr, fit,
                                                     fit.ppf(lowdisc_seq_halt))
            est[obs_size]["qis(halton)"].append(evidence_from_importance_weights(hqis_w, num_evid_samp))

        ## draw standard importance samples        
        (is_w, is_w_norm) = importance_weights(D, llhood_func, pr, fit,
                                               fit.rvs(num_imp_samples))
        est[obs_size]["is"].append(evidence_from_importance_weights(is_w, num_evid_samp))
        
        ## draw importance samples from the prior       
        (prior_is_w, prior_is_w_norm) = importance_weights(D, llhood_func, pr, pr,
                                               pr.rvs(num_imp_samples))
        est[obs_size]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_evid_samp))
        
    for key in est[obs_size]:
        est[obs_size][key] = np.array(est[obs_size][key])
        
    res[obs_size] = {"bias^2":{}, "var": {}, "mse":{}}
    
    # now calculate bias, variance and mse of estimators when compared
    # to analytic evidence
    for estim in estimator_names:
        estimate = est[obs_size][estim]
        analytic = est[obs_size]["an"].reshape((len(est[obs_size]["an"]), 1))
        
        res[obs_size]["bias^2"][estim] = eststat.logbias2exp(analytic, estimate, axis = 0).flat[:]
        res[obs_size]["mse"][estim] =  eststat.logmseexp(analytic, estimate, axis = 0).flat[:]
        res[obs_size]["var"][estim] =  eststat.logvarexp(estimate, axis = 0).flat[:]


res_file_name = ("Scalar_Normal_" 
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_evid_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_evid_samp), "Scalar Normal", num_post_samples, num_imp_samples, 1, outfname = "results/"+res_file_name+".pdf")