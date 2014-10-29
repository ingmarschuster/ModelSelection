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
import synthdata
from plotting import plot_var_bias_mse



def sample_params(num_samples, D, mu_pr, sd_pr, sd_li):
    
    rval = []
    prior = stats.norm(mu_pr, sd_pr)
    theta = prior.rvs((1,))
    for _ in range(num_samples):
        slice_sample_all_components(theta, lambda: stats.norm.logpdf(D, theta, sd_li).sum(), prior)
        rval.append(theta.copy())
    return np.array(rval)

def analytic_logevidence(D, mu_pr, sd_pr, sd_li):
    v_pr = sd_pr**2 #tau^2 in Murphy
    v_li = sd_li**2 #sigma^2 in Murphy
    D_mean = np.mean(D)
    D_mean_sq = D_mean**2
    
    fact = [ (log(sd_li) - ( len(D) *log(sqrt(2*np.pi) *sd_li) #1st factor
                           + log(sqrt(len(D) * v_pr + v_li))   )),
             (- (  np.power(D, 2).sum() / (2 * v_li)   #2nd factor
                 + mu_pr**2             / (2 * v_pr))),
             (( (v_pr*len(D)**2 *D_mean_sq)/v_li   #numerator of 3rd factor
                 + (v_li * D_mean_sq)/v_pr
                 + 2 * len(D) * D_mean * mu_pr
               ) / (2 * (len(D) * v_pr + v_li)) # denominator of 3rd factor
             )            
           ]
    #print(fact)
    return np.sum(fact)

def importance_weights(D, sd_li, prior, proposal_dist, imp_samp):
    w = ((prior.logpdf(imp_samp) # log prior of samples
           + np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                           for mean in imp_samp])) # log likelihood of samples
            - proposal_dist.logpdf(imp_samp) # log pdf of proposal distribution
         )
    w_norm = w - logsumexp(w)
    return (w, w_norm)
         


## Data generation ##

datasets = synthdata.simple_gaussian(dims = 1,
                                     observations_range = range(10,101,10),
                                     num_datasets = 10)

## MODEL Likelihood 

# mu_li ~ N(mu_pr, sd_pr)
sd_li = 1.


## MODEL  prior ##

mu_pr = 0.
sd_pr = 10.
pr = stats.norm(mu_pr, sd_pr)

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 1000
lowdisc_seq = i4_sobol_generate(1, num_imp_samples + 2, 2).flat[:]


est = {}
res = {}


for obs_size in datasets:
    est[obs_size] = {"an":[], "qis":[], "is":[]}
    for ds in datasets[obs_size]:
        D = ds["obs"]
        (mu_D, sd_d) = ds["params"]
        
        

        ## Sample from and fit gaussians to the posteriors ##
        samp_post = sample_params(num_post_samples, D, mu_pr, sd_pr, sd_li)
        param_fit = stats.t.fit(samp_post)
        fit = stats.t(param_fit[0], param_fit[1])
        
        
        ## Analytic evidence
        
        est[obs_size]["an"].append(analytic_logevidence(D, mu_pr, sd_pr, sd_li))
        
        # draw quasi importance samples using the percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf 
        
        (qis_w, qis_w_norm) = importance_weights(D, sd_li, pr, fit,
                                                 fit.ppf(lowdisc_seq))
        est[obs_size]["qis"].append(logsumexp(qis_w)-log(len(qis_w)))
        
        ## An finally standard importance samples        
        (is_w, is_w_norm) = importance_weights(D, sd_li, pr, fit,
                                               fit.rvs(num_imp_samples))
        est[obs_size]["is"].append(logsumexp(is_w)-log(len(is_w)))
        
    for key in est[obs_size]:
        est[obs_size][key] = np.array(est[obs_size][key])
        
    res[obs_size]={}
    
    # now calculate bias, variance and mse of estimators when compared
    # to analytic evidence
    for estim in ("qis", "is"):
        diff = exp(est[obs_size][estim]) - exp(est[obs_size]["an"])
        bias = np.mean(diff)
        mse = np.mean(np.power(diff, 2))
        var = np.var(diff)
        
        res[obs_size][estim] = {"bias^2":bias**2, "var": var, "mse":mse}

res_file_name = "res_"+str(time.clock())
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "est": est}, f)
plot_var_bias_mse(res, "results/"+res_file_name+".pdf")