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
from evidence import importance_weights, analytic_postparam_logevidence_mvnorm_known_K_li, evidence_from_importance_weights
from distributions import norm_invwishart, invwishart_logpdf, invwishart_rv, invwishart
from distributions import mvnorm
#from scipy.stats import norm as mvnorm

import estimator_statistics as eststat
from linalg import ensure_2d



def sample_params_known_K_li(num_samples, D, prior, K_li):
    rval = []
    mu = ensure_2d(prior.rvs())
    def llhood():
        return mvnorm(mu, K_li).logpdf(D).sum()
    for i in range(num_samples):
        print("Posterior sample", i)
        slice_sample_all_components(mu, llhood, prior)        
        rval.append(mu.copy())
    return np.array(rval)
    
def sample_params_unknown_K_li(num_samples, D, K_pr, nu_pr, mu_pr, kappa_pr):
    rval_K = []
    rval_mu = []
    K = ensure_2d(invwishart_rv(K_pr, nu_pr))
    mu = ensure_2d(mvnorm(mu_pr, K / kappa_pr).rvs())
    if len(mu.shape) == 0:
        mu = mu.reshape((1, 1))
        K = K.reshape((1, 1))
    def llhood():
        return mvnorm.logpdf(D).sum()
    for i in range(num_samples):        
        slice_sample_all_components(K, llhood, invwishart(K_pr, nu_pr))
        slice_sample_all_components(mu, llhood, mvnorm(mu_pr, K / kappa_pr))
        print("Posterior sample", i)
        rval_K.append(K.copy())
        rval_mu.append(mu.copy())
    return (rval_mu, rval_K)



## Dimension of Gaussian ##
dims = 2
num_obs = 100

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 10000

num_datasets = 50

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
        samp = sample_params_known_K_li(num_post_samples, D,
                                           mvnorm(mu_pr, K_pr),
                                           K_li).reshape((num_post_samples, dims))
                        
        param_fit = mvnorm.fit(samp)
        #print(ds["params"], param_fit)
        fit = mvnorm(param_fit[0], param_fit[1])
        
        ## Analytic evidence
        ((mu_post, K_post, Ki_post),
         evid) = analytic_postparam_logevidence_mvnorm_known_K_li(D, mu_pr, K_pr, K_li)
        #print("Analytic",mu_post, K_post, "\nFit", param_fit,"\n")
        est[obs_size]["an"].append(evid)
        
        
        
        # draw quasi importance samples using the pointwise percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf
        def llhood_func(posterior_samples):
            return np.array([mvnorm(mean, K_li).logpdf(D).sum()
                           for mean in posterior_samples])
        
        (qis_sob_w, qis_sob_w_norm) = importance_weights(D, llhood_func, pr, fit,
                                                         fit.ppf(lowdisc_seq_sob).reshape((num_imp_samples, dims)))
        est[obs_size]["qis(sobol)"].append(evidence_from_importance_weights(qis_sob_w, num_evid_samp))
        ## draw standard importance samples
        (is_w, is_w_norm) = importance_weights(D, llhood_func, pr, fit,
                                               fit.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[obs_size]["is"].append(evidence_from_importance_weights(is_w, num_evid_samp))
        
        
        ## draw importance samples from the prior
        (prior_is_w, prior_is_w_norm) = importance_weights(D, llhood_func, pr, pr,
                                               pr.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[obs_size]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_evid_samp))

    for key in est[obs_size]:
        est[obs_size][key] = np.array(est[obs_size][key])
        
    res[obs_size] = {"bias^2":{}, "var": {}, "mse":{}, "bias^2{ }(relat)":{}, "var{ }(relat)": {}, "mse{ }(relat)":{}}
    
    # now calculate bias, variance and mse of estimators when compared
    # to analytic evidence
    for estim in estimator_names:
        estimate = est[obs_size][estim]
        analytic = est[obs_size]["an"].reshape((len(est[obs_size]["an"]), 1))
        est_rel = estimate - analytic
        
        bias2 = eststat.logbias2exp(analytic, estimate, axis = 0)
        bias2_rel = eststat.logbias2exp(ensure_2d(0), est_rel, axis = 0)
        var = eststat.logvarexp(estimate, axis = 0)
        var_rel = eststat.logvarexp(est_rel, axis = 0)
        mse = eststat.logmseexp(analytic, estimate, axis = 0)
        mse_rel = eststat.logmseexp(ensure_2d(0), est_rel, axis = 0)
        
        res[obs_size]["bias^2"][estim] = bias2.flat[:]
        res[obs_size]["bias^2{ }(relat)"][estim] = bias2_rel.flat[:]
        res[obs_size]["var"][estim] =  var.flat[:]
        res[obs_size]["var{ }(relat)"][estim] =  var_rel.flat[:]
        res[obs_size]["mse"][estim] =  mse.flat[:]
        res[obs_size]["mse{ }(relat)"][estim] =  mse_rel.flat[:]
        #print(eststat.logsubtrexp(eststat.logaddexp(bias2, var)[0], mse)[0],"\n",
        #      eststat.logsubtrexp(logsumexp(np.vstack((bias2, var)), 0), mse)[0])
        decomp_err = eststat.logmeanexp(eststat.logsubtrexp(logsumexp(np.vstack((bias2, var)), 0), mse)[0])[0]
      
        if decomp_err >= -23: # error in original space >= 1e-10 
            print("large mse decomp error, on average", decomp_err)
        



res_file_name = ("MV-Normal_" + str(dims)+"d_"
                 + str(num_obs) + "_Observations_"
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_evid_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_evid_samp), "MV-Normal", num_post_samples, num_imp_samples, dims, outfname = "results/"+res_file_name+".pdf")