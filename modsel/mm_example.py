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

from modsel.mixture import GMM, TMM


from mc import mcmc

import estimator_statistics as eststat



## Dimension of Gaussian ##
dims = 3
num_obs = 100

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 10000

num_datasets = 50

if True:
    dims = 1
    num_obs=10
    num_post_samples = 100
    num_imp_samples=1000
    num_datasets=20


## Data generation ##
datasets = synthdata.simple_gaussian(dims = dims,
                                     observations_range = range(num_obs, num_obs + 1,10),
                                     num_datasets = num_datasets)

#assert()
## MODEL Likelihood 

# mu_li ~ N(mu_pr, sd_pr)
K_li = np.eye(dims) #+ np.ones((dims,dims))
lmodel = mvnorm([0]*dims, K_li)


## MODEL  prior ##

mu_pr = np.zeros(dims)
nu_pr = 5
K_pr = np.eye(dims) * 100
kappa_pr = 5
pr = mvnorm(mu_pr, K_pr)


lowdisc_seq_sob = i4_sobol_generate(dims + 1, num_imp_samples , 2).T


est = {}
num_est_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)

for num_obs in datasets:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["qis(sobol)","is","priorIs"]:
        est[num_obs][estim] = []
    for ds in datasets[num_obs]:
        
        def lpost(x):
            llhood = []
            for mean in x:
                lmodel.set_mu(mean)
                llhood.append(lmodel.logpdf(ds["obs"]).sum())
            llhood = np.array(llhood).flatten()  
            assert(len(llhood) == len(x))
            return  llhood + pr.logpdf(x)
        
        ## Sample from and fit gaussians to the posteriors ##
        samp = (samp, trace) = mcmc.sample(num_post_samples, pr.rvs(), mcmc.ComponentWiseSliceSamplingKernel(lpost))
                        
        #param_fit = mvnorm.fit(samp)
        #fit = mvnorm(param_fit[0], param_fit[1])
        fit = GMM(2, dims, samples = samp)
        
        ## Analytic evidence
        ((mu_post, K_post, Ki_post),
         evid) = analytic_postparam_logevidence_mvnorm_known_K_li(ds["obs"], mu_pr, K_pr, K_li)
        #print("Analytic",mu_post, K_post, "\nFit", param_fit,"\n")
        est[num_obs]["GroundTruth"].append(evid)
        
        
        
        # draw quasi importance samples using the pointwise percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf        
        qis_samples = fit.ppf(lowdisc_seq_sob).reshape((num_imp_samples, dims))
        (qis_sob_w, qis_sob_w_norm) = importance_weights(lpost, fit,
                                                         qis_samples)
        est[num_obs]["qis(sobol)"].append(evidence_from_importance_weights(qis_sob_w, num_est_samp))
        
        ## draw standard importance samples
        
        
        (is_w, is_w_norm) = importance_weights(lpost, fit,
                                               fit.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["is"].append(evidence_from_importance_weights(is_w, num_est_samp))
        
        
        ## draw importance samples from the prior
        (prior_is_w, prior_is_w_norm) = importance_weights(lpost, pr,
                                               pr.rvs(num_imp_samples).reshape((num_imp_samples, dims)))
        est[num_obs]["priorIs"].append(evidence_from_importance_weights(prior_is_w, num_est_samp))

    for key in est[num_obs]:
        est[num_obs][key] = np.array(est[num_obs][key])
        
    


res = eststat.logstatistics(est)
        



res_file_name = ("MM_" + str(dims)+"d_"
                 + str(num_obs) + "_Observations_"
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_est_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_est_samp), "MV-Normal", num_post_samples, num_imp_samples, dims, outfname = "results/"+res_file_name+".pdf")