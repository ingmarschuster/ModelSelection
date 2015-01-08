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

from modsel.mixture import NPGMM


from mc import mcmc

import estimator_statistics as eststat


rqmc = True

np.random.seed(1)


## Dimension of Gaussian ##
dims = 2
num_obs = 100



            
    

## Number of posterior samples to draw ##
num_post_samples = 1000


## Number of (Quasi-)Importance samples and precomputed low discrepancy sequence ##
num_imp_samples = 10000

num_datasets = 50

if False:
    dims = 3
    num_obs=10
    num_post_samples = 100
    num_imp_samples=1000
    num_datasets=30


## Data generation ##
print("generating Data")
logpost_ev = synthdata.gen_mm_lpost(num_datasets, dims) #gen_gauss_lpost(num_datasets, dims, cov_var_const=8)
#exit(0)
pr = mvnorm(np.zeros(dims),  np.eye(dims) * 10)

print("Low discr sequence")
#lowdisc_seq_sob = i4_sobol_generate(dims + 1, num_imp_samples , 2).T
%load_ext rmagic
%R require(randtoolbox)

%R -i num_imp_samples,dims -o lowdisc_seq_sob lowdisc_seq_sob <- sobol(num_imp_samples, dims + 1, scrambling = 0)
#%R -i num_imp_samples,dims -o rdzd_lowdisc_seq_sob rdzd_lowdisc_seq_sob <- sobol(num_imp_samples, dims + 1, seed = sample(1:30000, 1, TRUE), scrambling = 1)



est = {}
num_est_samp = np.logspace(1, np.log10(num_imp_samples), 15, base=10).astype(int)

ds = 0
for num_obs in [0]:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["rqis(sobol)", "qis(sobol)","is","priorIs"]:
        est[num_obs][estim] = []
    for (lpost,ev) in logpost_ev:
        ds += 1
        print("Dataset", ds)
        est[num_obs]["GroundTruth"].append(ev)
        
        
        ## Sample from and fit gaussians to the posteriors ##
        samp = (samp, trace) = mcmc.sample(num_post_samples, pr.rvs(), mcmc.ComponentWiseSliceSamplingKernel(lpost))
                        
        #param_fit = mvnorm.fit(samp)
        #fit = mvnorm(param_fit[0], param_fit[1])
        fit = NPGMM(dims, samples = samp)
        
        
        
        # draw quasi importance samples using the pointwise percent point function
        # (PPF, aka quantile function) where cdf^-1 = ppf        
        qis_samples = fit.ppf(lowdisc_seq_sob).reshape((num_imp_samples, dims))
        (qis_sob_w, qis_sob_w_norm) = importance_weights(lpost, fit,
                                                         qis_samples)
        est[num_obs]["qis(sobol)"].append(evidence_from_importance_weights(qis_sob_w, num_est_samp))
        
        # randomized quasi importance samples
        
        #rdzd_lowdisc_seq_sob = np.mod(lowdisc_seq_sob + stats.uniform(0,1).rvs(dims + 1), 1)
        %R -i num_imp_samples,dims -o rdzd_lowdisc_seq_sob rdzd_lowdisc_seq_sob <- sobol(num_imp_samples, dims + 1, seed = sample(1:30000, 1, TRUE), scrambling = 1)
        rqis_samples = fit.ppf(rdzd_lowdisc_seq_sob).reshape((num_imp_samples, dims))
        (rqis_sob_w, rqis_sob_w_norm) = importance_weights(lpost, fit,
                                                           rqis_samples)
        est[num_obs]["rqis(sobol)"].append(evidence_from_importance_weights(rqis_sob_w, num_est_samp))
        
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
        

print(est, "\n\n", res)

res_file_name = ("MM_or_" + str(dims)+"d_"
                 + str(num_obs) + "_Observations_"
                 + str(num_datasets) + "_Datasets_"
                 + str(num_post_samples)  + "_McmcSamp_"
                 + str(num_imp_samples) + "_ImpSamp_" + str(time.clock()))
print(res_file_name)
with open("results/" + res_file_name + ".pickle", "wb") as f:
    pickle.dump({"res":res, "#is-samp": num_est_samp, "est": est}, f)
plot_var_bias_mse(res, log(num_est_samp), "MV-Normal", num_post_samples, num_imp_samples, dims, outfname = "results/"+res_file_name+".pdf")