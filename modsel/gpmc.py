# -*- coding: utf-8 -*-
"""

Test code for gradient-based PMC

Created on Wed Nov 26 14:21:15 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp



from distributions import categorical
from synthdata import simple_gaussian
import mcmc, pmc

from pmc_proposals import *



def mean_of_samples(samples, num_est_samp = None):
    if num_est_samp is None:
        return samples.mean(0)
    return [samples[:N, :].mean(0) for N in num_est_samp]


num_post_samp = 100

num_obs = 100
num_dims = 2 #30
num_datasets = 3#50
datasets = simple_gaussian(dims = num_dims, observations_range = range(num_obs, num_obs + 1, 10), num_datasets = num_datasets, cov_var_const = 40)
nograd_sqerr = []
grad_sqerr = []
slice_sqerr = []
ll_count = np.atleast_1d(0)

llg_count = np.atleast_1d(0)

grad_llc = (0,0)
ng_llc = (0,0)
ss_llc = 0
ds_c = 0
est = {}
num_est_samp = np.logspace(1, np.log10(num_post_samp), 15, base=10).astype(int)

prior = mvnorm(np.array([0]*num_dims), np.eye(num_dims) * 50)

for num_obs in datasets:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["pmc","gpmc","slicesamp"]:
        est[num_obs][estim] = []
    for Data in datasets[num_obs]:        
        ds_c += 1
        obs = Data["obs"]
        
        truth = Data["params"][0]
        
        print("Dataset",ds_c)        
        
        est[num_obs]["GroundTruth"].append(truth)
        o_m = obs.mean(0)
        obs_K =  Data["params"][1]
        obs_Ki = np.linalg.inv(obs_K)
        obs_L = np.linalg.cholesky(obs_K)
        obs_logdet_K = np.linalg.slogdet(obs_K)[1]
        

        def llhood_and_grad(theta, grad = True):
            ll_count[:] = ll_count[:] + 1
            d = mvnorm(theta,  obs_K, Ki = obs_Ki, logdet_K = obs_logdet_K, L = obs_L)
            if grad:                
                llg_count[:] = llg_count[:] + 1
                (llh, gr) = d.log_pdf_and_grad(obs, grad = grad)
                return (llh.sum(), -gr.sum(0).flatten())
            else:
                return d.log_pdf_and_grad(obs, grad = grad).sum()
        
        def llhood(theta):
            return llhood_and_grad(theta, False)
        
        def lpost_and_grad(theta, grad = True):
            if not grad:
                return prior.logpdf(theta) + llhood(theta)
            else:
                (lpr, pr_grad) = prior.log_pdf_and_grad(theta)
                (llh, lh_grad) = llhood_and_grad(theta)
                return (lpr + llh, pr_grad + lh_grad)
        
        def lpost(theta):
             return lpost_and_grad(theta, False)
        
        if False:
            obs_m = obs.mean(0)+10
            (lp, lg) = mvnorm(obs_m, obs_K).log_pdf_and_grad(obs)
            lp = lp.sum()
            lg = -lg.sum(0)
            print("llhood", llhood(obs_m) - lp, "grad", llhood_grad(obs_m) - lg )
            exit(0)
            #print("gradient check", optimize.check_grad(llhood, llhood_grad, np.array([0]*num_dims)), llhood_grad(obs.mean(0)))
            #assert()
            
        theta = np.array([0]*num_dims)
        #s_gibbs_slice = []
        #for i in range(int(num_post_samp/3)):
        #    slice_sampling.slice_sample_all_components_mvprior(theta, lambda:llhood(theta), prior)
        #    s_gibbs_slice.append(theta.copy())
        #print("Slice", ll_count, llg_count)
        s_gibbs_slice = mcmc.sample(num_post_samp, theta, mcmc.ComponentWiseSliceSamplingKernel(lpost))#np.array(s_gibbs_slice)
        est[num_obs]["slicesamp"].append(mean_of_samples(s_gibbs_slice, num_est_samp))
        ss_llc += int(ll_count)
        ll_count[:] = 0
        llg_count[:] = 0
        
        #ga_theta = gradient_ascent(prior.rvs(), lpost_and_grad)
        #print( o_m, ga_theta[0])
        #continue #exit(0)
        s_nograd = pmc.sample(num_post_samp*3,
                                [prior.rvs() for _ in range(10)],
                                pmc.NaiveRandomWalkProposal(lpost, mvnorm([0]*num_dims, np.eye(num_dims)*3)),
                                population_size = 4)
    
        est[num_obs]["pmc"].append(mean_of_samples(s_nograd, num_est_samp))
        ng_llc = (ng_llc[0] + int(ll_count), ng_llc[1] + int(llg_count))
        ll_count[:] = 0
        llg_count[:] = 0
        
        s_grad = pmc.sample(num_post_samp,
                              [prior.rvs() for _ in range(10)],
                              pmc.GradientAscentProposal(lpost_and_grad, num_dims),
                              population_size = 4)
        est[num_obs]["gpmc"].append(mean_of_samples(s_grad, num_est_samp))
        grad_llc = (grad_llc[0] + int(ll_count), grad_llc[1]+ int(llg_count))
        ll_count[:] = 0
        llg_count[:] = 0
        
        

        

        
        nograd_sqerr.append(((o_m - s_nograd.mean(0))**2).mean())
        grad_sqerr.append(((o_m - s_grad.mean(0))**2).mean())
        slice_sqerr.append(((o_m - s_gibbs_slice.mean(0))**2).mean())
        print("Nograd mse", np.mean(nograd_sqerr),ng_llc, "grad", np.mean(grad_sqerr), grad_llc, "slice", np.mean(slice_sqerr), ss_llc)
print()
print("Nograd mse", np.mean(nograd_sqerr),ng_llc, "\ngrad mse", np.mean(grad_sqerr), grad_llc, "\nslice", np.mean(slice_sqerr), ss_llc)
#print()