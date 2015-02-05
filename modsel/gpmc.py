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



from modsel.distributions import categorical, mvnorm
from modsel.synthdata import simple_gaussian
import modsel.mc.mcmc as mcmc
import modsel.mc.pmc as pmc
import modsel.mc.flags as flags

#from pmc_proposals import *

np.random.seed(9)

def mean_of_samples(samples, num_est_samp = None):
    if num_est_samp is None:
        return samples.mean(0)
    else:
        assert(np.min(num_est_samp) >= -np.inf and np.max(num_est_samp) <= 0)
    num_s = len(samples)
    return [samples[:int(exp(N + log(num_s))), :].mean(0) for N in num_est_samp]


num_post_samp = 100

num_obs = 100
num_dims = 2 #30
num_datasets = 50#50
datasets = simple_gaussian(dims = num_dims, observations_range = range(num_obs, num_obs + 1, 10), num_datasets = num_datasets, cov_var_const = 40)
nograd_sqerr = []
grad_sqerr = []
cgrad_sqerr = []
slice_sqerr = []
ll_count = np.atleast_1d(0)

llg_count = np.atleast_1d(0)

grad_llc = (0,0)
cgrad_llc = (0,0)
ng_llc = (0,0)
ss_llc = 0
ds_c = 0
est = {}
num_est_samp = -np.linspace(-100.,0.,15,) # None #-np.logspace(1, np.log10(num_post_samp), 15, base=10).astype(int)

prior = mvnorm(np.array([0]*num_dims), np.eye(num_dims) * 50)

ad = []

for num_obs in datasets:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["pmc","cgpmc","gpmc","slicesamp", "slice_half"]:
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
        
        stop_flag = flags.LikelihoodEvalsFlag()
        

        def llhood_and_grad(theta, grad = True):
            stop_flag.inc_lhood()
            ll_count[:] = ll_count[:] + 1
            d = mvnorm(theta,  obs_K, Ki = obs_Ki, logdet_K = obs_logdet_K, L = obs_L)
            if grad:                
                llg_count[:] = llg_count[:] + 1
                stop_flag.inc_grad()
                (llh, gr) = d.log_pdf_and_grad(obs, grad = grad)
                return (llh.sum(), -gr.sum(0).flatten())
            else:
                return d.log_pdf_and_grad(obs, grad = grad).sum()
        
        def llhood(theta):
            stop_flag.inc_lhood()
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

            
        theta = np.array([0]*num_dims)
        #s_gibbs_slice = []
        #for i in range(int(num_post_samp/3)):
        #    slice_sampling.slice_sample_all_components_mvprior(theta, lambda:llhood(theta), prior)
        #    s_gibbs_slice.append(theta.copy())
        #print("Slice", ll_count, llg_count)
        

        
        (s_gibbs_slice, t_gibbs_slice) = mcmc.sample(num_post_samp, theta, mcmc.ComponentWiseSliceSamplingKernel(lpost), stop_flag = stop_flag)#np.array(s_gibbs_slice)
        est[num_obs]["slicesamp"].append(mean_of_samples(s_gibbs_slice, num_est_samp))
        est[num_obs]["slice_half"].append(mean_of_samples(s_gibbs_slice[len(s_gibbs_slice)//2:], num_est_samp))
        ss_llc += stop_flag.lhood
        stop_flag.max_both_from_current_counts()
        stop_flag.reset()
        
        
        ad.append(pmc.AdGrAsProposal(lpost_and_grad, num_dims))
        (s_cgrad, t_cgrad) = pmc.sample(num_post_samp**2,
                              [prior.rvs() for _ in range(10)],
                              ad[-1],
                              population_size = 4, stop_flag = stop_flag)
        est[num_obs]["cgpmc"].append(mean_of_samples(s_cgrad, num_est_samp))
        cgrad_llc = (cgrad_llc[0] + stop_flag.lhood, cgrad_llc[1]+ int(stop_flag.grad))
        
        stop_flag.reset()
        
        #ga_theta = gradient_ascent(prior.rvs(), lpost_and_grad)
        #print( o_m, ga_theta[0])
        #continue #exit(0)
        (s_nograd, t_nograd) = pmc.sample(num_post_samp**2,
                                [prior.rvs() for _ in range(10)],
                                pmc.NaiveRandomWalkProposal(lpost, mvnorm([0]*num_dims, np.eye(num_dims)*3)),
                                population_size = 4, stop_flag = stop_flag)
    
        est[num_obs]["pmc"].append(mean_of_samples(s_nograd, num_est_samp))
        ng_llc = (ng_llc[0] + int(stop_flag.lhood), ng_llc[1] + int(stop_flag.grad))
        stop_flag.reset()
        
        
        
        (s_grad, t_grad) = pmc.sample(num_post_samp**2,
                              [prior.rvs() for _ in range(10)],
                              pmc.GrAsProposal(lpost_and_grad, num_dims),
                              population_size = 4, stop_flag = stop_flag)
        est[num_obs]["gpmc"].append(mean_of_samples(s_grad, num_est_samp))
        grad_llc = (grad_llc[0] + int(stop_flag.lhood), grad_llc[1]+ int(stop_flag.grad))
        stop_flag.reset()
        


        

        
        nograd_sqerr.append(((o_m - s_nograd.mean(0))**2).mean())
        grad_sqerr.append(((o_m - s_grad.mean(0))**2).mean())
        cgrad_sqerr.append(((o_m - s_cgrad.mean(0))**2).mean())
        slice_sqerr.append(((o_m - s_gibbs_slice[num_post_samp//2:].mean(0))**2).mean())
        print("Nograd mse", np.mean(nograd_sqerr),ng_llc, "cgrad", np.mean(cgrad_sqerr), cgrad_llc, "grad", np.mean(grad_sqerr), grad_llc, "slice", np.mean(slice_sqerr), ss_llc)
print()
print("Nograd mse", np.mean(nograd_sqerr),ng_llc, "\ncgrad mse", np.mean(cgrad_sqerr), cgrad_llc, "\ngrad mse", np.mean(grad_sqerr), grad_llc, "\nslice", np.mean(slice_sqerr), ss_llc)
#print()