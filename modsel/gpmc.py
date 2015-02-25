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



from distributions import categorical, mvnorm
from modsel.synthdata import gen_gauss_lpost
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

num_obs = 1
num_dims = 5 #30
num_datasets = 50#50
np.random.seed(1)
posteriors = gen_gauss_lpost(num_datasets, num_dims, cov_var_const = 4, with_grad=True)
#simple_gaussian(dims = num_dims, observations_range = range(num_obs, num_obs + 1, 10), num_datasets = num_datasets, cov_var_const = 4)
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
num_est_samp = np.linspace(-100.,0.,15,) # None #-np.logspace(1, np.log10(num_post_samp), 15, base=10).astype(int)

prior = mvnorm(np.array([0]*num_dims), np.eye(num_dims) * 50)

ad = []



for num_obs in [1]:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["pmc","cgpmc","gpmc","slicesamp", "slice_half"]:
        est[num_obs][estim] = []
    for lpost in posteriors:
        naive_proposals = pmc.CategoricalOracle(pmc.GaussRwProposal(None,np.eye(num_dims)*15))

        grad_proposals = pmc.CategoricalOracle(pmc.GrAsStupidProposal(None,10, 15))

        #naive_proposals = pmc.CategoricalOracle(pmc.NaiveRandomWalkProposal(None, mvnorm([0]*num_dims, np.eye(num_dims))))

        #grad_proposals = pmc.CategoricalOracle(pmc.GrAsProposal(None, num_dims, prop_mean_on_line = 0.5, main_var_scale = 1, other_var = 0.5))
        ds_c += 1
        
        print("Dataset",ds_c)        
        
        stop_flag = flags.LikelihoodEvalsFlag()
        
        est[num_obs]["GroundTruth"].append(lpost.mean)
        

        
        def l_pdf_and_grad(x, include_grad=True):
            stop_flag.inc_lhood()
            if include_grad:
                stop_flag.inc_grad()
            return lpost.lpdf_and_grad(x, pdf=True, grad=include_grad)
            
        def lp(x):            
            return l_pdf_and_grad(x, False)

        naive_proposals.set_lpost_and_grad(l_pdf_and_grad)
        grad_proposals.set_lpost_and_grad(l_pdf_and_grad)
        
        theta = np.array([0]*num_dims)
        #s_gibbs_slice = []
        #for i in range(int(num_post_samp/3)):
        #    slice_sampling.slice_sample_all_components_mvprior(theta, lambda:llhood(theta), prior)
        #    s_gibbs_slice.append(theta.copy())
        #print("Slice", ll_count, llg_count)
        

        
        (s_gibbs_slice, t_gibbs_slice) = mcmc.sample(num_post_samp, theta, mcmc.ComponentWiseSliceSamplingKernel(lp), stop_flag = stop_flag)#np.array(s_gibbs_slice)
        est[num_obs]["slicesamp"].append(mean_of_samples(s_gibbs_slice, num_est_samp))
        est[num_obs]["slice_half"].append(mean_of_samples(s_gibbs_slice[len(s_gibbs_slice)//2:], num_est_samp))
        ss_llc += stop_flag.lhood
        stop_flag.max_both_from_current_counts()
        stop_flag.reset()
        
        

        
        #ga_theta = gradient_ascent(prior.rvs(), lpost_and_grad)
        #print( o_m, ga_theta[0])
        #continue #exit(0)
        (s_nograd, t_nograd) = pmc.sample(num_post_samp**2,
                                [prior.rvs() for _ in range(10)],
                                naive_proposals,
                                population_size = 4, stop_flag = stop_flag)
    
        est[num_obs]["pmc"].append(mean_of_samples(s_nograd, num_est_samp))
        ng_llc = (ng_llc[0] + int(stop_flag.lhood), ng_llc[1] + int(stop_flag.grad))
        stop_flag.reset()
        
        
        
        (s_grad, t_grad) = pmc.sample(num_post_samp**2,
                              [prior.rvs() for _ in range(10)],
                              grad_proposals,
                              population_size = 4, stop_flag = stop_flag)
        est[num_obs]["gpmc"].append(mean_of_samples(s_grad, num_est_samp))
        grad_llc = (grad_llc[0] + int(stop_flag.lhood), grad_llc[1]+ int(stop_flag.grad))
        stop_flag.reset()
        


        

        
        nograd_sqerr.append(((lpost.mean - s_nograd.mean(0))**2).mean())
        grad_sqerr.append(((lpost.mean - s_grad.mean(0))**2).mean())
        #cgrad_sqerr.append(((o_m - s_cgrad.mean(0))**2).mean()) #"cgrad", np.mean(cgrad_sqerr), cgrad_llc,
        slice_sqerr.append(((lpost.mean - s_gibbs_slice[num_post_samp//2:].mean(0))**2).mean())
        print("Nograd mse", np.mean(nograd_sqerr),ng_llc,  "grad", np.mean(grad_sqerr), grad_llc, "slice", np.mean(slice_sqerr), ss_llc)
        #print("sum Ng mse", np.mean(nograd_sqerr),ng_llc,  cgrad_llc, "grad", np.mean(grad_sqerr), grad_llc, "slice", np.mean(slice_sqerr), ss_llc)#"cgrad mse", np.mean(cgrad_sqerr),
print()
print("Nograd mse", np.mean(nograd_sqerr),ng_llc,  cgrad_llc, "\ngrad mse", np.mean(grad_sqerr), grad_llc, "\nslice", np.mean(slice_sqerr), ss_llc)#"\ncgrad mse", np.mean(cgrad_sqerr),
#print()