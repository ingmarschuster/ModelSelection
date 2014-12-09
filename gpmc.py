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

from copy import copy, deepcopy


from distributions import mvnorm, mvt, categorical
from distributions.linalg import pdinv
from synthdata import simple_gaussian
import slice_sampling
from gs_basis import ideal_covar

from scipy import optimize


def mean_of_samples(samples, num_est_samp = None):
    if num_est_samp is None:
        return samples.mean(0)
    return [samples[:N, :].mean(0) for N in num_est_samp]


def plot_current_prop(current, proposal_mu, proposal_dist, fig_name="Gradient_PMC.pdf"):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    f = plt.figure()
    a = f.gca()
    a.set_aspect("equal")
    
    p = proposal_dist.rvs(500)# + current
    
    
    
    a.scatter(p[:,0],p[:,1], s=  np.ones(p[:,0].shape)*2,  c = "blue", linewidths = 0, alpha = 0.5, label = "samples from proposal distr")
    a.scatter(current[0], current[1], s=  np.ones(p[:,0].shape) * 50, c = "red", marker = "x", label = "theta_old")
    
    #a.arrow(current[0], current[1], (gradient)[0], (gradient)[1], color="black", hatch="*", head_width=0.3, head_length=0.3, label = "gradient")
    a.scatter(proposal_mu[0], proposal_mu[1], s =  np.ones(p[:,0].shape) * 50, c = "black", marker = "x", label = "f * gradient(theta_old) + theta_old")
    a.legend()
    f.tight_layout()
    
    #f.show()
    f.savefig(fig_name)

def best_step_size(theta, gradient, llhood_and_grad_func):
    step_sizes = 10**(-np.linspace(1,10, num=4)) #[10**-i for i in range(1,10)]
    best = np.argmax( [llhood_and_grad_func(theta + f * gradient)[0] for f in step_sizes])
    return step_sizes[best]


def construct_covar_in_direction(direction, scale=False, mean="min", shrink_fact = 1):
    direction = direction.flat[:]
    var = np.linalg.norm(direction)
    vs = np.vstack([direction.flat[:], np.eye(len(direction))[:-1,:]])
    bas = gs(vs)
    ew = np.eye(len(direction))*0.5
    ew[0,0] = var * 0.5
    return bas.T.dot(ew).dot(np.linalg.inv(bas.T))

def find_step_size(theta, f, lpost, grad, lpost_and_grad_func):
    lpost_1 = -np.inf
    while lpost_1 < lpost:
        theta_1 = theta + f * grad
        (lpost_1, grad_1) = lpost_and_grad(theta_1)
        if lpost > lpost_1:
            f = f * 0.5
        else:
            f = f* 1.05
            break
    return (f, theta_1, lpost_1, grad_1)

def gradient_ascent(theta, lpost_and_grad, momentum = 0):
    (lpost, grad) = lpost_and_grad(theta)
    i = 0
    f = 0.1
    stayed = 0
    moved = 0
    while True:
        (f, theta_1, lpost_1, grad_1) = find_step_size(theta, f, lpost, grad, lpost_and_grad)
        (lpost, grad, theta) = (lpost_1, grad_1, theta_1) 
        if np.abs(grad.mean()) < 0.1:
            print("small gradient", np.abs(grad.mean()))
            break
        assert(not( np.any(np.isnan(theta)) or np.any(np.isnan(grad))))
            
        i += 1
    print(moved, stayed)
    return (theta, lpost)
            
        
        
        

def pmc_sampling(num_samples, lpost_and_grad, initial_particles, momentum = 0, population_size = 20, grad_proposal = True):
    rval = list(initial_particles)
    dim = rval[0].size
    num_initial = len(rval)
    rval = [prior.rvs().flatten() for _ in range(num_initial)]
    (lpost, post_gr) = [list(i) for i in 
                         zip(*[lpost_and_grad(j) for j in rval])]
    rval_step = [0.1] * num_initial
    i = -5
    while len(rval) - num_initial < num_samples:
        pop = []
        prop_lpdf = []
        prop_lpost = []
        prop_grad= []
        prop_logw = []
        
        prop_f = []
        prev_prob = np.array(lpost)
        prev_prob = prev_prob - logsumexp(prev_prob)
        
        
        while len(pop) < population_size: #idx in part_idx:
            idx = categorical(prev_prob, p_in_logspace = True).rvs()
            
            
            if not grad_proposal:
                dim = np.prod(rval[idx].shape)
                prop_dist = stats.multivariate_normal(rval[idx].flatten(), np.eye(dim))
            else:
                if np.abs(post_gr[idx].mean()) < 0.1:
                    #we are close to a local maximum
                    if stats.bernoulli.rvs(0.1):
                        # take a random step with large variance
                        f = 0.1
                        prop_dist = mvt(rval[idx], np.eye(dim)*10, dim)
                    else:
                        # take a random step with small variance
                        #(stay in region of high posterior probability)
                        f = rval_step[idx]
                        prop_dist = mvt(rval[idx], np.eye(dim)*0.5, dim)
                else:
                    #we are at a distance to a local maximum
                    #step in direction of gradient.
                    (f, theta_1, lpost_1, grad_1)  = find_step_size(rval[idx], rval_step[idx], lpost[idx], post_gr[idx], lpost_and_grad)
                    cov = ideal_covar(f * 0.5 * post_gr[idx], main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
                    prop_dist = mvnorm(rval[idx] + f * 0.5 * post_gr[idx], cov)
                if  i > 0 and i < 5:
                    i += 1
                    #print(cov)
                    #plot_current_prop(rval[idx], mu, prop_dist, "Gradient_PMC_"+str(len(rval))+"_"+str(len(pop))+".pdf")
           
            samp = prop_dist.rvs()
            pop.append(samp)
            (samp_lpost, samp_grad) = lpost_and_grad(samp)
            prop_lpost.append(samp_lpost)
            prop_grad.append(samp_grad)
            prop_lpdf.append(prop_dist.logpdf(samp))
            
            prop_logw.append(prop_lpost[-1] - prop_lpdf[-1])
            if grad_proposal:
                prop_f.append(f)

                    
        prop_w = exp(np.array(prop_logw) - logsumexp(prop_logw))
        while True:
            try:
                draws = np.random.multinomial(population_size, prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
                
        for idx in range(len(draws)):
            (s, lp, p_gr)    = (np.copy(pop[idx]),
                                np.copy(prop_lpost[idx]), 
                                np.copy(prop_grad[idx]))
            for _ in range(draws[idx]):
                rval.append(s)
                lpost.append(lp)
                
                post_gr.append(p_gr)
                if grad_proposal:
                    rval_step.append(prop_f[idx])
        #f = np.mean(f_samp[-20:])
        #print(np.floor(100 * len(rval) / num_samples), "% of samples")
        i += 1
    rval = np.array(rval)
    #print(rval,"\n", f_samp, np.mean(f_samp), f_samp[0])
    return np.array(rval[num_initial:])


num_post_samp = 1000

num_obs = 100
num_dims = 2 #30
num_datasets = 5#50
datasets = simple_gaussian(dims = num_dims, observations_range = range(num_obs, num_obs + 1, 10), num_datasets = num_datasets, cov_var_const = 4)
nograd_sqerr = []
grad_sqerr = []
slice_sqerr = []
ll_count = np.atleast_1d(0)

llg_count = np.atleast_1d(0)

grad_llc = 0
ng_llc = 0
ss_llc = 0
ds_c = 0
est = {}
num_est_samp = np.logspace(1, np.log10(num_post_samp), 15, base=10).astype(int)

print("Starting simulation")
for num_obs in datasets:
    est[num_obs] = {"GroundTruth":[]}
    for estim in ["pmc","gpmc","slicesamp"]:
        est[num_obs][estim] = []
    for Data in datasets[num_obs]:        
        ds_c += 1
        obs = Data["obs"]
        
        truth = Data["params"][0]
        
        print("Dataset",ds_c, "Truth:", truth)        
        
        est[num_obs]["GroundTruth"].append(truth)
        o_m = obs.mean(0)
        obs_K =  Data["params"][1]
        obs_Ki = np.linalg.inv(obs_K)
        obs_L = np.linalg.cholesky(obs_K)
        obs_logdet_K = np.linalg.slogdet(obs_K)[1]
        
        
        prior = mvnorm(np.array([0]*num_dims), np.eye(num_dims) * 50)
        
        
        def llhood(theta):
            ll_count[:] = ll_count[:] + 1
            d = mvnorm(theta,  obs_K, Ki = obs_Ki, logdet_K = obs_logdet_K, L = obs_L)
            return d.logpdf(obs).sum()
        
        def llhood_and_grad(theta):
            llg_count[:] = llg_count[:] + 1
            d = mvnorm(theta,  obs_K, Ki = obs_Ki, logdet_K = obs_logdet_K, L = obs_L)
            (llh, gr) = d.log_pdf_and_grad(obs)
            return (llh.sum(), -gr.sum(0).flatten())
        
        def lpost_and_grad(theta):
            (lpr, pr_grad) = prior.log_pdf_and_grad(theta)
            (llh, lh_grad) = llhood_and_grad(theta)
            return (lpr + llh, pr_grad + lh_grad)
        
        if False:
            obs_m = obs.mean(0)+10
            (lp, lg) = mvnorm(obs_m, obs_K).log_pdf_and_grad(obs)
            lp = lp.sum()
            lg = -lg.sum(0)
            print("llhood", llhood(obs_m) - lp, "grad", llhood_grad(obs_m)- lg )
            exit(0)
            #print("gradient check", optimize.check_grad(llhood, llhood_grad, np.array([0]*num_dims)), llhood_grad(obs.mean(0)))
            #assert()
            
        #ga_theta = gradient_ascent(prior.rvs(), lpost_and_grad)
        #print( o_m, ga_theta[0])
        #continue #exit(0)
        s_nograd = pmc_sampling(num_post_samp, lpost_and_grad, [prior.rvs() for _ in range(10)], population_size = 4, grad_proposal = False)
    
        est[num_obs]["pmc"].append(mean_of_samples(s_nograd, num_est_samp))
        ng_llc = int(llg_count)
        ll_count[:] = 0
        llg_count[:] = 0
        
        s_grad = pmc_sampling(num_post_samp, lpost_and_grad, [prior.rvs() for _ in range(10)], population_size = 4, grad_proposal = True)
        est[num_obs]["gpmc"].append(mean_of_samples(s_grad, num_est_samp))
        grad_llc =  int(llg_count)
        ll_count[:] = 0
        llg_count[:] = 0
        
        

        
        theta = np.array([0]*num_dims)
        s_gibbs_slice = []
        for i in range(num_post_samp):
            slice_sampling.slice_sample_all_components_mvprior(theta, lambda:llhood(theta), prior)
            s_gibbs_slice.append(theta.copy())
            #print(np.floor(100*i/num_post_samp),"% of samples")
        #print("Slice", ll_count, llg_count)
        s_gibbs_slice = np.array(s_gibbs_slice)
        est[num_obs]["slicesamp"].append(mean_of_samples(s_gibbs_slice, num_est_samp))
        ss_llc = int(ll_count)
        ll_count[:] = 0
        llg_count[:] = 0
        
        nograd_sqerr.append(((o_m - s_nograd.mean(0))**2).mean())
        grad_sqerr.append(((o_m - s_grad.mean(0))**2).mean())
        slice_sqerr.append(((o_m - s_gibbs_slice.mean(0))**2).mean())
print()
print("Nograd mse", np.mean(nograd_sqerr),ng_llc, "\ngrad mse", np.mean(grad_sqerr), grad_llc, "\nslice", np.mean(slice_sqerr), ss_llc)
#print()