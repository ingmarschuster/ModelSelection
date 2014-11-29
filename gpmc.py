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


from distributions import mvnorm
from distributions.linalg import pdinv
from synthdata import simple_gaussian
import slice_sampling 

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

#np.sum(gs_g(test) - gs(test))

def ideal_covar(direction, main_var_scale = 1, other_var = 0.5, fix_main_var=None):
    direction = direction.flat[:]
    if fix_main_var is not None:
        var = fix_main_var
    else:
        var = np.linalg.norm(direction)*main_var_scale
    vs = np.vstack([direction.flat[:], np.eye(len(direction))[:-1,:]])
    bas = gs(vs)
    ew = np.eye(len(direction))*other_var
    ew[0,0] = var
    return bas.T.dot(ew).dot(np.linalg.inv(bas.T))

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

def best_step_size(theta, gradient, llhood_func):
    step_sizes = [10**-i for i in range(1,10)]
    best = np.argmax( [llhood_func(theta + f * gradient) for f in step_sizes])
    return step_sizes[best]
   

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)
 
def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)
 
def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)
 
def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        Y.append(temp_vec)
    Y = np.array(Y)
    for row in range(Y.shape[0]):
        Y[row,:] = Y[row,:] / np.linalg.norm(Y[row,:])
    return Y


def construct_covar_in_direction(direction, scale=False, mean="min", shrink_fact = 1):
    direction = direction.flat[:]
    var = np.linalg.norm(direction)
    vs = np.vstack([direction.flat[:], np.eye(len(direction))[:-1,:]])
    bas = gs(vs)
    ew = np.eye(len(direction))*0.5
    ew[0,0] = var * 0.5
    return bas.T.dot(ew).dot(np.linalg.inv(bas.T))

def pmc_sampling(num_samples, theta, prior, llhood, llhood_grad, momentum = 0, population_size = 20, grad_proposal = True):
    initial_particles = population_size * 3
    rval = [prior.rvs() for _ in range(initial_particles)]
    lpr = [prior.logpdf(theta) for theta in rval]
    lpr_gr = [prior.logpdf_grad(theta) for theta in rval]
    llh = [llhood(theta) for theta in rval]
    llh_gr = [llhood_grad(theta) for theta in rval]
    f_samp = [best_step_size(rval[i], lpr_gr[i] + llh_gr[i], llhood) for i in range(len(rval))]
    
    i = -5
    while len(rval) - initial_particles < num_samples:
        part_idx = np.random.permutation(range(len(rval)) * (population_size + 1))[:population_size]
        pop = []
        prop_lpdf = []
        prop_llhood = []
        prop_lprior = []
        prop_logw = []
        #prop_scal_proj = []
        prop_f = []
        prev_prob = np.array(lpr) + np.array(llh)
        prev_prob = exp(prev_prob - logsumexp(prev_prob))
        
        
        while len(pop) < population_size: #idx in part_idx:
            while True:
                try:
                    idx = np.argmax(np.random.multinomial(1, prev_prob))
                    break
                except ValueError:
                    prev_prob /= prev_prob.sum()
            
            
            if not grad_proposal:
                dim = np.prod(rval[idx].shape)
                prop_dist = stats.multivariate_normal(rval[idx], np.eye(dim))
            else:
                f = f_samp[idx]
                scaled_grad = f * (lpr_gr[idx] + llh_gr[idx])
                mu = rval[idx] + scaled_grad * 0.8 # FIXME: used to be without the 0.5
                cov = ideal_covar(scaled_grad, 0.8, 0.4)
                prop_dist = stats.multivariate_normal(mu, cov)
                if  i > 0 and i < 5:
                    i += 1
                    #print(cov)
                    #plot_current_prop(rval[idx], mu, prop_dist, "Gradient_PMC_"+str(len(rval))+"_"+str(len(pop))+".pdf")
           
            samp = prop_dist.rvs()
            pop.append(samp)
            prop_lpdf.append(prop_dist.logpdf(samp))
            prop_llhood.append(llhood(samp))
            prop_lprior.append(prior.logpdf(samp))
            prop_logw.append(prop_llhood[-1] + prop_lprior[-1] - prop_lpdf[-1])

            step = samp - rval[idx]
            
            if grad_proposal:
                #proj_rel_step = step.T.dot(scaled_grad)/np.linalg.norm(scaled_grad)**2
                #prop_scal_proj.append(f*proj_rel_step)
                if prop_llhood[-1] + prop_lprior[-1] < (lpr[idx] + llh[idx]):
                    #This learning rate was too bold.
                    prop_f.append(f*0.5)
                else:
                    prop_f.append(f*1.05)
                    
        prop_w = exp(np.array(prop_logw) - logsumexp(prop_logw))
        while True:
            try:
                draws = np.random.multinomial(1, prop_w, population_size)
                break
            except ValueError:
                prop_w /= prop_w.sum()
        samp_idxs = np.argmax(draws, 1)
        for idx in samp_idxs:
            samp = pop[idx]
            rval.append(samp)
            lpr.append(prop_lprior[idx])
            lpr_gr.append(prior.logpdf_grad(samp))
            llh.append(prop_llhood[idx])
            llh_gr.append(llhood_grad(pop[idx]))
            if grad_proposal:
                f_samp.append(prop_f[idx])
        #f = np.mean(f_samp[-20:])
        #print(np.floor(100 * len(rval) / num_samples), "% of samples")
        i += 1
    rval = np.array(rval)
    #print(rval,"\n", f_samp, np.mean(f_samp), f_samp[0])
    return rval[initial_particles:]    


num_post_samp = 1000

num_obs = 100
num_dims = 2 #30
num_datasets = 100#50
datasets = simple_gaussian(dims = num_dims, observations_range = range(num_obs, num_obs + 1, 10), num_datasets = num_datasets, cov_var_const = 4)
nograd_sqerr = []
grad_sqerr = []
slice_sqerr = []
for num_obs in datasets:
    for Data in datasets[num_obs]:        
        obs = Data["obs"]
        
        
        obs_K =  Data["params"][1]
        (obs_Ki, obs_L, obs_Li, obs_logdet) = pdinv(obs_K)
        
        
        prior = mvnorm(np.array([0]*num_dims), np.eye(num_dims) * 50)
        
        
        def llhood(theta):
            return stats.multivariate_normal.logpdf(obs, theta, obs_K).sum()
        
        def llhood_grad(theta):
            return obs_Ki.dot((np.atleast_2d(obs) - theta).T).sum(1).T
            
        
        
        

        
        s_nograd = pmc_sampling(num_post_samp, np.array([0]*num_dims), prior, llhood, llhood_grad, population_size = 4, grad_proposal = False)
        s_grad = pmc_sampling(num_post_samp, np.array([0]*num_dims), prior, llhood, llhood_grad, population_size = 4, grad_proposal = True)
        
        theta = np.array([0]*num_dims)
        s_gibbs_slice = []
        for i in range(num_post_samp):
            slice_sampling.slice_sample_all_components(theta, lambda:llhood(theta), prior)
            s_gibbs_slice.append(theta.copy())
            #print(np.floor(100*i/num_post_samp),"% of samples")
        o_m = obs.mean(0)
        nograd_sqerr.append(((o_m - s_nograd)**2).sum())
        grad_sqerr.append(((o_m - s_grad)**2).sum())
        slice_sqerr.append(((o_m - s_gibbs_slice)**2).sum())
print("Nograd mse", np.mean(nograd_sqerr), "\ngrad mse", np.mean(grad_sqerr), "\nslice", np.mean(slice_sqerr))
#print()