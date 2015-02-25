# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:43:18 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats
import cPickle as pickle

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

from distributions import mvnorm, mvt, norm_invwishart

from itertools import cycle
import itertools
import functools

def check_tails(post, post_mode, prop):
    for of in range(10000, 100000, 10000):
        for sign in (-1,1):
            assert(post(post_mode+sign*of) - prop(post_mode+sign*of) <= 0)
    

def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def indices(samps, n_var):
    n_var = int(n_var)
    samps = int(samps)
    assert(n_var > 0 and samps > 0)
    sr = range(samps)
    rval = []
    for i in range(samps**(n_var-1)):
        itr = []
        for v in range(n_var):
            itr.append(cycle(sr))
            if v == n_var - 1:
                skip = i % samps
            else:
                skip = i // samps**(n_var - 1 - v)
            for _ in range(skip):
                itr[-1].next()
        rval.append([[j.next() for j in itr] for _ in range(samps)])
    return np.array(rval)

def permutations(orig):
    samps = orig.shape[0]
    n_var = orig.shape[1]
    ind = indices(samps, n_var)
    rval = []
    for perm in range(len(ind)):
        rval.append(np.array([orig[ind[perm,:,i],i] for i in range(ind.shape[2])]).T)
    return np.array(rval)


def test_indices():
    for (n_var, n_samp) in ((2, 3), (3, 3), (4,4)):
        ind = indices(n_samp, n_var)
        a = np.vstack(ind)
        assert(unique_rows(a).shape == a.shape)
        
        assert(not np.any(ind >= n_samp))
        assert(not np.any(ind < 0))

def test_permutations():
    for (n_var, n_samp) in ((2, 3), (3, 3), (4,4)):
        ind = indices(n_samp, n_var)
        assert(np.all(ind == permutations(ind[0])))



def log_imp_weight(samp, post, prop):
    return (post(samp)
                 - prop(samp) )

def inflate(samp):
    return np.vstack(permutations(samp))

def ess(samp, post, prop):
    weights = log_imp_weight(samp, post, prop)
    weights = weights - logsumexp(weights) # normalize
    return exp(-logsumexp(2*weights))

def perpl(samp, post, prop):
    weights = log_imp_weight(samp, post, prop)
    weights = weights - logsumexp(weights) # normalize
    return -np.sum(exp(weights)*weights)

def importance_weighting(samp, post, prop, limp_w = None):
    if limp_w is None:
        limp_w = log_imp_weight(samp, post, prop)
    factor = np.atleast_2d(limp_w).T
    #print(np.var(factor))
    estimates = samp * np.hstack([exp(factor)]*samp.shape[1])
    return estimates

def est_plain(samp, post, prop):
    return importance_weighting(samp, post, prop).mean(0)
    
def est_bu(samp, post, prop):
    return est_plain(inflate(samp), post, prop)

def est_bu_indiv_sets(samp, post, prop):
    var = []
    #for s in permutations(samp):
    #    isamp = importance_weighting(samp, post, prop) 
    #    e = isamp.mean(0)
    #    var.append((isamp-e).var(0))
   # assert()
    #print(np.var(var,0))   
    return np.vstack([est_plain(samp_set, post, prop)
                        for samp_set in permutations(samp)])

def est_bu_indiv_sets_prec_reweight(samp, post, prop):
    var = []
    #for s in permutations(samp):
    #    isamp = importance_weighting(samp, post, prop) 
    #    e = isamp.mean(0)
    #    var.append((isamp-e).var(0))
   # assert()
    #print(np.var(var,0))
    perm = permutations(samp)
    wprec = [1./importance_weighting(samp, post, prop).var() for samp_set in perm]
    prec_sum = np.sum(wprec)
    #assert(False)
    return np.vstack([(wprec[i]/prec_sum) * est_plain(perm[i], post, prop) for i in range(len(perm))]).sum(0)

def est_bu_indiv_sets_best_recomb(samp, post, prop):
    var = []
    #for s in permutations(samp):
    #    isamp = importance_weighting(samp, post, prop) 
    #    e = isamp.mean(0)
    #    var.append((isamp-e).var(0))
   # assert()
    #print(np.var(var,0))
    perm = permutations(samp)
    wprec = [1./importance_weighting(samp, post, prop).var() for samp_set in perm]
    best = np.argmax(wprec)
    #assert(False)
    return est_plain(perm[best], post, prop)# for i in range(len(perm))]).sum(0)


def est_stats(estimates):
    print("mse", (estimates**2).mean()) 

n_estimates = []
bu_estimates = []
bu_indiv_sets_estimates =[]
bu_rew_estimates = []

if True:
    M = 100
    K = 2
    log_evid = -1000
    (mu_true, K_true, offset) = (np.ones(K), np.eye(K)*2, 5)
    
    post_param = (mu_true, K_true)
    post = mvnorm(*post_param)
    post_lpdf = lambda x: post.logpdf(x) + log_evid

    prop_param = (mu_true+offset, K_true, 20)
    prop = mvt(*prop_param)
    prop_lpdf = lambda x: prop.logpdf(x)

    #check_tails(post, mu_true, prop)


    perm_x = []
    perm_weights = []
    for x in [prop.rvs(M) for _ in range(200)]:
        #plain_weights = log_imp_weight(x)
        perm_x.append(permutations(x))
        perm_weights.append([log_imp_weight(p, post_lpdf, prop_lpdf) for p in perm_x[-1]])
    with open("Gaussian_test_standard_imp_samp_200_M100_with_logevid_off_center_"+str(offset)+".pickle", "w") as f:
        obj = {"post":post_param, "prop":prop_param,
               "perm_x":perm_x,
               "log_importance_weights":perm_weights,
               "M": M, "K":K,
               "log_evid":log_evid }
        pickle.dump(obj, f)
else:
    
    for dim in range(2,3):
        param_prior = norm_invwishart(np.eye(dim)*3, dim, np.zeros(dim), 1)
        
        for i in range(20):
            (mu_true, K_true) = param_prior.rv()        
            post = mvnorm(mu_true, K_true)
            prop = mvt(mu_true, K_true*10, dim)
            check_tails(post, mu_true, prop)

            x = prop.rvs(100)
            n_estimates.append(np.linalg.norm(est_plain(x, post, prop) - mu_true,2))
            bu_estimates.append(np.linalg.norm(est_bu(x, post, prop) - mu_true,2))
            bu_rew_estimates.append(np.linalg.norm(est_bu_indiv_sets_prec_reweight(x, post, prop) - mu_true,2))
            bu_indiv_sets_estimates.append(np.linalg.norm(est_bu_indiv_sets(x, post, prop) - mu_true,2,1).mean(0))

    

    
#est_stats(np.array(n_estimates))
#est_stats(np.array(bu_estimates)) 
#est_stats(np.array(bu_indiv_sets_estimates)) 
#est_stats(np.array(bu_rew_estimates))
