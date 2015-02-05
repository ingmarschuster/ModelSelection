# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:43:18 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

from distributions import mvnorm, mvt

from itertools import cycle
import itertools
import functools

def check_tails(post, post_mode, prop):
    for of in range(10000, 100000, 10000):
        for sign in (-1,1):
            assert(post.logpdf(post_mode+sign*of) - prop.logpdf(post_mode+sign*of) <= 0)
    

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
    return (post.logpdf(samp)
                 - prop.logpdf(samp) )

def importance_weighting(samp, post, prop):
    factor = np.atleast_2d(log_imp_weight(samp, post, prop)).T
    #print(np.var(factor))
    estimates = samp * np.hstack([exp(factor)]*samp.shape[1])
    return estimates

def est_plain(samp, post, prop):
    return importance_weighting(samp, post, prop).mean(0)
    
def est_bu(samp, post, prop):
    var = []
    #for s in permutations(samp):
    #    isamp = importance_weighting(samp, post, prop) 
    #    e = isamp.mean(0)
    #    var.append((isamp-e).var(0))
   # assert()
    #print(np.var(var,0))   
    return est_plain(np.vstack(permutations(samp)), post, prop)

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

def est_stats(estimates):
    print("mse", (estimates**2).mean()) 

(mu_true, K_true, offset) = (np.ones(2)*-10, np.eye(2)*2, 2)


class Post(object):
    def __init__(self, mu1, K1, p, mu2, K2):
        assert(p > 0 and p < 1)
        self.p = stats.bernoulli(p)
        self.d1 = mvnorm(mu1, K1)
        self.d2 = mvnorm(mu2, K2)
    
    def logpdf(self, x):
        return logsumexp([self.p.logpmf(1)+ self.d1.logpdf(x), self.p.logpmf(0)+ self.d2.logpdf(x)])
        
post = mvnorm(mu_true, K_true)# Post(mu_true, K_true, 0.3, mu_true + offset, K_true)# 
exp_true = mu_true

prop = mvt(mu_true , K_true, 2.0000001)

check_tails(post, exp_true, prop)

n_estimates = []
bu_estimates = []
bu_indiv_sets_estimates = []

for x in [prop.rvs(100) for _ in range(20)]:
    n_estimates.append(np.linalg.norm(est_plain(x, post, prop) - exp_true,2))
    bu_estimates.append(np.linalg.norm(est_bu(x, post, prop) - exp_true,2))
    bu_indiv_sets_estimates.append(np.linalg.norm(est_bu_indiv_sets(x, post, prop) - exp_true,2,1).mean(0))


n_estimates = np.array(n_estimates)
bu_estimates = np.array(bu_estimates)
bu_indiv_sets_estimates = np.array(bu_indiv_sets_estimates)
    
est_stats(n_estimates)
est_stats(bu_estimates) 
est_stats(bu_indiv_sets_estimates)
print((bu_estimates**2).mean() <= (bu_indiv_sets_estimates**2).mean())


#print(x.shape, est(x, post, prop, mu_true))
#print(x_bu.shape, est(x_bu, post, prop, mu_true))