# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:21:00 2014

@author: arbeit
"""
from __future__ import division, print_function
from numpy import exp, log
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats

import datetime
import numpy.random as npr
from copy import deepcopy,copy
import pickle
import sys
from slice_sampling import slice_sample_all_components
from ADDAuxVar import AuxVarDimensionalityModel


def sample(theta, data, lv_prior,
           w_prior,
           remvar_prior,
           num_samples,):
    
        
    rval = []
    pre_llhood = theta["model"][theta["idx"]]["llhood"]
    for i in range(num_samples):
        print("## Sample %d; \n" % i, file=sys.stderr)
        idx_cur = theta["idx"]
        cur = theta["model"][idx_cur]
        candidates = theta["model"].keys()
        #candidates.remove(idx_cur) - dont remove current 
        # - thus sometimes the current model is resampled
        idx_prop = np.random.permutation(candidates)[0]
        prop = theta["model"][idx_prop]
        for c in (prop,):
            llhood = llhood_closure(data, prop)
            slice_sample_all_components(prop["w"], llhood, w_prior)
            slice_sample_all_components(prop["lv"], llhood, lv_prior)
            slice_sample_all_components(prop["rv"], llhood, remvar_prior)
        if stats.bernoulli.rvs(exp(min((0, prop["llhood"]- cur["llhood"])))) == 1:
            print("move from %d to %d accepted" % (idx_cur, idx_prop), file=sys.stderr)
            theta["idx"] = idx_prop
        else:
            print("move from %d to %d rejected" % (idx_cur, idx_prop), file=sys.stderr)
        print("Model %d\n\n" % theta["idx"], file=sys.stderr)
        rval.append(deepcopy(theta))
        
    return rval

def count_dim(samp):
    dimensions = [s["idx"] for s in samp]
    c = {}
    for i in range(1, np.max(dimensions)+1):
        c[i] = dimensions.count(i)
    return c

def llhood_closure(data, model_candidate):    
    def rval():
        mean = (data - model_candidate["lv"].dot(model_candidate["w"]))
        ll = np.sum(stats.norm(0, model_candidate["rv"]).logpdf(mean))
        
        model_candidate["llhood"] = ll
        #print("llhood %f" %model_candidate["llhood"], file=sys.stderr)
        return ll
    return rval
    
    
def test_all(num_obs = 100, num_samples = 100,
             dim_lv = 2, dim_obs = 5,
             interleaved_fix_dim_sampling = False,
             lv_prior = stats.t(500),
             w_prior = stats.t(2.099999),
             remvar_prior = stats.gamma(1, scale=1),
             fix_dim_moves = 0, dim_removed_resamples = 1, dim_added_resamples=1):

    assert(dim_lv < dim_obs)
    
    true_lv = lv_prior.rvs((num_obs, dim_lv))
    true_w = w_prior.rvs((dim_lv, dim_obs))
    data = true_lv.dot(true_w)
    noise_data = data + stats.norm.rvs(0,0.4, size=data.shape)
    
    
    theta = {"model": {}, "idx": max((1, dim_lv - 1))}    
    
    for cand in range(max((1, dim_lv - 1)), min((dim_obs - 1, dim_lv + 1)) + 1):
        theta["model"][cand] = {}
        m = theta["model"][cand]
        m["lv"] = stats.norm.rvs(0,lv_prior.var(), size=(num_obs, cand))
        m["w"] = stats.norm.rvs(0,w_prior.var(), size=(cand, dim_obs))
        m["rv"] =  remvar_prior.rvs((1,1))
        m["llhood"] = 0
        m["lprior"] = 0
        llhood = llhood_closure(noise_data, m)
        
        for _ in range(2):        
            slice_sample_all_components(m["w"], llhood, w_prior)
            slice_sample_all_components(m["lv"], llhood, lv_prior)
            slice_sample_all_components(m["rv"], llhood, remvar_prior)
    
    samp = sample(theta, data, lv_prior, w_prior, remvar_prior, num_samples)
    
    print(count_dim(samp), file=sys.stderr)

    return (data, samp)
    