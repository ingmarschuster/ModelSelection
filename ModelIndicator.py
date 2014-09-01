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


class MatrixFact:
    def __init__(self, num_obs = 100,
             dim_lv = 2, dim_obs = 5,
             lv_prior = stats.t(500),
             w_prior = stats.t(2.099999),
             remvar_prior = stats.gamma(1, scale=1)):
        m = {}
        m["lv"] = stats.norm.rvs(0,lv_prior.var(), size=(num_obs, cand))
        m["w"] = stats.norm.rvs(0,w_prior.var(), size=(cand, dim_obs))
        m["rv"] =  remvar_prior.rvs((1,1))
        m["llhood"] = 0
        m["lprior"] = 0
        
        self.dim_lv = dim_lv
        self.dim_obs = dim_obs
        self.lv_prior = lv_prior
        self.w_prior = w_prior
        self.remvar_prior = remvar_prior
        self.m = m
        
    def __getstate__(self):
        rval = self.__dict__.copy()
        rval["m"] = self.m.copy()
        for rand_var in ("lv","w", "rv"):
            rval["m"][rand_var] = self.__dict__["m"][rand_var].copy()
        return rval
        
    def deepcopy(self):
        rval = MatrixFact()
        rval.__setstate__(self.__getstate__())
        return rval

def sample(theta, data, lv_prior,
           w_prior,
           remvar_prior,
           num_samples,):
    
        
    rval = []
    pre_llhood = theta["model"][theta["idx"]]["llhood"]
    count = -10
    for i in range(num_samples):
        
        print("## Sample %d; \n" % i, file=sys.stderr)
        if count < 10:
            idx_cur = theta["idx"]
            cur = theta["model"][idx_cur]
            candidates = theta["model"].keys()
            #candidates.remove(idx_cur) #- dont remove current 
            # - thus sometimes the current model is resampled
            cand_lprob = np.array([1] * len(candidates))
            cand_lprob -= logsumexp(cand_lprob)
            idx_prop = candidates[np.argmax(np.random.multinomial(1, exp(cand_lprob)))]
            idx_nprop = [k for k in candidates if k != idx_prop]
            prop = theta["model"][idx_prop]
            prop_orig = deepcopy(prop)
            
            llhood = llhood_closure(data, prop)
            slice_sample_all_components(prop["w"], llhood, w_prior)
            slice_sample_all_components(prop["lv"], llhood, lv_prior)
            slice_sample_all_components(prop["rv"], llhood, remvar_prior)
            prop["lprior"] = (w_prior.logpdf(prop["w"]).sum() +
                              lv_prior.logpdf(prop["lv"]).sum() +
                              remvar_prior.logpdf(prop["rv"]).sum())
            print(candidates, idx_prop, idx_nprop, file=sys.stderr)
            other_lpost = logsumexp([theta["model"][k]["lprior"] + theta["model"][k]["llhood"] 
                                         for k in idx_nprop])
            numer = logsumexp((other_lpost - prop["llhood"], -log(len(candidates))))
            denom = logsumexp((other_lpost - prop_orig["llhood"], -log(len(candidates))))
            
            if stats.bernoulli.rvs(exp(min((0, numer - denom)))) == 1:
                print("resampling for %d accepted" % (idx_prop), file=sys.stderr)
            else:
                print("resampling for %d NOT accepted" % (idx_prop), file=sys.stderr)
                prop = theta["model"][idx_prop] = prop_orig
                
               
            orig_dim_move_accept_logprob = min((0, prop["llhood"] - cur["llhood"]))
            if stats.bernoulli.rvs(exp(orig_dim_move_accept_logprob)) == 1:
                print("move from %d to %d accepted" % (idx_cur, idx_prop), file=sys.stderr)
                theta["idx"] = idx_prop
            elif False:
                print("move from %d to %d rejected" % (idx_cur, idx_prop), file=sys.stderr)
                
                #propose resampled current dimension
                orig = deepcopy(cur)
                
                llhood = llhood_closure(data, cur)
                slice_sample_all_components(cur["w"], llhood, w_prior)
                slice_sample_all_components(cur["lv"], llhood, lv_prior)
                slice_sample_all_components(cur["rv"], llhood, remvar_prior)
                
                resamp_dim_move_accept_logprob = min((0, prop["llhood"]- cur["llhood"]))
                resamp_logratio = (  logsumexp([1,-resamp_dim_move_accept_logprob])
                                   - logsumexp([1,-orig_dim_move_accept_logprob]) )
                                  
                if stats.bernoulli.rvs(exp(min([0, resamp_logratio]))) == 1:
                    print("- accepted resample move for %d" % (idx_cur), file=sys.stderr)
                else:
                    print("- rejected resample move for %d" % (idx_cur), file=sys.stderr)
                    theta["model"][idx_cur] = orig
            if count >= 0:
                count = count + 1
        else:
            count = 0
            cand_lprob = np.array([1] * len(candidates))
            cand_lprob -= logsumexp(cand_lprob)
            idx_prop = candidates[np.argmax(np.random.multinomial(1, exp(cand_lprob)))]
            print("move to %d" % (idx_prop), file=sys.stderr)
            theta["idx"] = idx_prop
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
             lv_prior = stats.t(500),
             w_prior = stats.t(2.099999),
             remvar_prior = stats.gamma(1, scale=1)):

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
        
        for _ in range(1):        
            slice_sample_all_components(m["w"], llhood, w_prior)
            slice_sample_all_components(m["lv"], llhood, lv_prior)
            slice_sample_all_components(m["rv"], llhood, remvar_prior)
        m["lprior"] = (w_prior.logpdf(m["w"]).sum() +
                       lv_prior.logpdf(m["lv"]).sum() +
                       remvar_prior.logpdf(m["rv"]).sum())
    samp = sample(theta, data, lv_prior, w_prior, remvar_prior, num_samples)
    
    print(count_dim(samp), file=sys.stderr)

    return (data, samp)
    