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
        m["llhood_candidate"] = 0
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
    
    candidates = theta["model"].keys()
    dir_par = np.array([1] * len(candidates))
    pre_llhood = theta["model"][candidates[theta["idx"]]]["llhood_candidate"]
    for i in range(num_samples):
        print("## Sample %d; \n" % i, file=sys.stderr)
        idx_cur_dir = theta["idx"]
        idx_cur = candidates[idx_cur_dir]
        cur = theta["model"][idx_cur]
        
        #candidates.remove(idx_cur) #- dont remove current 
        # - thus sometimes the current model is resampled
        cand_prob = np.array([1./len(candidates)] * len(candidates))
        #cand_lprob -= logsumexp(cand_lprob)
        idx_prop_dir = np.argmax(np.random.multinomial(1, cand_prob))
        idx_prop = candidates[idx_prop_dir]
        idx_nprop_dir = [k for k in range(len(candidates)) if k != idx_prop_dir]
        prop = theta["model"][idx_prop]
        
        print(candidates, idx_cur_dir, idx_prop_dir, idx_nprop_dir, file = sys.stderr)
        #prop_orig = deepcopy(prop) # for uniform prior, every resampling of a submodel is accepted
        
        llhood = llhood_closure(data, cmm_proposal = (theta, idx_prop))
        rvs = {"w": w_prior, "lv": lv_prior, "rv": remvar_prior}
        lpri_factors = {}
        print("pre", int(prop["llhood_candidate"]), theta["llhood_collapsed"], file = sys.stderr)
        for var in rvs:
            lpri_tmp = prop["lprior"] - rvs[var].logpdf(prop[var]).sum()
            class lpri_closure:
                def logpdf(self, value):
                    lpri_factors[var] = rvs[var].logpdf(value).sum()
                    prop["lprior"] = lpri_tmp + lpri_factors[var]
                    return prop["lprior"]
            slice_sample_all_components(prop[var], llhood, lpri_closure())
        print("post", int(prop["llhood_candidate"]), theta["llhood_collapsed"], file = sys.stderr)
            
        
        print("proposed llhood:", int(prop["llhood_candidate"]), "current llhood:", int(cur["llhood_candidate"]), file = sys.stderr)
        orig_dim_move_accept_logprob = min((0, prop["llhood_candidate"] - cur["llhood_candidate"]))
        if stats.bernoulli.rvs(exp(orig_dim_move_accept_logprob)) == 1:
            print("move from %d to %d accepted" % (idx_cur, idx_prop), file=sys.stderr)
            dir_par[idx_prop_dir] += 1
            theta["idx"] = idx_prop_dir
        else:
            dir_par[idx_cur_dir] += 1
            print("move from %d to %d rejected" % (idx_cur, idx_prop), file=sys.stderr)
        print("Model %d\n\n" % candidates[theta["idx"]], file=sys.stderr)
        rval.append(deepcopy(theta))
        
    return rval

def count_dim(samp):
    dimensions = [s["idx"] for s in samp]
    c = {}
    for i in range(1, np.max(dimensions)+1):
        c[i] = dimensions.count(i)
    return c



def llhood_closure(data, model = None, cmm_proposal = (None, None)):
    if model != None: 
        assert(cmm_proposal == (None, None))
        def rval():
            mean = (data - model["lv"].dot(model["w"]))
            ll = np.sum(stats.norm(0, model["rv"]).logpdf(mean))
            
            model["llhood_candidate"] = ll
            #print("llhood %f" %model_candidate["llhood_candidate"], file=sys.stderr)
            return ll
        return rval
    elif cmm_proposal != (None, None):
        (cmm, proposal) = cmm_proposal
        prop_mdl = cmm["model"][proposal]
        
        candidates = cmm["model"].keys()
        cd_prior_lprob = -log(len(candidates))
        lpriors = 0
        llhoods = []
        for c in candidates:
            if c == proposal:
                continue
            lpriors = lpriors + cmm["model"][c]["lprior"]
            llhoods.append(cd_prior_lprob + cmm["model"][c]["llhood_candidate"])
        llhoods = logsumexp(llhoods)
        def rval():            
            mean = (data - prop_mdl["lv"].dot(prop_mdl["w"]))
            ll = np.sum(stats.norm(0, prop_mdl["rv"]).logpdf(mean))
            
            prop_mdl["llhood_candidate"] = ll
            cmm["llhood_collapsed"] = logsumexp((llhoods, ll + cd_prior_lprob))
            #print("llhood %f" %model_candidate["llhood_candidate"], file=sys.stderr)
            return cmm["llhood_collapsed"] # +lpriors
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
    
    
    theta = {"model": {}, "idx": 0, "llhood_collapsed":-5e6}    
    
    for cand in range(max((1, dim_lv - 1)), min((dim_obs - 1, dim_lv + 1)) + 1):
        theta["model"][cand] = {}
        m = theta["model"][cand]
        m["lv"] = stats.norm.rvs(0,lv_prior.var(), size=(num_obs, cand))
        m["w"] = stats.norm.rvs(0,w_prior.var(), size=(cand, dim_obs))
        m["rv"] =  remvar_prior.rvs((1,1))
        m["llhood"] = -5e6
        m["llhood_candidate"] = -5e6
        m["lprior"] = 0
        llhood = llhood_closure(noise_data, m)
        
        for _ in range(3):        
            slice_sample_all_components(m["w"], llhood, w_prior)
            slice_sample_all_components(m["lv"], llhood, lv_prior)
            slice_sample_all_components(m["rv"], llhood, remvar_prior)
        m["lprior"] = (w_prior.logpdf(m["w"]).sum() +
                       lv_prior.logpdf(m["lv"]).sum() +
                       remvar_prior.logpdf(m["rv"]).sum())
        m["llhood_candidate"] = m["llhood"]
    samp = sample(theta, data, lv_prior, w_prior, remvar_prior, num_samples)
    
    
    
    print(count_dim(samp), file=sys.stderr)

    return (data, samp)
    