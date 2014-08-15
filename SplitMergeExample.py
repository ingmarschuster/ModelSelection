# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:21:00 2014

@author: arbeit
"""
from __future__ import division, print_function
import numpy as np
import datetime
import numpy.random as npr
import scipy.stats as stats
from numpy import exp, log
from scipy.misc import logsumexp
from copy import deepcopy,copy
import pickle
import sys
from slice_sampling import slice_sample_all_components
from ADDAuxVar import AuxVarDimensionalityModel


def sample(theta, lv_prior,
           w_prior,
           remvar_prior,
           dim_m,
           llhood,
           num_samples,
           dim_added_resamples  = 1,
           fix_dim_moves = 1,
           dim_removed_resamples = 0):
    
        
    rval = []
    post_llhood = llhood()
    for i in range(num_samples):
        print("## Sample %d; \n\n" % i, file=sys.stderr)
        pre_llhood = post_llhood
        
        current_dims = theta["w"].shape[0]
        
        move_type = 2 * stats.bernoulli.rvs(0.5) - 1
        
        dim_log = "Not changing dim"
        
        if dim_m.logpmf(current_dims + move_type) == -np.Infinity:
            slice_sample_all_components(theta["w"], llhood, w_prior)
            slice_sample_all_components(theta["lv"], llhood, lv_prior)
        else:
            orig_model = deepcopy(theta)
            orig_dims = current_dims
            orig_llhood = pre_llhood
            orig_lprior = (dim_m.logpmf(orig_dims) +
                           w_prior.logpdf(theta["w"]).sum() +
                           lv_prior.logpdf(theta["lv"]).sum() +
                           remvar_prior.logpdf(theta["rv"]).sum())
            
            if move_type == +1:
                p = np.array([1] * (current_dims + 1))
                p = p - logsumexp(p)                
                i = np.argmax(np.random.multinomial(1, exp(p)))
                
                current_dims += 1
                dim_log = "Insert at %d; " % i
                theta["w"] = np.insert(theta["w"], i,
                                       np.zeros((theta["w"].shape[1], )),
                                       axis=0)
                theta["lv"] = np.insert(theta["lv"], i,
                                       np.zeros((theta["lv"].shape[0], )),
                                       axis=1)
                for _ in range(dim_added_resamples):        
                    slice_sample_all_components(theta["w"], llhood, w_prior)
                    slice_sample_all_components(theta["lv"], llhood, lv_prior)
            else: # move_type == -1
                p = np.array([1] * current_dims)
                p = p - logsumexp(p)                
                i = np.argmax(np.random.multinomial(1, exp(p)))
                
                current_dims -= 1
                dim_log  = "Remove at %d; " % i
                theta["w"] = np.delete(theta["w"], i,
                                       axis=0)
                theta["lv"] = np.delete(theta["lv"], i,
                                       axis=1)
                for _ in range(dim_removed_resamples):        
                    slice_sample_all_components(theta["w"], llhood, w_prior)
                    slice_sample_all_components(theta["lv"], llhood, lv_prior)
            
            # now construct a pseudo-proposal which is designed to
            # assign the same probability to the forward and reverse move
            # the same forth or back
            #theta["lv"][0, 0] += stats.norm.rvs(0, 0.0001)
            

            prop_llhood = llhood()
            prop_lprior = (dim_m.logpmf(current_dims) +
                           w_prior.logpdf(theta["w"]).sum() +
                           lv_prior.logpdf(theta["lv"]).sum() +
                           remvar_prior.logpdf(theta["rv"]).sum())            
            
            ratio = orig_lprior + orig_llhood - prop_lprior - prop_llhood
            #dim_log += "%f, %f" % (exp(ratio), np.min((1, exp(ratio))))
            
            if stats.bernoulli.rvs(exp(np.min((0, ratio)))) == 1:
                dim_log += " - accepted"
            else:
                #proposal not accepted
                dim_log += " - not accepted"
                theta["w"] = orig_model["w"]
                theta["lv"] = orig_model["lv"]
                theta["rv"] = orig_model["rv"]
                current_dims = orig_dims
        
        dim_m.update()
        ######## END resample dimensionality ########
        slice_sample_all_components(theta["rv"], llhood, remvar_prior)
        post_llhood = llhood()
        print("%s         \npre: %.2f \n\t post: %.2f \n\t  %d \n==========\n"
              % (dim_log, pre_llhood,   post_llhood,        current_dims),
              file=sys.stderr)
        rval.append((deepcopy(theta), copy(pre_llhood), copy(post_llhood)))
        
    return rval

def count_dim(samp):
    dimensions = [s[0]["lv"].shape[1] for s in samp]
    c = {}
    for i in range(1, np.max(dimensions)+1):
        c[i] = dimensions.count(i)
    return c

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
    remvar = remvar_prior.rvs((1,1))
    data = true_lv.dot(true_w)
    noise_data = data + stats.norm.rvs(0,0.4, size=data.shape)
    

    lv = stats.norm.rvs(0,lv_prior.var(), size=(num_obs, 1))
    w = stats.norm.rvs(0,w_prior.var(), size=(1, dim_obs))
    
    theta = {"lv": lv, "w": w, "rv": remvar}
    
    llhood = lambda: np.sum(stats.norm(0, theta["rv"]).logpdf(noise_data - theta["lv"].dot(theta["w"])))
    
    dim_m = AuxVarDimensionalityModel(dim_obs - 1,
                                      confidence = 1,
                                      current_dim_callback = lambda: theta["lv"].shape[1])
    samp = sample(theta, lv_prior, w_prior, remvar_prior, dim_m,
                  llhood, num_samples, fix_dim_moves = fix_dim_moves, dim_removed_resamples = dim_removed_resamples, dim_added_resamples=dim_added_resamples)
    
    print(count_dim(samp), file=sys.stderr)

    return (data, samp)
    