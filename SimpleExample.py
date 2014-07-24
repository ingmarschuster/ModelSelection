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
        
        ######## BEGIN resample dimensionality ########
        if dim_m.new_dim():
            # sample auxiliary theta'
            current_dims += 1
            idx = dim_m.new_dim_idx()
            print("Insert at %d; " % idx, file=sys.stderr)
            theta["w"] = np.insert(theta["w"], idx,
                                   w_prior.rvs((theta["w"].shape[1], )),
                                   axis=0)
            theta["lv"] = np.insert(theta["lv"], idx,
                                   lv_prior.rvs((theta["lv"].shape[0], )),
                                   axis=1)
            for j in range(dim_added_resamples):        
                slice_sample_all_components(theta["w"], llhood, w_prior)
                slice_sample_all_components(theta["lv"], llhood, lv_prior)
                slice_sample_all_components(theta["rv"], llhood, remvar_prior)

        largest_model = deepcopy(theta)
        
        removal_prior = [dim_m.logpmf(current_dims - 1) - log(current_dims)] * current_dims
        removal_prior.append(dim_m.logpmf(current_dims))
        removal_prior = np.array(removal_prior) #- logsumexp(removal_prior)
        
        removal_llhood = []
            
        for idx in range(current_dims):
            theta["w"]  = np.delete(theta["w"],  idx, axis = 0)
            theta["lv"] = np.delete(theta["lv"], idx, axis = 1)
            for j in range(dim_removed_resamples):        
                slice_sample_all_components(theta["w"], llhood, w_prior)
                slice_sample_all_components(theta["lv"], llhood, lv_prior)
                slice_sample_all_components(theta["rv"], llhood, remvar_prior)
            removal_llhood.append(llhood())
            theta["w"] = np.copy(largest_model["w"])
            theta["lv"] = np.copy(largest_model["lv"])
            theta["rv"] = np.copy(largest_model["rv"])
        removal_llhood.append(llhood())      
        removal_llhood = np.array(removal_llhood) #- logsumexp(removal_llhood)
        
        removal_posterior = removal_prior + removal_llhood
        removal_posterior = removal_posterior - logsumexp(removal_posterior)
        
        remove_idx = np.argmax(np.random.multinomial(1, exp(removal_posterior)))
        
        if remove_idx != current_dims:
            print("Remove at %d; " % remove_idx, file=sys.stderr)
            theta["w"]  = np.delete(theta["w"],  remove_idx, axis = 0)
            theta["lv"] = np.delete(theta["lv"], remove_idx, axis = 1)
        else:
           print("Keeping; ", file=sys.stderr) 
        current_dims = theta["w"].shape[0]
        
        dim_log = "\n%s\n"  % np.hstack((np.array((("prior", "lhood", "post "),)).T,
                                                                        np.around(np.vstack((removal_prior,
                                                                                             removal_llhood,
                                                                                             removal_posterior)), 1)
                                                                        ))        
        post_dim = llhood()
        dim_m.update()
        ######## END resample dimensionality ########
        post_w = post_lv = post_remvar = post_dim                           
        for j in range(fix_dim_moves):        
            slice_sample_all_components(theta["w"], llhood, w_prior)
            if j == fix_dim_moves - 1:
                post_w = llhood()
            slice_sample_all_components(theta["lv"], llhood, lv_prior)
            if j == fix_dim_moves - 1:
                post_lv = llhood()
            slice_sample_all_components(theta["rv"], llhood, remvar_prior)
            if j == fix_dim_moves - 1:
                post_remvar = llhood()
        post_llhood = post_remvar
        print("%s         pre: %.2f \n\t dim: %.2f \n\t w:   %.2f \n\t lv:  %.2f \n\t rv:  %.2f \n %d \n==========\n"
              % (dim_log, pre_llhood,   post_dim,      post_w,          post_lv,    post_remvar, current_dims),
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
    