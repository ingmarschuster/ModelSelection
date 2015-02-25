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


import modsel.deflation_is as dis
import modsel.mc.pmc as pmc
import modsel.estimator_statistics as es
import distributions as dist

import matplotlib.pyplot as plt



def test_DirCatTMM():
    num_obs = 1000
    for dim in range(2,4):
        mu = np.array([11 * (i+1) for i in range(dim)])
        K = np.eye(dim) * 5
        df = dim + 1
        obs_dist = dist.mvt(mu, K, df)
        obs = obs_dist.rvs(num_obs)
        dctmm = dis.DirCatTMM(obs, [1]*dim, obs_dist,
                                      dist.invwishart(np.eye(dim) * 5, dim + 1),
                                      stats.gamma(1, scale=1, loc=dim+1))
        orig_cat_param = dctmm.cat_param
        dctmm.cat_param = np.zeros(dim)
        for i in range(dim):
            dctmm.cat_param[i] = 1
            ### Test DirCatTMM.lpost_comp_indic ###
            for j in range(dim):
                c_indic = np.zeros(dim)
                c_indic[j] = 1
                for o in range(obs.shape[0]):
                    if i == j:
                        assert(dctmm.lpost_comp_indic(c_indic, o) > -np.inf)
                    else:
                        assert(dctmm.lpost_comp_indic(c_indic, o) == -np.inf)
                c_indic[j] = 0
            ### Test DirCatTMM.llhood_comp_param ###
            highest = dctmm.llhood_comp_param((mu, K, df), i)
            assert(highest >= dctmm.llhood_comp_param((-mu, K, df), i))
            assert(highest >= dctmm.llhood_comp_param((mu, K*5, df), i))
            assert(highest >= dctmm.llhood_comp_param((mu, K/2, df), i))
            assert(highest >= dctmm.llhood_comp_param((mu, K, df+10), i))
            dctmm.cat_param[i] = 0
        
        
        ### Test DirCatTMM.lprior ###
        dctmm.cat_param = np.array(dctmm.dir_param / dctmm.dir_param.sum())
        dctmm.comp_indic = dist.categorical(dctmm.cat_param).rvs(num_obs, indic = True)
        dctmm.update_comp_dists([(mu, K, df)] * dim)
        highest = dctmm.lprior()
        
        c_param = dctmm.dir_param + np.arange(dim)
        dctmm.cat_param = np.array(c_param / c_param.sum())
        ch_cat_param = dctmm.lprior()
        assert(highest > ch_cat_param)
        dctmm.update_comp_dists([(-mu, K, df)] * dim)
        assert(ch_cat_param > dctmm.lprior())

    

def test_DirCatTMMProposal():
    num_loc_proposals = 2
    num_imp_samp = 1000
    n_comp = 2
    p_comp = np.array([0.7, 0.3])
    dim = 1
    num_obs = 100
    obs = None
    
    means = []
    
    for i in range(n_comp):
        means.append([20*i]*dim)
        if obs is None:            
            obs = dist.mvt(means[-1], np.eye(dim),30).rvs(np.int(np.round(num_obs*p_comp[i])))
        else:
            obs = np.vstack([obs, dist.mvt(means[-1], np.eye(dim),30).rvs(np.int(np.round(num_obs*p_comp[i])))])

    count = {"local_lpost" :0, "local_llhood" :0, "naive_lpost" :0 ,"naive_llhood" :0,"standard_lpost" :0 ,"standard_llhood" :0}
    print(means)
    #return
    def count_closure(name):
        def rval():
            count[name] = count[name] + 1
        return rval
    
    initial_samples = []
    for _ in range(10):
        initial_samples.append(dis.DirCatTMM(obs, [1]*n_comp, dist.mvt(np.mean(means,0), np.eye(dim)*5, dim),
                                  dist.invwishart(np.eye(dim) * 5, dim+1 ),
                                  stats.gamma(1,scale=1)))
#    (naive_samp, naive_lpost) = pmc.sample(num_imp_samp, initial_samples,
#                               dis.DirCatTMMProposal(naive_multi_proposals = num_loc_proposals,
#                                                     lpost_count = count_closure("naive_lpost"),
#                                                     llhood_count =  count_closure("naive_llhood")),
#                               population_size = 4)
    (infl_samp, infl_lpost) = pmc.sample(num_imp_samp, initial_samples,
                               dis.DirCatTMMProposal(num_local_proposals = num_loc_proposals,
                                                     lpost_count = count_closure("local_lpost"),
                                                     llhood_count =  count_closure("local_llhood")),
                               population_size = 4)
                               
    (stand_samp, stand_lpost) = pmc.sample(num_imp_samp * num_loc_proposals, initial_samples,
                               dis.DirCatTMMProposal(lpost_count = count_closure("standard_lpost"),
                                                     llhood_count =  count_closure("standard_llhood")),
                               population_size = 4)

    print("===============\n",p_comp, means,
#          "\n\n--NAIVE--\n",
#          naive_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, naive_samp[-1].comp_indic.sum(0))+1, count["naive_llhood"], count["naive_lpost"],
          "\n\n--LOCAL--\n",
          infl_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, infl_samp[-1].comp_indic.sum(0))+1, count["local_llhood"], count["local_lpost"],
          "\n\n--STANDARD--\n",
          stand_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, stand_samp[-1].comp_indic.sum(0))+1, count["standard_llhood"], count["standard_lpost"],"\n\n")   
    #return {"infl":(infl_samp, infl_lpost), "standard":(stand_samp, stand_lpost)}
          

