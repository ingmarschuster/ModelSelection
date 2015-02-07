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


def cat_param_average_loss(truth, samples):
    truth = sorted(truth)
    return np.average([stats.entropy(truth, sorted(s.cat_param.flatten())) for s in samples])

def mixture_means_over_time(samples):
    return np.array([[float(s.comp_param[j][0])  for j in range(len(s.comp_param))] for s in samples])
 

def plot_mixture_means_over_time(res, outfname = "plot.pdf", asymptotes = [], steps=100):
    infl = mixture_means_over_time(res["infl"][0])
    std = mixture_means_over_time(res["standard"][0])
    i_idx = np.round(np.linspace(1,infl.shape[0]-1, num=steps)).astype(int)
    s_idx = np.round(np.linspace(1,std.shape[0]-1, num=steps)).astype(int)
    fig, axes = plt.subplots(ncols=3, nrows = 1, figsize=(9,3))
    i_n = []
    i_v = []
    i_l = []
    s_n = []
    s_v = []
    s_l = []
    for i in range(len(i_idx)):
        i_n.append(np.average(infl[:i_idx[i]], axis = 0))
        i_v.append(np.var(infl[:i_idx[i]], axis = 0))
        i_l.append(es.logmeanexp(res["infl"][1][:i_idx[i]])[0])
        s_n.append(np.average(std[:s_idx[i]], axis = 0))
        s_v.append(np.var(std[:s_idx[i]], axis = 0))
        s_l.append(es.logmeanexp(res["standard"][1][:s_idx[i]])[0])
    i_n = np.array(i_n)
    i_v = np.array(i_v)
    i_l = np.array(i_l)
    s_n = np.array(s_n)
    s_v = np.array(s_v)
    s_l = np.array(s_l)
    for a in axes:
        a.set_title("")
        a.set_xlabel("log # lhood evals")
        a.autoscale("both")
        a.set_aspect("auto", adjustable="datalim")
    
    if len(asymptotes) > 0:
        axes[0].set_ylim(min(i_n.min(),s_n.min(),min(asymptotes))-2, max(i_n.min(),s_n.min(),max(asymptotes))+2)
    axes[0].set_ylabel("estim. component means")
    axes[0].plot(s_idx, s_n, "--", s_idx, i_n)
    for l in asymptotes:
        axes[0].axhline(y = l, color="black", ls="dotted")
    axes[1].set_ylabel("variance of estimate")
    axes[1].plot(s_idx, s_v, "--", s_idx, i_v)
    axes[2].set_ylabel("avg. log likelihood")
    axes[2].plot(s_idx, s_l, "--", s_idx, i_l)
   # assert()
    fig.tight_layout()
    fig.savefig(outfname, bbox_inches='tight')
    plt.close(fig)
    return (s_idx, i_n, s_n)
    

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
        initial_samples.append(dis.DirCatTMM(obs, [1]*n_comp, dist.mvt([0]*dim, np.eye(dim)*5, dim),
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
    return {"infl":(infl_samp, infl_lpost), "standard":(stand_samp, stand_lpost)}
          

def test_DirCatTMMProposal_Iris():
    from sklearn.datasets import load_iris
    num_imp_samp = 5
    num_loc_proposals = 3
    n_comp = 3
    p_comp = np.array([1/n_comp] * n_comp)
    dim = 4
    iris = load_iris()
    obs = iris["data"]
    labels = iris["target"]
    means = np.array([obs[i*50:(i+1)*50].mean(0) for i in range(3)])

    count = {"local_lpost" :0, "local_llhood" :0, "naive_lpost" :0 ,"naive_llhood" :0,"standard_lpost" :0 ,"standard_llhood" :0}

    def count_closure(name):
        def rval():
            count[name] = count[name] + 1
        return rval
    
    initial_samples = []
    for _ in range(10):
        initial_samples.append(dis.DirCatTMM(obs, [1]*n_comp, dist.mvt(obs.mean(0), np.diag(obs.var(0)), 20),
                                  dist.invwishart(np.eye(dim) * 5, dim + 1),
                                  stats.gamma(1, scale=1)))
    (naive_samp, naive_lpost) = pmc.sample(num_imp_samp, initial_samples,
                               dis.DirCatTMMProposal(naive_multi_proposals = num_loc_proposals,
                                                     lpost_count = count_closure("naive_lpost"),
                                                     llhood_count =  count_closure("naive_llhood")),
                               population_size = 4)
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
          "\n\n--NAIVE--\n",
          naive_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, naive_samp[-1].comp_indic.sum(0))+1, count["naive_llhood"], count["naive_lpost"],
          "\n\n--LOCAL--\n",
          infl_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, infl_samp[-1].comp_indic.sum(0))+1, count["local_llhood"], count["local_lpost"],
          "\n\n--STANDARD--\n",
          stand_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, stand_samp[-1].comp_indic.sum(0))+1, count["standard_llhood"], count["standard_lpost"],"\n\n")   