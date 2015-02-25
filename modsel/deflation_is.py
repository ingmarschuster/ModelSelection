# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:53:38 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

from copy import copy, deepcopy


import modsel.mc.pmc as pmc
import distributions as dist
import modsel.mc.bijections as bij
import itertools



class DirCatTMM(object):    
    def __init__(self, data, dir_param, mean_prior, cov_prior, df_prior):
        self.data = data
        self.num_obs = data.shape[0]
        self.dim_obs = data.shape[1]
        self.dir_param = np.array(dir_param).flatten()
        self.cat_param = np.random.dirichlet(self.dir_param)
        self.comp_indic = dist.categorical(self.cat_param).rvs(size=self.num_obs, indic=True)
        
        self.mean_prior = mean_prior
        self.cov_prior = cov_prior
        self.df_prior = df_prior
        self.update_comp_dists([[mean_prior.rvs(), cov_prior.rv(), df_prior.rvs()]
                                    for _ in range(len(dir_param))])
        
    def update_comp_dists(self, comp_params):
        assert(len(comp_params) == self.comp_indic.shape[1])
        self.comp_param = comp_params
        self.comp_dist = [dist.mvt(*p) for p in self.comp_param]
    
    def lprior(self):
        rval = dist.dirichlet(self.dir_param).logpdf(self.cat_param)
        assert(rval != -np.inf)
        rval = rval + dist.categorical(self.cat_param).logpdf(self.comp_indic,
                                                        indic = True).sum()
        assert(rval != -np.inf)
        for i in range(len(self.comp_param)):
            rval = rval + self.mean_prior.logpdf(self.comp_param[i][0])
            assert(rval != -np.inf)
            rval = rval + self.cov_prior.logpdf(self.comp_param[i][1])
            assert(rval != -np.inf)
            rval = rval + self.df_prior.logpdf(self.comp_param[i][2])
            assert(rval != -np.inf)
        return rval
                
    
    def lpost_comp_indic(self, x, observation_idx):
        assert(observation_idx is not None)
        comp_idx = np.argmax(x.flat)
        return (dist.categorical(self.cat_param).logpdf(comp_idx) +
                self.comp_dist[comp_idx].logpdf(self.data[observation_idx]))
                    
    def llhood_comp_param(self, x, component_idx):
        candidate_dist = dist.mvt(*x)
        relevant_data = (component_idx == np.argmax(self.comp_indic, 1))
        return np.sum(candidate_dist.logpdf(self.data[relevant_data]))


class DirCatTMMProposal(pmc.PmcProposalDistribution):
    def __init__(self, num_local_proposals = 1, naive_multi_proposals = 1, lpost_count = None, llhood_count =  None):
        # - normal PMC proposal: naive_multi_proposals = 1 and num_local_proposals = 1
        # - 2^k PMC proposals with equal component indicators but different component parameters, 
        #       where k is the number of components:
        #       - naive_multi_proposals = 1 and num_local_proposals = 2
        # - 2 PMC proposals with equal component indicators but different component parameters: 
        #    - naive_multi_proposals = 2 and num_local_proposals = 1
        assert(not (num_local_proposals > 1 and naive_multi_proposals > 1))
        if lpost_count is None:
            self.lpc = lambda:None
        else:
            self.lpc = lpost_count
        if llhood_count is None:
            self.llhc =lambda:None
        else:
            self.llhc = llhood_count
        self.num_local_proposals = num_local_proposals
        self.naive_multi_proposal = naive_multi_proposals
        
    def process_initial_samples(self, samples):
        return [pmc.PmcSample(sample = s) for s in samples]
        
    def process_new_ancestors(self, ancestors):
        pass
    
    def gen_proposal(self, ancestor):
        if not isinstance(ancestor, pmc.PmcSample):
            ancestor = pmc.PmcSample(sample = ancestor)
        rval = []
        lprop_prob = 0
        proto = deepcopy(ancestor.sample)
        
        dirichl = dist.dirichlet(proto.dir_param + proto.cat_param.sum(0))
        proto.cat_param = dirichl.rvs()
        lprop_prob = lprop_prob + dirichl.logpdf(proto.cat_param)
        
        for o in range(ancestor.sample.num_obs):
            def lpost_func(comp_indic):
                self.lpc()
                return proto.lpost_comp_indic(comp_indic, observation_idx = o)
            cp = pmc.LatentClassProposal(lpost_func,
                                         ancestor.sample.comp_indic.shape[1]).gen_proposal()
            proto.comp_indic[o] = cp.sample
            lprop_prob = lprop_prob + cp.lprop      
        
        for i in range(self.naive_multi_proposal):
            
            comp_param_lprops = []
            for comp_idx in range(len(proto.comp_param)):
                comp_param_lprops.append([])
                for k in range(self.num_local_proposals):
                    (anc_mu, anc_cov, anc_df) = proto.comp_param[comp_idx]
                    dim = anc_mu.size
                    prop_mu = pmc.NaiveRandomWalkProposal(lambda x: None,
                                                          dist.mvt(np.zeros(dim), np.eye(dim)*5,20)).gen_proposal(mean=proto.comp_param[comp_idx][0])
                    prop_cov = pmc.InvWishartRandomWalkProposal(anc_cov.shape[0] + 2, anc_cov.shape[0]).gen_proposal(mean=proto.comp_param[comp_idx][1])
                    pdist_df = stats.gamma(proto.comp_param[comp_idx][2] + 1 , scale=1)
                    prop_df = pdist_df.rvs()
                    lprop_df = pdist_df.logpdf(prop_df)
                    param_all = [prop_mu.sample, prop_cov.sample, prop_df]
                    self.llhc()
                    prop = {"param": param_all,
                            "llhood": proto.llhood_comp_param(param_all, comp_idx),
                            "lprop": prop_mu.lprop + prop_cov.lprop + lprop_df}
                    comp_param_lprops[-1].append(prop)
            
            
            for combination in itertools.product(*comp_param_lprops):
                rval.append(pmc.PmcSample(ancestor))
                rval[-1].sample = copy(proto)
                rval[-1].sample.update_comp_dists([c["param"] for c in combination])
                rval[-1].lprop = lprop_prob + np.sum([c["lprop"] for c in combination])
                rval[-1].lpost = rval[-1].sample.lprior() + np.sum([c["llhood"] for c in combination])
                rval[-1].lweight = rval[-1].lpost - rval[-1].lprop
    #            assert()
        if False:
            if self.naive_multi_proposal > 1:
                print("Multiple proposals:", len(rval))
            elif self.num_local_proposals > 1:
                print("Combined proposals:", len(rval))
        return rval


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


def approximate_mixture_data():
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
        initial_samples.append(DirCatTMM(obs, [1]*n_comp, dist.mvt(np.mean(means,0), np.eye(dim)*5, dim),
                                  dist.invwishart(np.eye(dim) * 5, dim+1 ),
                                  stats.gamma(1,scale=1)))
#    (naive_samp, naive_lpost) = pmc.sample(num_imp_samp, initial_samples,
#                               DirCatTMMProposal(naive_multi_proposals = num_loc_proposals,
#                                                     lpost_count = count_closure("naive_lpost"),
#                                                     llhood_count =  count_closure("naive_llhood")),
#                               population_size = 4)
    (infl_samp, infl_lpost) = pmc.sample(num_imp_samp, initial_samples,
                               DirCatTMMProposal(num_local_proposals = num_loc_proposals,
                                                     lpost_count = count_closure("local_lpost"),
                                                     llhood_count =  count_closure("local_llhood")),
                               population_size = 4)
                               
    (stand_samp, stand_lpost) = pmc.sample(num_imp_samp * num_loc_proposals, initial_samples,
                               DirCatTMMProposal(lpost_count = count_closure("standard_lpost"),
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


def approximate_iris_mixture_data():
    from sklearn.datasets import load_iris
    num_imp_samp = 100
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
        initial_samples.append(DirCatTMM(obs, [1]*n_comp, dist.mvt(obs.mean(0), np.diag(obs.var(0)), 20),
                                  dist.invwishart(np.eye(dim), 50),
                                  stats.gamma(500, scale=0.1)))
        
    (infl_samp, infl_lpost) = pmc.sample(num_imp_samp, initial_samples,
                               DirCatTMMProposal(num_local_proposals = num_loc_proposals,
                                                     lpost_count = count_closure("local_lpost"),
                                                     llhood_count =  count_closure("local_llhood")),
                               population_size = 4)
                               
    (stand_samp, stand_lpost) = pmc.sample(num_imp_samp * num_loc_proposals, initial_samples,
                               DirCatTMMProposal(lpost_count = count_closure("standard_lpost"),
                                                     llhood_count =  count_closure("standard_llhood")),
                               population_size = 4)

    print("===============\n",p_comp, means,
          "\n\n--LOCAL--\n",
          infl_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, infl_samp[-1].cat_param.flatten()), count["local_llhood"], count["local_lpost"],
          "\n\n--STANDARD--\n",
          stand_samp[-1].comp_indic.sum(0), stats.entropy(p_comp, stand_samp[-1].cat_param.flatten()), count["standard_llhood"], count["standard_lpost"],"\n\n")   
    return {"infl":(infl_samp, infl_lpost), "standard":(stand_samp, stand_lpost)}
          

if __name__ == "__main__":
    import scipy.io as io
    
    of3 = io.loadmat("data/oilFlow3Class.mat")
    of3_lab = np.vstack((of3["DataTrnLbls"], of3["DataTstLbls"],of3["DataVdnLbls"],))
    of3 = np.vstack((of3["DataTrn"], of3["DataTst"],of3["DataVdn"],))*100

    initial = [DirCatTMM(of3,
                         [1]*3, 
                         dist.mvnorm([0]*12, 
                         np.eye(12)), 
                         dist.invwishart(np.eye(12)*5, 12), 
                         stats.gamma(1,scale=1)) for _ in range(10)]
                         
    count = {"local_lpost" :0, "local_llhood" :0, "naive_lpost" :0 ,"naive_llhood" :0}
        
    def count_closure(name):
        def rval():
            count[name] = count[name] + 1
        return rval
            
    samps = pmc.sample(50,
                       initial,
                       DirCatTMMProposal(lpost_count = count_closure("naive_lpost"), llhood_count = count_closure("naive_lpost")),
                       population_size=5,
                       quiet=False)
    
