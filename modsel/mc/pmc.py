# -*- coding: utf-8 -*-
"""
Proposal classes for PMC
Created on Thu Dec 11 09:24:27 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import sys
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp

from copy import copy, deepcopy


from distributions import mvnorm, mvt, categorical, invwishart
from modsel.mc.gs_basis import ideal_covar
from modsel.mc.optimization import find_step_size
from modsel.mc import flags

import modsel.mc.conjugate_models as cm
import modsel.mc.bijections as bij

__all__ = ["sample", "sample_lpost_based",
           "PmcSample", "PmcProposalDistribution",
           "NaiveRandomWalkProposal", "LatentClassProposal", 
           "WishartRandomWalkProposal",
           "GrAsProposal", "ConGrAsProposal"]

def sample(num_samples, initial_particles, proposal_method, population_size = 20, stop_flag = flags.NeverStopFlag(), quiet = True):
    rval = proposal_method.process_initial_samples(initial_particles)
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples and not stop_flag.stop():
        #print(len(rval))
        anc_cand = np.min((len(rval), 2 * population_size))
        
        ancest_dist = categorical([1./anc_cand] * anc_cand)
        
        #choose ancestor uniformly at random from previous samples
        pop = []
        for _ in range(population_size):
            tmp = proposal_method.gen_proposal(rval[-ancest_dist.rvs()])
            if hasattr(tmp, "__iter__"):
                pop.extend(tmp)
            else:
                pop.append(tmp)
                

        prop_w = np.array([s.lweight for s in pop])
        prop_w = exp(prop_w - logsumexp(prop_w))
        
        
        # Importance Resampling
        while True:
            try:
                draws = np.random.multinomial(population_size, prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
        
        new_samp = []
        for idx in range(len(draws)):
            new_samp.extend([pop[idx]] * draws[idx])
        proposal_method.process_new_ancestors(new_samp)
        rval.extend(new_samp)
        if not quiet:
            print(len(rval), "samples", file=sys.stderr)
        
    try:
        pass
        #print("jump model", proposal_method.jump_mdl.ddist.K, "thres model",proposal_method.jump_thres_mdl.rv(),
        #      "gr_covar_mdl", proposal_method.gr_covar_mdl.rv())
    except:
        pass
    return (np.array([s.sample for s in rval[num_initial:]]), np.array([s.lpost for s in rval[num_initial:]]))


def sample_sis(num_samples, initial_particles, proposal_method, stop_flag = flags.NeverStopFlag(), quiet = True):
    part = proposal_method.process_initial_samples(initial_particles)
    rval = []
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples and not stop_flag.stop():
        #print(len(rval))
        
        #choose ancestor uniformly at random from previous samples
        pop = []
        part_new = []
        for p in part:
            tmp = proposal_method.gen_proposal(p)
            if hasattr(tmp, "__iter__"):
                pop.extend(tmp)
                lposts = np.array([t.lpost for t in tmp])
                cd = categorical(lposts - logsumexp(lposts), True)
                part_new.append(tmp[cd.rvs()])
            else:
                pop.append(tmp)
                part_new.append(tmp)
                

        prop_w = np.array([s.lweight for s in pop])
        prop_w = exp(prop_w - logsumexp(prop_w))
        
        
        # Importance Resampling
        while True:
            try:
                draws = np.random.multinomial(len(initial_particles), prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
        
        new_samp = []
        for idx in range(len(draws)):
            new_samp.extend([pop[idx]] * draws[idx])
        proposal_method.process_new_ancestors(new_samp)
        rval.extend(new_samp)
        if not quiet:
            print(len(rval), "samples", file=sys.stderr)
        part = part_new

    return (np.array([s.sample for s in rval]), np.array([s.lpost for s in rval]))



def sample_lpost_based(num_samples, initial_particles, proposal_method, population_size = 20):
    rval = []
    anc = proposal_method.process_initial_samples(initial_particles)
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples:
        ancest_dist = np.array([a.lpost for a in anc])
        ancest_dist = categorical(ancest_dist - logsumexp(ancest_dist), p_in_logspace = True)
        
        #choose ancestor uniformly at random from previous samples
        pop = [proposal_method.gen_proposal(anc[ancest_dist.rvs()])
                for _ in range(population_size)]

        prop_w = np.array([s.lweight for s in pop])
        prop_w = exp(prop_w - logsumexp(prop_w))
        
        
        # Importance Resampling
        while True:
            try:
                draws = np.random.multinomial(population_size, prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
                
        for idx in range(len(draws)):
            rval.extend([pop[idx]] * draws[idx])
            anc.append(pop[idx])
    
    return (np.array([s.sample for s in rval]), np.array([s.lpost for s in rval]))



class PmcSample(object):    
    def __init__(self, ancestor = None, sample = None, lpost = None, lweight = None, other = {}, lprop = None):
        self.sample = sample
        self.lpost = lpost
        self.lprop = lprop
        self.ancestor = ancestor
        self.lweight = lweight
        self.other = other
        

class PmcProposalDistribution(object):
    
    def gen_proposal(self, ancestor = None):
        raise NotImplementedError()
    
    def process_initial_samples(self, samples):
        raise NotImplementedError()
    
    def process_new_ancestors(self, ancestors):
        raise NotImplementedError()
        

class NaiveRandomWalkProposal(PmcProposalDistribution):
    def __init__(self, lpost_func, proposal_dist):
        self.lpost = lpost_func
        self.pdist = proposal_dist


    def process_initial_samples(self, samples):
        return [PmcSample(sample = s, lpost = self.lpost(s)) for s in samples]    
        
        
    def gen_proposal(self, ancestor = None, mean = None):
        rval = PmcSample(ancestor)        
        if mean is None and ancestor is not None:
            if ancestor.sample is not None:
                mean = ancestor.sample
            else:
                mean = np.zeros(self.pdist.rvs().shape)

        step = self.pdist.rvs()
        rval.sample = mean + step
        rval.lpost = self.lpost(rval.sample)
        rval.lprop = self.pdist.logpdf(step)
        if rval.lpost is not None:
            rval.lweight = rval.lpost - rval.lprop
        return rval
    
    def process_new_ancestors(self, ancestors):
        pass


class InvWishartRandomWalkProposal(PmcProposalDistribution):
    def __init__(self, df, dim, lpost_func = lambda x: None):
        assert(df > dim + 1)
        self.df = df
        self.dim = dim
        self.lpost = lpost_func


    def process_initial_samples(self, samples):
        return [PmcSample(sample = s, lpost = self.lpost(s)) for s in samples]    
        
        
    def gen_proposal(self, ancestor = None, mean = None):
        assert((mean is not None and
                np.prod(mean.shape) == self.dim**2) or
               (ancestor is not None and
               ancestor.sample is not None and
               np.prod(ancestor.sample.shape) == self.dim**2))
        rval = PmcSample(ancestor)        
        if mean is None and ancestor is not None:
            if ancestor.sample is not None:
                mean = ancestor.sample
            else:
                mean = np.zeros(self.pdist.rvs().shape)
        scale_matr = mean * (self.df - self.dim - 1)
        pdist = invwishart(scale_matr, self.df)
        rval.sample = pdist.rv()
        rval.lpost = self.lpost(rval.sample)
        rval.lprop = pdist.logpdf(rval.sample)
        if rval.lpost is not None:
            rval.lweight = rval.lpost - rval.lprop
        else:
            rval.lweight = None
            
        return rval
    
    def process_new_ancestors(self, ancestors):
        pass  


class LatentClassProposal(PmcProposalDistribution):
    """
        Latent Class Proposal for 1-of-n coding
    """
    
    def __init__(self, lpost_func, dim):
        self.lpost = lpost_func
        self.dim = dim


    def process_initial_samples(self, samples):
        return [PmcSample(sample = s, lpost = self.lpost(s)) for s in samples]    
        
        
    def gen_proposal(self, ancestor = None):
        rval = PmcSample(ancestor)
        rval.sample = np.zeros(self.dim)
        lpost_vals = []
        for i in range(rval.sample.size):
            rval.sample.flat[i] = 1
            lpost_vals.append(self.lpost(rval.sample))
            rval.sample.flat[i] = 0
        lpost_vals = np.array(lpost_vals)
        lpost_norm = lpost_vals - logsumexp(lpost_vals)
        
        chosen = np.argmax(np.random.multinomial(1, exp(lpost_norm)))
        rval.sample.flat[chosen] = 1
        rval.lpost = lpost_vals[chosen]
        rval.lprop = lpost_norm[chosen]
        rval.lweight = rval.lpost - rval.lprop
        return rval
    
    def process_new_ancestors(self, ancestors):
        pass

        
   
class GrAsProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1, prop_mean_on_line = 0.5, main_var_scale = 1, other_var = 0.5):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.jump_dist = mvt([0]*dim, np.eye(dim)*5, dim)
        self.back_off_count = 0
        self.prop_mean_on_line = prop_mean_on_line
        self.main_var_scale = main_var_scale
        self.other_var = other_var

    
    def process_initial_samples(self, samples):
        s_lp_gr = [(s, self.lpost_and_grad(s)) for s in samples]
        return [PmcSample(sample = s, lpost = lp, other = {"gr":gr, "lrate":self.lrate})
                   for (s, (lp, gr)) in s_lp_gr]
        
    def gen_proposal(self, ancestor = None):
        assert(ancestor is not None)
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            print("jump")
            f = ancestor.other["lrate"]
            prop_dist = self.jump_dist
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            (f, theta_1, lpost_1, back_off_tmp)  = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["gr"], func = self.lpost)
            self.back_off_count += back_off_tmp
            step_mean = f * self.prop_mean_on_line * ancestor.other["gr"]
            cov = ideal_covar(step_mean, main_var_scale = self.main_var_scale, other_var = self.other_var) # , fix_main_var=1
            prop_dist = mvnorm(step_mean, cov)
        step = prop_dist.rvs()
        samp = ancestor.sample + step
        (lp, gr) = self.lpost_and_grad(samp)
        lprop =  prop_dist.logpdf(step)
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop,
                           lweight = lp - lprop,
                           other = {"lrate":f, "gr":gr})
        return rval
        
        
    def process_new_ancestors(self, ancestors):
        pass     



class ConGrAsProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.jump_dist = mvt([0]*dim, np.eye(dim)*5, dim)
        self.compute_conj_dir = lambda anc, current: max(0, np.float(current.sample.T.dot(current.sample - anc.sample) / anc.sample.T.dot(anc.sample)))
        self.back_off_count = 0


    def process_initial_samples(self, samples):
        s_lp_gr = [(s, self.lpost_and_grad(s)) for s in samples]
        return [PmcSample(sample = s, lpost = lp, other = {"gr":gr, "lrate":self.lrate, "conj":gr})
                         for (s, (lp, gr)) in s_lp_gr]

        
    def gen_proposal(self, ancestor = None):
        assert(ancestor is not None)
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            f = ancestor.other["lrate"]
            prop_dist = self.jump_dist
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            (f, theta_1, fval_1, back_off_tmp) = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["conj"], func = self.lpost)
            self.back_off_count += back_off_tmp
            step_mean = f * 0.5 * ancestor.other["conj"]
            cov = ideal_covar(step_mean, main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
            prop_dist = mvnorm(step_mean, cov)
          
        step = prop_dist.rvs()
        samp = ancestor.sample + prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        momentum = max(0, np.float(gr.T.dot(gr - ancestor.other["gr"]) / ancestor.other["gr"].T.dot(ancestor.other["gr"])))
        conj_dir_1 = gr + momentum * ancestor.other["conj"]
        lprop =  prop_dist.logpdf(step)
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop,
                           lweight = lp - prop_dist.logpdf(step),
                           other = {"lrate":f, "gr":gr, "conj":conj_dir_1})
                           
        return rval
        
    def process_new_ancestors(self, ancestors):
        pass   

