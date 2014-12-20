# -*- coding: utf-8 -*-
"""
Proposal classes for PMC
Created on Thu Dec 11 09:24:27 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp

from copy import copy, deepcopy


from modsel.distributions import mvnorm, mvt, categorical
from modsel.mc.gs_basis import ideal_covar
from modsel.mc.optimization import find_step_size


def sample(num_samples, initial_particles, proposal_method, population_size = 20):
    rval = proposal_method.process_initial_samples(initial_particles)
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples:
        
        ancest_dist = categorical([1./len(rval)] * len(rval))
        
        #choose ancestor uniformly at random from previous samples
        pop = [proposal_method.gen_proposal(rval[ancest_dist.rvs()])
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
    
    return (np.array([s.sample for s in rval[num_initial:]]), np.array([s.lpost for s in rval[num_initial:]]))


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
    def __init__(self, ancestor = None, sample = None, lpost = None, lweight = None, other = {}):
        self.sample = sample
        self.lpost = lpost
        self.ancestor = ancestor
        self.lweight = lweight
        self.other = other
        

class PmcProposalDistribution(object):
    
    def gen_proposal(self, ancestor = None):
        raise NotImplementedError()
    
    def process_initial_samples(self, samples):
        raise NotImplementedError()
        

class NaiveRandomWalkProposal(PmcProposalDistribution):
    def __init__(self, lpost_func, proposal_dist):
        self.lpost = lpost_func
        self.pdist = proposal_dist


    def process_initial_samples(self, samples):
        return [PmcSample(sample = s, lpost = self.lpost(s)) for s in samples]    
        
        
    def gen_proposal(self, ancestor = None):
        rval = PmcSample(ancestor)
        dim = ancestor.sample.size
        self.pdist.set_mu(ancestor.sample)
        rval.sample = self.pdist.rvs()
        rval.lpost = self.lpost(rval.sample)
        rval.lweight = rval.lpost - self.pdist.logpdf(rval.sample)
        return rval




class GradientAscentProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.far_jump_dist = mvt([0]*dim, np.eye(dim)*100, dim)
        self.near_jump_dist = mvt([0]*dim, np.eye(dim)*3, dim)
        

    
    def process_initial_samples(self, samples):
        s_lp_gr = [(s, self.lpost_and_grad(s)) for s in samples]
        return [PmcSample(sample = s, lpost = lp, other = {"gr":gr, "lrate":self.lrate})
                   for (s, (lp, gr)) in s_lp_gr]

        
    def gen_proposal(self, ancestor = None):
        assert(ancestor is not None)
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            if stats.bernoulli.rvs(0.1):
                # take a random step with large variance
                f = self.lrate
                prop_dist = self.far_jump_dist
            else:
                # take a random step with small variance
                #(stay in region of high posterior probability)
                f = ancestor.other["lrate"]
                prop_dist = self.near_jump_dist
            prop_dist.set_mu(ancestor.sample)
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            (f, theta_1, lpost_1, foo)  = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["gr"], func = self.lpost)
            step_mean = f * 0.5 * ancestor.other["gr"]
            cov = ideal_covar(step_mean, main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
            prop_dist = mvnorm(ancestor.sample + step_mean, cov)
                    
        samp = prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lweight = lp - prop_dist.logpdf(samp),
                           other = {"lrate":f, "gr":gr})
        return rval



class ConjugateGradientAscentProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.far_jump_dist = mvt([0]*dim, np.eye(dim)*100, dim)
        self.near_jump_dist = mvt([0]*dim, np.eye(dim)*3, dim)
        self.compute_conj_dir = lambda anc, current: max(0, np.float(current.sample.T.dot(current.sample - anc.sample) / anc.sample.T.dot(anc.sample)))


    def process_initial_samples(self, samples):
        s_lp_gr = [(s, self.lpost_and_grad(s)) for s in samples]
        return [PmcSample(sample = s, lpost = lp, other = {"gr":gr, "lrate":self.lrate, "conj":gr})
                         for (s, (lp, gr)) in s_lp_gr]
        

        
    def gen_proposal(self, ancestor = None):
        assert(ancestor is not None)
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            if stats.bernoulli.rvs(0.1):
                # take a random step with large variance
                f = self.lrate
                prop_dist = self.far_jump_dist
            else:
                # take a random step with small variance
                #(stay in region of high posterior probability)
                f = ancestor.other["lrate"]
                prop_dist = self.near_jump_dist
            prop_dist.set_mu(ancestor.sample)
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            (f, theta_1, fval_1, back_off_tmp) = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["conj"], func = self.lpost)
            step_mean = f * 0.5 * ancestor.other["conj"]
            cov = ideal_covar(step_mean, main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
            prop_dist = mvnorm(ancestor.sample + step_mean, cov)
            
        samp = prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        momentum = max(0, np.float(gr.T.dot(gr - ancestor.other["gr"]) / ancestor.other["gr"].T.dot(ancestor.other["gr"])))
        conj_dir_1 = gr + momentum * ancestor.other["conj"]
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lweight = lp - prop_dist.logpdf(samp),
                           other = {"lrate":f, "gr":gr, "conj":conj_dir_1})
                           
        return rval


