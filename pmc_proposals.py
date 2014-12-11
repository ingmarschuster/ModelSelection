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


from distributions import mvnorm, mvt, categorical
from distributions.linalg import pdinv
from synthdata import simple_gaussian
import slice_sampling
from gs_basis import ideal_covar
from optimization import find_step_size

from scipy import optimize


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
            (f, theta_1, lpost_1, foo)  = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["gr"], self.lpost_and_grad)
            cov = ideal_covar(f * 0.5 * ancestor.other["gr"], main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
            prop_dist = mvnorm(ancestor.sample + f * 0.5 * ancestor.other["gr"], cov)
                    
        samp = prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lweight = lp - prop_dist.logpdf(samp),
                           other = {"lrate":f, "gr":gr})
        return rval