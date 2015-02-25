# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:52:11 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

from distributions import categorical


__all__ = ["ProposalOracle", "CategoricalOracle"]

class ProposalOracle(object):
    def observe(self, population):
        raise NotImplementedError()
        
    def gen_proposal(self, ancestor = None):
        raise NotImplementedError()

class CategoricalOracle(object):
    def __init__(self, *predefined_proposals):
        """
        Choose proposals to decrease variance of weights
        
        Parameters
        ==========
        predefined_proposals: a set of given proposals among which to choose
        """
        
        length = len(predefined_proposals)
        self.prop2idx = {}
        self.idx2prop = []
        for i in range(len(predefined_proposals)):
            self.prop2idx[predefined_proposals[i]] = i
            self.idx2prop.append(predefined_proposals[i])
        self.num_samp = np.zeros(length) # number of samples for weights
        self.sum = -np.inf*np.ones(length) # sum of weights
        self.sqr_sum = -np.inf*np.ones(length) # sum of squares
        self.var = np.zeros(length) # weight variance estimate

        self.prop_dist = categorical(np.array([1./length] * length))
    
    def set_lpost(self, func):
        for prop in self.idx2prop:
            prop.lpost = func
            
    def set_lpost_and_grad(self, func):
        for prop in self.idx2prop:
            prop.lpost_and_grad = func
        
    def observe(self, population):
        lweights = np.array([s.lweight for s in population])
        #print(lweights)
        lweights = lweights - logsumexp(lweights) #+ 1000
        #print(lweights)
        indices = np.array([self.prop2idx[s.prop_obj] for s in population])
        for i in range(len(lweights)):
            prop_idx = indices[i]
            self.num_samp[prop_idx] = self.num_samp[prop_idx] + 1
            self.sum[prop_idx] = logsumexp((self.sum[prop_idx], lweights[i]))
            self.sqr_sum[prop_idx] = logsumexp((self.sqr_sum[prop_idx], 2*lweights[i]))
        lnum_samp = log(self.num_samp)
        self.var = exp(logsumexp([self.sum, self.sqr_sum - lnum_samp], 0) - lnum_samp)
        #self.var = exp(self.var - logsumexp(self.var))
        
        if self.var.size > 1:
            tmp = self.var.sum()
            if tmp == 0 or np.isnan(tmp):
                prop_prob = np.array([1./self.var.size] * self.var.size)
            else:
                prop_prob = (self.var.sum() - self.var)
                prop_prob = prop_prob/prop_prob.sum()/2 + np.random.dirichlet(1 + self.num_samp)/2
        else:
            prop_prob = np.array([1./self.var.size] * self.var.size)
        self.prop_dist = categorical(prop_prob)
    
    def gen_proposal(self, ancestor = None):
        return self.idx2prop[self.prop_dist.rvs()].gen_proposal(ancestor)