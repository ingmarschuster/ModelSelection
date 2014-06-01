# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:03:21 2014

@author: arbeit
"""
from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import scipy.stats as stats
import scipy.io
import sys
from sys import stderr,stdout,stdin
from scipy.misc import logsumexp
from numpy.random import permutation
from copy import copy,deepcopy
import logging
import pickle

import collections



def slice_sample_component(rv, idx, log_likelihood, prior, cur_ll = None, width = None, max_steps = 10):
    #rv: Matrix, deren Komponentnen gesampled werden sollen
    
    #idx: indices der Komponente, die gesampled werden sollen, für komponente 0,0: idx =([0],[0])
    
    #log_likelihood: Funktion ohne Parameter, aufrufbar als log_likelihood()
    ##class Model:
    #    self.bedeutung = np.array(.....)
    #    self.gewichte = np.array(....) #oder stats.norm(0,1).rvs((i,j))
    #    def sample_model(self, data):
    #        self.bedeutung = .... #irgend ein wert
    #        def ll_with_caching():
    #            return np.sum([stats.norm(self.bedeutung.dot(self.gewichte)[k,l], 5).logpdf(data[k,l]) for k in range(...) for l in range(...)]) + ... #irgendwas 
    #                   #python list comprehensions
    #        slice_sample_component(.., .., ll_with_caching, ...
    
    #prior: ein distributions-Objekt aus scipy.stats, z.B. stats.norm(mean,var) (alle prior gleich)
    
    #width: suchbreite
    
    
    
    # returns new log likelihood
    
    log = logging.getLogger("sampling")
    
    
    if log_likelihood == None:
        #if we don't have data to evaluate the likelihood, just sample from the prior
        log.debug("No data, sampling from prior")
        rv[idx] = prior.rvs()
    else:
        cur = rv[idx]
        if cur_ll == None:
            cur_ll = log_likelihood()
        cur_log_post = cur_ll + prior.logpdf(cur)
        
        if width == None:
            try:
                width = prior.var()
            except:
                # prior might not have the var() method
                width = 1.
        left = rv[idx] -  npr.uniform(0, width)
        right = left + width
        
        rv[idx] = left                    
        left_log_post = log_likelihood() + prior.logpdf(left)
        rv[idx] = right
        right_log_post = log_likelihood() + prior.logpdf(right)

        aux = cur_log_post + np.log(npr.rand(1)) #auxiliary variable called 'u' in Bishops "Pattern Recognition and Machine Learning", Section 11.4
        while True:
            if right_log_post > aux and max_steps > 0:
                right += width
                rv[idx] = right
                right_log_post = log_likelihood() + prior.logpdf(right)
                log.debug("Right bound below posterior, resampling as " + str((right)) +".")
                max_steps -= 1
                continue
            if left_log_post > aux and max_steps > 0:
                left -= width
                rv[idx] = left
                left_log_post = log_likelihood() + prior.logpdf(left)
                log.debug("Left bound below posterior, resampling as " + str((left)) +".")
                max_steps -= 1
                continue
            candidate = left + npr.rand(1) * (right - left)
            rv[idx] = candidate
            cur_ll = log_likelihood()
            cand_log_post = cur_ll + prior.logpdf(candidate)

            if (cand_log_post > aux) or (cur == candidate and abs(left - right) < 10**(-7)):
                #(cand_log_post > aux): candidate within slice, found our new value.
                #(cur == candidate and abs(left - right) < 10**(-9)): prior and posterior are (almost) the same
                #       this might occur when data likelihood doesn't favour any value and we chose aux close to the top
                rv[idx] = candidate
                log.debug("Found new sample value at " + str((candidate)))
                return cur_ll
            else:
                #candidate outside slice
                if candidate < cur:
                    left = candidate
                    left_log_post = cand_log_post
                    log.debug("Candidate outside slice, cutting left bound to " + str((candidate)))
                elif candidate > cur:
                    right = candidate
                    right_log_post = cand_log_post
                    log.debug("Candidate outside slice, cutting right bound to " + str((candidate)))
                else:
                    log.info("Slice sampler shrank too far.", file=sys.stderr)
                    rv[idx] = candidate
                    return cur_ll

def slice_interval_doubling(rv, idx, aux, log_likelihood, prior, max_power = 15, width = None):
    cur = rv[idx]
    
    if width == None:
        try:
            width = prior.var()
        except:
            # prior might not have the var() method
            width = 10.
    left = cur - width * stats.uniform(0,1).rvs()
    rv[idx] = left
    left_log_post = log_likelihood()
    
    right = left + width
    rv[idx] = right
    right_log_post = log_likelihood()
    
    while max_power > 0 and (aux < left_log_post or aux < right_log_post):
        if stats.bernoulli.rvs(0.5):
            left = left - (right - left)
            rv[idx] = left
            left_log_post = log_likelihood()
        else:
            right = right + (right - left)
            rv[idx] = right
            right_log_post = log_likelihood()
        max_power -= 1
    return (left, right)
    
    
def slice_acceptable_doubling(rv, idx, cur, cand, aux, left, right, log_likelihood, prior, width):
    assert(left < right)
    different_interval = False
    rv[idx] = right
    right_log_post = log_likelihood()
    rv[idx] = left
    left_log_post = log_likelihood()
    while right - left > 1.1*width:
        mid = (left + right) / 2
        if ((cur <  mid and cand >= mid) or
            (cur >= mid and cand <  mid)):
            different_interval = True
        if cand < mid:
            right = mid
            rv[idx] = right
            right_log_post = log_likelihood()
        else:
            left = mid
            rv[idx] = left
            left_log_post = log_likelihood()
        if different_interval and aux >= left_log_post and aux >= right_log_post:
            return False
    return True

def slice_sample_component_double(rv, idx, log_likelihood, prior, cur_ll = None, width = None):
    #rv: Matrix, deren Komponentnen gesampled werden sollen
    
    #idx: indices der Komponente, die gesampled werden sollen, für komponente 0,0: idx =([0],[0])
    
    #log_likelihood: Funktion ohne Parameter, aufrufbar als log_likelihood()
    ##class Model:
    #    self.bedeutung = np.array(.....)
    #    self.gewichte = np.array(....) #oder stats.norm(0,1).rvs((i,j))
    #    def sample_model(self, data):
    #        self.bedeutung = .... #irgend ein wert
    #        def ll_with_caching():
    #            return np.sum([stats.norm(self.bedeutung.dot(self.gewichte)[k,l], 5).logpdf(data[k,l]) for k in range(...) for l in range(...)]) + ... #irgendwas 
    #                   #python list comprehensions
    #        slice_sample_component(.., .., ll_with_caching, ...
    
    #prior: ein distributions-Objekt aus scipy.stats, z.B. stats.norm(mean,var) (alle prior gleich)
    
    #width: suchbreite
    
    
    
    # returns new log likelihood
    
    log = logging.getLogger("sampling")
    
    
    if log_likelihood == None:
        #if we don't have data to evaluate the likelihood, just sample from the prior
        log.info("No data, sampling from prior")
        rv[idx] = prior.rvs()
    else:
        find_interval = slice_interval_doubling
        acceptable_candidate= slice_acceptable_doubling
        cur = rv[idx]
        if cur_ll == None:
            cur_ll = log_likelihood()
        cur_log_post = cur_ll + prior.logpdf(cur)
        
        if width == None:
            try:
                width = prior.var()
            except:
                # prior might not have the var() method
                width = 100.
        
        aux = cur_log_post - stats.expon.rvs() #p.712 in Neal (2003), "- stats.expon.rvs()" same as "+ np.log(npr.rand(1))"
        (left, right) = find_interval(rv, idx, aux, log_likelihood, prior, max_power = 10, width = width)
        

        while True:
            candidate = stats.uniform(left, (right - left)).rvs()
            rv[idx] = candidate
            cur_ll = log_likelihood()
            cand_log_post = cur_ll + prior.logpdf(candidate)

            if (cand_log_post > aux) or (cur == candidate and abs(left - right) < 10**(-7)):
                #(cand_log_post > aux): candidate within slice, found our new value.
                #(cur == candidate and abs(left - right) < 10**(-9)): prior and posterior are (almost) the same
                #       this might occur when data likelihood doesn't favour any value and we chose aux close to the top
                if acceptable_candidate(rv, idx, cur, candidate, aux, left, right, log_likelihood, prior, width):
                    rv[idx] = candidate
                    log.debug("Found new sample value at " + str((candidate)))
                    return cur_ll
            #candidate outside slice or not accepted
            if candidate < cur:
                left = candidate
                left_log_post = cand_log_post
                log.debug("Candidate outside slice, cutting left bound to " + str((candidate)))
            elif candidate > cur:
                right = candidate
                right_log_post = cand_log_post
                log.debug("Candidate outside slice, cutting right bound to " + str((candidate)))
            else:
                log.info("Slice sampler shrank too far.", file=sys.stderr)
                rv[idx] = candidate
                return cur_ll


def slice_sample_all_components(rv, log_likelihood, prior, width = None):
    if len(rv.shape) > 1:
        rv = rv.flat
    log = logging.getLogger("sampling")
    """Slice sample components of rv according to a single 'prior' over all components or a prior for each component in 'prior_list'.
       'rv' is expected to be a flat numpy array (ie as returned by the .flat property of an array).
       if 'log_likelihood' is None, it's expected to be equal everywhere and the methods just samples from the prior."""
    if isinstance(prior, collections.Sequence):
        if len(prior) == 1:
            prior_list = [prior[0]] * len(rv)
        elif len(prior) == len(rv):
            prior_list = prior
        else:
            raise IndexError("Expected either one prior for all rv or one for each")
    else:
        prior_list = [prior] * len(rv)
        
    if log_likelihood != None:
        cur_ll = log_likelihood()
    else:
        cur_ll = None
        
    for i in npr.permutation(len(rv)):
        pre_ll = cur_ll
        i_pre = rv[i]
        cur_ll = slice_sample_component(rv,
                                        i,
                                        log_likelihood,
                                        prior_list[i],
                                        cur_ll = cur_ll,
                                        width=width)
        #print( pre_ll,"  ", i_pre, "->", cur_ll,"  ", rv[i], file=sys.stderr)
        

def slice_sample_all_components_optimized(rv_mdl, glob_mdl, data, prior, rows = None, cols = None , width = None):
    log = logging.getLogger("sampling")
    rv = rv_mdl.get()
    
    if isinstance(prior, collections.Sequence):
        if len(prior) == 1:
            prior_list = [prior[0]] * len(rv.flat)
        elif len(prior) == len(rv.flat):
            prior_list = prior
        else:
            raise IndexError("Expected either one prior for all rv or one for each")
    else:
        prior_list = [prior] * len(rv.flat)
        
    cur_ll = glob_mdl.log_likelihood(data)
    
    if rows == None:
        rows = npr.permutation(rv.shape[0])
    if cols == None:
        cols = npr.permutation(rv.shape[1])
    
    for row in rows:
        for col in cols:
            log_likelihood = glob_mdl.llike_function(data, rv_mdl, (row, col))
            idx = np.ravel_multi_index((row, col), rv.shape)
            cur_ll =  slice_sample_component(rv.flat,
                                             idx,
                                             log_likelihood,
                                             prior_list[idx],
                                             cur_ll,
                                             width)
