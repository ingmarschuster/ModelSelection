# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:58:18 2014

@author: Ingmar Schuster
"""

import numpy as np
import slice_sampling


def sample(num_samples, initialization, markov_kernel):
    rval = []
    theta = np.copy(initialization)
    for i in range(num_samples):        
        rval.append(markov_kernel.step(theta))
    #assert()
    return np.array(rval)


class MarkovKernel(object):
    def step(self, samples):
        raise NotImplementedError()


class ComponentWiseSliceSamplingKernel(MarkovKernel):
    def __init__(self, lpost_func):
        class dummy_prior(object):
            def logpdf(self, x):
                return 0
                
        self.lpost = lpost_func
        self.pr = dummy_prior()
        
    def step(self, theta):
        tc = theta.copy()
        rval = slice_sampling.slice_sample_all_components_mvprior(tc, lambda:self.lpost(tc), self.pr)
        #assert ()
        return rval