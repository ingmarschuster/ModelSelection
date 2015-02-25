# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:33:04 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv
import time


class Flag(object):
    def stop(self):
        raise(NotImplementedException())

class NeverStopFlag(Flag):
    def stop(self):
        return False


class ProcessorTimeFlag(Flag):
    def __init__(self, max_lhood = np.inf, max_grad = np.inf, max_both = np.inf):
        self.max_lhood = max_lhood
        self.max_grad = max_grad
        self.max_both = max_both
        self.maxtime = np.inf
        self.reset()
    
    def reset(self):
        self.start = time.clock()    
        self.lhood = 0
        self.grad = 0
    
    def elapsed(self):
        return (time.clock() - self.start)
    
    def set_max_time(self):
        self.max_lhood = self.lhood
        self.maxtime =  time.clock() - self.start 
    
    def inc_lhood(self):
        self.lhood += 1
        
    def inc_grad(self):
        self.grad += 1
        
    def stop(self):
        return self.elapsed() >= self.maxtime or self.lhood > self.max_lhood


class LikelihoodEvalsFlag(Flag):
    def __init__(self, max_lhood = np.inf, max_grad = np.inf, max_both = np.inf):
        self.max_lhood = max_lhood
        self.max_grad = max_grad
        self.max_both = max_both
        self.reset()
    
    def reset(self):        
        self.lhood = 0
        self.grad = 0
    
    def max_both_from_current_counts(self):
        self.max_both = self.lhood + self.grad
    
    def inc_lhood(self):
        self.lhood += 1
        
    def inc_grad(self):
        self.grad += 1
        
    def stop(self):
        return ( self.grad >= self.max_grad or
                 self.lhood >= self.max_lhood or
                (self.grad + self.lhood) >= self.max_both)