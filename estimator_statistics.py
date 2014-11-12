# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:56:05 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats

def log_sign(a):    
    sign_indicator = ((a < 0 ) * -2 + 1)
    return (log(np.abs(a)), sign_indicator)

def exp_sign(a, sign_indicator):
    return exp(a) * sign_indicator

def logsubtrexp(a, c):
    d = c - a
    
    sign_indicator = ((d > 0 ) * -2 + 1)
    
    return (a + log(1.-exp(d *  sign_indicator)), -sign_indicator)

def logabssubtrexp(a, c):
    return logsubtrexp(a, c)[0]

def logmeanexp(a, sign_indicator = None, axis = None):    
    a = np.array(a)
    if axis is None:
        norm = log(np.prod(a.shape))
    else:
        norm = log(a.shape[axis])
        
    if sign_indicator is None:
        return logsumexp(a, axis = axis) - norm
    else:
        pos_sum = logsumexp(a * (sign_indicator == 1), axis = axis)
        neg_sum = logsumexp(a * (sign_indicator == -1), axis = axis)
        (res, signs) =  logsubtrexp(pos_sum, neg_sum)
        res = res - norm
        return (res, signs)

def logvarexp(a):
    raise NotImplementedError()
    a = np.array(a)
    m_ml = logmeanexp(a)
    return logsumexp(2 * logsubtrexp(a, m_ml)) - log(np.prod(a.shape) - 1)
    

def log_abs_bias(log_true_theta, log_estimates):
    assert(log_estimates.shape[0] == log_true_theta.shape[0] and
           log_estimates.shape[0] == np.prod(log_true_theta.shape))
    true_tiled = log_true_theta.reshape((np.prod(log_true_theta.shape), 1))
    true_tiled = np.tile(true_tiled, (1, log_estimates.shape[1]))
    (diff, sign) = logsubtrexp(log_estimates, true_tiled)
    return logmeanexp(diff, sign, 0)

def log_bias_sq(log_true_theta, log_estimates):
    return 2*log_abs_bias(log_true_theta, log_estimates)[0]
    
    
    
    