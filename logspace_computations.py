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

def logsubtrexp(a, c):
    d = c - a
    
    #switch sign for cases where c[i] > a[i]
    #see http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    d = d * ((d > 0 ) * -2 + 1)
    raise NotImplementedError("the line before this is incorrect")
    
    return a + log(1.-exp(d))

def logmeanexp(a, axis = None):
    a = np.array(a)
    if axis == None:
        norm = log(np.prod(a.shape))
    else:
        norm = log(a.shape[axis])
    return logsumexp(a, axis = axis) - norm

def logvarexp(a):
    raise NotImplementedError()
    a = np.array(a)
    m_ml = logmeanexp(a)
    return logsumexp(2 * logsubtrexp(a, m_ml)) - log(np.prod(a.shape) - 1)