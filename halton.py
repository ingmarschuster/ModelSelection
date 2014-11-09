from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats




def sequence(n=None,b=None):

    hs = np.zeros(n)
    for idx in range(n):
        hs[idx] = single(idx + 1, b)
    return hs
    
def single(n=None, b=None):
    n0 = n
    hn = 0
    f = 1. / b
    while (n0 > 0):
        n1 = np.floor(n0 / b)
        r = n0 - n1 * b
        hn = hn + f * r
        f = f / b
        n0 = n1
    return hn