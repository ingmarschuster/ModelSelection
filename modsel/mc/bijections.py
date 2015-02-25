from __future__ import absolute_import,print_function, division

import numpy as np
import scipy.stats as stats
from math import log, exp



relu = np.vectorize(lambda x:max(0, x))  

def int_logistic(x):
    #avoid overflows
    if x >= 37:
        return 1.
    elif x <= -710:
        return 0.
    else:
        return 1./(1.+exp(-x))

logistic = np.vectorize(int_logistic)

def int_invlogistic(y):
    if y <= 0. or y >= 1.:
        raise ValueError("inverse logistic function defined only for 0.< y < 1.")
    return -log(1. / y - 1.)

def int_invlogistic_perturb(y):
    if y < 0. or y > 1.:
        raise ValueError("inverse logistic function defined only for 0.< y < 1.")
    elif y == 0.:
        return -36.5
    elif y == 1.:
        return 36.5
    return -log(1. / y - 1.)


invlogistic = np.vectorize(int_invlogistic_perturb)

logistic.i = invlogistic
invlogistic.i = logistic

def int_softplus(x):
    if x >= 34:
        # in this case, limitations in floating-point
        # precision result in log(exp(y) - 1) == y
        return x
    elif x <= -37:
        # this also results from precision limits
        return 10**-38
    else:
        return log(1 + exp(x))
        
softplus = np.vectorize(int_softplus)

def int_invsoftplus(y):
    y = float(y)
    if y >= 34:
        # in this case, limitations in floating-point
        # precision result in log(exp(y) - 1) == y
        return y 
    elif y < 0:
        raise ValueError("Function defined only for y >= 0")
    elif y == 0:
        #perturb input so as not to return -inf
        y += stats.gamma(100,scale=0.00001).rvs()
    return log(exp(y) - 1)
    
invsoftplus = np.vectorize(int_invsoftplus)

softplus.i = invsoftplus
invsoftplus.i = softplus

def composed_bijection(components, apply_components_to_rows = False):
    n_comp = len(components)
    
    def apply_comp_closure(comp_funcs):
        def closure(matr):
            matr = np.array(matr)
            if apply_components_to_rows == False:
                matr = matr.T
            assert(   (len(matr.shape) == 1 and matr.shape[0] == n_comp)
                   or (len(matr.shape) == 2 and matr.shape[1] == n_comp))
            rval = np.vstack([comp_funcs[j](matr[j]) for j in range(n_comp)])
            if apply_components_to_rows == False:
                return rval.T
            else:
                return rval
                
        return closure
    
    
    
    forw = apply_comp_closure(components)            
    forw.i = apply_comp_closure([c.i for c in components])
    
    return forw
