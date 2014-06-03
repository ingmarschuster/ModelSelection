from __future__ import print_function

import numpy
import scipy.stats as stat
from math import log, exp



relu = numpy.vectorize(lambda x:max(0, x))  

def int_logistic(x):
    #avoid overflows
    if x >= 37:
        return 1.
    elif x <= -710:
        return 0.
    else:
        return 1./(1.+exp(-x))

logistic = numpy.vectorize(int_logistic)

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


invlogistic = numpy.vectorize(int_invlogistic_perturb)

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
        
softplus = numpy.vectorize(int_softplus)

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
        y += stat.gamma(100,scale=0.00001).rvs()
    return log(exp(y) - 1)
invsoftplus = numpy.vectorize(int_invsoftplus)
