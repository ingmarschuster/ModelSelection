from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.random as npr

from numpy import exp, log
from numpy.linalg import inv, cholesky, det
from scipy.special import multigammaln
from scipy.stats import chi2
import scipy.stats as stats

from ..estimator_statistics import logsubtrexp


def is_equal(a, b, tolerance = 1e-14):
    return (np.abs(a - b) <= tolerance).all()

def assert_equal(a, b, tolerance = 1e-14):
    assert(is_equal(a, b, tolerance = tolerance))

def log_is_equal(a, b, sign_a = None, sign_b = None, tolerance = -23):
    return (logsubtrexp(a, b, sign_minuend = sign_a, sign_subtrahend = sign_b) <= tolerance).all()