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
from modsel.estimator_statistics import *

def test_log_subtr_add_exp():    
    for (m, s) in [(np.array([[-10.,   3.],[ -1.,   5.]]), np.array([-5.5,  4. ])),
                   (np.array([[10.,   -3.],[ 1.,   -5.]]), np.array([5.5,  -4. ])),
                   (3, 10),
                   (3, 0),
                   (-30, 0), 
                   (np.arange(3).astype(float), 1),
                   ((np.arange(12).reshape((2,2,3)) - 4).astype(float),1)]:
        (lm, sm) = log_sign(m)
        (ls, ss) = log_sign(s)
        (subtr, subtr_s) = logsubtrexp(lm, ls, sm, ss)
        (add, add_s) = logaddexp(lm, ls, sm, ss)
        assert((np.abs((m - s) - exp_sign(subtr, subtr_s)) < 1e-14).all())
        assert((np.abs((m + s) - exp_sign(add, add_s)) < 1e-14).all())

def test_logmeanexp():
    b = np.array([[-10.22911746,   3.68323883],[ -0.41504275,   5.68779   ]])

    for a in (np.array([[ 6.5,  1. ],[ 2.5,  3. ]]),
              np.array([(-10, 3), (-1, 5)]),
              np.arange(4).reshape(2, 2) - 2,
              stats.norm.rvs(0,10, (2,2))):
        (la, sa) = log_sign(a)
        for ax in range(2):
            (abs_, sign_) = logmeanexp(la, sa, ax)
            
            assert(abs_.shape[ax] ==  1)
            assert((np.abs(a.mean(ax).flat[:] - exp_sign(abs_, sign_).flat[:]) < 1e-10).all())

def test_logvarexp():
    b = np.array([[-10.22911746,   3.68323883],[ -0.41504275,   5.68779   ]])

    for a in (np.array([(-10, -11), (1, 2)]).T,
              np.array([(-10, 3), (-1, 5)]),
              np.arange(4).reshape(2, 2) - 2,
              stats.norm.rvs(0,10, (2,2))
              ):
        (la, sa) = log_sign(a)
        for ax in range(2):
            lvar = logvarexp(la, sa, ax).flatten()
            var = np.var(a, ax)
            assert((np.abs(var - exp(lvar)) < 1e-10).all()) 
    

def test_log_mse_bias_sq_exp():
    for (estim, truth)  in [(np.array([(-10, -11), (1, 2)]).T , np.array([(-10, 1.5)])),
                            (stats.norm.rvs(0,10, (5,3)), stats.norm.rvs(0, 10, (1,3))),
                            (stats.norm.rvs(0,55, (10,2)), stats.norm.rvs(0, 55, (1,2)))]:
        (le, se) = log_sign(estim)
        (lt, st) = log_sign(truth)
        for ax in range(1):
            lmse = logmseexp(lt, le, signs_truth = st, signs_estim = se, axis = ax)
            mse = ((estim - truth)**2).mean(axis = ax)
            assert((np.abs(mse - exp(lmse)) < 1e-10).all())
            lbias = logbias2exp(lt, le, signs_truth = st, signs_estim = se, axis = ax)
            bias = (estim - truth).mean(axis = ax)**2
            assert((np.abs(bias - exp(lbias)) < 1e-10).all())
            lvar = logvarexp(le, sign_indicator = se, axis = ax)
            assert((np.abs(logaddexp(lbias, lvar)[0] - lmse) < 1e-10).all())