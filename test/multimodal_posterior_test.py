# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:26:44 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp

import matplotlib

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


delta = 0.025

lim = 5
X = np.arange(-lim, lim, delta)
(X,Y) = np.meshgrid(X, X)
X = X
Y = Y







if False:

    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')


def mm_dens(x):
    uvn = stats.norm(0,1)
    mvn = stats.multivariate_normal(np.zeros(2),5*np.eye(2))
    return exp(uvn.logcdf(np.sin(x.sum(1)))+mvn.logpdf(x))

def apply_to_mg(func, *mg):
    x = np.vstack([e.flat for e in mg]).T
    return func(x).reshape(mg[0].shape)

plt.figure()
CS = plt.contour(X,Y,apply_to_mg(mm_dens, X,Y))
