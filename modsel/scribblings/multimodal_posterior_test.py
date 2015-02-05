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

from modsel.mc import mcmc
from modsel.mc import bijections



matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


delta = 0.025

lim = 10
X = np.arange(-10, 10, delta)
(X,Y) = np.meshgrid(X, X)
X = X
Y = Y

dims = 2


def mm_dens(x):
    norm_const = 1
    # Some multimodal density
    x = np.atleast_2d(x)
    uvn = stats.norm(0,1)
    mvn = stats.multivariate_normal(np.zeros(dims),5*np.eye(dims))
    rval = uvn.logcdf(np.sin(x.sum(1)))+ mvn.logpdf(x)
    return log(norm_const)+rval

def hole_dens(x):
    norm_const = 1
    # Some multimodal density
    x = np.atleast_2d(x)
    uvn = stats.beta(1,1)
    mvn = stats.multivariate_normal(np.zeros(dims),5*np.eye(dims))
    rval = uvn.logcdf(np.linalg.norm(x)**2)+ mvn.logpdf(x)
    return log(norm_const)+rval


def sample_params(num_samples):    
    (s, tr) = mcmc.sample(num_samples, -2*np.ones(dims), mcmc.ComponentWiseSliceSamplingKernel(mm_dens))

    return s

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



def apply_to_mg(func, *mg):
    x = np.vstack([e.flat for e in mg]).T
    return func(x).reshape(mg[0].shape)

s = sample_params(1000)


plt.figure()
CS = plt.contour(X,Y,exp(apply_to_mg(hole_dens, X,Y)))
plt.show()
plt.close()

print(mm_dens([[-2.1,-2.1], [0.8,0.8], [0,0]]))
