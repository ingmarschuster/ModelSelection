
from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp

from copy import copy, deepcopy


from distributions import mvnorm, mvt
from distributions.linalg import pdinv
from synthdata import simple_gaussian
import slice_sampling
from gs_basis import ideal_covar

from scipy import optimize

dim = 6
num_obs = 1000

theta = np.array([4] * dim) #np.array([-4.90095387,  8.31233585]) #np.array([4] * 2)

obs_K = np.atleast_2d(np.eye(np.prod(dim))) *10# np.array([[ 1.93563577,  0.10419466], [ 0.10419466,  0.59682879]]) #np.eye(np.prod(theta.shape))*4
obs_Ki = np.linalg.inv(obs_K)
obs = stats.multivariate_normal.rvs(theta, obs_K, size=num_obs).reshape((num_obs, dim))

obs_m = obs.mean(0)

def llhood(theta):
    return stats.multivariate_normal.logpdf(np.atleast_2d(obs), np.array(theta), obs_K).sum()

def llhood_grad(theta):
    #assert()
    return obs_Ki.dot((np.atleast_2d(obs) - np.array(theta)).T).sum(1).T

print(optimize.check_grad(llhood, llhood_grad, obs_m), optimize.check_grad(llhood, llhood_grad, obs_m+100), llhood_grad(obs_m+100))
th2 = np.zeros(theta.shape)
a=optimize.minimize( lambda th:-llhood(th), th2,method="BFGS", jac = lambda th:-llhood_grad(th))
print(theta, th2,a)
