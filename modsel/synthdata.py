# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:24:23 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats
from modsel.distributions import norm_invwishart, invwishart_rv, mvnorm
from modsel.mixture import GMM

def simple_gaussian(dims = 1, observations_range = range(10,101,10), num_datasets = 10, cov_var_const = 4):
    ds = {}
    for nobs in observations_range:
        ds[nobs] = []
        for n in range(num_datasets):
            m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*100)
            K = invwishart_rv(np.eye(dims) * cov_var_const, dims + 2)
            ds[nobs].append({"params":(m, K),
                            "obs":stats.multivariate_normal.rvs(m, K, size=nobs).reshape((nobs, dims))})
    return ds


def gen_gauss_lpost(num_datasets, dims, ev_params = [(80, 10), (40,10)], cov_var_const = 4):
    def gen_lp_unnorm_ev(lev, distr_norm):
        print(distr_norm.mu, distr_norm.K)
        rval = lambda x:distr_norm.logpdf(x) + lev
        rval.log_evidence = lev
        return (rval, lev)
        
    rval = []
    for ep in ev_params:
        lev_distr = stats.gamma(ep[0], scale=ep[1])
        for i in range(int(num_datasets//len(ev_params))):
            while True:
                try:
                    m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*1000)
                    K = invwishart_rv(np.eye(dims) * cov_var_const, dims + 1 )
                    val = gen_lp_unnorm_ev(-lev_distr.rvs(), mvnorm(m, K))
                    val.mean = m
                    val.cov = K
                    rval.append(val)
                    break
                except np.linalg.LinAlgError:
                    import sys
                    #The Matrix from the niw was not invertible. Try again.
                    print("np.linalg.LinAlgError - trying again", file=sys.stderr)
                    pass
            
    return rval


def gen_gauss_diag_lpost(num_datasets, dims, ev_params = [(80, 10), (40,10)], cov_var_const = 4):
    def gen_lp_unnorm_ev(lev, distr_norm):
        print(distr_norm.mu, distr_norm.K)
        rval = lambda x:distr_norm.logpdf(x) + lev
        rval.log_evidence = lev
        return (rval, lev)
        
        
    rval = []
    for ep in ev_params:
        lev_distr = stats.gamma(ep[0], scale=ep[1])
        for i in range(int(num_datasets//len(ev_params))):
            while True:
                try:
                    m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*1000)
                    K = np.eye(dims)
                    val = gen_lp_unnorm_ev(-lev_distr.rvs(), mvnorm(m, K))
                    val.mean = m
                    val.cov = K
                    rval.append(val)
                    break
                except np.linalg.LinAlgError:
                    import sys
                    #The Matrix from the niw was not invertible. Try again.
                    print("np.linalg.LinAlgError - trying again", file=sys.stderr)
                    pass
            
    return rval


## Multimodal density, see Lemma 3 of Azzalini 2005, "The Skew-normal Distribution and Related Multivariate Families" ##

def gen_mm_lpost(num_datasets,num_modes, dims, ev_params = [(80, 10), (40,10)], cov_var_const = 1.5):
    def gen_lp_unnorm_ev(lev, mixt):
        rval = lambda x:mixt.logpdf(x) + lev
        rval.log_evidence = lev
        return (rval, lev)
        
    rval = []
    for ep in ev_params:
        lev_distr = stats.gamma(ep[0], scale=ep[1])
        for i in range(int(num_datasets//len(ev_params))):
            mode_p = np.random.dirichlet([100] * num_modes)
            mode_d = []
            m = stats.multivariate_normal.rvs([0] * dims, np.eye(dims)*10)
            while True:
                try:
                    K = invwishart_rv(np.eye(dims) * cov_var_const , dims)
                    print(K)
                    mode_mean_dist = stats.multivariate_normal(m, K)
                    break
                except:
                    pass
            
            
            while len(mode_d) != num_modes:
                try:                    
                    mode_d.append(mvnorm(mode_mean_dist.rvs(),
                                         invwishart_rv(K, dims)))
                except:
                    #The Matrix from the niw was not invertible. Try again.
                    pass
            mixt = GMM(num_modes, dims)
            mixt.comp_lprior = np.log(mode_p)
            mixt.comp_dist = mode_d
            rval.append(gen_lp_unnorm_ev(-lev_distr.rvs(), mixt))
    return rval
    
    

def gen_mm_dens(dims, cov_var_const = None, log_norm_const = 0, peakyness = None, conv_width = None, dist_modes = None):
    if cov_var_const is None:
        cov_var_const = stats.gamma.rvs(5,scale=1)
    if peakyness is None:
        peakyness = stats.gamma.rvs(2,scale=1)
    if conv_width is None:
        conv_width = stats.gamma.rvs(5,scale=1)
    if dist_modes is None:
        dist_modes = stats.gamma.rvs(5,scale=1)
    K = invwishart_rv(np.eye(dims) * cov_var_const, dims)    
    uvn = stats.norm(0, conv_width)
    mvn = stats.multivariate_normal(np.zeros(dims),K*10)
    mu = np.atleast_2d(stats.multivariate_normal.rvs(np.zeros(dims),np.eye(dims)*100))
    def mm_dens(x):
        # Some multimodal density
        x = np.atleast_2d(x)-mu
        return log_norm_const+uvn.logcdf(np.sin(x.sum(1)*peakyness)*dist_modes)+ mvn.logpdf(x)
    return (mm_dens, log_norm_const)
    
def gen_mm_lpost(num_datasets, dims, ev_params = [(80, 10), (40,10)], cov_var_const = 4):
    def gen_lp_unnorm_ev(lev, mixt):
        rval = lambda x:mixt.logpdf(x) + lev
        rval.log_evidence = lev
        return (rval, lev)
        
    rval = []
    for ep in ev_params:
        lev_distr = stats.gamma(ep[0], scale=ep[1])
        for i in range(int(num_datasets//len(ev_params))):
            rval.append(gen_mm_dens(dims, cov_var_const, -lev_distr.rvs(), peakyness = 1, conv_width = 4, dist_modes = 1))
    return rval