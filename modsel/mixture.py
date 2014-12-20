# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:03:06 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp


from .distributions import mvnorm, mvt, categorical


class TMM(object):
    def __init__(self, num_components, dim, samples = None):
        self.num_components = num_components
        self.dim = dim
        if samples is not None:
            self.fit(samples)

    
    def fit(self, samples):
        import sklearn.mixture
        m = sklearn.mixture.GMM(self.num_components, "full")
        m.fit(samples)
        self.comp_lprior = log(m.weights_)
        self.dist_cat = categorical(exp(self.comp_lprior))
        self.comp_dist = [mvt(m.means_[i], m.covars_[i], 4) for i in range(self.comp_lprior.size)]
        self.dim = m.means_[0].size
    
    
    def ppf(self, component_cum_prob):
        assert(component_cum_prob.shape[1] == self.dim + 2)
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:])))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.comp_lprior[i]+ self.comp_dist[i].logpdf(x)
                              for i in range(self.comp_lprior.size)])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples):
        return self.ppf(stats.uniform.rvs(0, 1, (num_samples, self.dim + 2)))
    

class GMM(object):
    def __init__(self, num_components, dim, samples = None):
        self.num_components = num_components
        self.dim = dim
        if samples is not None:
            self.fit(samples)

    
    def fit(self, samples):
        import sklearn.mixture
        m = sklearn.mixture.GMM(self.num_components, "full")
        m.fit(samples)
        self.comp_lprior = log(m.weights_)
        self.dist_cat = categorical(exp(self.comp_lprior))
        self.comp_dist = [mvnorm(m.means_[i], m.covars_[i]) for i in range(self.comp_lprior.size)]
        self.dim = m.means_[0].size
        #self._e_step()
        if False:        
            old = -1
            i = 0
            while not np.all(old == self.resp):
                i += 1
                old = self.resp.copy()
                self._e_step()
                self._m_step()
                print(np.sum(old == self.resp)/self.resp.size)
            #print("Convergence after",i,"iterations")
            self.dist_cat = categorical(exp(self.comp_lprior))
    
    def _m_step(self):
        assert(self.resp.shape[0] == self.num_samp)
        pseud_lcount = logsumexp(self.resp, axis = 0).flat
        r = exp(self.resp)        
        
        self.comp_dist = []
        for c in range(self.num_components):
            norm = exp(pseud_lcount[c])
            mu = np.sum(r[:,c:c+1] * self.samples, axis=0) / norm
            diff = self.samples - mu
            scatter_matrix = np.zeros([self.samples.shape[1]]*2)
            for i in range(diff.shape[0]):
                scatter_matrix += r[i,c:c+1] *diff[i:i+1,:].T.dot(diff[i:i+1,:])
            scatter_matrix /= norm
            self.comp_dist.append(mvnorm(mu, scatter_matrix))
        self.comp_lprior = pseud_lcount - log(self.num_samp)
            
    
    def _e_step(self):
        lpdfs = np.array([d.logpdf(self.samples).flat[:] 
                              for d in self.comp_dist]).T + self.comp_lprior
        self.resp = lpdfs - logsumexp(lpdfs, axis = 1).reshape((self.num_samp, 1))
    
    def ppf(self, component_cum_prob):
        assert(component_cum_prob.shape[1] == self.dim + 1)
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:])))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.comp_lprior[i]+ self.comp_dist[i].logpdf(x)
                              for i in range(self.comp_lprior.size)])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples):
        return self.ppf(stats.uniform.rvs(0, 1, (num_samples, self.dim + 1)))