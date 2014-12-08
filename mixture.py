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


from distributions import mvnorm, mvt, categorical

class GMM(object):
    def __init__(self, samples, num_components):
        self.samples = samples
        
        self.dim = self.samples.shape[1]
        
        self.num_components = num_components
        self.num_samp = self.samples.shape[0]
        p = [1./num_components] * num_components
        self.comp_dist = []
        sample_perm = np.random.permutation(self.num_samp)
        for c in range(self.num_components):
            mu = samples[sample_perm[c], :]
            self.comp_dist.append(mvnorm(mu, np.eye(self.dim)))
        self.resp = np.argmax(np.random.multinomial(1, p, self.num_samp),
                                    axis = 1).flat[:]
        self.comp_lprior = np.zeros((1, self.num_components)) - log(self.num_components)
        self._fit()

    
    def _fit(self):        
        old = -1
        i = 0
        while not np.all(old == self.resp):
            i += 1
            old = self.resp.copy()
            self._e_step()
            self._m_step()
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
    

def test_GMM():
    from sobol.sobol_seq import i4_sobol_generate
    num_samp = 10
    samp = np.vstack((stats.multivariate_normal.rvs(np.ones(2)*10, np.eye(2), size=num_samp), stats.multivariate_normal.rvs(np.ones(2)*-10, np.eye(2), size=num_samp)))
    m = GMM(samp, 2)
    assert(np.all(exp(m.resp).round(1).sum(0) == np.array([num_samp] * 2)))
    quasi = m.ppf(i4_sobol_generate(3,num_samp).T)
    assert((quasi < 0).sum() == num_samp)
    
    