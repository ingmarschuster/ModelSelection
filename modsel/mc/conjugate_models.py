# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 11:11:58 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

from distributions import norm_invwishart, mvnorm




class BetaBernModel(object):
    def __init__(self, alpha, beta):
        self.pri_par = np.array((alpha, beta))
        assert(self.pri_par.size == 2)
        self.post_par = self.pri_par.copy()
    
    def observe(self, obs):
        """
        Observe Bernoulli data
        
        Parameters
        ==========
        obs - Bernoulli data observations (i.e. an array or list containing only 0 and 1)
        """
        obs = np.array(obs).flatten()
        
        s_obs = obs.sum()
        self.post_par =  self.post_par + np.array((s_obs, obs.size - s_obs))
    
    def rv(self):
        return stats.bernoulli.rvs(stats.beta.rvs(*self.post_par))
        
        

class MvnNiwModel(object):
    #an MVN - Normal Inverse Wishart model
    def __init__(self, K0, nu0, mu0, kappa0, bij = None):
        """
        Construct an object representing a normal - normal-inverse_wishart Model
        
        Parameters
        ==========
        K0 - positive definite inverse scale matrix (DxD)
        nu0 - degrees of freedom for the inverse wishart part (nu0 > D - 1)
        mu0 - location parameter
        kappa0 - prior measurements of scale (kappa0 > 0)
        transf - a (bijective) transform of random variables after they are drawn from the normal
        inv_transf - the inverse of the transform
        
        Returns
        =======
        object
        """
        
        
        if bij is not None and not hasattr(bij, "i"):
            raise ValueError("Bijection bij is expected to store its inverse in attribute bij.i")
                
            
        self.K = K0
        self.nu = nu0
        if bij is not None:
            self.mu = bij.i(mu0)
        else:
            self.mu = mu0
        self.kappa = kappa0
        
        self._update_data_dist()
        
        self.bij = bij
    
    def _update_data_dist(self):
        self.ddist = mvnorm(*norm_invwishart(self.K, self.nu, self.mu, self.kappa).rv())
    
        
    def rv(self):
        #self._update_data_dist()
        if self.bij is not None:
            return self.bij(self.ddist.rvs())
        else:
            return self.ddist.rvs()
        
    def observe(self, obs):
        """
        Observe data (possibly in a transformed space)
        
        Parameters
        ==========
        obs - observed data from a (possibly transformed) MV-Normal distribution
        """
        
        def centering_matr(n):
            return np.eye(n) - 1./n * np.ones((n,n))
        
        def scatter_matr(D, observations_in_rows = True):
            if observations_in_rows:
                M = D.T
            else:
                M = D
            return M.dot(centering_matr(M.shape[1])).dot(M.T)
        
        obs = np.array(obs)
        if self.bij is not None:
            obs = self.bij.i(obs)
        if obs.size == self.mu.size:
            obs.shape = (1, self.mu.size)
        assert(len(obs.shape) == 2)
        
        m_obs = obs.mean(0)
        n = obs.shape[0]
        
        mu = (self.kappa*self.mu + n*m_obs) / (self.kappa + n)
        
        K = self.K + scatter_matr(obs) + self.kappa*n/(self.kappa + n) * (m_obs-self.mu).T.dot(m_obs-self.mu)
        
        self.mu = mu
        self.K = K        
        self.kappa = self.kappa + n
        self.nu = self.nu + n
        self._update_data_dist()

    
    
        
        