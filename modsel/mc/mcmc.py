# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:58:18 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import exp, log
from distributions import mvnorm
from modsel.mc import slice_sampling, flags


def sample(num_samples, initialization, markov_kernel, stop_flag = flags.NeverStopFlag()):
    s = []
    lp = []
    theta_old = np.copy(initialization)
    lpost_old = -np.inf
    i = 0
    while i < num_samples and not stop_flag.stop():  
        (theta_new, lpost_new) = markov_kernel.step(theta_old, lpost_old)
        s.append(theta_new)
        lp.append(lpost_new)
        (theta_old, lpost_old) = (theta_new, lpost_new)
        #print s[-1], theta_old
        i += 1
    #assert()
    return (np.array(s), np.array(lp))


class MarkovKernel(object):
    def step(self, theta, current_lpost):
        raise NotImplementedError()
        

class ComponentWiseSliceSamplingKernel(MarkovKernel):
    def __init__(self, lpost_func):
        class dummy_prior(object):
            def logpdf(self, x):
                return 0
                
        self.lpost = lpost_func
        self.pr = dummy_prior()
    
    def __str__(self):
        return ("<ComponentWiseSliceSamplingKernel>")
        
    def step(self, theta, current_lpost):
        theta_new = theta.copy()
        (rval, lpost) = slice_sampling.slice_sample_all_components_mvprior(theta_new, lambda:self.lpost(theta_new), self.pr, cur_ll = current_lpost)
        #assert ()
        return (rval, lpost)



class MHKernel(MarkovKernel):
    
    def __init__(self, lpost_func, proposal_distr, component_wise):
        self.lpost = lpost_func
        self.prop_dist = proposal_distr
        self.comp_wise = component_wise
        
    def __str__(self):
        return ("<MHKernel comp_wise="+str(self.comp_wise)+ " prop_dist="+ str(self.prop_dist)+">")
        
    def step(self, theta, current_lpost):
        if not self.comp_wise:
            delta = self.prop_dist.rvs()
            theta_new = theta + delta
            lpost_new = self.lpost(theta_new)
            
            if not self.accept(delta, lpost_new, current_lpost):
                (theta_new, lpost_new) = (theta, current_lpost)
        else:
            theta_new = theta.copy()
            lpost_new = np.copy(current_lpost)
            lpost_old = np.copy(current_lpost)
            for i in range(theta_new.size):
                delta = self.prop_dist.rvs()
                old = theta_new.flat[i]
                theta_new.flat[i] = old + delta
                lpost_new = self.lpost(theta_new)
                if self.accept(delta, lpost_new, lpost_old):
                    lpost_old = lpost_new
                else:
                    lpost_new = lpost_old
                    theta_new.flat[i] = old
        return (theta_new, lpost_new)
        
    
    def accept(self, theta_new_minus_old, lpost_new, lpost_old):
        #probability of moving back to theta_old
        p_backw = lpost_new + self.prop_dist.logpdf(-theta_new_minus_old)
        #probability of moving forward to theta_new
        p_forw  = lpost_old + self.prop_dist.logpdf(+theta_new_minus_old)
        
        p = np.min((1,exp(p_backw - p_forw)))
        
        return np.random.binomial(1, p) == 1 
    
    @staticmethod
    def check_variance(var, component_wise):
        """Returns checks covariance matrix for proper shape (may try to reshape it)
        
        Parameters
        ----------
        var - if component_wise is True, this is a scalar variance, else a covariance matrix
        component_wise - whether to make multivariate step proposals or a proposal for each component
        
        Returns
        -------
        (var,sh)  - Covariance matrix and its shape
        """
        if component_wise is True:
            var = np.atleast_2d(np.array(var).flatten())
            if var.size != 1:
                raise TypeErrorv("var should be a single number when component_wise is True")
            s = 1
        else:
            var = np.atleast_2d(var)
            s =  np.int(np.sqrt(var.size))
            if var.size != s**2:
                raise TypeError("var is expected to be a covariance matrix, but is not square and cannot be cast into square form")
            var.shape = (s, s)
        
        return (var, s)
        


class GaussMHKernel(MHKernel):
    def __init__(self, lpost_func, var, component_wise = False):
        """Returns a Metropolis-Hastings Markov kernel with the given step
        (co)variance.
        
        Parameters
        ----------
        lpost_func - posterior measure we want to sample from
        var - if component_wise is True, this is a scalar variance, else a covariance matrix
        component_wise - whether to make multivariate step proposals or a proposal for each component
        
        Returns
        -------
        kernel  - The Metropolis-Hastings Markov kernel
        """
        (var, sh) = MHKernel.check_variance(var, component_wise)
        super(GaussMHKernel, self).__init__(lpost_func, mvnorm([0]*sh, var), component_wise)
    
    def accept(self, theta_new_minus_old, lpost_new, lpost_old):
        #As the proposal is symmetric, it reduces from the ratio
        p = exp(np.min((0, lpost_new - lpost_old)))
        
        return np.random.binomial(1, p) == 1   