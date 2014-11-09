# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:15:43 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from scipy.special import multigammaln

import numpy as np
import scipy.stats as stats

def centering_matr(n):
    return np.eye(n) - 1./n * np.ones((n,n))

def scatter_matr(D, observations_in_rows = True):
    if observations_in_rows:
        M = D.T
    else:
        M = D
    return M.dot(centering_matr(M.shape[1])).dot(M.T)

def analytic_logevidence_norm(D, mu_pr, sd_pr, sd_li):
    v_pr = sd_pr**2 #tau^2 in Murphy
    v_li = sd_li**2 #sigma^2 in Murphy
    D_mean = np.mean(D)
    D_mean_sq = D_mean**2
    
    fact = [ (log(sd_li) - ( len(D) *log(sqrt(2*np.pi) *sd_li) #1st factor
                           + log(sqrt(len(D) * v_pr + v_li))   )),
             (- (  np.power(D, 2).sum() / (2 * v_li)   #2nd factor
                 + mu_pr**2             / (2 * v_pr))),
             (( (v_pr*len(D)**2 *D_mean_sq)/v_li   #numerator of 3rd factor
                 + (v_li * D_mean_sq)/v_pr
                 + 2 * len(D) * D_mean * mu_pr
               ) / (2 * (len(D) * v_pr + v_li)) # denominator of 3rd factor
             )            
           ]
    #print(fact)
    return np.sum(fact)

def analytic_postparam_logevidence_mvnorm(D, mu_pr, prec_pr, kappa_pr, nu_pr):
    D_mean = np.mean(D, 0)  
    
    (n, dim) = D.shape
    (kappa_post, nu_post) = (kappa_pr + n, nu_pr + n)
    mu_post = (mu_pr * kappa_pr + D_mean * n) / (kappa_pr + n)    
    scatter = scatter_matr(D)
    m_mu_pr = (D_mean - mu_pr)
    m_mu_pr.shape = (1, np.prod(m_mu_pr.shape))
    prec_post = prec_pr + scatter + kappa_pr * n /(kappa_pr + n) * m_mu_pr.T.dot(m_mu_pr)
        
    (sign, ldet_pr) = np.linalg.slogdet(prec_pr)
    (sign, ldet_post) = np.linalg.slogdet(prec_post)
    
    evid = (-(log(np.pi)*n*dim/2)  + multigammaln(nu_post/2, dim)
                                  - multigammaln(nu_pr / 2, dim) 
                                  + ldet_pr * nu_pr/2
                                  - ldet_post * nu_post/2
                                  + dim/2 * (log(kappa_pr) - log(kappa_post))
                                  )

    return ((mu_post, prec_post, kappa_post, nu_post), evid)
    

def analytic_logevidence_scalar_gaussian(D, mu_pr, sd_pr, sd_li):
    v_pr = sd_pr**2 #tau^2 in Murphy
    v_li = sd_li**2 #sigma^2 in Murphy
    D_mean = np.mean(D)
    D_mean_sq = D_mean**2
    
    fact = [ (log(sd_li) - ( len(D) *log(sqrt(2*np.pi) *sd_li) #1st factor
                           + log(sqrt(len(D) * v_pr + v_li))   )),
             (- (  np.power(D, 2).sum() / (2 * v_li)   #2nd factor
                 + mu_pr**2             / (2 * v_pr))),
             (( (v_pr*len(D)**2 *D_mean_sq)/v_li   #numerator of 3rd factor
                 + (v_li * D_mean_sq)/v_pr
                 + 2 * len(D) * D_mean * mu_pr
               ) / (2 * (len(D) * v_pr + v_li)) # denominator of 3rd factor
             )            
           ]
    #print(fact)
    return np.sum(fact)

def evidence_from_importance_weights(weights, num_weights_range = None):
    if num_weights_range is None:
        logsumexp(weights)-log(len(weights))
    return [logsumexp(weights[:N]) - log(N) for N in num_weights_range]