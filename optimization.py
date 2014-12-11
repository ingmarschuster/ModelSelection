# -*- coding: utf-8 -*-
"""

Test code for gradient-based PMC

Created on Wed Nov 26 14:21:15 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np
import scipy as sp

from numpy import log, exp
from scipy.misc import logsumexp

from copy import copy, deepcopy


from distributions import mvnorm, mvt, categorical
from distributions.linalg import pdinv
from synthdata import simple_gaussian
import slice_sampling
from gs_basis import ideal_covar



import sobol.sobol_seq 



def find_step_size(theta, f, lpost, search_direc, func_and_grad):
    lpost_1 = -np.inf  
    back_off = 0
    while lpost_1 < lpost:
        theta_1 = theta + f * search_direc
        lpost_1 = func_and_grad(theta_1, grad = False)
        if lpost > lpost_1:
            f = f * 0.5
            back_off +=1
        else:
            f = f* 1.05
            
            break
    return (f, theta_1, lpost_1, back_off)

def gradient_ascent(theta_0, func_and_grad, maxiter = 1000, quiet = True):
    (fval_0, grad_0) = func_and_grad(theta_0)
    f = 0.1
    back_off = 0
    move = 0
    for i in range(maxiter):
        (f, theta_1, fval_1, grad_1, back_off_tmp) = find_step_size(theta_0, f, fval_0, grad_0, func_and_grad)
        back_off += back_off_tmp
        move += 1
        (fval_0, grad_0, theta_0) = (fval_1, grad_1, theta_1) 
        gnorm = np.linalg.norm(grad_0)
        if gnorm < 10**-10:
            break
        assert(not(np.any(np.isnan(theta_0)) or np.any(np.isnan(grad_0))))
        
    if not quiet:
        print(move, back_off)
    return (theta_0, fval_0) 
    


def conjugate_gradient_ascent(theta_0, func_and_grad, maxiter = 1000, quiet = True):
    (fval_0, grad_0) = func_and_grad(theta_0)
    conj_dir_0 = grad_0
    f = 0.1
    (f, theta_1, fval_1, grad_1, back_off_tmp) = find_step_size(theta_0, f, fval_0, conj_dir_0, func_and_grad)
    #
    back_off = back_off_tmp
    move = 1
    for i in range(maxiter):        
        momentum = max(0, np.float(grad_1.T.dot(grad_1 -grad_0) / grad_0.T.dot(grad_0)))
        conj_dir_1 = grad_1 + momentum * conj_dir_0
        
        (theta_0, fval_0, grad_0, conj_dir_0) = (theta_1, fval_1, grad_1, conj_dir_1)
        (f, theta_1, fval_1, grad_1, back_off_tmp) = find_step_size(theta_1, f, fval_1, conj_dir_1, func_and_grad)
        back_off += back_off_tmp
        move += 1
        gnorm = np.linalg.norm(grad_0)
        if gnorm < 10**-10:
            break
        assert(not(np.any(np.isnan(theta_0)) or np.any(np.isnan(grad_0))))
    if not quiet:
        print(move, back_off)
    return (theta_0, fval_0) 
    

def test_gradient_ascent():
    num_dims = 4
    lds = sobol.sobol_seq.i4_sobol_generate(num_dims, 100).T
    
    
    var = 5
    K  =  np.eye(num_dims) * var
    Ki = np.linalg.inv(K)
    L = np.linalg.cholesky(K)
    logdet_K = np.linalg.slogdet(K)[1]
    samp = mvnorm(np.array([1000]*num_dims), K, Ki = Ki, L = L, logdet_K = logdet_K).ppf(lds)
    m_samp = samp.mean(0)
    

        
    def lpost_and_grad(theta):
        (llh, gr) = mvnorm(theta, K, Ki = Ki, L = L, logdet_K = logdet_K).log_pdf_and_grad(samp)
        return (llh.sum(), -gr.sum(0).flatten())
    
    theta = np.array([0]*num_dims)
    
    mse_cga = ((m_samp - conjugate_gradient_ascent(theta, lpost_and_grad)[0])**2).mean()
    mse_ga = ((m_samp - gradient_ascent(theta, lpost_and_grad)[0])**2).mean()
    print(mse_ga, mse_cga)
    assert(mse_ga < 10**-5)
    assert(mse_cga < 10**-5)

def test_rosenbrock():
    def neg_rosen_and_grad(theta):
        return (-sp.optimize.rosen(theta), -sp.optimize.rosen_der(theta))
    print("= Conjugate Gradient Ascent =")
    conj_ga = conjugate_gradient_ascent(np.ones(2)*30, neg_rosen_and_grad, maxiter=100000, quiet = False)
    assert(np.abs(conj_ga[1]) <10**3)
    print(conj_ga)
    print("= Gradient Ascent =")
    ga = gradient_ascent(np.ones(2)*30, neg_rosen_and_grad, maxiter=100000,  quiet = False)
    assert(np.abs(ga[1]) <25) 
    print(ga)
    