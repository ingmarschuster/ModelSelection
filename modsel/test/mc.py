# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:08:28 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import scipy as sp
import numpy as np
import scipy.stats as stats
from modsel.distributions import mvnorm
from modsel.mc import mcmc, pmc
from modsel.mc.optimization import conjugate_gradient_ascent, gradient_ascent


from modsel.sobol import sobol_seq


def test_MCMC_PMC():
    np.random.seed(2)
    for dim in [3, 4]:
        theta = stats.multivariate_normal.rvs(np.array([0]*dim), np.eye(dim)*10)
        def lpost_and_grad(x, grad = True):
            diff = (theta-x)
            lp = -(100*diff**2).sum()
            if not grad:
                return lp
            else:
                gr = 200 * diff
                return (lp, gr)
        lpost = lambda x: lpost_and_grad(x, False)
        ###### MCMC ######
        for mk in [mcmc.GaussMHKernel(lpost, 1, True),
                   mcmc.GaussMHKernel(lpost, np.eye(dim), False),
                   mcmc.ComponentWiseSliceSamplingKernel(lpost)]:
            (samp, trace) = mcmc.sample(100, -theta, mk)
            samp_m = samp[len(samp)//2:].mean(0)
            print(mk, np.mean((samp_m - theta)**2))
            if np.mean((samp_m - theta)**2) > 4:
                print(mk, np.mean((samp_m - theta)**2), samp_m, theta)
                assert(False)
        ###### PMC ######
        for (prop, num_samp) in [(pmc.NaiveRandomWalkProposal(lpost, mvnorm([0]*dim, np.eye(dim)*5)), 1000),
                                 (pmc.GradientAscentProposal(lpost_and_grad, dim, lrate = 0.1), 100),
                                 (pmc.ConjugateGradientAscentProposal(lpost_and_grad, dim, lrate = 0.1), 100)]:
            (samp, trace) = pmc.sample(num_samp, [-theta]*10, prop) #sample_lpost_based
            samp_m = samp[len(samp)//2:].mean(0)#samp.mean(0)
            print(prop, np.mean((samp_m - theta)**2))
            if np.mean((samp_m - theta)**2) > 4:
                print(prop, np.mean((samp_m - theta)**2), samp_m, theta)
                assert(False)
        

def test_Rosenbrock():
    np.random.seed(2)
    def lpost_and_grad(theta, grad = True):
        fval = -sp.optimize.rosen(theta)
        if not grad:
            return fval
        else:
            return (fval, -sp.optimize.rosen_der(theta))
    lpost = lambda x: lpost_and_grad(x, False)
    theta=np.array((1, 1))
    dim = 2
    inits = mvnorm([0]*dim, np.eye(dim)*5).rvs(10)
    for i in len(inits):
        initial = inits[i]
        ###### MCMC ######
        for mk in [mcmc.GaussMHKernel(lpost, 1, True),
                   mcmc.GaussMHKernel(lpost, np.eye(dim), False),
                   mcmc.ComponentWiseSliceSamplingKernel(lpost)]:
            (samp, trace) = mcmc.sample(100, -theta, mk)
            samp_m = samp[len(samp)//2:].mean(0)
            print(mk, np.mean((samp_m - theta)**2))
            if np.mean((samp_m - theta)**2) > 4:
                print(mk, np.mean((samp_m - theta)**2), samp_m, theta)
                #assert(False)
        ###### PMC ######
        for (prop, num_samp) in [(pmc.NaiveRandomWalkProposal(lpost, mvnorm([0]*dim, np.eye(dim)*5)), 1000),
                                 (pmc.GradientAscentProposal(lpost_and_grad, dim, lrate = 0.1), 100),
                                 (pmc.ConjugateGradientAscentProposal(lpost_and_grad, dim, lrate = 0.1), 100)]:
            (samp, trace) = pmc.sample(num_samp, [-theta]*10, prop) #sample_lpost_based
            samp_m = samp[len(samp)//2:].mean(0)#samp.mean(0)
            print(prop, np.mean((samp_m - theta)**2))
            if np.mean((samp_m - theta)**2) > 4:
                print(prop, np.mean((samp_m - theta)**2), samp_m, theta)
                #assert(False)


def test_optimization_gradient_ascent():
    num_dims = 4
    lds = sobol_seq.i4_sobol_generate(num_dims, 100).T
    
    
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

def test_optimization_rosenbrock():
    def neg_rosen_and_grad(theta, grad = True):
        fval = -sp.optimize.rosen(theta)
        if not grad:
            return fval
        else:
            return (fval, -sp.optimize.rosen_der(theta))

    print("= Conjugate Gradient Ascent =")
    conj_ga = conjugate_gradient_ascent(np.ones(2)*30, neg_rosen_and_grad, maxiter=100000, quiet = False)
    assert(np.abs(conj_ga[1]) <10**3)
    print(conj_ga)
    print("= Gradient Ascent =")
    ga = gradient_ascent(np.ones(2)*30, neg_rosen_and_grad, maxiter=100000,  quiet = False)
    assert(np.abs(ga[1]) <25) 
    print(ga)