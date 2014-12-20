# -*- coding: utf-8 -*-
"""

Test code for gradient-based PMC

Created on Wed Nov 26 14:21:15 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import scipy as sp



def find_step_size(theta, f, lpost, search_direc, func_and_grad = None, func = None):
    assert(func_and_grad is not None or func is not None)
    lpost_1 = -np.inf  
    back_off = 0
    while lpost_1 < lpost:
        theta_1 = theta + f * search_direc
        if func_and_grad is not None:
            (lpost_1, grad_1) = func_and_grad(theta_1)
        else:
            lpost_1 = func(theta_1)
        if lpost > lpost_1:
            f = f * 0.5
            back_off +=1
        else:
            f = f* 1.05
            
            break
    if func_and_grad is not None:
        return (f, theta_1, lpost_1, grad_1, back_off)
    else:
        return (f, theta_1, lpost_1, back_off)

def gradient_ascent(theta_0, func_and_grad, maxiter = 1000, quiet = True):
    (fval_0, grad_0) = func_and_grad(theta_0)
    f = 0.1
    back_off = 0
    move = 0
    for i in range(maxiter):
        (f, theta_1, fval_1, grad_1, back_off_tmp) = find_step_size(theta_0, f, fval_0, grad_0, func_and_grad = func_and_grad)
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
    back_off = 0
    move = 0
    for i in range(maxiter): 
        (f, theta_1, fval_1, grad_1, back_off_tmp) = find_step_size(theta_0, f, fval_0, conj_dir_0, func_and_grad = func_and_grad)
        back_off += back_off_tmp
        momentum = max(0, np.float(grad_1.T.dot(grad_1 -grad_0) / grad_0.T.dot(grad_0)))
        conj_dir_1 = grad_1 + momentum * conj_dir_0
        
        (theta_0, fval_0, grad_0, conj_dir_0) = (theta_1, fval_1, grad_1, conj_dir_1)
        
        move += 1
        gnorm = np.linalg.norm(grad_0)
        if gnorm < 10**-10:
            break
        assert(not(np.any(np.isnan(theta_0)) or np.any(np.isnan(grad_0))))
    if not quiet:
        print(move, back_off)
    return (theta_0, fval_0) 
    

    