# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:56:05 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats

def log_sign(a):
    a = np.array(a)
    sign_indicator = ((a < 0 ) * -2 + 1)
    return (log(np.abs(a)), sign_indicator)

def exp_sign(a, sign_indicator):
    return exp(a) * sign_indicator

def logsubtrexp(minuend, subtrahend, sign_minuend = None, sign_subtrahend = None):

    if sign_minuend is None:
        sign_minuend = np.ones(minuend.shape)
    if sign_subtrahend is None:
        sign_subtrahend = np.ones(subtrahend.shape)
    if not (minuend.shape == sign_minuend.shape and subtrahend.shape == sign_subtrahend.shape):
        raise ValueError("sign arguments expected be of same shape as corresponding log-matrices")
    if not (np.abs(sign_minuend).all() and np.abs(sign_subtrahend).all()):
        raise ValueError("sign arguments expected to contain only +1 or -1 elements")
        
    b = np.broadcast(minuend, subtrahend)
    s_b = np.broadcast(sign_minuend, sign_subtrahend)
    abs_res = np.empty(b.shape)
    sign_res = np.empty(b.shape)
    
    for i in range(b.size):
        (m, s) = b.next()
        (sign_m, sign_s) = s_b.next()
        if sign_m > sign_s: # sign_m == 1 and sign_s == -1
            # this is equivalent to logsumexp(m, s)
            #print("sign_m > sign_s")
            sign_res.flat[i] = 1
            abs_res.flat[i] = logsumexp((m,s))
        elif sign_m < sign_s: # sign_m == -1 and sign_s == 1
            #print("sign_m < sign_s")
            sign_res.flat[i] = -1
            abs_res.flat[i] = logsumexp((m,s))
        else:
            #signs are eqal
            if m == s:                
                sign_res.flat[i] = 1
                abs_res.flat[i] = log(0)
            else:
                if sign_m == -1:
                    if m > s:
                        #print("m >= s")
                        sign_res.flat[i] = -1
                        abs_res.flat[i] = log(1 - exp(s - m)) + m
                    elif m < s:
                        #print("m < s")
                        sign_res.flat[i] = 1
                        abs_res.flat[i] = log(1 - exp(m - s)) + s
                else:# sign_m == 1
                    if m > s:
                        #print("m >= s")
                        sign_res.flat[i] = 1
                        abs_res.flat[i] = log(1 - exp(s - m)) + m
                    elif m < s:
                        #print("m < s")
                        sign_res.flat[i] = -1
                        abs_res.flat[i] = log(1 - exp(m - s)) + s
        #print(sign_m*exp(m),  sign_s*exp(s),  sign_m*exp(m) - sign_s*exp(s), sign_res.flat[i] * exp(abs_res.flat[i]))
    
    return (abs_res, sign_res)
    
def logaddexp(minuend, subtrahend, sign_minuend = None, sign_subtrahend = None):
    if sign_subtrahend is None:
        sign_subtrahend = np.ones(subtrahend.shape)
    return logsubtrexp(minuend, subtrahend, sign_minuend = sign_minuend, sign_subtrahend = -sign_subtrahend)



def logmeanexp(a, sign_indicator = None, axis = None): 
    def conditional_logsumexp(where, axis):
        masked = -np.ones(a.shape) * np.inf
        np.copyto(masked, a, where = where)
        masked_sum = logsumexp(masked, axis = axis)
        #np.copyto(masked_sum,  -np.ones(masked_sum.shape) * np.inf, where = np.isnan(masked_sum)) 
        np.place(masked_sum, np.isnan(masked_sum), -np.inf)
        return masked_sum
    
    a = np.array(a)
    if axis is None:
        norm = log(np.prod(a.shape))
    else:
        norm = log(a.shape[axis])
        
    if sign_indicator is None:
        res = np.array(logsumexp(a, axis = axis)) - norm
        signs = np.ones(res.shape)
    else:
        pos_sum = conditional_logsumexp(sign_indicator == 1, axis)
        neg_sum = conditional_logsumexp(sign_indicator == -1, axis)
        
        #print("axis", axis, "\narray", exp_sign(a, sign_indicator),"\n",(a, sign_indicator), "\npos_sum", pos_sum, "\nneg_sum", neg_sum)
        (res, signs) =  logsubtrexp(pos_sum, neg_sum)
        #np.copyto(res,  -np.ones(res.shape) * np.inf, where = np.isnan(res))
        np.place(res, np.isnan(res), -np.inf)
        res = res - norm
    try:
        if axis != None:
            sh = list(a.shape)
            sh[axis] = 1
            res = res.reshape(sh)
            signs = signs.reshape(sh)
    except Exception as e:
        print("Exception when trying to reshape:", e)
    return (res, signs)


def logvarexp(a, sign_indicator = None, axis = None):
    # the unbiased estimatior (dividing by (n-1))
    # is only definded for sample sizes >=2)
    assert(a.shape[axis] >= 2) 
    a = np.array(a)
    if axis is None:
        norm = log(np.prod(a.shape))
    else:
        norm = log(a.shape[axis])
    (mean, mean_s) = logmeanexp(a, sign_indicator = sign_indicator, axis = axis)
    (diff, diff_s) = logsubtrexp(a, mean, sign_minuend = sign_indicator, sign_subtrahend = mean_s)
    var = logsumexp(2*diff, axis = axis) - log(diff.shape[axis])
    var = var.reshape(mean.shape)
    return var
    
def logmseexp(log_truth, log_estim, signs_truth = None, signs_estim = None, axis = 0):
    (diff, sign) = logsubtrexp(log_estim, log_truth, sign_minuend = signs_estim, sign_subtrahend = signs_truth)
    return logmeanexp(diff * 2, axis = axis)[0]


def logbiasexp(log_truth, log_estim, signs_truth = None, signs_estim = None, axis = 0):
    #assert(log_estim.shape[axis] == log_truth.T.shape[axis])
        
    #(true_tiled, log_estimates) = np.broadcast_arrays(log_true_theta, log_estimates)
    (diff, sign) = logsubtrexp(log_estim, log_truth, sign_minuend = signs_estim, sign_subtrahend = signs_truth)
    return logmeanexp(diff, sign, axis = axis)

def logbias2exp(log_true_theta, log_estimates, signs_truth = None, signs_estim = None, axis = 0):
    return 2*logbiasexp(log_true_theta, log_estimates, signs_estim = signs_estim, signs_truth = signs_truth, axis = axis)[0]


def logstatistics(est):
    res = {}
    for num_samples in est:  
        res[num_samples] = {"bias^2":{},
                         "variance": {},
                         "mse":{},
                         "bias^2{ }(relat)":{}, 
                         "variance{ }(relat)": {}, 
                         "mse{ }(relat)":{}
                         }
        # now calculate bias, variance and mse of estimators when compared
        # to analytic evidence
        for estim in est[num_samples]:
            if estim == "GroundTruth":
                #Analytical evidence is not to be evaluated
                continue
            estimate = est[num_samples][estim]
            analytic = est[num_samples]["GroundTruth"].reshape((len(est[num_samples]["GroundTruth"]), 1))
            est_rel = estimate - analytic
            
            bias2 = logbias2exp(analytic, estimate, axis = 0)
            bias2_rel = logbias2exp(np.atleast_2d(0), est_rel, axis = 0)
            var = logvarexp(estimate, axis = 0)
            var_rel = logvarexp(est_rel, axis = 0)
            mse = logmseexp(analytic, estimate, axis = 0)
            mse_rel = logmseexp(np.atleast_2d(0), est_rel, axis = 0)
            
            res[num_samples]["bias^2"][estim] = bias2.flat[:]
            res[num_samples]["bias^2{ }(relat)"][estim] = bias2_rel.flat[:]
            res[num_samples]["variance"][estim] =  var.flat[:]
            res[num_samples]["variance{ }(relat)"][estim] =  var_rel.flat[:]
            res[num_samples]["mse"][estim] =  mse.flat[:]
            res[num_samples]["mse{ }(relat)"][estim] =  mse_rel.flat[:]
            #print(logsubtrexp(logaddexp(bias2, var)[0], mse)[0],"\n",
            #      logsubtrexp(logsumexp(np.vstack((bias2, var)), 0), mse)[0])
            decomp_err = logmeanexp(logsubtrexp(logsumexp(np.vstack((bias2, var)), 0), mse)[0])[0]
          
            if decomp_err >= -23: # error in original space >= 1e-10 
                print("large mse decomp error, on average", decomp_err)
    return res


def statistics(est, take_logarithm = True):
    res = {}
    if take_logarithm:
        transform = np.log
        prestr = "log "
    else:
        transform = lambda x: x
        prestr = ""
    for num_samples in est:  
        res[num_samples] = {prestr+"bias^2":{},
                         prestr+"variance": {},
                         prestr+"mse":{},
                         prestr+"bias^2{ }(relat)":{}, 
                         prestr+"variance{ }(relat)": {}, 
                         prestr+"mse{ }(relat)":{}
                         }
        # now calculate bias, variance and mse of estimators when compared
        # to analytic evidence
        for estim in est[num_samples]:
            if estim == "GroundTruth":
                #Analytical evidence is not to be evaluated
                continue
            estimate = est[num_samples][estim]
            analytic = est[num_samples]["GroundTruth"].reshape((len(est[num_samples]["GroundTruth"]), 1))
            est_rel = estimate / analytic
            
            bias2 = (estimate - analytic).mean(0).mean()**2
            bias2_rel = (estimate - 1).mean(0).mean()**2
            var = estimate.var(0).mean()
            var_rel = est_rel.var(0).mean()
            mse = ((estimate - analytic)**2).mean()
            mse_rel = ((est_rel - 1)**2).mean()
            
            res[num_samples][prestr+"bias^2"][estim] = transform(bias2.flat[:])
            res[num_samples][prestr+"bias^2{ }(relat)"][estim] = transform(bias2_rel.flat[:])
            res[num_samples][prestr+"variance"][estim] =  transform(var.flat[:])
            res[num_samples][prestr+"variance{ }(relat)"][estim] =  transform(var_rel.flat[:])
            res[num_samples][prestr+"mse"][estim] =  transform(mse.flat[:])
            res[num_samples][prestr+"mse{ }(relat)"][estim] =  transform(mse_rel.flat[:])
            #print(logsubtrexp(logaddexp(bias2, var)[0], mse)[0],"\n",
            #      logsubtrexp(logsumexp(np.vstack((bias2, var)), 0), mse)[0])
            decomp_err = (bias2 + var - mse).mean()
          
            if decomp_err >= 0.01: # error in original space >= 1e-10 
                print("large mse decomp error, on average", decomp_err)
    return res