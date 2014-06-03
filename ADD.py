# -*- coding: utf-8 -*-
"""
Created on Sat May 31 07:31:34 2014

@author: Ingmar Schuster
"""


from __future__ import division, print_function
import numpy as np
import datetime
import numpy.random as npr
import scipy.stats as stats
from numpy import exp, log
from scipy.misc import logsumexp
from copy import deepcopy,copy
import pickle
import sys

from basicmodels import MatrixWithComponentPriorModel,IsotropicRemainingVarModel


import collections

from slice_sampling import slice_sample_component#, slice_sample_component_double

class DimensionalityModel:
    def __init__(self, upper_dim_bound):
        self.upper_bound = upper_dim_bound
        self.matr = np.array((1,))
        # our prior belief is that each dimension might be important
        confidence = 40 # this has to be an int
        self.bin_param = np.array((upper_dim_bound, 1./upper_dim_bound))
        #The following sets the binomial prior to belief in dimensionality 1
        #to a degree determined by 'confidence'
        self.beta_param = np.array([confidence, (upper_dim_bound - 1) * confidence])
    
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
    
    def get(self):
        return np.int(self.matr)
            
    def sample(self, data = None,  dimrange = None):
        #print("weight pre",llike(), self.matr)
        logpost = np.ones((np.min([self.upper_bound + 1, self.matr + 2 ]),)) * np.NINF
        if dimrange == None:
            dimrange = (np.int(np.max([1, self.matr - 1])),
                        np.int(np.min([self.upper_bound + 1, self.matr + 2 ])))
                       
        prior = stats.binom(self.bin_param[0], self.bin_param[1])
        
        for dim in range(*dimrange):
            log_likelihood = self.glob_mdl.llike_function(data, rv_mdl=self, idx = (dim,dim))
            logpost[dim] = log_likelihood() + prior.logpmf(dim)
        logpost -= logsumexp(logpost)
        print( self.matr, " ",self.upper_bound, range(*dimrange), "\n", exp(logpost))
        self.matr = np.argmax(np.random.multinomial(1, exp(logpost)))
        self.beta_param[0] += self.matr
        self.beta_param[1] += self.upper_bound - self.matr
        self.bin_param[1] = stats.beta(self.beta_param[0], self.beta_param[1]).rvs()



class ADDTwoFactorModel:
    def __getstate__(self):
        rval = copy(self.__dict__)
        for attr in ("new_prior_func","prior", "kernel"):
            try:
                rval[attr] = None
            except:
                pass
        return rval

    def __setstate__(self, state):
        self.__dict__ = state
        if self.__dict__["kernel"] == None:
            self.kernel = np.dot
            
    def __init__(self, latent_dim_bound,
                       data,
                       lv_mdl = None,
                       weight_mdl = None,
                       remainvar_mdl = None,
                       kernel = np.dot,
                       additional_dim_policy = "inline",
                       quiet = False):
        #observations are expected in colums.
        if lv_mdl != None:
            self.lvm = lv_mdl
        else:
            self.lvm = MatrixWithComponentPriorModel((data.shape[0], latent_dim_bound),
                                                     prior = ("norm", (0., 1.), {}),
                                                     estimate_mean = False)
        if weight_mdl != None:
            self.wm = weight_mdl
        else:
            self.wm = MatrixWithComponentPriorModel((latent_dim_bound, data.shape[1]),
                                                    prior = ("norm", (0., 5.), {}),
                                                    estimate_mean = True,
                                                    new_prior_func = "lambda mean: stats.norm(mean, 5.)")
                                                    
        self.dim_m = DimensionalityModel(latent_dim_bound)
        
        if remainvar_mdl != None:
            self.rvm = remainvar_mdl
        else:
            self.rvm = IsotropicRemainingVarModel(data.shape[1])
        self.additional_dim_policy = additional_dim_policy
        self.kernel = kernel
        self.lvm.set_global_model(self)
        self.wm.set_global_model(self)
        self.rvm.set_global_model(self)
        self.dim_m.set_global_model(self)
        self.last_ll = np.infty
        self.quiet = quiet #Output likelihood in during sampling?
        #self.log_likelihood(data)
        
        self.data_mean = np.average(data,0)
        self.pred = {}
        self.ll_factors = {}
        self.last_ll = {}
        self.recompute_cache(data)
        self.recompute_cache(data, dim = self.dim()+1)
        
    def sample(self, data = None, num_samples = 1):
        rval = []        
        for it in range(num_samples):
            log_msg = "Iteration " + str(it) +" pre "+ str(self.last_ll)
            pre_ll = self.last_ll
            # randomize resampling order to counter selection bias
            lv_range = self.dim_lv_range()
            for rv in npr.permutation((self.wm, self.lvm)):
                log_msg = log_msg + "; sampled "
                if rv == self.wm:
                    row_col_ranges = (lv_range, None)
                    log_msg = log_msg + "WM "
                else:
                    row_col_ranges = (None, lv_range)
                    log_msg = log_msg + "LVM "                        
                rv.sample(data, row_col_ranges = row_col_ranges)
                log_msg = log_msg + str(self.last_ll)
            self.rvm.sample(data)
            log_msg = log_msg + "; sampled RV: "+ str(self.last_ll)
            self.dim_m.sample(data)
            log_msg = log_msg + "; sampled Relev: "+ str(self.last_ll)
            log_msg = log_msg + "\n(dim %d)\n" % self.dim_m.get()
            if not self.quiet:
                print(log_msg)
            rval.append(deepcopy(self))
        return rval
   

    def notify(self, data, rv_mdl, idx, dim = None):
        if dim == None:
            print("No dim given")
            dim = self.dim()
        dim = np.int(dim)
        (row, col) = idx
        if rv_mdl == self.lvm:
            self.pred[dim][row,:] = (self.lvm.get()[:, :dim][row,:].dot(self.wm.get()[:dim]) + self.data_mean)
            tmp = np.array([stats.norm(self.pred[dim][row,j], self.rvm.get()[j,j]).logpdf(data[row,j])
                                               for j in range(self.pred[dim].shape[1])])
            self.ll_factors[dim][row,:] = tmp
        elif rv_mdl == self.wm:
            self.pred[dim][:, col] = (self.lvm.get()[:, :dim].dot(self.wm.get()[:dim][:, col]) + self.data_mean[col])
            self.ll_factors[dim][:, col] = np.array([stats.norm(self.pred[dim][i,col], self.rvm.get()[col,col])
                                                        .logpdf(data[i,col]) for i in range(self.pred[dim][:, col].shape[0])])
        elif rv_mdl == self.dim_m:
            assert(dim == row)
            self.recompute_cache(data, dim)
        else:
            self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred[dim].shape[1])]
                                        for i in range(self.pred[dim].shape[0])])
        self.last_ll[dim] = np.sum(self.ll_factors[dim])

    def prediction(self, data = None, dim = None):
        if dim == None:
            dim = self.dim()
        pred = self.kernel(self.lvm.get()[:, :dim], self.wm.get()[:dim])
        if data != None:
            if data.shape != pred.shape:
                raise IndexError("Data and Model of different dimensions", data.shape, "vs", pred.shape)
            else:
                return pred + self.data_mean
        else:
            return pred
    
    def dim(self):
        return self.dim_m.get()
        
    def dim_lv_range(self):
        if self.additional_dim_policy == "inline":
            (0,np.min((self.dim()+1, self.dim_m.upper_bound)))
        else:
            None

    def recompute_cache(self, data, dim = None):
        if dim == None:
            print("recompute_cache: No dim given")
            dim = self.dim_m.get()
        self.pred[dim] = self.prediction(data, dim)
        
        self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred[dim].shape[1])]
                                        for i in range(self.pred[dim].shape[0])])
        
        self.last_ll[dim] = self.ll_factors[dim].sum()
        
        
    def log_likelihood(self, data):
        self.recompute_cache(data)
        return self.last_ll
    
    def llike_function(self, data, rv_mdl = None, idx = None):
        if rv_mdl == None or idx == None:
            assert(rv_mdl == None and idx == None)
            ll_full_computation =  lambda: self.log_likelihood(data)
            return ll_full_computation
        else:
            if rv_mdl == self.dim_m:
                dim,_ = idx
                def ll_dim_with_caching():
                    #FIXME: work with presummed loglikelihood, only sum the changes
                    self.notify(data, rv_mdl, idx, dim = dim)
                    return self.last_ll[dim]
                return ll_dim_with_caching
            else:
                if self.additional_dim_policy == "inline":
                    d = np.int(np.min((self.dim_m.upper_bound, self.dim()+1)))
                    if d not in self.pred:
                        self.recompute_cache(data, d)
                    def ll_with_caching_inline_dim():
                        #FIXME: work with presummed loglikelihood, only sum the changes
                        self.notify(data, rv_mdl, idx, dim = d)
                        return self.last_ll[d]
                    return ll_with_caching_inline_dim
                else:
                    def ll_with_caching():
                        #FIXME: work with presummed loglikelihood, only sum the changes
                        self.notify(data, rv_mdl, idx)
                        return self.last_ll[self.dim()]
                    return ll_with_caching


def test_DimensionalityModel():
    from basicmodels import FixMatrixModel

    class DummyGlobModel:
        def __init__(self):
            self.data = stats.norm(3, 2).rvs((50, 3))
            self.too_much_data = FixMatrixModel(np.hstack((self.data, stats.norm(-30,1).rvs((50, 3)))))
            self.data = np.hstack((self.data, np.zeros((50, 3))))
            self.dim_m = DimensionalityModel(self.data.shape[1])
            self.dim_m.set_global_model(self)
            
        def llike_function(self, data, rv_mdl=None, idx = None):
            _, idx = idx
            print(idx)
            def llike():
                shape = self.too_much_data.get().shape
                dim_mask = np.zeros(shape)
                dim_mask[:, :idx] = 1.
                masked = self.too_much_data.get() * dim_mask
                #assert(False)
                #likelihood is negative squared error
                return  - np.sum((masked - self.data)**2)
            return llike
            
        def sample(self, num_samples):
            rval = []
            for i in range(num_samples):
                self.dim_m.sample(self.too_much_data)
                #print(self.dim_m.get())
                rval.append(deepcopy(self.dim_m.get()))
            return rval
    
    np.set_printoptions(precision=3, suppress = True)
    dgm = DummyGlobModel()
    relevance = dgm.sample(1000)
    print(np.average(relevance))
    return (relevance, dgm)


def test_ADDTwoFactorModel_dim_only():
    from basicmodels import FixMatrixModel
    
    dim_lv = 3
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, lv_mdl = FixMatrixModel(lv),
                              weight_mdl = FixMatrixModel(w),
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s

def test_ADDTwoFactorModel_dim_wm():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, lv_mdl = FixMatrixModel(lv),
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s

def test_ADDTwoFactorModel_dim_lvm():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, weight_mdl = FixMatrixModel(w),
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s
   
def test_ADDTwoFactorModel_dim_rv():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, lv_mdl = FixMatrixModel(lv),
                              weight_mdl = FixMatrixModel(w),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s

def test_ADDTwoFactorModel_dim_wm_rv():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, lv_mdl = FixMatrixModel(lv),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s

def test_ADDTwoFactorModel_dim_lvm_rv():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, weight_mdl = FixMatrixModel(w),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s
    
def test_ADDTwoFactorModel_all():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ADDTwoFactorModel(upper_bound, data, quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s
    

def test_ADDTwoFactorModel_dim_lvm_wm_laplace():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 6
    upper_bound = dim_obs - 1
    
    num_obs = 50
    
    lv = stats.norm(0,1).rvs((num_obs, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.uniform(0,1).rvs((num_obs, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    lvm = MatrixWithComponentPriorModel((num_obs, upper_bound), prior = ("uniform", (0, 1), {}), estimate_mean = False)
    
    mdl = ADDTwoFactorModel(upper_bound, data, lv_mdl = lvm,
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s