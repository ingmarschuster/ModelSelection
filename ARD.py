# -*- coding: utf-8 -*-
"""
Created on Sat May 31 07:31:34 2014

@author: arbeit
"""


from __future__ import division, print_function
import numpy as np
import datetime
import numpy.random as npr
import scipy.stats as stats
from scipy.misc import logsumexp
from copy import deepcopy,copy
import pickle
import sys
from bijections import logistic

from basicmodels import MatrixWithComponentPriorModel,IsotropicRemainingVarModel


import collections

from slice_sampling import slice_sample_component#, slice_sample_component_double

class RelevanceModel:
    def __init__(self, dim):
        self.matr = np.zeros((dim,1))
        # our prior belief is that each dimension might be important
        self.norm_param = np.array([[30, 10] ] * dim)
        
        #first dimension: we want to take each dimension as important in the beginnig
        #second dimension: after 20 samples, prior and data have equal say about wether the
        #                 dimension is important
        #print("new")
        self.norm_gamma_param = np.array([[50, 30, 40, 1/40] ] * dim) 
    
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
    
    def get(self, dim = None):
        if dim == None:
            return logistic(self.matr)
        else:
            return logistic(self.matr[dim])
            
    def sample(self, data = None,  dimrange = None):
        #print("weight pre",llike(), self.matr)
        if dimrange == None:
            dimrange = (0, len(self.matr))
        for dim in np.random.permutation(range(*dimrange)):
            log_likelihood = self.glob_mdl.llike_function(data, rv_mdl=self, idx = (dim,dim))
            cur_ll =  slice_sample_component(self.matr,
                                             dim,
                                             log_likelihood,
                                             stats.norm(self.norm_param[dim, 0], self.norm_param[dim, 1]),
                                             None,
                                             100)
            x = self.matr[dim] # this is the one posterior sample we drew
            (o_mu, o_nu, o_alpha, o_beta) = tuple(self.norm_gamma_param[dim, :])
            mu = (o_nu* o_mu + x) / (o_nu + 1)
            nu = o_nu + 1
            alpha = o_alpha + 0.5
            beta = o_beta + (1*o_nu + (x-o_mu)**2)/((o_nu + 1) *2)
            self.norm_gamma_param[dim] = np.array((mu, nu, alpha, beta))
            
            #now resample parameters of the normal
            
            var = stats.invgamma(alpha, scale=beta).rvs()
            self.norm_param[dim, 1] = np.sqrt(var)
            self.norm_param[dim, 0] = stats.norm(mu, np.sqrt(var/nu)).rvs()



class ARDTwoFactorModel:
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
                                                    
        self.relevm = RelevanceModel(latent_dim_bound)
        
        if remainvar_mdl != None:
            self.rvm = remainvar_mdl
        else:
            self.rvm = IsotropicRemainingVarModel(data.shape[1])
        self.kernel = kernel
        self.lvm.set_global_model(self)
        self.wm.set_global_model(self)
        self.rvm.set_global_model(self)
        self.relevm.set_global_model(self)
        self.last_ll = np.infty
        self.quiet = quiet #Output likelihood in during sampling?
        #self.log_likelihood(data)
        
        self.data_mean = np.average(data,0)
        self.pred = []
        self.w_cache = self.wm.get() * self.relevm.get()
        self.ll_factors = []
        self.last_lls = []
        self.recompute_cache(data)
        print(self.ll_factors.shape, self.pred.shape)
        
    def sample(self, data = None, num_samples = 1):
        rval = []        
        for it in range(num_samples):
            log_msg = "Iteration " + str(it) +" pre "+ str(self.last_ll)
            pre_ll = self.last_ll
            # randomize resampling order to counter selection bias
            for rv in npr.permutation((self.wm, self.lvm)):
                rv.sample(data)
                log_msg = log_msg + "; sampled "
                if rv == self.wm:
                    log_msg = log_msg + "WM "
                else:
                    log_msg = log_msg + "LVM "
                log_msg = log_msg + str(self.last_ll)
            self.rvm.sample(data)
            log_msg = log_msg + "; sampled RV: "+ str(self.last_ll)
            self.relevm.sample(data)
            log_msg = log_msg + "; sampled Relev: "+ str(self.last_ll)
            if not self.quiet:
                print(log_msg)
            print(np.hstack((self.relevm.get(), self.relevm.norm_param)))
            rval.append(deepcopy(self))
        return rval
    
    def prediction(self, data = None):
        pred = self.kernel(self.lvm.get(), self.w_cache)
        if data != None:
            if data.shape != pred.shape:
                raise IndexError("Data and Model of different dimensions", data.shape, "vs", pred.shape)
            else:
                return pred + self.data_mean
        else:
            return pred
   

    def notify(self, data, rv_mdl, idx):
        (row, col) = idx
        if rv_mdl == self.lvm:
            self.pred[row,:] = (self.lvm.get()[row,:].dot(self.w_cache) + self.data_mean)
            tmp = np.array([stats.norm(self.pred[row,j], self.rvm.get()[j,j]).logpdf(data[row,j])
                                               for j in range(self.pred.shape[1])])
            self.ll_factors[row,:] = tmp
        elif rv_mdl == self.wm:
            self.pred[:, col] = (self.kernel(self.lvm.get(), self.w_cache[:, col]) + self.data_mean[col])
            self.ll_factors[:, col] = np.array([stats.norm(self.pred[i,col], self.rvm.get()[col,col])
                                                        .logpdf(data[i,col]) for i in range(self.pred[:, col].shape[0])])
        elif rv_mdl == self.relevm:
            self.w_cache[row, :] = self.wm.get()[row, :] * self.relevm.get(row)
            self.pred = self.prediction(data)
            self.ll_factors = np.array( [[stats.norm(self.pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred.shape[1])]
                                        for i in range(self.pred.shape[0])])
        
            self.last_ll = self.ll_factors.sum()
        else:
            self.ll_factors = np.array( [[stats.norm(self.pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred.shape[1])]
                                        for i in range(self.pred.shape[0])])
        self.last_ll = np.sum(self.ll_factors)
    
    def recompute_cache(self, data):
        self.w_cache = self.wm.get() * self.relevm.get()
        self.pred = self.prediction(data)
        
        self.ll_factors = np.array( [[stats.norm(self.pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred.shape[1])]
                                        for i in range(self.pred.shape[0])])
        
        self.last_ll = self.ll_factors.sum()
        
    def log_likelihood(self, data):
        self.recompute_cache(data)
        return self.last_ll
    
    def llike_function(self, data, rv_mdl = None, idx = None):
        if rv_mdl == None or idx == None:
            assert(rv_mdl == None and idx == None)
            ll_full_computation =  lambda: self.log_likelihood(data)
            return ll_full_computation
        else:
            def ll_with_caching():
                #FIXME: work with presummed loglikelihood, only sum the changes
                self.notify(data, rv_mdl, idx)
                return self.last_ll
            return ll_with_caching


def test_RelevanceModel():
    from transdim_jeff import FixMatrixModel

    class DummyGlobModel:
        def __init__(self):
            self.data = stats.norm(3, 2).rvs((4, 3))
            self.too_much_data = FixMatrixModel(np.hstack((self.data, stats.norm(-30,1).rvs((4, 3)))))
            self.data = np.hstack((self.data, np.zeros((4, 3))))
            self.relevm = RelevanceModel(self.data.shape[1])
            self.relevm.set_global_model(self)
            
        def llike_function(self, data, rv_mdl=None, idx = None):
            def llike():
                relevance_weighted = self.too_much_data.get() * self.relevm.get().T
                #likelihood is negative squared error
                return  - np.sum((relevance_weighted - self.data)**2)
            return llike
            
        def sample(self, num_samples):
            rval = []
            for i in range(num_samples):
                self.relevm.sample(self.too_much_data)
                print(self.relevm.get().T)
                rval.append(deepcopy(np.hstack((self.relevm.get(), self.relevm.norm_param))))
            return rval
    
    np.set_printoptions(precision=3, suppress = True)
    dgm = DummyGlobModel()
    relevance = dgm.sample(1000)
    avg = relevance[0]/ len(relevance)
    for s in relevance[1:]:
        avg += s/ len(relevance)
    #avg / len(relevance)
    rel = np.average(avg[:3],0)
    irrel = np.average(avg[3:],0)
    print("Relevant", rel)
    assert(rel[0] > 0.5 and rel[1] > 0)
    print("Irrelevant", irrel)
    assert(irrel[0] < 0.5 and irrel[1] < 0)
    return relevance


def test_ARDTwoFactorModel():
    from transdim_jeff import FixMatrixModel
    
    lv = stats.norm(0,1).rvs((50,2))
    w = stats.norm(0,10).rvs((2,5))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50,2))))
    w = np.vstack((w, stats.norm(0,10).rvs((2,5))))
    mdl = ARDTwoFactorModel(4, data, lv_mdl = FixMatrixModel(lv),
                              weight_mdl = FixMatrixModel(w),
                              remainvar_mdl = FixMatrixModel(np.eye(5)),
                              quiet = False)
    s = mdl.sample(data, 500)
    return s

def test_ARDTwoFactorModel_dim_lvm():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ARDTwoFactorModel(upper_bound, data, weight_mdl = FixMatrixModel(w),
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 100)
    return s

def test_ARDTwoFactorModel_dim_lvm_wm():
    from basicmodels import FixMatrixModel
    
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.norm(0,1).rvs((50, dim_lv))
    w = stats.norm(0,10).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, stats.norm(0,1).rvs((50, upper_bound - dim_lv))))
    w = np.vstack((w, stats.norm(0,10).rvs((upper_bound - dim_lv, dim_obs))))
    
    mdl = ARDTwoFactorModel(upper_bound, data, 
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 50)
    assert(dim_lv == np.round(np.average([samp.dim_m.matr for samp in s])))
    return s