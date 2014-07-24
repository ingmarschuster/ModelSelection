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

class AuxVarDimensionalityModel:
    def __init__(self, upper_dim_bound,
                       confidence = 40, # how confident are we that we need few dimensions. This has to be an int
                       current_dim_callback = None):
        self.current_dim_callback = current_dim_callback
        self.upper_bound = upper_dim_bound
        # our prior belief is that we should use few dimensions
        self.bin_param = np.array((upper_dim_bound, 1./upper_dim_bound))
        #The following sets the binomial prior to belief in dimensionality 1
        #to a degree determined by 'confidence'
        self.beta_param = np.array([confidence, (upper_dim_bound - 1) * confidence])
    
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
    
    def get(self):
        if self.current_dim_callback != None:
            return self.current_dim_callback()
        else:
            return self.glob_mdl.latent_dim()
            
    
    def new_dim(self):
        if self.get() >= self.upper_bound:
            return False
        else:
            return (stats.bernoulli.rvs(0.5) == 1)
            
    def new_dim_idx(self):
        p = np.array([1] * (self.get()+1))
        p = p - logsumexp(p)
        return np.argmax(np.random.multinomial(1, exp(p)))
    
    def logpmf(self, dim):
        if dim < 1 or dim > self.upper_bound:
            return -np.infty
        else:
            lpmf = stats.binom.logpmf(range(1, self.upper_bound+1), self.bin_param[0], self.bin_param[1])
            lpmf = lpmf - logsumexp(lpmf)
            return lpmf[dim - 1]
            
    def update(self):
        #update the beta prior on "success probability" binomial parameter 
        self.beta_param[0] += self.get()
        self.beta_param[1] += self.upper_bound - self.get()
        
        #sample binomial parameters from beta posterior
        self.bin_param[1] = stats.beta(self.beta_param[0], self.beta_param[1]).rvs()


def test_AuxVarDimensionalityModel():
    class DummyGlobalModel:
        dim = 9
        def latent_dim(self):
            return self.dim
    dm = AuxVarDimensionalityModel(10)
    gm = DummyGlobalModel()
    dm.set_global_model(gm)
    idxs = [dm.new_dim_idx() for i in range(5000)]
    u = np.unique(idxs)
    bucket_counts = np.array([(i, abs(idxs.count(i)/5000-0.1) < 10**-2) for i in u])
    
    ## test that bounds are inside
    assert(0 in u)
    assert(gm.latent_dim() in u)
    assert(bucket_counts.sum(0)[1] >= 8)
    
    ## test if logpmf respects dimensionality bounds
    assert(dm.logpmf(0) == - np.inf)
    assert(dm.logpmf(dm.upper_bound + 1) == - np.inf)
    ## test if dimensionality prior sums to one
    assert(logsumexp([dm.logpmf(i) for i in range(1,dm.upper_bound+1)]) == 0)
    
    ## test if new dimension is created in half of the cases
    assert((np.average([dm.new_dim() for i in range(1000)]) - 0.5) **2 < 10**-2)
    gm.dim = 10
    ## test if new dimension is never created when upper bound is hit
    assert(np.sum([dm.new_dim() for i in range(1000)]) == 0)
    lpmfs = [dm.logpmf(i) for i in range(0,dm.upper_bound+1)]
    
    ## test if model assigns maximum probability to 1 dimension
    ## before data is taken into account
    assert(np.argmax(lpmfs) == 1)
    
    ## test if prior adapts to new dimensionality
    gm.dim = 4
    for i in range(1000):
        dm.update()    
    lpmfs = [dm.logpmf(i) for i in range(0,dm.upper_bound+1)]
    assert(np.argmax(lpmfs) == 4)
    

##############################################################################


class ADDAuxVarTwoFactorModel:
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
    
    def __getcache__(self):
        return copy((self.wm.get(), self.lvm.get(), self.pred, self.ll_factors, self.last_ll))
    
    def __setcache__(self, cache, cp = False):
        if cp:
            cache = copy(cache)
        (self.wm.matr, self.lvm.matr, self.pred, self.ll_factors, self.last_ll) = cache
            
    def __init__(self, latent_dim_bound,
                       data,
                       lv_mdl = None,
                       weight_mdl = None,
                       remainvar_mdl = None,
                       kernel = np.dot,
                       quiet = False,
                       interleaved_fix_dim_sampling = False):
        #observations are expected in colums.
        if lv_mdl != None:
            self.lvm = lv_mdl
        else:
            self.lvm = MatrixWithComponentPriorModel((data.shape[0], 1),
                                                     prior = ("t", (500,), {}),
                                                     estimate_mean = False, var_dim_axis = 1)
        self.interleaved_fix_dim_sampling = interleaved_fix_dim_sampling
        if weight_mdl != None:
            self.wm = weight_mdl
        else:
            self.wm = MatrixWithComponentPriorModel((1, data.shape[1]),
                                                    prior = ("t", (2.099999,), {}),
                                                    var_dim_axis = 0,
                                                    estimate_mean = False)#,
                                                    #new_prior_func = "lambda mean: stats.t(2.099999, loc=mean)")
                                                    
        self.dim_m = AuxVarDimensionalityModel(latent_dim_bound)
        
        if remainvar_mdl != None:
            self.rvm = remainvar_mdl
        else:
            self.rvm = IsotropicRemainingVarModel(data.shape[1])
        self.kernel = kernel
        self.lvm.set_global_model(self)
        self.wm.set_global_model(self)
        self.rvm.set_global_model(self)
        self.dim_m.set_global_model(self)
        self.last_ll = np.infty
        self.quiet = quiet #Output likelihood in during sampling?
        #self.log_likelihood(data)
        
        self.data_mean = np.average(data,0)
        self.cache = {}
        self.recompute_cache(data)
        (self.pred, self.ll_factors, self.last_ll) = self.cache[None]
    

    
    def sample(self, data = None, num_samples = 1):
        rval = []        
        for it in range(num_samples):
            log_msg = "Iteration " + str(it) +" pre "+ str(self.last_ll)
            pre_ll = self.last_ll
            
            ######## BEGIN resample dimensionality ########
            if self.dim_m.new_dim():                
                idx = self.dim_m.new_dim_idx()
                log_msg += "; Adding dimension at %d" % idx
                self.wm.sample_new_dim(idx)
                self.lvm.sample_new_dim(idx)
            
            
            removal_llhood = []
            removal_prior = [self.dim_m.logpmf(self.latent_dim() - 1)
                                  - np.log(self.latent_dim())] * self.latent_dim()
            largest_model = self.__getcache__()
            cache = []
            for idx in range(self.latent_dim()):
                for rv in (self.wm, self.lvm):
                    rv.delete_dim(idx)
                self.log_likelihood(data)
                if self.interleaved_fix_dim_sampling:
                    log_msg += "; INTERLEAVING SAMPLING; "
                    for i in range(1):                        
                        for rv in np.random.permutation((self.wm, self.lvm)):
                            rv.sample(data)
                        self.rvm.sample(data)
                removal_llhood.append(self.last_ll)
                cache.append(self.__getcache__())
                self.__setcache__(largest_model, True)

            removal_prior.append(self.dim_m.logpmf(self.latent_dim()))
            removal_prior = np.array(removal_prior) #-  logsumexp(removal_prior)            
            
            removal_llhood.append(self.last_ll)            
            removal_llhood = np.array(removal_llhood) #- logsumexp(removal_llhood)
            
            removal_posterior = removal_prior + removal_llhood
            removal_posterior = removal_posterior - logsumexp(removal_posterior)
            
            log_msg +=  "\n%s\n"  % np.hstack((np.array((("prior", "lhood", "post "),)).T,
                                                                        np.around(np.vstack((removal_prior,
                                                                                             removal_llhood,
                                                                                             removal_posterior)), 1)
                                                                        ))
            remove_idx = np.argmax(np.random.multinomial(1, exp(removal_posterior)))
            if remove_idx == self.latent_dim(): #FIXME????? maybe > latent_dim()
                log_msg += "; Keep"
            else:
                log_msg += "; Removing %d " % remove_idx
                self.__setcache__(cache[remove_idx - 1])
                
            self.dim_m.update()
            ######## END resample dimensionality ########
            
            for i in range(2):
                for rv in np.random.permutation((self.wm, self.lvm, self.rvm)):
                    rv.sample(data)
                    if rv == self.wm:
                        log_msg += "; sampled WM: "
                    elif rv == self.lvm:
                        log_msg += "; sampled LVM: "
                    else:
                        log_msg += "; sampled RV: "
                    log_msg += str(self.last_ll)
            log_msg += "\t(dim %d)\n" % self.dim_m.get()
            if not self.quiet:
                print(log_msg)
            rval.append(deepcopy(self))
        return rval
   

    def notify(self, data, rv_mdl, idx):
        (row, col) = idx
        if rv_mdl == self.lvm:
            self.pred[row,:] = (self.lvm.get()[row,:].dot(self.wm.get()) + self.data_mean)
            tmp = np.array([stats.norm(self.pred[row,j], self.rvm.get()[j,j]).logpdf(data[row,j])
                                               for j in range(self.pred.shape[1])])
            self.ll_factors[row,:] = tmp
        elif rv_mdl == self.wm:
            self.pred[:, col] = (self.lvm.get().dot(self.wm.get()[:, col]) + self.data_mean[col])
            self.ll_factors[:, col] = np.array([stats.norm(self.pred[i,col], self.rvm.get()[col,col])
                                                        .logpdf(data[i,col]) for i in range(self.pred[:, col].shape[0])])
        else:
            self.ll_factors = np.array( [[stats.norm(self.pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred.shape[1])]
                                        for i in range(self.pred.shape[0])])
        self.last_ll = np.sum(self.ll_factors)

    def prediction(self, data = None, exclude_dim = None):
        pred = self.kernel(self.lvm.get(exclude_dim), self.wm.get(exclude_dim))
        if data != None:
            if data.shape != pred.shape:
                raise IndexError("Data and Model of different dimensions", data.shape, "vs", pred.shape)
            else:
                return pred + self.data_mean
        else:
            return pred
    
    def latent_dim(self):
        return self.wm.get().shape[0]

    def recompute_cache(self, data, exclude_dim = None):
        assert(exclude_dim == None or
               (np.int(exclude_dim) == exclude_dim and exclude_dim >= 0))
        pred = self.prediction(data, exclude_dim)
        ll_factors = np.array( [[stats.norm(pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(pred.shape[1])]
                                        for i in range(pred.shape[0])])
        last_ll = ll_factors.sum()
        
        self.cache[exclude_dim] = (pred, ll_factors, last_ll)
        return self.cache[exclude_dim]
        
        
    def log_likelihood(self, data):        
        self.recompute_cache(data, exclude_dim = None)
        (self.pred, self.ll_factors, self.last_ll) = self.cache[None]
        return self.last_ll
    
    def llike_function(self, data, rv_mdl = None, idx = None, exclude_dim = None):
        if rv_mdl == None or idx == None:
            assert(rv_mdl == None and idx == None)
            ll_full_computation =  lambda: self.log_likelihood(data)
            return ll_full_computation
        else:
            if rv_mdl == self.dim_m:
                assert(exclude_dim != None)
                def ll_dim_with_caching():
                    self.recompute_cache(data, exclude_dim = exclude_dim)
                    #FIXME: work with presummed loglikelihood, only sum the changes
                    return self.cache[exclude_dim][-1]
                return ll_dim_with_caching
            else:
                def ll_with_caching():
                    #FIXME: work with presummed loglikelihood, only sum the changes
                    self.notify(data, rv_mdl, idx)
                    return self.last_ll
                return ll_with_caching


def test_AuxVarDimensionalityModel():
    from basicmodels import FixMatrixModel

    class DummyGlobModel:
        def __init__(self):
            self.data = stats.norm(3, 2).rvs((50, 3))
            self.too_much_data = FixMatrixModel(np.hstack((self.data, stats.norm(-30,1).rvs((50, 3)))))
            self.data = np.hstack((self.data, np.zeros((50, 3))))
            self.dim_m = AuxVarDimensionalityModel(self.data.shape[1])
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


def test_ADDAuxVarTwoFactorModel_dim_only():
    from basicmodels import FixMatrixModel
    
    num_obs = 50
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.t(500).rvs((num_obs, dim_lv))
    w = stats.t(2.099999).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, np.zeros((num_obs, upper_bound - dim_lv))))
    w = np.vstack((w, np.zeros((upper_bound - dim_lv, dim_obs))))
    
    lv_mdl = FixMatrixModel(lv, var_dim_axis = 1)
    w_mdl  = FixMatrixModel(w, var_dim_axis = 0)
    
    while w_mdl.get().shape[0] > 1:
        idx = w_mdl.get().shape[0] - 1
        lv_mdl.delete_dim(idx)
        w_mdl.delete_dim(idx)
    
    mdl = ADDAuxVarTwoFactorModel(upper_bound, data, lv_mdl = lv_mdl,
                              weight_mdl = w_mdl,
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              quiet = False)
    s = mdl.sample(data, 500)
    assert(dim_lv == np.round(np.average([samp.dim_m.get() for samp in s])))
    return s

def test_ADDAuxVarTwoFactorModel_dim_wm(interleaved_fix_dim_sampling= False):
    from basicmodels import FixMatrixModel
    
    num_obs = 50
    dim_lv = 2
    dim_obs = 5
    upper_bound = dim_obs - 1
    
    lv = stats.t(500).rvs((num_obs, dim_lv))
    w = stats.t(2.099999).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    
    lv = np.hstack((lv, np.zeros((50, upper_bound - dim_lv))))
    
    lv_mdl = FixMatrixModel(lv, var_dim_axis = 1)
    
    while lv_mdl.get().shape[1] > 1:
        idx = lv_mdl.get().shape[1] - 1
        lv_mdl.delete_dim(idx)
    
    mdl = ADDAuxVarTwoFactorModel(upper_bound, data, lv_mdl = lv_mdl,
                              remainvar_mdl = FixMatrixModel(np.eye(dim_obs)),
                              interleaved_fix_dim_sampling = interleaved_fix_dim_sampling,
                              quiet = False)
    s = mdl.sample(data, 100)
    assert(dim_lv == np.round(np.average([samp.dim_m.get() for samp in s])))
    return s

    
def test_ADDAuxVarTwoFactorModel_all(num_obs = 100, num_samples = 100,
                                     dim_lv = 2, dim_obs = 5,
                                     interleaved_fix_dim_sampling = False):
    from basicmodels import FixMatrixModel

    assert(dim_lv < dim_obs)
    upper_bound = dim_obs - 1
    
    lv = stats.t(500).rvs((num_obs, dim_lv))
    w = stats.t(2.099999).rvs((dim_lv, dim_obs))
    data = lv.dot(w)
    noise_data = data + stats.norm.rvs(0,0.4, size=data.shape)

    
    mdl = ADDAuxVarTwoFactorModel(upper_bound, noise_data, quiet = False, interleaved_fix_dim_sampling = interleaved_fix_dim_sampling)
    print(np.average([mdl.dim_m.new_dim_idx() for _ in range(5000)]))
    s = mdl.sample(noise_data, num_samples)
    avg_inferred_dim = np.average([samp.dim_m.get() for samp in s])
    if dim_lv != np.round(avg_inferred_dim):
        msg = "FAILED. "
    else:
        msg = "passed. "
    msg += "%d true dimensions, %.2f inferred." % (dim_lv, avg_inferred_dim)
    print(msg, file=sys.stderr)
    return s

## 100 obs, 100 samples
#update theta twice before resampling dimensionality: too low (1)
#update theta a) before b) after resampling dimensionality: too low (1)
#update theta twice after resampling dimensionality: too high (4)
#update theta a) before b) twice after resampling dimensionality: too low (1)