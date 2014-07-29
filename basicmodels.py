# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:11:47 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function
import scipy.stats as stats
import numpy as np
from copy import copy, deepcopy

from slice_sampling import slice_sample_component

class MatrixWithComponentPriorModel:
    def __init__(self, shape, prior = ("norm", (0, 1), {}), estimate_mean = False, new_prior_func = None, var_dim_axis = None):
        self.prior_pickle = prior
        self.new_prior_func_pickle = new_prior_func
        if prior != None:
            self.prior = eval("stats."+prior[0])(*prior[1], **prior[2])
        self.matr = self.prior.rvs(shape)
        self.estimate_mean = estimate_mean
        if estimate_mean:
            assert(self.new_prior_func_pickle != None)            
            self.new_prior_func = eval(self.new_prior_func_pickle)
        assert(var_dim_axis < 2)
        self.var_dim_axis = var_dim_axis
    
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
            
    def sample(self, data = None,  row_col_ranges = (None, None)):
        rowrange, colrange = row_col_ranges
        #print("weight pre",llike(), self.matr)
        prior = [self.prior] * np.prod(self.matr.shape)
        if self.estimate_mean:
            prior = [self.new_prior_func(cur) for cur in self.matr.flat]
        if rowrange == None:
            rowrange = (0, self.matr.shape[0])
        if colrange == None:
            colrange = (0, self.matr.shape[1])
        for row in np.random.permutation(range(*rowrange)):
            for col in np.random.permutation(range(*colrange)):
                log_likelihood = self.glob_mdl.llike_function(data, self, (row, col))
                idx = np.ravel_multi_index((row, col), self.matr.shape)
                cur_ll =  slice_sample_component(self.matr.flat,
                                                 idx,
                                                 log_likelihood,
                                                 prior[idx],
                                                 None,
                                                 100)
        #slice_sample_all_components(self.matr, self.glob_mdl.llike_function(data), prior)#

    def get(self, exclude_idx = None, axis = None):
        if exclude_idx == None or exclude_idx < 0:
            return self.matr
        assert(self.var_dim_axis != None or
               axis != None)
        if axis == None:
            axis = self.var_dim_axis
        assert(axis < len(self.matr.shape))
        return np.delete(self.matr, exclude_idx, axis=axis)
    
    def sample_new_dim(self, insert_at, axis = None):
        assert(self.var_dim_axis != None or
               axis != None)
        if axis == None:
            axis = self.var_dim_axis
        assert(axis < len(self.matr.shape))
        if self.estimate_mean:
            prior = self.new_prior_func(np.average(self.matr))
        else:
            prior = self.prior
        self.matr = np.insert(self.matr, insert_at, prior.rvs((np.prod(self.matr.shape)//self.matr.shape[axis], )), axis=axis)
    
    def delete_dim(self, idx, axis = None):
        self.matr = self.get(exclude_idx = idx, axis = axis)
        
    def __getstate__(self):
        rval = copy(self.__dict__)
        for attr in ("new_prior_func", "prior"):
            try:
                rval.pop(attr)
            except:
                pass
        return rval
    
    def __setstate__(self, state):
        self.__dict__ = state
        try:
            if state["new_prior_func_pickle"] != None:
                self.new_prior_func = eval(state["new_prior_func_pickle"])
        except:
            pass
        self.prior = eval("stats."+state["prior_pickle"][0])(*state["prior_pickle"][1], **state["prior_pickle"][2])



#For random variables you dont want to sample
class FixMatrixModel:
    def __init__(self, matrix, var_dim_axis = None):
        self.orig = matrix
        self.matr = matrix
        self.var_dim_axis = var_dim_axis
        
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
    
    def get(self, exclude_idx = None, axis = None):
        if exclude_idx == None or exclude_idx < 0:
            return self.matr
        assert(self.var_dim_axis != None or
               axis != None)
        if axis == None:
            axis = self.var_dim_axis
        assert(axis < len(self.matr.shape))
        return np.delete(self.matr, self.matr.shape[axis]-1, axis=axis)
    
    def sample_new_dim(self, insert_at, axis = None):
        assert(self.var_dim_axis != None or
               axis != None)
        if axis == None:
            axis = self.var_dim_axis
        assert(axis < len(self.matr.shape))
        new_dims = self.matr.shape[axis] + 1
        self.matr = self.orig
        while self.matr.shape[axis] != new_dims:
            self.delete_dim(3)
    
    def delete_dim(self, idx, axis = None):
        if axis == None:
            axis = self.var_dim_axis
        self.matr = self.get(exclude_idx = idx, axis = axis)
        
    def sample(self, data = None, row_col_ranges=None):
        pass
    

class IsotropicRemainingVarModel:
    var = np.array([1.])
    vshape, vscale = 1,1
    glob_mdl = None
    
    def __init__(self, dim):
        self.matr = np.eye(dim)
        self.var = MatrixWithComponentPriorModel((1,1), prior = ("gamma", (1.,), {"scale":1.}))
    
    def get(self):
        return self.matr * self.var.matr[0]
    
    def set_global_model(self, glob_mdl):
        self.glob_mdl = glob_mdl
        self.var.set_global_model(glob_mdl)
    
    def sample(self, data = None):
        self.var.sample(data)
        


class GlobalTwoFactorModel:
    def __init__(self, latent_dim,
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
            self.lvm = MatrixWithComponentPriorModel((data.shape[0], latent_dim),
                                                     prior =  ("t", (500,), {}),
                                                     estimate_mean = False)
        if weight_mdl != None:
            self.wm = weight_mdl
        else:
            self.wm = MatrixWithComponentPriorModel((latent_dim, data.shape[1]),
                                                    prior =  ("t", (2.099999,), {}),
                                                    estimate_mean = False)

        if remainvar_mdl != None:
            self.rvm = remainvar_mdl
        else:
            self.rvm = IsotropicRemainingVarModel(data.shape[1])
        self.kernel = kernel
        self.lvm.set_global_model(self)
        self.wm.set_global_model(self)
        self.rvm.set_global_model(self)
        self.last_ll = np.infty
        self.quiet = quiet #Output likelihood in during sampling?
        self.log_likelihood(data)
        
    def sample(self, data = None, num_samples = 1):
        rval = []        
        for it in range(num_samples):
            log_msg = "Iteration " + str(it) +" pre "+ str(self.last_ll)
            pre_ll = self.last_ll
            # randomize resampling order to counter selection bias
            for rv in np.random.permutation((self.rvm, self.wm, self.lvm)):
                rv.sample(data)
                log_msg = log_msg + "; sampled "+rv.__class__.__name__+": "+ str(self.last_ll)
            if not self.quiet:
                print(log_msg)
            rval.append(deepcopy(self))
        return rval
    
    def prediction(self, data = None):
        pred = self.kernel(self.lvm.get(), self.wm.get())
        if data != None:
            if data.shape != pred.shape:
                raise IndexError("Data and Model of different dimensions", data.shape, "vs", pred.shape)
            else:
                return pred + np.average(data,0)
        else:
            return pred
   

    def notify(self, data, rv_mdl, idx):
        (row, col) = idx
        if rv_mdl == self.lvm:
            self.pred[row,:] = (self.lvm.get()[row,:].dot(self.wm.get()) + self.data_mean)
            tmp = np.array([stats.norm(self.pred[row,j], self.rvm.get()[j,j]).logpdf(data[row,j])
                                               for j in range(self.pred.shape[1])])
            self.ll_factors[row,:] = tmp
        elif rv_mdl == self.wm:
            self.pred[:, col] = (self.kernel(self.lvm.get(), self.wm.get()[:, col]) + self.data_mean[col])
            self.ll_factors[:, col] = np.array([stats.norm(self.pred[i,col],
                                                           self.rvm.get()[col,col]).logpdf(data[i,col])
                                                            for i in range(self.pred[:, col].shape[0])])
            
        else:
            self.ll_factors = np.array( [[stats.norm(self.pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred.shape[1])]
                                        for i in range(self.pred.shape[0])])
        self.last_ll = np.sum(self.ll_factors)
        
    def neg_sq_error(self, data):
        self.recompute_cache(data)
        self.last_ll = - np.sum((self.pred - (data - self.data_mean))**2)        
        return self.last_ll
    
    def recompute_cache(self, data):
        self.data_mean = np.average(data,0)
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
        
        if False:
            if rv_mdl == self.lvm or rv_mdl == self.wm:
                assert(rv_mdl == self.lvm or rv_mdl == self.wm)
                (row, col) = idx
                data_mean = np.average(data,0)
                pred = self.prediction(data)
                precomp_ll = 0
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[1]):
                        if ((rv_mdl == self.lvm and i == row) or
                            (rv_mdl == self.wm and j == col)):
                            continue
                        
                        precomp_ll += stats.norm(pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                        
            
                if rv_mdl == self.lvm:
                    def ll_fix_lv_variable():
                        pred = (self.lvm.get()[row,:].dot(self.wm.get()) + data_mean).flat
                        self.last_ll = precomp_ll + np.sum([stats.norm(pred[i], self.rvm.get()[i,i]).logpdf(data[row, i])
                                                            for i in range(len(pred))])
                                        #multiv_norm_logpdf(, self.lvm.get()[row, :].dot(self.wm.get()) + data_mean,  self.rvm.get())
                        return self.last_ll
                    return ll_fix_lv_variable
                elif rv_mdl == self.wm:
                    def ll_fix_weight_variable():
                        pred = (self.kernel(self.lvm.get(), self.wm.get()[:, col]) + data_mean[col]).flat
                        d = data[:,col].flat
                        self.last_ll = precomp_ll + np.sum([stats.norm(pred[i], self.rvm.get()[col,col]).logpdf(d[i]) for i in range(len(d))])
                        return self.last_ll
                    return ll_fix_weight_variable
            else:
                pred = self.prediction(data)
                def ll_for_remain_var_variable():
                    self.last_ll = np.sum([stats.  fnorm(pred[i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                           for i in range(pred.shape[0])
                                           for j in range(pred.shape[1])])
                    return self.last_ll
                return ll_for_remain_var_variable
                

def test_precomputed_ll():
    rng = np.random.RandomState(42)
    num_obs = 10
    dim_obs = 6
    dim_lv = 3
    orig_lv = rng.standard_normal((num_obs, dim_lv))
    orig_w = rng.normal(0, 10, (orig_lv.shape[1], dim_obs))
    
    orig_data = orig_lv.dot(orig_w)
    
    mdl = GlobalTwoFactorModel(dim_lv, orig_data, quiet = True)
    unopt_ll = mdl.llike_function(orig_data)
    opt_lv_ll = mdl.llike_function(orig_data, mdl.lvm, (0,0))
    opt_w_ll = mdl.llike_function(orig_data, mdl.wm, (0,0))
    ll = {}
    for func in (opt_lv_ll, opt_w_ll, unopt_ll):
        ll[func] = func()
    print([ll[k] for k in (unopt_ll, opt_lv_ll, opt_w_ll)])
    assert(np.abs(ll[unopt_ll] - ll[opt_lv_ll]) < 0.001)
    assert(np.abs(ll[unopt_ll] - ll[opt_w_ll]) < 0.001)
    
    orig_lvm = mdl.lvm.matr[0,0]
    orig_wm = mdl.wm.matr[0,0]
    
    for func in (unopt_ll, opt_lv_ll, opt_w_ll):
        mdl.lvm.matr[0,0] = orig_lvm
        mdl.recompute_cache(orig_data)
        mdl.lvm.matr[0,0] = 4
        ll[func] = func()
    print([ll[k] for k in (unopt_ll, opt_lv_ll, opt_w_ll)])
    assert(np.abs(ll[unopt_ll] - ll[opt_lv_ll]) < 0.001)
    assert(np.abs(ll[unopt_ll] - ll[opt_w_ll]) > 0.001)
    
    for func in (unopt_ll, opt_lv_ll, opt_w_ll):
        mdl.wm.matr[0,0] = orig_wm
        mdl.recompute_cache(orig_data)
        mdl.wm.matr[0,0] = 4
        ll[func] = func()
    print([ll[k] for k in (unopt_ll, opt_lv_ll, opt_w_ll)])
    assert(np.abs(ll[unopt_ll] - ll[opt_lv_ll]) > 0.001)
    assert(np.abs(ll[unopt_ll] - ll[opt_w_ll]) < 0.001)
 
 
def test_all(num_obs = 100, num_samples = 100,
             dim_lv = 2, dim_obs = 5,
             interleaved_fix_dim_sampling = False,
             lv_prior = stats.t(500),
             w_prior = stats.t(2.099999),
             remvar_prior = stats.gamma(1, scale=1),
             fix_dim_moves = 0, dim_removed_resamples = 1, dim_added_resamples=1):

    assert(dim_lv < dim_obs)
    true_lv = lv_prior.rvs((num_obs, dim_lv))
    true_w = w_prior.rvs((dim_lv, dim_obs))
    remvar = remvar_prior.rvs((1,1))
    data = true_lv.dot(true_w)
    noise_data = data + stats.norm.rvs(0,0.4, size=data.shape)
    

    lv = stats.norm.rvs(0,lv_prior.var(), size=(num_obs, 1))
    w = stats.norm.rvs(0,w_prior.var(), size=(1, dim_obs))
    
    theta = {"lv": lv, "w": w, "rv": remvar}
    
    #llhood = lambda: np.sum(stats.norm(0, theta["rv"]).logpdf(noise_data - theta["lv"].dot(theta["w"])))
    mdl = GlobalTwoFactorModel(dim_lv, data)
    samp = mdl.sample(noise_data, num_samples)
    
    #print(count_dim(samp), file=sys.stderr)

    return (data, samp)

if __name__ == "__main__":
    test_all(num_samples=10)