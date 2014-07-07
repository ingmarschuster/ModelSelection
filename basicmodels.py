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