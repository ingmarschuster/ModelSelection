# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:23:58 2014

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

from ARD import RelevanceModel, ARDTwoFactorModel
from ADD import DimensionalityModel, ADDTwoFactorModel
from basicmodels import *

import collections

from slice_sampling import slice_sample_component

# <codecell>



        

class NaiveTransDimModel:
    
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
                       dim_spec_variance = True,
                       integr_rvar = True,
                       remainvar_mdl = None,
                       streaming_evidence = True,
                       kernel = np.dot,
                       quiet = False):
        #observations are expected in colums.
        print(latent_dim_bound,"dim bound", data.shape[1], "observables")
        assert(latent_dim_bound >= 1)
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
        if dim_spec_variance:
            self.rvm = []
            self.rvm_to_dim = {}
            for dim in range(latent_dim_bound):
                if remainvar_mdl != None:
                    self.rvm.append(deepcopy(remainvar_mdl))
                else:
                    self.rvm.append(IsotropicRemainingVarModel(data.shape[1]))
                self.rvm[dim].set_global_model(self)
                self.rvm_to_dim[self.rvm[dim]] = dim
                self.rvm_to_dim[self.rvm[dim].var] = dim
        else:
            if remainvar_mdl != None:
                self.rvm = remainvar_mdl
            else:
                self.rvm = IsotropicRemainingVarModel(data.shape[1])
            self.integr_rvar = integr_rvar
            self.rvm.set_global_model(self)
            
        self.kernel = kernel
        self.streaming_evidence = streaming_evidence
        self.lvm.set_global_model(self)
        self.wm.set_global_model(self)
        
        self.last_ll = np.infty
        self.quiet = quiet #Output likelihood in during sampling?
        #self.log_likelihood(data)
        
        self.data_mean = np.average(data,0)
        self.pred = []
        self.ll_factors = []
        self.last_lls = []
        
        
        for dim in range(latent_dim_bound):            
            self.recompute_cache(data, dim)
    
    def dimension_specific_variance(self):
        return isinstance(self.rvm, collections.Sequence)
        
    def sample(self, data = None, num_samples = 1, random_order = True):
        model_evidence = None
        rval = []
        for it in range(num_samples):
            log_msg = "Iteration " + str(it) +" pre "+ str(self.last_ll) 
            pre_ll = self.last_ll
            msg = {}
            order = range(self.lvm.get().shape[1])
            if random_order:
                # randomize resampling order to counter selection bias
                log_msg += "(random order)"
                order = npr.permutation(self.lvm.get().shape[1])
            log_msg += "\n"
            for dim in order:
                self.recompute_cache(data, dim)
                self.wm.sample(data,((dim,dim+1), None))
                msg[dim] = " dim " + str(dim) + "\tsampled Weight: "+ str(self.last_lls[dim])
                self.lvm.sample(data,(None, (dim,dim+1)))
                msg[dim] = msg[dim]+  " sampled LV: "+ str(self.last_lls[dim]) + "\n"
                if self.dimension_specific_variance():
                    #each dimension has its individual variance matrix
                    self.rvm[dim].sample(data)
            for dim in range(self.lvm.get().shape[1]):
                log_msg += msg[dim]
                
            if not self.dimension_specific_variance():
                #all dimensions have a common variance matrix
                self.rvm.sample(data)
                if not self.integr_rvar:
                    #print("resampled non-specific non-integrated variance, now updating likelihoods", file=sys.stderr)
                    self.notify(data, self.rvm, (0,0))
                
            #self.dim = npr.multinomial(1, np.exp(np.array(self.last_lls) - logsumexp(self.last_lls)))
            rval.append(deepcopy(self))
            
            if not self.quiet:
                if self.streaming_evidence:
                    if model_evidence == None:
                        model_evidence = np.array(self.last_lls)
                    else:
                        model_evidence = logsumexp(np.vstack((model_evidence, np.array(self.last_lls))),0)   
                    #print(np.vstack((model_evidence, np.array(self.last_lls))), np.log(len(rval)), model_evidence)
                    norm_evidence = model_evidence - np.log(len(rval))
                    log_msg += "Model Evidence " + str(np.round(np.exp(norm_evidence - logsumexp(norm_evidence)), decimals = 3))+ "\n" # 
                print(log_msg)
        return rval
    
    def prediction(self, data, dim):
        pred = self.kernel(self.lvm.get()[:, :dim+1], self.wm.get()[:dim+1, :])
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
            dim = col
            if not self.dimension_specific_variance():
                variance = self.rvm.get()
            else:
                variance = self.rvm[dim].get()
            self.pred[dim][row,:] = (self.lvm.get()[row,:dim+1].dot(self.wm.get()[:dim+1,:]) + self.data_mean)
            tmp = np.array([stats.norm(self.pred[dim][row,j], variance[j,j]).logpdf(data[row,j])
                                               for j in range(self.pred[dim].shape[1])])
            self.ll_factors[dim][row,:] = tmp
        elif rv_mdl == self.wm:
            dim = row
            if not self.dimension_specific_variance():
                variance = self.rvm.get()
            else:
                variance = self.rvm[dim].get()
            self.pred[dim][:, col] = (self.kernel(self.lvm.get()[:, :dim+1], self.wm.get()[:dim+1, col]) + self.data_mean[col])
            self.ll_factors[dim][:, col] = np.array([stats.norm(self.pred[dim][i,col],
                                                           variance[col, col]).logpdf(data[i,col])
                                                            for i in range(self.pred[dim][:, col].shape[0])])
        else:
            if not self.dimension_specific_variance():
                for dim in range(len(self.ll_factors)):
                    self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                              for i in range(self.pred[dim].shape[0])]
                                                for j in range(self.pred[dim].shape[1])])
            elif rv_mdl in self.rvm_to_dim:
                dim = self.rvm_to_dim[rv_mdl]
                self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], self.rvm[dim].get()[j,j]).logpdf(data[i,j])
                                          for i in range(self.pred[dim].shape[0])]
                                            for j in range(self.pred[dim].shape[1])])
            else:
                raise ValueError("Did not recognize submodel given as 'rv_mdl' parameter")
        for dim in range(len(self.ll_factors)):
            self.last_lls[dim] = np.sum(self.ll_factors[dim])
        
    def neg_sq_error(self, data):
        self.recompute_cache(data, dim)
        self.last_ll = - np.sum((self.pred - (data - self.data_mean))**2)        
        return self.last_ll
    
    def recompute_cache(self, data, dim):
        # ensure proper length of cache lists
        for cache in (self.pred, self.ll_factors, self.last_lls):
            if len(cache) <= dim:
                cache.append(0)
                
        self.pred[dim] = self.prediction(data, dim)
        
        if self.dimension_specific_variance():
            variance = self.rvm[dim].get()
        else:
            variance = self.rvm.get()
            
        self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], variance[j,j]).logpdf(data[i,j])
                                      for j in range(self.pred[dim].shape[1])]
                                        for i in range(self.pred[dim].shape[0])])
        self.last_lls[dim] = self.ll_factors[dim].sum()
        
    def log_likelihood(self, data, dim):
        self.recompute_cache(data, dim)
        return self.last_lls[dim]
    
    def llike_function(self, data, rv_mdl = None, idx = None):
        if rv_mdl == None or idx == None:
            assert(rv_mdl == None and idx == None)
            ll_full_computation =  lambda: self.log_likelihood(data, dim)
            return ll_full_computation
        else:
            
            if rv_mdl == self.lvm:
                def ll_lvm_caching():
                    #FIXME: work with presummed loglikelihood, only sum the changes
                    self.notify(data, rv_mdl, idx)
                    #print("lvm_chaching")
                    return self.last_lls[idx[1]]
                return ll_lvm_caching
            elif rv_mdl == self.wm:
                def ll_wm_caching():
                    #FIXME: work with presummed loglikelihood, only sum the changes
                    self.notify(data, rv_mdl, idx)
                    #print("wm_chaching")
                    return self.last_lls[idx[0]]
                return ll_wm_caching
            else:
                if self.dimension_specific_variance():
                    #print("likelihood function for specific variance", file=sys.stderr)
                    dim = self.rvm_to_dim[rv_mdl]
                    def ll_specific_remainvar_caching():
                        #FIXME: work with presummed loglikelihood, only sum the changes
                        self.notify(data, rv_mdl, idx)
                        #print("remain_chaching")
                        return self.last_lls[dim]
                    return ll_specific_remainvar_caching
                else:
                    if self.integr_rvar:
                        #print("likelihood function for integrated variance", file=sys.stderr)
                        def ll_common_integr_remainvar_caching():
                            #FIXME: work with presummed loglikelihood, only sum the changes
                            self.notify(data, rv_mdl, idx)
                            #average over dimensions likelihoods with this
                            #specific common variance
                            return logsumexp(self.last_lls) - np.log(len(self.last_lls))
                        return ll_common_integr_remainvar_caching
                    else:
                        #print("likelihood function for non-specific non-integrated variance", file=sys.stderr)
                        def ll_commmon_unintegr_remainvar_noncaching():
                            dim = len(self.ll_factors) - 1
                            self.ll_factors[dim] = np.array( [[stats.norm(self.pred[dim][i,j], self.rvm.get()[j,j]).logpdf(data[i,j])
                                                  for i in range(self.pred[dim].shape[0])]
                                                    for j in range(self.pred[dim].shape[1])])
                            return logsumexp(self.ll_factors[dim])
                        return ll_commmon_unintegr_remainvar_noncaching


                        

 # </codecell>


 
# <codecell> 

#now the experimental part



def model_evidence(samples, start = 0, stop = None, max_idx = False):
    lll = np.array([s.last_lls for s in samples])
    if stop == None:
        stop = lll.shape[0]
    l1 = logsumexp(lll[start:stop, :],0)
    l1 -= logsumexp(l1)
    if max_idx:
        return (l1, np.argmax(l1.flat)+1)
    else:
        return l1


def ppca_synth_data(num_obs = 10, dim_obs = 6, dim_lv = 4, lv_var = 1, w_var = 10, noise_var = 0.5):
    rng = np.random.RandomState()
    orig_lv = rng.normal(0, lv_var, (num_obs-1, dim_lv))
    orig_w = rng.normal(0, w_var, (orig_lv.shape[1], dim_obs))
    
    orig_data = orig_lv.dot(orig_w)
    noise = rng.normal(0, noise_var, orig_data.shape)
    noisy_data = orig_data+noise

    return (noisy_data, orig_data, noise)



def trans_dim_sampling(noisy_data, num_samples = 1000,
                       dim_spec_variance = True, integr_rvar = True,
                       random_order = True, quiet = False,
                       streaming_evidence = True):

    
    mdl = NaiveTransDimModel(noisy_data.shape[1]-1, noisy_data,
                               streaming_evidence = streaming_evidence,
                               dim_spec_variance = dim_spec_variance,
                               integr_rvar = integr_rvar,
                               quiet = quiet)
    return mdl.sample(noisy_data, num_samples, random_order = random_order)

def once(out_path, num_samples = 1000, burnin = 400, num_obs = 50, dim_obs = 6,
         dim_lv = 4, lv_var = 1, w_var = 10, noise_var = 0.5,
         dim_spec_variance = True, integr_rvar = False):
    import pickle
    import numpy as np
    
    rval = {"num_obs": num_obs, "dim_obs": dim_obs, "dim_lv": dim_lv}
    (noisy_data, orig_data, noise) = ppca_synth_data(num_obs = num_obs, dim_obs = dim_obs, dim_lv = dim_lv, lv_var = lv_var, w_var = w_var, noise_var = noise_var)
    rval["noisy_data"] = noisy_data
    rval["noise"] = noise
    rval["samples"] = trans_dim_sampling(noisy_data, num_samples = num_samples,
                                         dim_spec_variance = dim_spec_variance,
                                         quiet = False, streaming_evidence = True,
                                         integr_rvar = integr_rvar)
    rval["evidence"] = model_evidence(rval["samples"], max_idx=True)
    print(rval["dim_lv"], "No Burn-in\t\t", rval["evidence"])
    if len(rval["samples"]) > burnin:
        bi_key = "evidence_%d" % burnin
        rval[bi_key] = model_evidence(rval["samples"], burnin, max_idx=True)
        print("With Burn-In\t\t", rval[bi_key])
    fn = ("%d_samples__%d_obs__%d_dim__%d_lv__%.2f_lvar__%.2f_wvar__%.2f_nvar_%s.pickle"
                         % (num_samples, num_obs, dim_obs, dim_lv, lv_var, w_var, noise_var, datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%s")))
    f = open(out_path+"/"+fn, "w")
    pickle.dump(rval, f)
    rval["filename"] = fn
    return rval
    

def ard_sampling(noisy_data, num_samples = 1000,
                       quiet = False,
                       streaming_evidence = True,
                       dim_bound = None):
    if dim_bound == None:
        dim_bound = noisy_data.shape[1]-1
    mdl = ARDTwoFactorModel(dim_bound, noisy_data, quiet = quiet)
    return mdl.sample(noisy_data, num_samples)

def ard_once(out_path, num_samples = 1000, burnin = 400, num_obs = 50, dim_obs = 6,
         dim_lv = 4, lv_var = 1, w_var = 10, noise_var = 0.5):
    import pickle
    import numpy as np
    import transdim
    
    rval = {"num_obs": num_obs, "dim_obs": dim_obs, "dim_lv": dim_lv}
    (noisy_data, orig_data, noise) = ppca_synth_data(num_obs = num_obs, dim_obs = dim_obs, dim_lv = dim_lv, lv_var = lv_var, w_var = w_var, noise_var = noise_var)
    rval["noisy_data"] = noisy_data
    rval["noise"] = noise
    rval["samples"] = transdim.ard_sampling(noisy_data, num_samples = num_samples,
                                         quiet = False, dim_bound = dim_lv+2)
    rval["evidence"] = model_evidence(rval["samples"], max_idx=True)
    print(rval["dim_lv"], "No Burn-in\t\t", rval["evidence"])
    if len(rval["samples"]) > burnin:
        bi_key = "evidence_%d" % burnin
        rval[bi_key] = model_evidence(rval["samples"], burnin, max_idx=True)
        print("With Burn-In\t\t", rval[bi_key])
    fn = ("%d_samples__%d_obs__%d_dim__%d_lv__%.2f_lvar__%.2f_wvar__%.2f_nvar_%s.pickle"
                         % (num_samples, num_obs, dim_obs, dim_lv, lv_var, w_var, noise_var, datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%s")))
    f = open(out_path+"/"+fn, "w")
    pickle.dump(rval, f)
    rval["filename"] = fn
    return rval


def add_sampling(noisy_data, num_samples = 1000,
                       quiet = False,
                       streaming_evidence = True,
                       dim_bound = None):
    if dim_bound == None:
        dim_bound = noisy_data.shape[1]-1
    mdl = ADDTwoFactorModel(dim_bound, noisy_data, quiet = quiet)
    return mdl.sample(noisy_data, num_samples)

def add_once(out_path, num_samples = 1000, burnin = 400, num_obs = 50, dim_obs = 6,
         dim_lv = 4, lv_var = 1, w_var = 10, noise_var = 0.5):
    import pickle
    import numpy as np
    import transdim
    
    rval = {"num_obs": num_obs, "dim_obs": dim_obs, "dim_lv": dim_lv}
    (noisy_data, orig_data, noise) = ppca_synth_data(num_obs = num_obs, dim_obs = dim_obs, dim_lv = dim_lv, lv_var = lv_var, w_var = w_var, noise_var = noise_var)
    rval["noisy_data"] = noisy_data
    rval["noise"] = noise
    rval["samples"] = transdim.add_sampling(noisy_data, num_samples = num_samples,
                                         quiet = False, dim_bound = dim_lv+2)
    rval["evidence"] = model_evidence(rval["samples"], max_idx=True)
    print(rval["dim_lv"], "No Burn-in\t\t", rval["evidence"])
    if len(rval["samples"]) > burnin:
        bi_key = "evidence_%d" % burnin
        rval[bi_key] = model_evidence(rval["samples"], burnin, max_idx=True)
        print("With Burn-In\t\t", rval[bi_key])
    fn = ("%d_samples__%d_obs__%d_dim__%d_lv__%.2f_lvar__%.2f_wvar__%.2f_nvar_%s.pickle"
                         % (num_samples, num_obs, dim_obs, dim_lv, lv_var, w_var, noise_var, datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%s")))
    f = open(out_path+"/"+fn, "w")
    pickle.dump(rval, f)
    rval["filename"] = fn
    return rval
    
def test_10_obs_2lv():
    rval = {}
    (noisy_data, orig_data, noise) = ppca_synth_data(num_obs = 10, dim_obs = 6, dim_lv = 2, lv_var = 1, w_var = 10, noise_var = 0.5)
 
    rval["dimensionality_specific_variance"] = trans_dim_sampling(noisy_data, num_samples = 1000,
                                                                dim_spec_variance = True, quiet = False,
                                                                streaming_evidence = True)
#    rval["fix_order_spec_var"] = trans_dim_sampling(noisy_data, num_samples = 1000,
#                            dim_spec_variance = True, random_order = False, quiet = False,
#                            streaming_evidence = True)
#    rval["common_variance"] = trans_dim_sampling(noisy_data, num_samples = 1000,
#                                                dim_spec_variance = False, quiet = False,
#                                                streaming_evidence = True)
    for k in rval:
        print(k, "No Burn-in\t\t", model_evidence(rval[k], max_idx=True),"\n", "With Burn-In\t\t", model_evidence(rval[k], 400, max_idx=True))
    return rval
# </codecell>

# <codecell>
def run_multiple(result_dir, parameters, one_run_func = once, parallel = False):
    if parallel:
        from IPython.parallel import Client
        
        c = Client()
        v = c.load_balanced_view()
    rval = []
    for par in parameters:
#       Lopes and West: 1000 times, Burn-in 10_000, then 10_000 samples with thining of 10 -> 1000 final samples
#       DUNSON: 100 times, 5,000 iterations, Burn-in 1,000
#         -   {"dim_obs" : 7, "dim_lv" : 1, "num_obs":50}
#            - upper bound : 3
#         -   {"dim_obs" : 9, "dim_lv" : 3, "num_obs":50}
#            - for special factor loadings & noise variance
#            - upper bound: 5
    
#        DUNSON only
#         -   {"dim_obs" : 26, "dim_lv" : 3, "num_obs":500}
#            - for special factor loadings & noise variance
#            - upper bound: 10
    
        if parallel:
            rval.append(v.apply_async(one_run_func, *(result_dir,), **par))
        else:
            rval.append(one_run_func(*(result_dir,), **par))
    distances = []        
    if parallel:
        asyncs = rval
        rval = []
        while len(asyncs) > 0:
            v.wait(asyncs, 120)
            for val in asyncs:
                if val.ready():
                    asyncs.remove(val)
                    rval.append(val.get())
                    r = rval[-1]
                    try:
                        ev = r["evidence_400"][1]
                    except:
                        ev = r["evidence"][1]
                    print(r["filename"], ev, r["dim_lv"])
                    distances.append(np.abs(r["dim_lv"] - ev))
    else:
        for r in rval:
            try:
                ev = r["evidence_400"][1]
            except:
                ev = r["evidence"][1]
            distances.append(np.abs(r["dim_lv"] - ev))
            print(r["filename"], ev, r["dim_lv"])
    distances = np.array(distances)
    print("average",np.average(distances), "TP", np.sum(distances == 0)/np.prod(distances.shape))
    return rval
#run_multiple("results", [{"dim_obs" : 6, "dim_lv" : 2, "num_obs":100, "num_samples":2000, "burnin":1000}]*5, one_run_func =  transdim.ard_once, parallel = True)
#run_multiple("results", [{"dim_obs" : 6, "dim_lv" : 2, "num_obs":10, "num_samples":2, "burnin":1}]*2, parallel = True)

# </codecell>
