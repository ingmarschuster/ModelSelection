# -*- coding: utf-8 -*-
"""
Proposal classes for PMC
Created on Thu Dec 11 09:24:27 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import
import sys
import scipy.stats as stats
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp

from copy import copy, deepcopy


from distributions import mvnorm, mvt, categorical, invwishart
from modsel.mc.gs_basis import ideal_covar, scalar_projection
from modsel.mc.optimization import find_step_size
from modsel.mc import flags

import modsel.mc.conjugate_models as cm
import modsel.mc.bijections as bij

__all__ = ["sample", "sample_lpost_based",
           "PmcSample", "PmcProposalDistribution",
           "gen_sample_prototype",
           "NaiveRandomWalkProposal", "GaussRwProposal", "LatentClassProposal", 
           "InvWishartRandomWalkProposal",
           "GrAsProposal", "GrAsStupidProposal", "ConGrAsProposal"]


def gr_ass(samps):
    for s in samps:
        if "gr" in s.other:
            assert(s.other["gr"].size == s.sample.size)

def compute_ess(logweights, normalize = False, ret_logval = False):
    if normalize:
        logweights = logweights - logsumexp(logweights)
    rval = -logsumexp(2*logweights)
    if ret_logval:
        return rval
    else:
        return exp(rval)
            
def importance_resampling(resampled_size, pop, ess = False):
    prop_w = np.array([s.lweight for s in pop])
    prop_w = prop_w - logsumexp(prop_w)
    if ess:
        rval_ess = compute_ess(prop_w)
    prop_w = exp(prop_w)
    # Importance Resampling
    while True:
        try:
            dist = categorical(prop_w)
            break
        except ValueError:
            prop_w /= prop_w.sum()
    
    new_samp = []
    for idx in range(resampled_size):
        new_samp.append(pop[dist.rvs()])
        
    if ess:
        return (new_samp, rval_ess)
    else:
        return new_samp
    
    

def sample(num_samples, initial_guesses, proposal_method, population_size = 20, stop_flag = flags.NeverStopFlag(), quiet = True, ess = True):
    num_initial = len(initial_guesses)
    rval =  [PmcSample(sample=s) for s in initial_guesses]
    if ess:
        list_ess = []
    
    while len(rval) - num_initial < num_samples and not stop_flag.stop():

        #print(len(rval))
        anc_cand = np.min((len(rval), 2 * population_size))
        
        ancest_dist = categorical([1./anc_cand] * anc_cand)
        
        #choose ancestor uniformly at random from previous samples
        pop = []
        for _ in range(population_size):
            #assert(len(idx) ==1 and  idx.)
            tmp = proposal_method.gen_proposal(rval[-int(ancest_dist.rvs())])
            if not hasattr(tmp, "__iter__"):
                tmp = [tmp]
            pop.extend(tmp)

        proposal_method.observe(pop) # adapt proposal
        if ess:
            (samps, cur_ess) = importance_resampling(population_size, pop, ess = True)
            list_ess.append(cur_ess)
        else:
            samps = importance_resampling(population_size, pop, ess = True)
        rval.extend(samps)
        if not quiet:
            print(len(rval), "samples", file=sys.stderr)
        
    try:
        pass
        #print("jump model", proposal_method.jump_mdl.ddist.K, "thres model",proposal_method.jump_thres_mdl.rv(),
        #      "gr_covar_mdl", proposal_method.gr_covar_mdl.rv())
    except:
        pass
    #assert()
    rval = [np.array([s.sample for s in rval[num_initial:]]), np.array([s.lpost for s in rval[num_initial:]])]
    if ess:
        rval.append(np.mean(list_ess))
    return rval
    


def sample_sis(num_samples, initial_particles, proposal_method, stop_flag = flags.NeverStopFlag(), quiet = True):
    part = proposal_method.process_initial_samples(initial_particles)
    rval = []
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples and not stop_flag.stop():
        #print(len(rval))
        
        #choose ancestor uniformly at random from previous samples
        pop = []
        part_new = []
        for p in part:
            tmp = proposal_method.gen_proposal(p)
            if hasattr(tmp, "__iter__"):
                pop.extend(tmp)
                lposts = np.array([t.lpost for t in tmp])
                cd = categorical(lposts - logsumexp(lposts), True)
                part_new.append(tmp[cd.rvs()])
            else:
                pop.append(tmp)
                part_new.append(tmp)
                

        prop_w = np.array([s.lweight for s in pop])
        prop_w = exp(prop_w - logsumexp(prop_w))
        
        
        # Importance Resampling
        while True:
            try:
                draws = np.random.multinomial(len(initial_particles), prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
        
        new_samp = []
        for idx in range(len(draws)):
            new_samp.extend([pop[idx]] * draws[idx])
        proposal_method.process_new_ancestors(new_samp)
        rval.extend(new_samp)
        if not quiet:
            print(len(rval), "samples", file=sys.stderr)
        part = part_new

    return (np.array([s.sample for s in rval]), np.array([s.lpost for s in rval]))



def sample_lpost_based(num_samples, initial_particles, proposal_method, population_size = 20):
    rval = []
    anc = proposal_method.process_initial_samples(initial_particles)
    num_initial = len(rval)
    
    while len(rval) - num_initial < num_samples:
        ancest_dist = np.array([a.lpost for a in anc])
        ancest_dist = categorical(ancest_dist - logsumexp(ancest_dist), p_in_logspace = True)
        
        #choose ancestor uniformly at random from previous samples
        pop = [proposal_method.gen_proposal(anc[ancest_dist.rvs()])
                for _ in range(population_size)]

        prop_w = np.array([s.lweight for s in pop])
        prop_w = exp(prop_w - logsumexp(prop_w))
        
        
        # Importance Resampling
        while True:
            try:
                draws = np.random.multinomial(population_size, prop_w)
                break
            except ValueError:
                prop_w /= prop_w.sum()
                
        for idx in range(len(draws)):
            rval.extend([pop[idx]] * draws[idx])
            anc.append(pop[idx])
    
    return (np.array([s.sample for s in rval]), np.array([s.lpost for s in rval]))



class PmcSample(object):    
    def __init__(self, ancestor = None, sample = None, lpost = None, lweight = None, other = None, lprop = None, prop_obj = None):
        self.sample = sample
        self.lpost = lpost
        self.lprop = lprop
        self.ancestor = ancestor
        self.lweight = lweight
        self.prop_obj = prop_obj
        if other is None: #when directly having the default for other be {}, some strange behaviour was the result
            self.other = {}
        else:
            self.other = other

      

class PmcProposalDistribution(object):
    def gen_proposal(self, ancestor = None):
        raise NotImplementedError()
    
    def observe(self, population):
        pass



def gen_sample_prototype(ancestor, proposal_object,
                       prop_dist = None, step_dist = None,
                       lpost_func = None, lpost_and_grad_func = None,
                       other = None):
    """
    Generate a sample prototype by either taking a random step from the ancestors
    sample value or drawing from a given distribution.
    
    Parameters
    ==========
    ancestor - a PmcSample that acts as the ancestor
    proposal_object - the pmc proposal object calling this function
    prop_dist - the distribution to draw the new sample from OR
    step_dist - the distribution to draw the random step from 
    lpost_func - a function to evaluate the log posterior probability OR
    lpost_and_grad_func - a function to evaluate the log posterior and posterior gradient
    other - a value for the "other" property of the resulting PmcSample (will possibly be complemented with gradient)
    
    Returns
    =======
    pmc_samp - The newly constructed PmcSample prototype OR
    (pmc_samp, step) - the newly constructed PmcSample prototype and the drawn random step
    """
    assert(proposal_object is not None)
    if step_dist is not None:
        assert(ancestor is not None)
    assert((prop_dist is not None and step_dist is     None) or
           (prop_dist is     None and step_dist is not None))
    assert((lpost_func is not None and lpost_and_grad_func is     None) or
           (lpost_func is     None and lpost_and_grad_func is not None))
           
    (lp, gr, step) = (None, None, None)
    if other is None:
        other = {}
        
    if prop_dist is not None:
        samp = prop_dist.rvs()
        lprop = prop_dist.logpdf(samp)
    else:
        step = step_dist.rvs()
        samp = ancestor.sample + step
        lprop = step_dist.logpdf(step)
    if lpost_func is not None:
        lp = lpost_func(samp)
    else:
        (lp, other["gr"]) = lpost_and_grad_func(samp)
        
    pmc_samp = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop,
                           lweight = lp - lprop,
                           prop_obj = proposal_object,
                           other = other)
    if step is None:
        return pmc_samp
    else:
        return (pmc_samp, step)

class PmcProposalDistribution(PmcProposalDistribution):
    def get_proposal_distr(self, ancestor = None):
        raise NotImplementedError()
        
    def gen_proposal(self, ancestor = None):
        prop_dist = self.get_proposal_distr(ancestor)
        samp = prop_dist.rvs()
        if self.lpost_and_grad is not None:
            (lp, gr) = self.lpost_and_grad(samp)
        else:
            lp = self.lpost(samp)
        lprop =  prop_dist.logpdf(samp)
        return (PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop,
                           lweight = lp - lprop,
                           prop_obj = self,
                           other = {"lrate":f, "gr":gr, "old":True}))
    
    def observe(self, population):
        pass
    



class NaiveRandomWalkProposal(PmcProposalDistribution):
    def __init__(self, lpost_func, step_dist):
        self.lpost = lpost_func
        self.sdist = step_dist   
        
        
    def gen_proposal(self, ancestor = None, mean = None):  
        if mean is None and ancestor is not None:
            if ancestor.sample is not None:
                return gen_sample_prototype(ancestor,
                                            self,
                                            step_dist = self.sdist,
                                            lpost_func = self.lpost)[0]
            else:
                mean = np.zeros(self.pdist.rvs().shape)
        
        class dummy_proposal_dist(object):
            def logpdf(self, x):
                self.sdist.logpdf(x - mean)
            def rvs(self):
                self.sdist.rvs() + mean
        return gen_sample_prototype(ancestor,
                                    self,
                                    prop_dist = dummy_proposal_dist(),
                                    lpost_func = self.lpost)
        

class GaussRwProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, cov):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.step_dist = mvnorm([0]*cov.shape[0], cov)
        
    def gen_proposal(self, ancestor = None):
        return gen_sample_prototype(ancestor,
                                    self,
                                    step_dist = self.step_dist,
                                    lpost_func = self.lpost)[0]


            


class InvWishartRandomWalkProposal(PmcProposalDistribution):
    def __init__(self, df, dim, lpost_func = lambda x: -np.inf):
        assert(df > dim + 1)
        self.df = df
        self.dim = dim
        self.lpost = lpost_func 
        
        
    def gen_proposal(self, ancestor = None, mean = None):
        assert((mean is not None and
                np.prod(mean.shape) == self.dim**2) or
               (ancestor is not None and
               ancestor.sample is not None and
               np.prod(ancestor.sample.shape) == self.dim**2))
               
        if mean is None and ancestor is not None:
            mean = ancestor.sample
                
        scale_matr = mean * (self.df - self.dim - 1)
        pdist = invwishart(scale_matr, self.df)
        if not hasattr(pdist, "rvs"):
            pdist.__dict__["rvs"] = pdist.rv #some trickery

        return gen_sample_prototype(ancestor,
                                    self,
                                    prop_dist = pdist,
                                    lpost_func = self.lpost)


class LatentClassProposal(PmcProposalDistribution):
    """
        Latent Class Proposal for 1-of-n coding
    """
    
    def __init__(self, lpost_func, dim):
        self.lpost = lpost_func
        self.dim = dim  
        
        
    def gen_proposal(self, ancestor = None):
        rval = PmcSample(ancestor, prop_obj = self)
        rval.sample = np.zeros(self.dim)
        lpost_vals = []
        for i in range(rval.sample.size):
            rval.sample.flat[i] = 1
            lpost_vals.append(self.lpost(rval.sample))
            rval.sample.flat[i] = 0
        lpost_vals = np.array(lpost_vals)
        lpost_norm = lpost_vals - logsumexp(lpost_vals)
        
        chosen = np.argmax(np.random.multinomial(1, exp(lpost_norm)))
        rval.sample.flat[chosen] = 1
        rval.lpost = lpost_vals[chosen]
        rval.lprop = lpost_norm[chosen]
        rval.lweight = rval.lpost - rval.lprop
        return rval

        
   
class GrAsStupidProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, main_var, other_var, lrate = 0.1):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.back_off_count = 0
        self.main_var = main_var
        self.other_var = other_var
    
    def prepare_ancestor(self, ancestor):
        assert(ancestor is not None)        
        assert(ancestor.sample is not None)
        if ancestor.lpost is None:
            if "gr" not in ancestor.other:
                (ancestor.lpost, ancestor.other["gr"]) = self.lpost_and_grad(ancestor.sample)
            else:
                ancestor.lpost = self.lpost_and_grad(ancestor.sample)[0]
        elif "gr" not in ancestor.other:
            ancestor.other["old"] = False
            ancestor.other["gr"] = self.lpost_and_grad(ancestor.sample)[1]
            assert(ancestor.other["gr"].size == ancestor.sample.size)
        if "lrate" not in ancestor.other:
            ancestor.other["lrate"] = self.lrate
        assert(ancestor.other["gr"].size == ancestor.sample.size)
        
    def gen_proposal(self, ancestor = None):
        self.prepare_ancestor(ancestor)
        f = ancestor.other["lrate"]
        
        if True:
            (f, theta_1, lpost_1,
             grad_1, foo)  = find_step_size(ancestor.sample, f, ancestor.lpost,
                                            ancestor.other["gr"],
                                            func_and_grad = self.lpost_and_grad)
            sc_gr = f * 0.5* ancestor.other["gr"]
            cov = ideal_covar(sc_gr, fix_main_var = self.main_var, other_var = self.other_var) # , fix_main_var=1
            step_dist = mvnorm(sc_gr, cov)
            #prop_dist = mvnorm(ancestor.sample + sc_gr, self.cov)
            #print(ancestor.lpost, lpost_1)
        else:
            step_dist = mvnorm(np.zeros(ancestor.sample.size), np.eye(ancestor.sample.size)*self.main_var)
        
        (new_samp, step) = gen_sample_prototype(ancestor, self,
                                                step_dist = step_dist,
                                                lpost_and_grad_func = self.lpost_and_grad)
        new_samp.other["lrate"] = f
        return new_samp


class GrAsProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1, prop_mean_on_line = 0.5, main_var_scale = 1, other_var = 0.5, fix_main_var = None):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.jump_dist = mvt([0]*dim, np.eye(dim)*5, dim)
        self.back_off_count = 0
        self.prop_mean_on_line = prop_mean_on_line
        self.main_var_scale = main_var_scale
        self.other_var = other_var
        self.fix_main_var = fix_main_var
        
    def gen_proposal(self, ancestor = None):
        assert(ancestor is not None)        
        assert(ancestor.sample is not None)
        
        rval = []
        if ancestor.lpost is None:
            if "gr" not in ancestor.other:
                (ancestor.lpost, ancestor.other["gr"]) = self.lpost_and_grad(ancestor.sample)
            else:
                ancestor.lpost = self.lpost_and_grad(ancestor.sample)[0]
        elif "gr" not in ancestor.other:
            ancestor.other["old"] = False
            ancestor.other["gr"] = self.lpost_and_grad(ancestor.sample)[1]
            assert(ancestor.other["gr"].size == ancestor.sample.size)
        if "lrate" in ancestor.other:
            f = ancestor.other["lrate"]
        else:
            f = self.lrate
        
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            print("jumping")
            prop_dist = self.jump_dist
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            assert(ancestor.other["gr"].size == ancestor.sample.size)
            (f, theta_1, lpost_1, grad_1, back_off_tmp)  = find_step_size(ancestor.sample, f, ancestor.lpost, ancestor.other["gr"], func_and_grad = self.lpost_and_grad)
            self.back_off_count += len(back_off_tmp)
            if False and ancestor.lprop is not None:
                for (f, samp, lp, gr) in back_off_tmp:
                    rval.append(PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = ancestor.lprop,
                           lweight = lp - ancestor.lprop,
                           prop_obj = ancestor.prop_obj,
                           other = {"lrate":f, "gr":gr, "old":True}))
            mean_step = f * self.prop_mean_on_line * ancestor.other["gr"]
            prop_mean = ancestor.sample + mean_step
            cov = ideal_covar(mean_step, main_var_scale = self.main_var_scale, other_var = self.other_var, fix_main_var = self.fix_main_var) # , fix_main_var=1
            prop_dist = mvnorm(prop_mean, cov)
            #print(ancestor.lpost, lpost_1)
        samp = prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        print(ancestor.lpost, self.lpost_and_grad(prop_mean)[0])
        lprop =  prop_dist.logpdf(samp)
        assert(ancestor.other["gr"].size == ancestor.sample.size)
        assert(gr.size == samp.size)
        rval.append(PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop,
                           lweight = lp - lprop,
                           prop_obj = self,
                           other = {"lrate":f, "gr":gr, "old":True}))
        return rval


class ConGrAsProposal(PmcProposalDistribution):        
    def __init__(self, lpost_and_grad_func, dim, lrate = 0.1):
        self.lpost_and_grad = lpost_and_grad_func
        self.lpost = lambda x:self.lpost_and_grad(x, False)
        self.lrate = lrate
        self.jump_dist = mvt([0]*dim, np.eye(dim)*5, dim)
        self.compute_conj_dir = lambda anc, current: max(0, np.float(current.sample.T.dot(current.sample - anc.sample) / anc.sample.T.dot(anc.sample)))
        self.back_off_count = 0


    def process_initial_samples(self, samples):
        s_lp_gr = [(s, self.lpost_and_grad(s)) for s in samples]
        return [PmcSample(sample = s, lpost = lp, other = {"gr":gr, "lrate":self.lrate, "conj":gr})
                         for (s, (lp, gr)) in s_lp_gr]

        
    def gen_proposal(self, ancestor = None):
        assert()
        #this needs to be updated
        assert(ancestor is not None)
        if np.linalg.norm(ancestor.other["gr"]) < 10**-10:
            #we are close to a local maximum
            f = ancestor.other["lrate"]
            prop_dist = self.jump_dist
        else:
            #we are at a distance to a local maximum
            #step in direction of gradient.
            (f, theta_1, fval_1, back_off_tmp) = find_step_size(ancestor.sample, ancestor.other["lrate"], ancestor.lpost, ancestor.other["conj"], func = self.lpost)
            self.back_off_count += back_off_tmp
            step_mean = f * 0.5 * ancestor.other["conj"]
            cov = ideal_covar(step_mean, main_var_scale = 1, other_var = 0.5) # , fix_main_var=1
            prop_dist = mvnorm(step_mean, cov)
          
        step = prop_dist.rvs()
        samp = ancestor.sample + prop_dist.rvs()
        (lp, gr) = self.lpost_and_grad(samp)
        momentum = max(0, np.float(gr.T.dot(gr - ancestor.other["gr"]) / ancestor.other["gr"].T.dot(ancestor.other["gr"])))
        conj_dir_1 = gr + momentum * ancestor.other["conj"]
        lprop =  prop_dist.logpdf(step)
        rval = PmcSample(ancestor = ancestor,
                           sample = samp,
                           lpost = lp,
                           lprop = lprop, 
                           prop_obj = self,
                           lweight = lp - prop_dist.logpdf(step),
                           other = {"lrate":f, "gr":gr, "conj":conj_dir_1})
                           
        return rval

