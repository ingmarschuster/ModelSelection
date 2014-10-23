# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:14:34 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function
from numpy import exp, log
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats

from slice_sampling import slice_sample_all_components

from sobol.sobol_seq import i4_sobol, i4_sobol_generate


def sample_params(num_samples,
                                  D,
                                  mu_pr, s2_pr, s2_li):
    
    rval = []
    prior = stats.norm(mu_pr, s2_pr)
    theta = prior.rvs((1,))
    for _ in range(num_samples):
        slice_sample_all_components(theta, lambda: stats.norm.logpdf(D, theta, s2_li).sum(), prior)
        rval.append(theta.copy())
    return np.array(rval)

def analytic_logevidence(D, mu_pr, s2_pr, s2_li):
    return stats.norm.logpdf(D, mu_pr, s2_pr + s2_li).sum()
    


## Data generation ##
mu_gen = 10.
s2_gen = 1.

D = stats.norm(mu_gen, s2_gen).rvs(1000)

mu_D = D.mean()


## Likelihood (common to both models)

# mu_li ~ N(mu_pr, s2_pr)
s2_li = 1.


## MODEL 1 prior ##

mu_pr1 = 0.
s2_pr1 = 10.
pr1 = stats.norm(mu_pr1, s2_pr1)

evid1 = analytic_logevidence(D, mu_pr1, s2_pr1, s2_li)


## MODEL 2 prior ##

mu_pr2 = 10.
s2_pr2 = 3.
pr2 = stats.norm(mu_pr2, s2_pr2)

evid2 = analytic_logevidence(D, mu_pr2, s2_pr2, s2_li)

print("Analytic evidence model 1", evid1)
print("Analytic evidence model 2", evid2)

## Sample from and fit gaussians to the posteriors ##
num_post_samples = 1000

samp_post1 = sample_params(num_post_samples, D, mu_pr1, s2_pr1, s2_li)
param_fit1 = stats.norm.fit(samp_post1)
fit1 = stats.norm(param_fit1[0], param_fit1[1])

samp_post2 = sample_params(num_post_samples, D, mu_pr2, s2_pr2, s2_li)
param_fit2 = stats.norm.fit(samp_post2)
fit2 = stats.norm(param_fit2[0], param_fit2[1])


print("Fitted posterior params 1", param_fit1)
print("Fitted posterior params 2", param_fit2)

## Now for QMC-Sampling from distributions defined by param_fit1 and param_fit2
## and importance approximation of evidence

num_qmc_samples = 1000

lowdisc_seq = i4_sobol_generate(1, num_qmc_samples + 2, 2).flat[:]

# draw quasi importance samples using the percent point function (PPF, aka quantile function)
# where cdf^-1 = ppf 
imp_samp1 =  fit1.ppf(lowdisc_seq)
imp_samp2 =  fit2.ppf(lowdisc_seq)

imp_w1 = ((pr1.logpdf(imp_samp1) # log prior of samples
           + np.array([stats.norm.logpdf(D, mean, s2_li).sum()
                           for mean in imp_samp1])) # log likelihood of samples
            - fit1.logpdf(imp_samp1) # log pdf of proposal distribution
         )
# normalized importance weights in case we need them:
imp_w_norm1 = imp_w1 - logsumexp(imp_w1)

imp_w2 =  ((pr2.logpdf(imp_samp2) # log prior of samples
            + np.array([stats.norm.logpdf(D, mean, s2_li).sum()
                            for mean in imp_samp2])) # log likelihood of samples
            - fit2.logpdf(imp_samp2) # log pdf of proposal distribution
          )
# normalized importance weights in case we need them:
imp_w_norm2 = imp_w2 - logsumexp(imp_w2)

print("QMC-IS estimate of evidence model 1:", logsumexp(imp_w1)-log(len(imp_w1)))
print("QMC-IS estimate of evidence model 2:", logsumexp(imp_w2)-log(len(imp_w2)))

#print("QMC importance samples and normalized weights 1:\n",np.array(zip(imp_samp1,exp(imp_w_norm1))))
#print("QMC importance samples and normalized weights 2:\n",np.array(zip(imp_samp2,exp(imp_w_norm2))))