# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:14:34 2014

@author: arbeit
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

mu_pr2 = 0.
s2_pr2 = 3.
pr2 = stats.norm(mu_pr2, s2_pr2)

evid2 = analytic_logevidence(D, mu_pr2, s2_pr2, s2_li)

print("Analytic evidence model 1", evid1)
print("Analytic evidence model 2", evid2)

## Sample from the posterior ##
num_post_samples = 10

samp_post1 = sample_params(num_post_samples, D, mu_pr1, s2_pr1, s2_li)
gaus_fit1 = stats.norm.fit(samp_post1)

samp_post2 = sample_params(num_post_samples, D, mu_pr2, s2_pr2, s2_li)
gaus_fit2 = stats.norm.fit(samp_post2)


## Now for QMC-Sampling from distributions defined by gaus_fit1 and gaus_fit2
## and importance approximation of evidence

# i4_sobol_generate(NUM_DIMENSIONS, NUM_SAMPLES, SKIPPED_SAMPLES)
