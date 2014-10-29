# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:14:34 2014

@author: Ingmar Schuster
"""

from __future__ import division, print_function
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats

from slice_sampling import slice_sample_all_components

from sobol.sobol_seq import i4_sobol, i4_sobol_generate


def sample_params(num_samples, D, mu_pr, sd_pr, sd_li):
    
    rval = []
    prior = stats.norm(mu_pr, sd_pr)
    theta = prior.rvs((1,))
    for _ in range(num_samples):
        slice_sample_all_components(theta, lambda: stats.norm.logpdf(D, theta, sd_li).sum(), prior)
        rval.append(theta.copy())
    return np.array(rval)

def analytic_logevidence(D, mu_pr, sd_pr, sd_li):
    v_pr = sd_pr**2 #tau^2 in Murphy
    v_li = sd_li**2 #sigma^2 in Murphy
    D_mean = np.mean(D)
    D_mean_sq = D_mean**2
    
    fact = [ (log(sd_li) - ( len(D) *log(sqrt(2*np.pi) *sd_li) #1st factor
                           + log(sqrt(len(D) * v_pr + v_li))   )),
             (- (  np.power(D, 2).sum() / (2 * v_li)   #2nd factor
                 + mu_pr**2             / (2 * v_pr))),
             (( (v_pr*len(D)**2 *D_mean_sq)/v_li   #numerator of 3rd factor
                 + (v_li * D_mean_sq)/v_pr
                 + 2 * len(D) * D_mean * mu_pr
               ) / (2 * (len(D) * v_pr + v_li)) # denominator of 3rd factor
             )            
           ]
    #print(fact)
    return np.sum(fact)

def importance_weights(D, sd_li, prior, proposal_dist, imp_samp):
    w = ((prior.logpdf(imp_samp) # log prior of samples
           + np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                           for mean in imp_samp])) # log likelihood of samples
            - proposal_dist.logpdf(imp_samp) # log pdf of proposal distribution
         )
    w_norm = w - logsumexp(w)
    return (w, w_norm)
         


## Data generation ##
num_obs = 10
mu_gen = 10.
sd_gen = 1.

D = stats.norm(mu_gen, sd_gen).rvs(num_obs)

mu_D = D.mean()


## Likelihood (common to both models)

# mu_li ~ N(mu_pr, sd_pr)
sd_li = 1.


## MODEL 1 prior ##

mu_pr1 = 0.
sd_pr1 = 10.
pr1 = stats.norm(mu_pr1, sd_pr1)




## MODEL 2 prior ##

mu_pr2 = 10.
sd_pr2 = 1.
pr2 = stats.norm(mu_pr2, sd_pr2)



## Sample from and fit gaussians to the posteriors ##
num_post_samples = 100

samp_post1 = sample_params(num_post_samples, D, mu_pr1, sd_pr1, sd_li)
param_fit1 = stats.t.fit(samp_post1)
fit1 = stats.t(param_fit1[0], param_fit1[1])

samp_post2 = sample_params(num_post_samples, D, mu_pr2, sd_pr2, sd_li)
param_fit2 = stats.t.fit(samp_post2)
fit2 = stats.t(param_fit2[0], param_fit2[1])


print("Fitted posterior params 1", param_fit1)
print("Fitted posterior params 2", param_fit2)
print("\n")

## Analytic evidence

evid1 = analytic_logevidence(D, mu_pr1, sd_pr1, sd_li)
evid2 = analytic_logevidence(D, mu_pr2, sd_pr2, sd_li)

print("Analytic evidence model 1:", evid1)
print("Analytic evidence model 2:", evid2)


## Now for QMC-Sampling from distributions defined by param_fit1 and param_fit2
## and importance approximation of evidence

num_imp_samples = 10

lowdisc_seq = i4_sobol_generate(1, num_imp_samples + 2, 2).flat[:]

# draw quasi importance samples using the percent point function (PPF, aka quantile function)
# where cdf^-1 = ppf 
qis_samp1 =  fit1.ppf(lowdisc_seq)
qis_samp2 =  fit2.ppf(lowdisc_seq)

(qis_w1, qis_w_norm1) = importance_weights(D, sd_li, pr1, fit1, qis_samp1)
(qis_w2, qis_w_norm2) = importance_weights(D, sd_li, pr2, fit2, qis_samp2)
if False:
    qis_w1 = ((pr1.logpdf(qis_samp1) # log prior of samples
               + np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                               for mean in qis_samp1])) # log likelihood of samples
                - fit1.logpdf(qis_samp1) # log pdf of proposal distribution
             )
    # normalized importance weights in case we need them:
    qis_w_norm1 = qis_w1 - logsumexp(qis_w1)
    
    qis_w2 =  ((pr2.logpdf(qis_samp2) # log prior of samples
                + np.array([stats.norm.logpdf(D, mean, sd_li).sum()
                                for mean in qis_samp2])) # log likelihood of samples
                - fit2.logpdf(qis_samp2) # log pdf of proposal distribution
              )
    # normalized importance weights in case we need them:
    qis_w_norm2 = qis_w2 - logsumexp(qis_w2)

print("QMC-IS estimate of evidence model 1:", logsumexp(qis_w1)-log(len(qis_w1)))
print("QMC-IS estimate of evidence model 2:", logsumexp(qis_w2)-log(len(qis_w2)))

## An finally standard importance samples
is_samp1 = fit1.rvs(num_imp_samples)
is_samp2 = fit2.rvs(num_imp_samples)

(is_w1, is_w_norm1) = importance_weights(D, sd_li, pr1, fit1, is_samp1)
(is_w2, is_w_norm2) = importance_weights(D, sd_li, pr2, fit2, is_samp2)

print("Standard IS estimate of evidence model 1:", logsumexp(is_w1)-log(len(is_w1)))
print("Standard IS estimate of evidence model 2:", logsumexp(is_w2)-log(len(is_w2)))