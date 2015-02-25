# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 09:47:47 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

import modsel.estimator_statistics as es
import cPickle as pick
from copy import copy

import matplotlib as mpl

from modsel.evidence import evidence_from_importance_weights


import matplotlib.pyplot as plt

def plot_var_bias_mse(res,  outfname = "plot.pdf", ylabel_pre = ""):
    ssize = sorted(res.keys())
    st = res[ssize[0]].keys()
    st_abs = []
    st_rel = []
    for s in st:
        if s.endswith("(relat)"):
            continue
            st_rel.append(s)
        else:
            st_abs.append(s)
    st_abs.sort()
    st_rel.sort()
    st = copy(st_abs)
    #st.extend(st_rel)
    estimators = res[ssize[0]][st[0]].keys()
    fig, axes = plt.subplots(ncols=max(len(st_abs), len(st_rel)), nrows = 1, figsize=(9,3))
    
    
    for i in range(len(st)):
        m = st[i]
        a = axes.flat[i]
        for e in estimators:
            x = np.log(sorted(res.keys()))
            y = np.array([res[i][m][e] for i in ssize]).flatten()
            #assert()
            a.plot(x, y, label=e)
        a.set_title("")
        a.set_xlabel("log # lhood evals")
        a.set_ylabel(ylabel_pre+m)
        a.autoscale("both")
        a.set_aspect("auto", adjustable="datalim")
    lgd = axes[len(st_abs)-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #fig.suptitle(title + "; dim=" + str(dims))
   # fig.figure(num=1, figsize=(1,3))
    
    fig.tight_layout()
    fig.savefig(outfname, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)


# {"post":post_param, "prop":prop_param,
#  "perm_x":perm_x,
#  "log_importance_weights":perm_weights,
#  "M": M, "K":K,
#  "log_evid":log_evid }

    

def estim_stats_progression(samp, log_weights, true_exp, true_ev = None, norm_w = True, steps = 10):
    idx_norm = np.round(np.logspace(1, np.log10(samp[0].shape[0]), steps, base=10)).astype(int)
    idx_infl = np.round(np.logspace(1, np.log10(np.vstack(samp).shape[0]), steps, base=10)).astype(int)
    est = {}
    ev_est = {}
    for i in range(steps):
        est[idx_norm[i]] = {"Standard":estim(samp[0][:idx_norm[i]], log_weights[0][:idx_norm[i]], norm_w = norm_w),
                            "Inflation": estim(np.vstack(samp)[:idx_infl[i]], np.hstack(log_weights)[:idx_infl[i]], norm_w = norm_w),
                            "GroundTruth": true_exp}
        ev_est[idx_norm[i]] = {"Standard": np.atleast_2d(log_weights[0][:idx_norm[i]]).T,
                            "Inflation": np.atleast_2d(np.hstack(log_weights)[:idx_infl[i]]).T,
                            "GroundTruth": true_ev}
    return (est, ev_est)

def construct_long_run(samp, log_weights):
    rval_samp = samp[0]
    rval_lw = log_weights[0]
    for i in range(len(samp)):
        rval_samp

    
def estim(samp, log_weights, norm_w = True):
    if norm_w is True:
        log_weights = log_weights - logsumexp(log_weights)
    (lsamp, lsamp_sign) = es.log_sign(samp)
    (lws, lws_sign) = es.logaddexp(lsamp, np.atleast_2d(log_weights).T, lsamp_sign)
    return es.exp_sign(lws, lws_sign).mean(0)


# {"post":post_param, "prop":prop_param,
#  "perm_x":perm_x,
#  "log_importance_weights":perm_weights,
#  "M": M, "K":K,
#  "log_evid":log_evid }

def plot(fname, num_runs = 100):
    with open(fname, "r") as f:
        res = pick.load(f)
    perm_x = np.hstack(res["perm_x"][:num_runs]) # stack up to long run
    liw = np.hstack(res["log_importance_weights"][:num_runs]) # stack up to long run
    std_ss = perm_x[0].shape[0]
    infl_ss = len(perm_x)*perm_x[0].shape[0]
    print("Standard IS:", std_ss, "samples, Inflated:", infl_ss)
    added = "__is_"+str(std_ss)+"_-_issi_"+str(infl_ss)+"_post"+str(res["post"][0])+"_prop"+str(res["prop"][0])+"_M"+str(res["M"])+"_K"+str(res["K"])+"_logevid"+str(res["log_evid"])
    print(fname+added)
    #return
    (s, ev_s) = estim_stats_progression(perm_x, liw, res["post"][0], np.atleast_2d(res["log_evid"]))
    s_stat = es.statistics(s)
    ev_stat = es.logstatistics(ev_s)
    #assert()
    plot_var_bias_mse(s_stat, outfname = fname+added+".pdf")
    plot_var_bias_mse(ev_stat, outfname = fname+added+"_evidence.pdf", ylabel_pre="log ")


