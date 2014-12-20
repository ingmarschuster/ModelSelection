# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:57:30 2014

@author: arbeit
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
from copy import copy
import scipy.stats as stats
import cPickle as pickle

import matplotlib as mpl

import matplotlib.pyplot as plt

def plot_var_bias_mse(res, num_evid_samp, title, num_post_samples, num_imp_samples, dims, logarithmic = True, outfname = "plot.pdf"):
    ssize = sorted(res.keys())
    st = res[ssize[0]].keys()
    st_abs = []
    st_rel = []
    for s in st:
        if s.endswith("(relat)"):
            st_rel.append(s)
        else:
            st_abs.append(s)
    st_abs.sort()
    st_rel.sort()
    st = copy(st_abs)
    st.extend(st_rel)
    estimators = res[ssize[0]][st[0]].keys()
    fig, axes = plt.subplots(ncols=max(len(st_abs), len(st_rel)), nrows = 2)
    
    some_val = res.values()[0]
    
    for i in range(len(st)):
        m = st[i]
        a = axes.flat[i]
        for e in estimators:
            if logarithmic:
                prestr = "log "
            else:
                prestr = ""
            x = num_evid_samp
            y = some_val[m][e]
            a.plot(x, y, label=e)
        a.set_title("$"+m+"$")
        a.set_xlabel(prestr + "# imp samp")
        a.set_ylabel(prestr + "$"+m+"$")
        a.autoscale("both")
        a.set_aspect("auto", adjustable="datalim")
    lgd = axes[0,-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.suptitle(title + "; dim=" + str(dims)+", "
                 + str(num_post_samples)  + " MCMC Samples, "
                 + str(num_imp_samples) + " Importance Samples")
    
    fig.tight_layout()
    fig.savefig(outfname, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)
            
    
                