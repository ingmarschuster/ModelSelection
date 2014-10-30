# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:57:30 2014

@author: arbeit
"""

from __future__ import division, print_function, absolute_import
from numpy import exp, log, sqrt
from scipy.misc import logsumexp
import numpy as np
import scipy.stats as stats
import cPickle as pickle

import matplotlib as mpl

import matplotlib.pyplot as plt

def plot_var_bias_mse(res, num_evid_samp, logarithmic = True, outfname = "plot.pdf"):
    ssize = sorted(res.keys())
    measures = res[ssize[0]].keys()
    estimators = res[ssize[0]][measures[0]].keys()
    fig, axes = plt.subplots(ncols=len(measures),nrows=1)
    
    for i in range(len(measures)):
        m = measures[i]
        a = axes[i]
        for e in estimators:
            if logarithmic:
                prestr = "log "
                x = log(num_evid_samp)
                y = log(res[10][m][e])                
            else:
                prestr = ""
                x = num_evid_samp
                y = res[10][m][e]
            a.plot(x, y, label=e)
        a.set_title("$"+m+"$")
        a.set_xlabel(prestr + "number of samples")
        a.set_ylabel(prestr + "$"+m+"$")
        a.autoscale("both")
        a.set_aspect("auto", adjustable="datalim")
    lgd = axes[-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(outfname, bbox_extra_artists=(lgd,), bbox_inches='tight')
            
    
                