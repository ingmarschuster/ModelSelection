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

def plot_var_bias_mse(res, outfname = "plot.pdf"):
    ssize = sorted(res.keys())
    estimators = res[ssize[0]].keys()
    measures = res[ssize[0]][estimators[0]].keys()
    fig, axes = plt.subplots(ncols=len(measures),nrows=1)
    collect = {}
    
    for m in measures:
        collect[m] = {}
        for e in estimators:
            collect[m][e] = log([res[s][e][m] for s in ssize])
    
    for i in range(len(measures)):
        m = measures[i]
        a = axes[i]
        for e in estimators:
            a.plot(ssize, collect[m][e], label=e)
        a.set_title("$"+m+"$")
        a.set_xlabel("# observations")
        a.set_ylabel("log $"+m+"$")
        a.autoscale("both")
        a.set_aspect("equal", adjustable="datalim")
        a.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outfname, bbox_inches='tight')
            
    
                