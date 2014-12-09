# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:00:29 2014

@author: arbeit
"""

from __future__ import division, print_function, absolute_import
import scipy.stats as stats
import numpy as np
from copy import copy, deepcopy

from gs_basis import *
 

def plot_current_prop(current, proposal, proposal_dists, fig_name="Gradient_PMC.pdf", half = None):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    f = plt.figure()
    a = f.gca()
    a.set_aspect("equal")    

    for (name, d) in proposal_dists:
        p = d.rvs(300)# + current
        a.scatter(p[:,0],p[:,1], s=  np.ones(p[:,0].shape)*2, linewidths = 0, c = np.random.rand(3,1), alpha = 1, label = name)

    a.scatter(current[0], current[1], s=  np.ones(p[:,0].shape) * 50, c = "black", marker = "x", label = "theta_old")
    
    #a.arrow(current[0], current[1], (gradient)[0], (gradient)[1], color="black", hatch="*", head_width=0.3, head_length=0.3, label = "gradient")
    a.scatter(proposal[0], proposal[1], s = np.ones(p[:,0].shape) * 50, c = "red", marker = "x", label = "f * gradient(theta_old) + theta_old")
    if half is not None:
        a.scatter(half[0], half[1], s = np.ones(p[:,0].shape) * 50, c = "green", marker = "x", label = "half step")
    #a.legend()
    lgd = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    f.tight_layout()
    
    f.show()
    f.savefig(fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')





current = np.array((4,5))
gradient = np.array((-5, -10))
mu_half = current + 0.5 * gradient
mu = gradient + current 


d = []
for (main, other) in [(0.8,0.4), (0.8, 0.8)]:
    dist = stats.multivariate_normal(mu, ideal_covar(gradient, main, other))
    d.append((str((main, other)), dist))

#d_arithm = stats.multivariate_normal(mu, construct_covar_in_direction(gradient, mean="arithm"))
plot_current_prop(current, mu, d, half = mu_half)


