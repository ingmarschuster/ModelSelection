from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.random as npr

from numpy import exp, log
from numpy.linalg import inv, cholesky, det
from scipy.special import multigammaln
from scipy.stats import chi2
import scipy.stats as stats
from linalg import pdinv, ensure_2d

from distributions import *
from test.util import eq_test


def test_mvnorm_fit():
    mu = np.array((2,3,4))
    cov = np.array([(20, 3, 2),
                    (3, 10, 1),
                    (2,  1, 7)])
    
    param_fit = mvnorm.fit(stats.multivariate_normal(mu, cov).rvs(2000000))
    #print(param_fit)
    assert(eq_test(param_fit[0], mu, 0.2))
    assert(eq_test(param_fit[1].diagonal(), cov.diagonal(), 0.1))
    
        
def test_invwishart_logpdf():
    # values from R-package bayesm, function lndIWishart(6.1, a, a)
    a = 4 * np.eye(5)
    assert(abs(invwishart_logpdf(a,a,6.1) + 40.526062) < 1*10**-5)
    
    a = np.eye(5) + np.ones((5,5))
    assert(abs(invwishart_logpdf(a,a,6.1) + 25.1069258) < 1*10**-6)
    
    a = 2 * np.eye(5)
    assert(abs(invwishart_logpdf(a,a,6.1) + 30.12885519) < 1*10**-7)



    
if __name__ == '__main__':
    npr.seed(1)
    nu = 5
    a = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
    #print invwishart_rv(nu,a)
    x = np.array([ invwishart_rv(nu,a) for i in range(20000)])
    nux = np.array([invwishart_prec_rv(nu,a) for i in range(20000)])
    print(x.shape)
    print(np.mean(x,0),"\n", inv(np.mean(nux,0)))
    #print inv(a)/(nu-a.shape[0]-1)
    