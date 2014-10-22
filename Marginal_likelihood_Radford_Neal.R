# Tests of harmonic mean estimator for the marginal likelihood, with
# comparisons to the true value and to estimating the marginal 
# likelihood by averaging the likelhood over parameters sampled from
# the prior.  The model is x ~ N(t,s1^2) with t ~ N(0,s0^2).
#
# Written by Radford M. Neal, August 2008.


# DO THE TESTS.

do.tests <- function ()
{
  set.seed(1)
  for (i in 1:5)
  { print(harmonic.mean.marg.lik(2,10,1,10^7))
  }

  par(mfrow=c(2,2))

#   set.seed(1);
#   test.est(1,10^(1:6),10,0.5,1,harmonic.mean.marg.lik,ylim=c(0.21,0.29))
#   abline(h=true.marg.lik(1,0.5,1))
#   title("harmonic mean estimate, x=1, s0=0.5, s1=1")
  
  set.seed(1);
  test.est(1,10^(1:6),10,0.5,1,cq.mean.marg.lik,ylim=c(0.21,0.29))
  abline(h=true.marg.lik(1,0.5,1))
  title("nq mean estimate, x=1, s0=0.5, s1=1")

  set.seed(1);
  test.est(1,10^(1:6),10,0.5,1,prior.mean.marg.lik,ylim=c(0.21,0.29))
  abline(h=true.marg.lik(1,0.5,1))
  title("prior mean estimate, x=1, s0=0.5, s1=1")

#   set.seed(1);
#   test.est(2,10^(2:7),10,10,1,harmonic.mean.marg.lik,ylim=c(0.02,0.25))
#   abline(h=true.marg.lik(2,10,1))
#   title("harmonic mean estimate, x=2, s0=10, s1=1")
  
  set.seed(1);
  test.est(2,10^(2:7),10,10,1,cq.mean.marg.lik,ylim=c(0.02,0.4))
  abline(h=true.marg.lik(2,10,1))
  title("nq mean estimate, x=2, s0=10, s1=1")

  set.seed(1);
  test.est(2,10^(2:7),10,10,1,prior.mean.marg.lik,ylim=c(0.02,0.4))
  abline(h=true.marg.lik(2,10,1))
  title("prior mean estimate, x=2, s0=10, s1=1")
}


# HARMONIC MEAN ESTIMATE OF MARGINAL LIKELIHOOD.  Arguments are the
# observed data, x, the values of the hyperparameters s0 and s1, and
# the sample size for the harmonic mean estimate.

harmonic.mean.marg.lik <- function (x, s0, s1, n)
{ post.prec <- 1/s0^2 + 1/s1^2
  t <- rnorm (n, (x/s1^2)/post.prec, sqrt(1/post.prec))
  lik <- dnorm(x,t,s1)
  1/mean(1/lik)
}

# NESTED QUADRATURE ESTIMATE OF MARGINAL LIKELIHOOD.  Arguments are the
# observed data, x, the values of the hyperparameters s0 and s1, and
# the sample size for the harmonic mean estimate.

nq.mean.marg.lik <- function (x, s0, s1, n)
{ post.prec <- 1/s0^2 + 1/s1^2
  t <- rnorm (n, (x/s1^2)/post.prec, sqrt(1/post.prec))
  d <- data.frame(lik=dnorm(x,t,s1),p= dnorm(t, 0, s0))
  d <- d[order(d$lik),]
  
  denom = sum(d$p)
  
  b1 <- sum(d$lik * d$p) / denom
  d.m <- tail(d, n=1)
  tmp.p <- d$p[1:(length(d$p)-1)]
  tmp.lik <- d$p[2:(length(d$p))]
  b2 <- sum(tmp.lik * tmp.p)  + d.m$lik * d.m$p
  b2 <- sum(b2) /denom
  print(c("b1:", b1, ", b2:",b2))
  (b1 + b2)/2
}

cq.mean.marg.lik <- function (x, s0, s1, n)
{ post.prec <- 1/s0^2 + 1/s1^2
  t <- rnorm (n, (x/s1^2)/post.prec, sqrt(1/post.prec))
  d <- data.frame(lik=dnorm(x,t,s1),p= dnorm(t, 0, s0))
  #order.t <- order(t)
  #d <- d[order.t,]
  #t <- t[order.t,]
  #unique.t <- !duplicated(t)
  #counts.t <- rle(t)$lengths
  
  #denom = rle(t)$lengths
  
  #du <- d[unique.t,]
  
  b1 <- mean(d$lik) #/ counts.t
  b1
}

# PRIOR MEAN ESTIMATE OF MARGINAL LIKELIHOOD.  Arguments are the
# observed data, x, the values of the hyperparameters s0 and s1, and
# the sample size for the prior mean estimate.

prior.mean.marg.lik <- function (x, s0, s1, n)
{ t <- rnorm(n,0,s0)
  lik <- dnorm(x,t,s1)
  mean(lik)
}


# TRUE VALUE OF MARGINAL LIKELIHOOD.   Arguments are the observed
# data, x, and the values of the hyperparameters s0 and s1.

true.marg.lik <- function (x,s0,s1)
{ dnorm(x,0,sqrt(s0^2+s1^2))
}


# PLOT THE PERFORMANCE OF A MARGINAL LIKELIHOOD ESTIMATOR.  Arguments
# are the observed data value, x, a vector, nvec, of sample sizes for 
# the estimator, the number of repetitions for each estimate, k, the
# values of the hyperparameters s0 and s1, and the function to do 
# the estimation.  Any additional arguments are passed on to the plot 
# function.

test.est <- function (x, nvec, k, s0, s1, est, ...)
{ r <- matrix(NA,k,length(nvec))
  for (i in 1:length(nvec))
  { for (j in 1:k)
    { r[j,i] <- est(x,s0,s1,nvec[i])
    }
  }
  plot (log10(rep(nvec,each=k)), as.vector(r),
        xlab="log10 of sample size",
        ylab="marginal likelihood estimates (log scale)",
        log="y", ...)
}

do.tests()