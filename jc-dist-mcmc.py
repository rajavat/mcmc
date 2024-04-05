#!/usr/bin/python3

import matplotlib.pyplot as plt   # we use matplotlib for plotting
import numpy as np                # we use numpy for random number generation 
import pandas as pd               # we use pandas for summary stats
import math                       # we use math for log and exp functions

# initialization of variables (defaults)
# python doesn't have constants, so we use all caps to indicate that these are not to be changed

N = 948                # number of sites
X = 90                 # number of variable sites
BURNIN = 0             # number of burnin iterations (discard first BURNIN samples)
SAMPLES = 500          # number of mcmc samples to generate
MU = 0.2               # mean of exponential prior for theta

# long run
# BURNIN = 5000
# SAMPLES = 20000

# autocorrelation function
# Note: This is a simple implementation of the autocorrelation function.
# It is equivalent to the following: 
# import statsmodels.api as sm
# pd.Series(sm.tsa.acf(x,nlags=10))
# results may differ slightly due to differences in floating point precision
def acf(x, maxlag=10):
  laglist = []
  maxlag = min(maxlag,len(x)-1)

  mean  = sum(x)/len(x)                                    # equiv. to: np.mean(x)
  stdev = math.sqrt(sum([(i-mean)**2 for i in x])/len(x))  # equiv. to: np.std(x)
  y = [(i-mean)/stdev for i in x]                          # equiv. to: (np.array(x)-mean)/stdev

  for lag in range(2,maxlag):

    # now calculate rho as the sum of products of y[i] and y[i+lag]
    rho = sum(y[i]*y[i+lag] for i in range(len(y)-lag)) / (len(y)-lag)

    if rho < 0:
      break 

    laglist.append(rho)

  return laglist

def logpriorlikelihood(theta, x, n, mu):
  # logarithm of prior and likelihood for JC69 distance. We calculate the log
  # to avoid underflows and oveflows when theta,x, or n are large.
  # x = number of observed nucleotide differences between two sequences
  # n = number of sites compared
  # theta = JC69 distance
  # mu = mean of exponential prior for theta
  p = 3/4 - (3/4)*math.exp(-4*theta/3)
  lnp = x*math.log(p) + (n-x)*math.log(1-p) - math.log(mu) - theta/mu

  # Note 1: For this particular example we can ignore the term "math.log(mu)" as
  # it will cancel out when calculating the ratio (or log-ratio) of posteriors
  #
  # In other words, the log-difference of posteriors will contain the terms:
  # math.log(mu) - math.log(mu)
  # which results in 0, and thus we can save some computation by not including it.

  return lnp


def mcmc(steps,n,x,w=0.01,mu=0.2,theta=0.01):
  sample = []
  accepted = 0

  # calculate the initial unnormalized posterior
  lnp = logpriorlikelihood(theta, x, n, mu)

  # add the initial theta to the sample (only useful to indicate the starting
  # state when plotting)
  sample.append(theta)

  for i in range(steps):
    # propose new theta
    theta_new = abs(np.random.uniform(low=theta-w/2,high=theta+w/2))

    # the above is equivalent to:
    # theta_new = abs(theta - w/2 + np.random.uniform()*w)

    # calculate unnormalized posterior for proposed theta, and posterior ratio
    lnpnew = logpriorlikelihood(theta_new, x, n, mu)
    logratio = lnpnew - lnp

    if logratio >= 0 or np.random.uniform() < math.exp(logratio):
      # we accept the proposal
      theta = theta_new
      lnp = lnpnew
      accepted = accepted + 1

    # we log a sample regardless whether we accepted the proposal or not
    sample.append(theta)

  # return the sample and the number of accepted proposals
  return sample, accepted


if __name__ == "__main__":
  
  # do three runs with different proposal widths and different starting values
  run1,acc1 = mcmc(SAMPLES, N, X, 0.01, MU, 0.09)
  run2,acc2 = mcmc(SAMPLES, N, X,  0.1, MU, 0.05)
  run3,acc3 = mcmc(SAMPLES, N, X,    1, MU, 0.01)

  # show summary statistics for the runs discarding the first BURNIN samples
  print("Summary for run1")
  print(pd.DataFrame(run1[BURNIN:]).describe()[0])
  eff = 1/(1+2*sum(acf(run1[BURNIN:],maxlag=2000)))
  print("Acceptance: {}".format(acc1/SAMPLES))
  print("Efficiency: {}\n".format(eff))

  print("Summary for run2")
  print(pd.DataFrame(run2[BURNIN:]).describe()[0])
  eff = 1/(1+2*sum(acf(run2[BURNIN:],maxlag=2000)))
  print("Acceptance: {}".format(acc2/SAMPLES))
  print("Efficiency: {}\n".format(eff))

  print("Summary for run3")
  print(pd.DataFrame(run3[BURNIN:]).describe()[0])
  eff = 1/(1+2*sum(acf(run3[BURNIN:],maxlag=2000)))
  print("Acceptance: {}".format(acc3/SAMPLES))
  print("Efficiency: {}\n".format(eff))

  # plot the three runs
  plt.plot(list(range(len(run1))),run1,label="w=0.01")
  plt.plot(list(range(len(run2))),run2,label="w=0.1")
  plt.plot(list(range(len(run3))),run3,label="w=1")
  plt.legend()
  plt.show()

  # now create a histogram of 100 bins for the samples of run1, 
  # discarding the first BURNIN samples
  plt.hist(run2[BURNIN:],bins=100,facecolor='none',edgecolor='black')
  plt.title("Histogram of MCMC samples for run2")
  plt.show()
