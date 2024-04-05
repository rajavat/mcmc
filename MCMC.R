# function 1: posterior
lnPf = function(tau, theta, time, # parameters
                mu.tau, mu.theta, # prior means
                x, n){ # data
  
  s = c() # init variable
  p = 3/4 - (3/4) * exp(-(8/3) * (tau + time))
  
  s = (log(2/theta) - (2/theta * time) + 
         x * log(p) + (n - x) * log(1 - p))
  
  lnP = -tau/mu.tau - theta/mu.theta + sum(s)
  
  return(lnP)
}

MSC = function(tau, theta, time, 
               N, L, x, n, 
               w.tau, w.theta, w.time, 
               mu.tau, mu.theta){
  
  # for sampling parameters
  s.tau = numeric(N+1)
  s.theta = numeric(N+1)
  s.time = matrix(nrow=N, 
                  ncol=L)
  
  # initial parameter values
  s.tau[1] = tau
  s.theta[1] = theta
  s.time[1,] = time
  
  # for iterating acceptances
  a.tau = 0
  a.theta = 0
  a.time = 0
  
  # calculate posterior
  lnP = lnPf(tau, theta, time,
             mu.tau, mu.theta,
             x, n);
  
  for (i in 1:N) {
    # 2b : tau values
    ntau = tau + (runif(1) - 0.5) * w.tau
    if (ntau < 0) ntau = -ntau;
    
    nlnP = lnPf(ntau, theta, time, 
                mu.tau, mu.theta, 
                x, n);
    ratio = nlnP - lnP;
    if(ratio>=0 || runif(1)<exp(ratio)) {
      tau = ntau;
      lnP = nlnP;
      a.tau = a.tau + 1
    }
    else {
      ;
    }
    s.tau[i + 1] = tau;
    
    # 2b : theta values
    ntheta = theta + (runif(1) - 0.5) * w.theta;
    if (ntheta < 0) ntheta = -ntheta;
    
    nlnP = lnPf(tau, ntheta, time, 
                mu.tau, mu.theta, 
                x, n);
    logratio = nlnP - lnP;
    if(logratio>=0 || runif(1)<exp(logratio)) {
      theta = ntheta;
      lnP = nlnP;
      a.theta = a.theta + 1
    }
    else {
      ;
    }
    s.theta[i + 1] = theta;
    
    # 2c : coalescence times 
    
    # function 3: calculate the log ratio for coalescent times
    tj.ratio <- function(j, time, new.time){
      p <- 3/4 - (3/4) * exp(-(8/3) * (tau + time))
      new.p <-3/4 - (3/4) * exp(-(8/3) * (tau + new.time))
      
      ratio = -(2/theta) * (new.time - time) + 
        x[j] * log(new.p/p) + (n[j] - x[j]) * 
        log((1 - new.p)/(1 - p))
      
      return(ratio)
    }
    
    for(j in 1:L){
      ntime = time + (runif(1) - 0.5) * w.time;
      if (ntime < 0) ntime = -ntime;
      
      nlnP = lnPf(tau, theta, ntime, 
                  mu.tau, mu.theta, x, n);
      ratio = nlnP - lnP;
      
      #ratio = tj.ratio(j, time, ntime)
      
      if(ratio >= 0 || runif(1) < exp(ratio)) {
        time = ntime;
        #lnP = lnP + ratio;
        lnP = nlnP;
        a.time = a.time + 1
      }
      s.time[i, j] = time
    }
  }
  
  # calculate acceptances
  a.tau = a.tau/(N - 1)
  a.theta = a.theta/(N - 1)
  a.time = a.time/(L * (N-1))
  
  return(list(s.tau, s.theta, s.time, a.tau, a.theta, a.time))
}


# optimise window size:
run = MSC(tau = 0.01,  theta = 0.001,  time = 0.001, 
          N = 1000, L = 1000, 
          x = data.x, n = data.n, 
          w.tau = 0.0005, w.theta = 0.00001, w.time = 0.000001,
          mu.tau = 0.005, mu.theta = 0.001)


# check acceptance proportions: 0.4734735 0.4474474 0.4230280
c(run[[4]],run[[5]],run[[6]])

# plot parameter traces
plot(run[[1]], type = "l", 
     col = "cornflowerblue", 
     xlab = "MCMC iterations", 
     ylab = "Tau and theta parameters", 
     ylim = c(0, 0.01), xlim = c(0, 1000))
lines(run[[2]], col = "coral")


# set burn-in levels
s.tau = run[[1]][100:length(run[[1]])]; # 100 for tau
s.theta = run[[2]][500 : length(run[[2]])]; # 500 for theta

# plot histogram of posteriors
hist(s.tau, xlab = "Tau values", main = NULL, 
     freq = FALSE, n = 20, col = "cornflowerblue")

hist(s.theta, xlab = "Theta values", main = NULL, 
     freq = FALSE, n = 20, col = "coral")

# determine means
mean(s.tau)
mean(s.theta)

# convert to dataframes
s.tau <- data.frame(s.tau)
s.theta <- data.frame(s.theta)

# determine 95% CI
l.model <- lm(s.tau ~ 1, s.tau)
confint(l.model, level = 0.95)

l.model <- lm(s.theta ~ 1, s.theta)
confint(l.model, level = 0.95)