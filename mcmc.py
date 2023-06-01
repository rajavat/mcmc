import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dgamma

# input gene info
n = 948 # length of pairwise alignment region
nts = 84 # number of transitions
ntv = 6 # number of transversions 


# log-likehood function in Kimura substitution: log f(D|d,k) for D(n_s, n_v)
def kimura(d, k, n, nts, ntv):
    p0 = 0.25 + 0.25 * math.exp(-4 * d / (k+2)) + 0.5 * math.exp(-2 * d * (k + 1) / (k + 2))
    p1 = 0.25 + 0.25 * math.exp(-4 * d / (k+2)) - 0.5 * math.exp(-2 * d * (k + 1) / (k + 2))
    p2 = 0.25 + 0.25 * math.exp(-4 * d / (k+2))
    
    return((n - nts - ntv * math.log(p0 / 4) + nts * math.log(p1 / 4) + ntv * math.log(p2 / 4)))

# plotting likelihood, prior, and posterior surfaces
dim = 100
dv = np.linspace(0, 0.3, dim)
kv = np.linspace(0, 100, dim)

# basically R expand.grid()
dk = np.array([(x, y) for x in dv for y in kv])

# prior surface, f(D) f(k)
pri = dgamma.pdf(dk[:,0], 2, scale = 20) * dgamma.pdf(dk[:,1], 2, scale = 0.1)