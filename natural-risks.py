"""
Created on Tue Jul  5 20:32:55 2022

@author: erick
"""

import numpy as np
import scipy.stats.distributions as dis
import matplotlib.pyplot as plt
from numpy.random import rand


# generate lognormal distribution from 90% confidence interval
def lognorm_from_conf(fifth, ninetyfifth):
    n5 = np.log(fifth)
    n95 = np.log(ninetyfifth)
    #params of related normal distribution
    μ = (n5 + n95)/2
    σ = (n95 - μ) / 1.64486
    scl = np.exp(μ)
    return dis.lognorm(σ, scale=scl)

# Generate beta distribution from confidence interval (5th and 95th percentile)
# Mainly intended for small probabilities, algorithm is not robust in all cases
def beta_from_conf(fifth, ninetyfifth):
    #mu = np.exp(np.log(fifth)/2 + np.log(ninetyfifth)/2)
    mu = (fifth+ninetyfifth)/2
    #ef = np.sqrt(ninetyfifth/fifth)
    stdev = (ninetyfifth-fifth)/(1.64486*2) # if it's normal, which it's not
    aplusb = mu*(1-mu)/stdev**2
    a = mu*(aplusb)
    b = (1-mu)*(aplusb)
    beta = dis.beta(a,b)
    
    #outof = 40 / ef
    #b = dis.beta(mu*outof, (1-mu)*outof)
    i0,i1 = beta.interval(0.9)
    #print(a, b, i0, i1)
    tol = 0.01
    i = 0
    while np.abs(i0-fifth) > fifth*tol or np.abs(i1-ninetyfifth) > ninetyfifth*tol:
        mu += (ninetyfifth-i1)/10 + (fifth-i0)/10
        stdev += ((ninetyfifth - fifth) - (i1-i0))/(1.64486*2)/10
        aplusb = mu*(1-mu)/stdev**2
        a = mu*(aplusb)
        b = (1-mu)*(aplusb)
        beta = dis.beta(a,b)
        i0,i1 = beta.interval(0.9)
        #print(a, b, i0, i1)
        i += 1
        if i > 1000:
            print("Failed to converge, solution is approximate:", i0, i1)
            break
    print(a,b,i0,i1,beta.mean())
    return beta

def generate_samples(params, n):
    samples = {}
    for k in params.keys():
        samples[k] = params[k].rvs(n)
    return samples 

def pick_sample(samples, i):
    g = {}
    for k in samples.keys():
        g[k] = np.array([samples[k][i]])
    return g

def pltdist(d,d2=None,plot="cdf"):
    i = d.interval(0.99)
    span = i[1] - i[0]
    x0 = i[0] - span*0.05
    x1 = i[1] + span*0.05
    x = np.linspace(x0, x1, 5000)
    if d2 is not None:
        if plot == "pdf":
            plt.plot(x, d.pdf(x), x, d2.pdf(x))
        else:
            plt.plot(x, d.cdf(x), x, d2.cdf(x))
    else:
        if plot == "pdf":
            plt.plot(x, d.pdf(x))
        else:
            plt.plot(x, d.cdf(x))

def simulate_all(samples):
    if len(samples.values()) < 1:
        return None
    n = len(list(samples.values())[0])
    results = []
    for i in range(n):
        results.append(simulate(samples, i))
    return results


p = {}

p["bolide>1km"] = beta_from_conf(0.3e-8,4e-8) # mean 1.7e-8
p["10%dead|bolide"] = beta_from_conf(0.78,0.99) # mean 0.9
p["supereruption"] = beta_from_conf(2e-5,9e-5) # mean 5e-5
p["10%dead|eruption"] = beta_from_conf(0.01,0.8) # mean 0.31
p["10dead_avg_change"] = dis.norm(-0.01,0.01)
p["10dead_change_stdev"] = lognorm_from_conf(1e-3,0.05)


def simulate(samples, i, seed=-1):
    #if seed == -1:
    #    seed = i
    #np.random.seed(seed)
    
    pbolide = samples["bolide>1km"][i]
    p10bolide = samples["10%dead|bolide"][i]
    pvolcano = samples["supereruption"][i]
    p10volcano_init = samples["10%dead|eruption"][i]
    p10volcano_avg_chg = samples["10dead_avg_change"][i]
    p10_chg = dis.norm(p10volcano_avg_chg, samples["10dead_change_stdev"][i]).rvs(78)
    
    p10volcano = [p10volcano_init]
    for year in range(2023, 2099):
        p10volcano.append(p10volcano[-1]+p10_chg[year-2023]*p10volcano[-1])
        
    p10v = np.array(p10volcano)
    
    pcatastrophe_per_year = p10v*pvolcano + pbolide*p10bolide #array
    psurvival = 1 - pcatastrophe_per_year # array, per year survival prob
    pcatastrophe_2100 = 1 - np.prod(psurvival)
    pcatastrophe_2050 = 1 - np.prod(psurvival[0:28])
    pcatastrophe_2030 = 1 - np.prod(psurvival[0:8])
    
    return pcatastrophe_2030, pcatastrophe_2050, pcatastrophe_2100#, pcatastrophe_per_year


np.random.seed(0)
g = generate_samples(p, 1000)

output = simulate_all(g)
p2030 = np.mean([o[0] for o in output])
p2050 = np.mean([o[1] for o in output])
p2100 = np.mean([o[2] for o in output])
print("\nProbability of 10% killed by 2030:", p2030, " by 2050:", p2050, " by 2100:", p2100)


#for i in range(1000):
#    if output[i][0] == 1:
#        print(i, output[i])
#g2 = pick_sample(g,252)
#print(simulate(g2,0,seed=252))
