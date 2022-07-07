# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 20:32:55 2022

@author: erick
"""

import numpy as np
from scipy import stats
import scipy.stats.distributions as dis
import matplotlib.pyplot as plt

np.random.seed(0)


def lognorm_from_conf(fifth, ninetyfifth):
    n5 = np.log(fifth)
    n95 = np.log(ninetyfifth)
    # params of related normal distribution
    μ = (n5 + n95)/2
    σ = (n95 - μ) / 1.64486
    scl = np.exp(μ)
    return dis.lognorm(σ, scale=scl)

def beta_from_conf(fifth, ninetyfifth):
    mu = np.exp(np.log(fifth)/2 + np.log(ninetyfifth)/2)
    ef = np.sqrt(ninetyfifth/fifth)
    outof = 40 / ef
    b = dis.beta(mu*outof, (1-mu)*outof)
    i0,i1 = b.interval(0.9)
    #while 
    return b


params = {}
#params["nuclear_war_frequency"] = lognorm_from_conf(2e-4, 7e-2)
#n1 = params["nuclear_war_frequency"]
μ = 0.0196
outof = 31
params["nwf_beta"] = dis.beta(μ*outof, (1-μ)*outof)
mu_winter = 0.235
outof = 10
params["pwinter"] = dis.beta(mu_winter*outof, (1-mu_winter)*outof)
mu_10dead = 0.705
outof = 54
params["p10dead"] = dis.beta(mu_10dead*outof, (1-mu_winter)*outof)
params["nwf_avg_percent_change"] = dis.norm(-0.04,0.1)
params["nwf_percent_change_stdev"] = lognorm_from_conf(0.01,0.15)


def generate_samples(params, n):
    samples = {}
    for k in params.keys():
        samples[k] = params[k].rvs(n)
    return samples 

g = generate_samples(params, 10)



def pltdist(d,d2=None):
    i = d.interval(0.99)
    span = i[1] - i[0]
    x0 = i[0] - span*0.05
    x1 = i[1] + span*0.05
    x = np.linspace(x0, x1, 5000)
    if d2 is not None:
        plt.plot(x, d.cdf(x), x, d2.cdf(x))
    else:
        plt.plot(x, d.cdf(x))


nwfb = params["nwf_beta"]
pltdist(nwfb)

#x=np.linspace(0,1,50000)
#plt.loglog(x,nb.pdf(x), x, n1.pdf(x))
#plt.axis([1e-5, 1, 1e-5, 1000])

#print(nb.mean())
#print(nb.interval(0.9))

def simulate(samples):
    if len(samples.values()) < 1:
        return None
    n = len(samples.values()[0])
    for i in range(n):
        nwf = samples["nwf_beta"][i] # Frequency of major nuclear war
        pwinter = samples["pwinter"][i] # p(nuclear winter | nwf). Doesn't change with time.
        p10dead = samples["p10dead"][i] # p(10% of humans die | nuclear winter)
        nwf_avg_change = samples["nwf_avg_percent_change"][i]*nwf
        nwf_chg_stdev = samples["nwf_percent_change_stdev"][i]*nwf
        

        for year in range(2022, 2029):
            



