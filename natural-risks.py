"""
Created on Tue Jul  5 20:32:55 2022

@author: erick
"""

import numpy as np
import scipy.stats.distributions as dis
import matplotlib.pyplot as plt
from numpy.random import rand


def lognorm_from_conf(fifth, ninetyfifth):
    n5 = np.log(fifth)
    n95 = np.log(ninetyfifth)
    #params of related normal distribution
    μ = (n5 + n95)/2
    σ = (n95 - μ) / 1.64486
    scl = np.exp(μ)
    return dis.lognorm(σ, scale=scl)

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


p = {}

p["bolide>1km"] = beta_from_conf(0.3e-8,4e-8)
p["10%dead|bolide"] = beta_from_conf(0.78,0.99)

#params["pwinter"] = beta_from_conf(0.01,0.4) #dis.beta(mu_winter*outof, (1-mu_winter)*outof)

#params["p10dead"] = beta_from_conf(0.6,0.8) # dis.beta(mu_10dead*outof, (1-mu_winter)*outof)
#params["nwf_avg_frac_change"] = dis.norm(-0.04,0.06)
#params["nwf_frac_change_stdev"] = lognorm_from_conf(0.01,0.25)


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


#nwfb = params["nwf_beta"]
#pltdist(nwfb)

#x=np.linspace(0,1,50000)
#plt.loglog(x,nb.pdf(x), x, n1.pdf(x))
#plt.axis([1e-5, 1, 1e-5, 1000])

#print(nb.mean())
#print(nb.interval(0.9))

def simulate_all(samples):
    if len(samples.values()) < 1:
        return None
    n = len(list(samples.values())[0])
    results = []
    for i in range(n):
        results.append(simulate(samples, i))
    return results
        
def simulate(samples, i, seed=-1):
    nwf = samples["nwf_beta"][i] # Frequency of major nuclear war
    pwinter = samples["pwinter"][i] # p(nuclear winter | nwf). Doesn't change with time.
    p10dead = samples["p10dead"][i] # p(10% of humans die | nuclear winter)
    nwf_avg_change = samples["nwf_avg_frac_change"][i]
    nwf_chg_stdev = samples["nwf_frac_change_stdev"][i]
    nwf_chg = dis.norm(nwf_avg_change, nwf_chg_stdev).rvs(80)
    
    if seed == -1:
        seed = i
    np.random.seed(seed)
    randnums = rand(80,3)
    for year in range(2022, 2099):
        i = year-2022
        # Determine what happens this year
        if randnums[i,0] < nwf: # There's a major nuclear war this year
            if randnums[i,1] < pwinter: # It causes a nuclear winter
                if randnums[i,2] < p10dead: # That kills more than 10% of humans
                    if year < 2029: # assume these deaths take a year or so
                        return (1, 1, 1) # occurred before 2030, 2050, and 2100
                    if year < 2049:
                        return (0, 1, 1) # occurred before 2050 and 2100
                    return (0, 0, 1) # occurred before 2100

        # Update the probabilites/frequencies for next year:
                # sample from a normal to see how much the frequency of nuclear war changes.
        nwf_change = nwf_chg[i]
        nwf += nwf_change*nwf # It's a fractional change
            
    
    # If nothing has happened, return all zeros
    return (0, 0, 0)

np.random.seed(0)
g = generate_samples(p, 1000)

output = simulate_all(g)
p2030 = len([s for s in output if s[0]==1])/len(output)
p2050 = len([s for s in output if s[1]==1])/len(output)
p2100 = len([s for s in output if s[2]==1])/len(output)
print("Probability of 10% killed by 2030:", p2030, " by 2050:", p2050, " by 2100:", p2100)


#for i in range(1000):
#    if output[i][0] == 1:
#        print(i, output[i])
#g2 = pick_sample(g,252)
#print(simulate(g2,0,seed=252))
