# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:02:20 2018

@author: Borislav
"""

"""
2 agents, 1 context c

History D of encounters E between the 2 agents and 
prior on Reputation - prior behavior of the evalutated agent from previous encouters with agents in A

trust is the expected probability of \theta given D
trust directly inflences the action {cooperate, deflect}
"""


#%%
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


#%%
np.random.seed(42)

def generate_history_beta(n, t, p_enc):
    """
    Generate history of encounters 
    """
    p = np.cumsum(p_enc)
    D = np.zeros((2,n))
    r = np.random.rand(n)
    
    """
    Doesn't work - generate the 4 cases with given probability
    
    """
    # (0,0)
    D[:, r < p[0]] = 0
    # (0,1)
    D[1, r > p[0]] = 1
    # (1, 0)
    D[0, r < p[2]] = 1
    # (1,1)
    D[:, r > p[2]] = 1
    
    
    D = np.where(D < t, 0, 1)
    return D


def generate_history(n):
    """
    Generate history of encounters 
    """
    D = np.random.randint(0,2,(2,n))
    return D

def rep(history):
    r = np.sum(history, axis=0)
    # (1,1) encouters / total_encouters
    print(r)
    reputation = np.sum(np.where(r >= 2, 1, 0))/ r.shape[0]
    return reputation

def reciprocity(history):
    r = np.sum(history, axis=0)
    print(r)
    # (1,1) + (0, 0) encouters / total_encouters
    rec = np.sum(np.where(r != 1, 1, 0))/ r.shape[0]
    return rec


# personal history between the 2 agents
d = generate_history(20)
# Calculate dyadic reciprocity (0,0) or (1,1) / total_encounters 
rec = reciprocity(d) # between the 2 agents



#%%
# Joint history with all other agents in A, first agent
R_prior_1 = generate_history(10)
# reputation is (1,1) / total encounters for All agents in A inlcuding agent 2 and D
rep1 = rep(np.append(R_prior_1, d, axis=1)) 

#R_prior_2 = generate_history(30)
#rep2 = rep(R_prior_2)



  

#%%
a =  2
b = 2


for a in range(1,5):
    for b in range(1,5):
       

theta = np.arange (0, 1.01, 0.01)
y = beta.pdf(theta,a,b)
plt.plot(theta,y)
plt.show()






