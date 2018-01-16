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


def gen_connection_matrix(n, p=0.5):
    m = np.random.rand(n,n)
    m = np.sqrt(m * m.T) - np.eye(n) # make symetric and the diagonal negative
    return np.where(m > p, 1, 0) # if bigger than p, make connection, else no connection


class Agent(object):
    
    def __init__(self, index, connections):
        self.index = index
        self.connections = connections 
        self.recip_actions = np.where(connections == 1, 0, -1) # init to 0 with people you know, -1 to people you don't know
        self.collaborations = np.where(connections == 1, 0, -1)
        self.n_encouters = np.where(connections == 1, 0, -1)
        self.a = 1 # priors on trust
        self.b = 1 # priors on trust
        
    def updateHistory(self, agent2_index, alpha1, alpha2):
        if alpha1 == alpha2:
            self.recip_actions[agent2_index] += 1
        if alpha2 == 1:
            self.collaborations[agent2_index] += 1
        self.n_encouters[agent2_index] += 1
        
    def estimateTrust(self, agent2_index):
        
        z = self.collaborations[agent2_index]
        N = self.n_encouters[agent2_index]
        theta = np.arange (0, 1.01, 0.01)
        # posterior on reputation given data
        p_theta = beta.pdf(theta, self.a + z, self.b + N - z)
        # return the expected value of the posterior probabilty of trust given data
        return np.sum(theta * p_theta)  / theta.shape[0] #Expected value is not between 0, 1 need some way to normalize
    
    def action(self, trust):
        # If trust is larger than some constant, return the 
        return 0 if np.random.rand() < trust else 1 # linear function
        #return 1 if trust > k else 0   # threshold function
    
    def printInfo(self):
        print("Agent index {}".format(self.index))
        print("Connections with agents    {}".format(self.connections))
        print("Number reciprocate actions {}".format(self.recip_actions))
        print("Number of collaborations   {}".format(self.collaborations))
        print("Total number of encounters {}".format(self.n_encouters))
        print("Trust priors a = {}, b = {}".format(self.a, self.b))
        

def encounter(agents, index1, index2, verbose=0):
    # compute trust
    trust01 = agents[index1].estimateTrust(index2)
    trust10 = agents[index2].estimateTrust(index1)
    # decide action
    action0 = agents[index1].action(trust01)
    action1 = agents[index2].action(trust10) 
    # update history
    agents[index1].updateHistory(index2, action0, action1)
    agents[index2].updateHistory(index1, action1, action0)
    
    if verbose:
        print("Encounter between agents {} and {}".format(index1, index2))
        print("Trust: ({},{})".format(trust01, trust10))
        print("Actions: ({},{})".format(action0, action1))
    
    
    
n = 5
    
connection_matrix = gen_connection_matrix(n, 0.3)

agents = []
for i, connections in enumerate(connection_matrix):
    agents.append(Agent(i, connections))
    
    
    
for _ in range(30):
    
    # Pick to agents to interact
    index1 = np.random.randint(n)
    know_indecies = [i for i, connection in enumerate(agents[index1].connections) if connection == 1]
    # TODO how to scale with reputation - more reputable more chance to be picked
    index2 = int(np.random.choice(know_indecies, 1))
    
    encounter(agents, index1, index2, verbose=1)
    
    
    
    
    #agents[0].printInfo()
    #agents[1].printInfo()
    #print("\n\n\n")
        
       


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
a_s = range(1000,1005)
b_s = range(1000,1005)

theta = np.arange (0, 1.001, 0.001)
 
 
f, ax = plt.subplots(len(a_s), len(b_s), figsize=(12,12))

for i,a in enumerate(a_s):
    for j,b in enumerate(b_s):
        
         
        
        y = beta.pdf(theta,a,b)
        
        E = [t * y_ for t,y_ in zip(theta, y)]
        #print(sum(E) / len(theta))
        
        ax[i][j].plot(theta,y)
        ax[i][j].set_title("a = {}, b = {}".format(a,b))
        
plt.show()





