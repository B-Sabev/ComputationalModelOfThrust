# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:02:20 2018

@author: Borislav
"""

#%%
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

np.random.seed(42)
#%%






class SocialNet(object):
    
    def __init__(self, num_agents, p_connectivity):
        self.n = num_agents
        self.connection_mat = self.gen_connection_matrix(num_agents, p_connectivity)
        self.agents = self.create_agents()
        self.know_index = self.know_indecies()
         
    def gen_connection_matrix(self, n, p):
        m = np.random.rand(n,n)
        m = np.sqrt(m * m.T) - np.eye(n) # make symetric and the diagonal negative
        return np.where(m > p, 1, 0) # if bigger than p, make connection, else no connection
    
    def create_agents(self):
        agents = []
        for i, connections in enumerate(self.connection_mat):
            agents.append(Agent(i, connections))
        return agents
    
    def know_indecies(self):
        indecies = []
        for i in range(self.n): 
            i_index = [i for i, connection in 
                             enumerate(self.agents[i].connections) if connection == 1]
            indecies.append(i_index)
        return indecies
    
    def encounter(self, index1, index2, verbose=0):
        # compute trust
        self.agents[index1].estimateTrust(index2)
        self.agents[index2].estimateTrust(index1)
        # decide action
        action0 = self.agents[index1].action(index2)
        action1 = self.agents[index2].action(index1) 
        # update history
        self.agents[index1].updateHistory(index2, action0, action1)
        self.agents[index2].updateHistory(index1, action1, action0)
        
        if verbose:
            print("Encounter between agents {} and {}".format(index1, index2))
            print("Trust: ({},{})".format(self.agents[index1].trusts[index2], self.agents[index2].trusts[index1]))
            print("Actions: ({},{})\n".format(action0, action1))
   


class Agent(object):
    
    def __init__(self, index, connections):
        self.index = index
        self.connections = connections 
        self.recip_actions = np.where(connections == 1, 0, -1) # init to 0 with people you know, -1 to people you don't know
        self.collaborations = np.where(connections == 1, 0, -1)
        self.n_encouters = np.where(connections == 1, 0, -1)
        self.trusts = np.where(connections == 1, 0.5, -1)
        self.a = 1 # priors on trust
        self.b = 1 # priors on trust
        
    def updateHistory(self, agent2_index, action1, action2):
        """
        Update reciprocial connections, collaborations and total number of encouters
        """
        if action1 == action2:
            self.recip_actions[agent2_index] += 1
        if action2 == 1:
            self.collaborations[agent2_index] += 1
        self.n_encouters[agent2_index] += 1
        
    def estimateTrust(self, agent2_index):
        """
        Estimate the trust towards agent2 and update the value in instance variable
        """
        z = self.collaborations[agent2_index]
        N = self.n_encouters[agent2_index]
        theta = np.arange (0, 1.01, 0.01)
        # posterior on reputation given data
        p_theta = beta.pdf(theta, self.a + z, self.b + N - z)
        # return the expected value of the posterior probabilty of trust given data
        self.trusts[agent2_index] = np.sum(theta * p_theta)  / theta.shape[0] #Expected value is not between 0, 1 need some way to normalize
    
    def action(self, agent2_index):
        """
        Get action towards agent2 with P(colaborate) = trust
        """
        return 0 if np.random.rand() < self.trusts[agent2_index] else 1 # linear function
        #return 1 if trust > k else 0   # threshold function
    
    def printInfo(self):
        print("Agent index {}".format(self.index))
        print("Connections with agents    {}".format(self.connections))
        print("Number reciprocate actions {}".format(self.recip_actions))
        print("Number of collaborations   {}".format(self.collaborations))
        print("Total number of encounters {}".format(self.n_encouters))
        print("Trust towards others       {}".format(self.trusts))
        print("Trust priors a = {}, b = {}".format(self.a, self.b))
        


   
   
#%%
 
net = SocialNet(5, 0.4)    
   


for _ in range(50):
    
    # Pick two agents to interact
    index1 = np.random.randint(net.n)
 
    # TODO how to scale with reputation - more reputable more chance to be picked
    index2 = int(np.random.choice(net.know_index[index1], 1))
    
    net.encounter(index1, index2, verbose=0)
    
    print(net.agents[0].trusts)
#%%
    
x = np.random.rand(5,5)

plt.imshow(x)
   
  
#%%


    
    
agents[0].printInfo()
agents[0].updateHistory(1, 0, 0)
agents[0].estimateTrust(1)
agents[2].printInfo()
agents[0].action(1)
    
    
    
    
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
    
"""
Visualization of beta distribution

"""
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





