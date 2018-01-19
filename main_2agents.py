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
    
    def __init__(self, num_agents, priors=None):
        self.n = num_agents
        self.agents = self.create_agents(priors)
        self.graph = self.build_connection_graph()
        self.cost = {(0, 0) : (0, 0),
                     (0, 1) : (0,-1),
                     (1, 0) : (-1,0),
                     (1, 1) : (1, 1)}
        
    
    def create_agents(self, priors):
        agents = []
        for i in range(self.n):
            if priors is None:
                agents.append(Agent(i, self.n))
            else:
                agents.append(Agent(i, self.n, priors[i][0], priors[i][1]))
        return agents
    
    def encounter(self, index1, index2, verbose=0, coop_agent=-1):
        
        # check for proper values for k and m
        z_rep1, N_rep1, z_rep2, N_rep2 = self.propagateRep(index1, index2, k=2, m=140)
        
        # compute trust
        self.agents[index1].estimateTrust(index2, (z_rep2, N_rep2))
        self.agents[index2].estimateTrust(index1, (z_rep1, N_rep1))
        # decide action
        if coop_agent == index1:
            action1 = 1
        else:
            action1 = self.agents[index1].action(index2)
        
        if coop_agent == index2:
            action2 = 1
        else: 
            action2 = self.agents[index2].action(index1) 
        # update history
        self.agents[index1].updateHistory(index2, action1, action2)
        self.agents[index2].updateHistory(index1, action2, action1)
        
        
        # Compute benefit and update it
        benefit1, benefit2 = self.cost[(action1, action2)]
        self.agents[index1].benefits[index2] += benefit1
        self.agents[index2].benefits[index1] += benefit2
        
        
        if verbose:
            print("Encounter between agents {} and {}".format(index1, index2))
            print("Trust: ({},{})".format(self.agents[index1].trusts[index2], self.agents[index2].trusts[index1]))
            print("Actions: ({},{})\n".format(action1, action2))
        
        # If the 2 agents interact for first time
        if self.agents[index1].n_encouters[index2] <= 1:
            self.build_connection_graph()
            
            
    def build_connection_graph(self):
        agents_index = np.arange(self.n)
        # For all agents, check all other agents, and it they interacted, add their index    
        agents_know = [[i for i,enc in enumerate(agent.n_encouters) if enc > 0] for agent in self.agents]
        # build graph    
        self.graph = { v : set(neighboor) for v, neighboor in zip(agents_index, agents_know)}
        return self.graph
        
    def bfs(self, agent1_index, agent2_index, k=0):
        """
        Breath-first search to list all possible paths between 2 agents
        k - at most steps to make, k=0 is only direct neighboors
        """
        queue = [(agent1_index, [agent1_index])]
        while queue:
            (vertex, path) = queue.pop(0)
            if len(path) > k+1:
                break
            for next in self.graph[vertex] - set(path):
                if next == agent2_index:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))
                    
                    
    def propagateRep(self, agent1_index, agent2_index, k, m):
        """
        Takes 2 indecies of agents, returns their reputation in the eyes of one another
        in the from of (z_rep, N_rep)
        Reputation is based on reciprocity
        """
        # get all possible paths between the 2 agents
        paths = list(self.bfs(agent1_index, agent2_index, k))
        if len(paths) == 0:
            return 0,0,0,0
        
        ws = []
        rep21 = [] # Reputation of 2 in eyes of 1
        rep12 = []
        # for all paths
        for path in paths:
            wi = 1.0
            # for every point in the path
            for i in range(len(path)-1):
                # number of collaborations between the 2 subsequent vertices
                m_ab = self.agents[path[i]].n_encouters[path[i+1]]
                # prod of w_ab
                wi *= float(m_ab) / float(m) if m_ab < m else 1.0  
            # z,N to be passed along every path
            rep21.append((self.agents[path[-2]].recip_actions[path[-1]], 
                        self.agents[path[-2]].n_encouters[path[-1]])) 
            rep12.append((self.agents[path[1]].recip_actions[path[0]], 
                        self.agents[path[1]].n_encouters[path[0]])) 
            ws.append(wi)
        # normalize
        w_sum = sum(ws)
        w_norm = [w/w_sum for w in ws]
       
        # Pass the information along the links
        # multiply colaborations * norm weight and sum all of them
        z_rep12 = sum([r[0] * w for r,w in zip(rep12, w_norm)])
        # multiply total_encounters * norm weight and sum all of them
        N_rep12 = sum([r[1] * w for r,w in zip(rep12, w_norm)])
        
        z_rep21 = sum([r[0] * w for r,w in zip(rep21, w_norm)])
        # multiply total_encounters * norm weight and sum all of them
        N_rep21 = sum([r[1] * w for r,w in zip(rep21, w_norm)])
        
        return z_rep12, N_rep12, z_rep21, N_rep21
    
    def selectAgents(self, method='random'):
        """
        Select 2 agents to interact:
            'random' - both agents are selected at random
            'trust-based' - first agent is selected at random
                            second is selected proprtional to the trust of the first agent
        """
        
        i1 = np.random.randint(net.n) 
        i2 = i1
        
        if method == 'trust':
            trust_i1 = net.agents[i1].trusts
            trust_i1[i1] = 0 # make sure the trust between agent and himself is 0
            trust_i1 = trust_i1 / np.sum(trust_i1) # normalized
            
            # select the second agent in proportion to the trust towards other agents
            i2 = int(np.random.choice(net.n, 1, p=trust_i1))
        else:
            while i1 == i2:
                i2 = np.random.randint(net.n)
                
        return i1,i2
    
    def setCost(self, cost):
        self.cost = cost
        
        
           
            


class Agent(object):
    
    def __init__(self, index, n, a=1,b=1):
        self.index = index
        self.recip_actions = np.zeros(n)
        self.collaborations = np.zeros(n)
        self.n_encouters = np.zeros(n)
        self.benefits = np.zeros(n)
        self.trusts = np.where(np.arange(n) == index, 0, 0.5) # agent trust itself with 1, everyone else init to 0.5
        self.a = a # priors on trust 
        self.b = b # priors on trust
        
    def updateHistory(self, agent2_index, action1, action2):
        """
        Update reciprocial connections, collaborations and total number of encouters
        """
        assert (self.index != agent2_index), "Can't update history with the agent itself!"
        if action1 == action2:
            self.recip_actions[agent2_index] += 1
        if action2 == 1:
            self.collaborations[agent2_index] += 1
        self.n_encouters[agent2_index] += 1     
        
        
      
    def estimateTrust(self, agent2_index, reputation):
        """
        Estimate the trust towards agent2 and update the value in instance variable
        """
        assert (self.index != agent2_index), "Can't change the agents trust with itself!"
        z = self.recip_actions[agent2_index] + reputation[0]
        N = self.n_encouters[agent2_index] + reputation[1]
        self.trusts[agent2_index] = (self.a + z) / (self.a + self.b + N)
    
    def action(self, agent2_index):
        """
        Get action towards agent2 with P(colaborate) = trust
        """
        assert (self.index != agent2_index), "Agent can't act on itself itself!"
        return 1 if np.random.rand() < self.trusts[agent2_index] else 0 # linear function
        #return 1 if trust > k else 0   # threshold function
        
    
    def printInfo(self):
        print("Agent index {}".format(self.index))
        print("Number reciprocate actions {}".format(self.recip_actions))
        print("Number of collaborations   {}".format(self.collaborations))
        print("Total number of encounters {}".format(self.n_encouters))
        print("Trust towards others       {}".format(self.trusts))
        print("Trust priors a = {}, b = {}".format(self.a, self.b))
        

#%%
    
"""
Between 2 agents - visualization of the posterior on reputation

- Modify the priors on trust to simulate different cases
"""

ab0 = (1,1) 
ab1 = (1,1)

theta = np.arange(0, 1.001, 0.001)

steps = 5

net = SocialNet(num_agents=2, priors=[ab0, ab1])

for i in range(steps):
    
    z0, N0 = net.agents[1].collaborations[0], net.agents[1].n_encouters[0]
    z1, N1 = net.agents[0].collaborations[1], net.agents[0].n_encouters[1]
    
    posterior0 = beta.pdf(theta, ab0[0]+z0, ab0[1]+N0-z0)
    posterior1 = beta.pdf(theta, ab1[0]+z1, ab1[1]+N1-z1)
    
    print("")    
    f, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].plot(theta, posterior0)
    ax[0].set_title("Agent 1 in the eyes of Agent2 z={}, N={}".format(z0,N0))
    ax[1].plot(theta, posterior1)
    ax[1].set_title("Agent 2 in the eyes of Agent1 z={}, N={}".format(z1,N1))
    plt.show()
    
    # encounter between the agents
    net.encounter(0, 1, verbose=1)
    

#%%
"""
Multiple agents - one agent cooperates all the time
"""
steps = 5000    
net = SocialNet(num_agents=20)

for i in range(steps):
    
    index1, index2 = net.selectAgents('trust')
    net.encounter(index1, index2, verbose=0, coop_agent=0) # if one of the agents is 0, he will always cooperate
        
# Plot the reciprocity between the cooperative angent and all the rest
recip =  net.agents[0].recip_actions[1:] / net.agents[0].n_encouters[1:]    
plt.hist(recip)
plt.show()


#%%
"""
Prisoner dilema

"""
# TODO - implement prisoner dilema net with 2 people, try different priors

ab0 = (1,1)
ab1 = (1,1)

steps = 5

cost = {(0, 0) : (0, 0),
        (0, 1) : (0,-1),
        (1, 0) : (-1,0),
        (1, 1) : (1, 1)}

net = SocialNet(num_agents=2, priors=[ab0,ab1])
net.setCost(cost)


for i in range(steps):
    # encounter between the agents
    net.encounter(0, 1, verbose=1)



#%%
"""
Net benefit
"""

num_agents = 10

# Init priors for 3 different societies
priors = { 'uniform'   : [(1,1) for _ in range(num_agents)],
           'naive'     : [(4,1) for _ in range(num_agents)],
           'suspicious': [(1,4) for _ in range(num_agents)]}


nets = { 'uniform'   : SocialNet(num_agents, priors['uniform']),
         'naive'     : SocialNet(num_agents, priors['naive']),
         'suspicious': SocialNet(num_agents, priors['suspicious'])}   

steps = 20000  
visual_step = 20

for i in range(steps):
    
    
    
    # Make encounter in all networks
    for prior_type, net in nets.items():
        # Pick two agents to interact
        index1, index2 = net.selectAgents('trust')
        #print(prior_type)
        net.encounter(index1, index2, verbose=0)
        
    print("\n") 
    
    # for every step, visualize the 3 benefit matrices
    if i % visual_step == 0:
        f, ax = plt.subplots(1,3, figsize=(14, 5))
        
        cax = f.add_axes([0.95, 0.25, 0.03, 0.5])
        
        for counter, (prior_type, net) in enumerate(nets.items()):
            # Compute benefit
            benefits = np.array([agent.benefits for agent in net.agents])
            n_encouters = np.array([agent.n_encouters for agent in net.agents])
            n_encouters = np.where(n_encouters == 0, 1, n_encouters)
            
            net_benefit = np.sum(benefits)
            # TODO - make a prettier colormap
            
            # Plot the benefit as image
            im = ax[counter].imshow(benefits / n_encouters, cmap='hot', origin='lower',  vmin=-1, vmax=1)
            ax[counter].set_title("{}\nnet benefit / encounter = {:.3f}".format(prior_type, net_benefit / (i+1)))
            ax[counter].set_xticks(np.arange(num_agents))
            ax[counter].set_yticks(np.arange(num_agents))
        f.colorbar(im, cax=cax)    
        plt.show()
        



#%%

"""
Estimate m
"""
from math import log

error = 0.05 # at most error 
confidence = 0.95 # how confident to be the agent of the measurement

def estimate_m(error, confidence):
    return - 1.0 / (2*(error**2)) * log(confidence/2.0)

for error in [0.01, 0.05, 0.1]:
    for confidence in [0.90, 0.95, 0.99]:
        m = estimate_m(error, confidence)
        print("Error = {}, Confidence = {}, then it must be m = {:.4f}".format(error, confidence,m))



    
#%%    
    
"""

STD on the trust:
    (a + z)*(b + N - z)
    (a+b+n-1) * (a+b+n)**2
    
Error - calculation in the paper


"""


#%%


        




