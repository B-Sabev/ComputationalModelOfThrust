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
    
    def __init__(self, num_agents):
        self.n = num_agents
        self.agents = self.create_agents()
        self.graph = self.build_connection_graph()
    
    def create_agents(self):
        agents = []
        for i in range(self.n):
            agents.append(Agent(i, self.n))
        return agents
    
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
        
        # If the 2 agents interact for first time
        if self.agents[index1].n_encouters[index2] <= 1:
            self.build_connection_graph()
            
            
    def build_connection_graph(self):
        agents_index = np.arange(self.n)
        # For all agents, check all other agents, and it they interacted, add their index    
        agents_know = [[i for i,enc in enumerate(agent.n_encouters) if enc > 0] for agent in self.agents]
        # build graph    
        self.graph = { v : set(neighboor) for v, neighboor in zip(agents_index, agents_know)}
        


class Agent(object):
    
    def __init__(self, index, n):
        self.index = index
        self.recip_actions = np.zeros(n)
        self.collaborations = np.zeros(n)
        self.n_encouters = np.zeros(n)
        self.trusts = np.where(np.arange(n) == index, 1, 0.5) # agent trust itself with 1, everyone else init to 0.5
        self.a = 1 # priors on trust 
        self.b = 1 # priors on trust
        
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
      
    def estimateTrust(self, agent2_index):
        """
        Estimate the trust towards agent2 and update the value in instance variable
        """
        assert (self.index != agent2_index), "Can't change the agents trust with itself!"
        z = self.collaborations[agent2_index]
        N = self.n_encouters[agent2_index]
        self.trusts[agent2_index] = (self.a + z) / (self.a + self.b + N)
    
    def action(self, agent2_index):
        """
        Get action towards agent2 with P(colaborate) = trust
        """
        assert (self.index != agent2_index), "Agent can't act on itself itself!"
        return 0 if np.random.rand() < self.trusts[agent2_index] else 1 # linear function
        #return 1 if trust > k else 0   # threshold function
        
    def bfs(self, graph, agent2_index, k=0):
        """
        Breath-first search to list all possible paths between self and another agent
        k - at most steps to make, k=0 is only direct neighboors
        """
        queue = [(self.index, [self.index])]
        while queue:
            (vertex, path) = queue.pop(0)
            if len(path) > k+1:
                break
            for next in graph[vertex] - set(path):
                if next == agent2_index:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))
                    
                    
    def propagateRep(self, agents, graph, agent2_index, k, m):
        
        paths = self.bfs(graph, agent2_index, k)
        #print(len(list(paths)))
        ws = []
        
        rep = []
        # for all paths
        for path in paths:
            
            wi = 1.0
            # for every point in the path
            for i in range(len(path)-1):
                # number of collaborations between the 2 subsequent vertices
                m_ab = agents[path[i]].n_encouters[path[i+1]]
                # prod of w_ab
                wi *= float(m_ab) / float(m) if m_ab < m else 1.0  
            
            # z,N to be passed along every path
            rep.append((agents[path[-2]].collaborations[path[-1]], 
                        agents[path[-2]].n_encouters[path[-1]]))
            
            ws.append(wi)
        """
        Now ws is a list of all weights wi
        Normalize the weights, then multiply by the passing information for the rep
        
        """
        w_sum = sum(ws)
        w_norm = [w/w_sum for w in ws]
        
        # multiply colaborations * norm weight and sum all of them
        z_rep = sum([r[0] * w for r,w in zip(rep, w_norm)])
        # multiply total_encounters * norm weight and sum all of them
        N_rep = sum([r[1] * w for r,w in zip(rep, w_norm)])
        
        return (z_rep, N_rep)
                
            
                
                
        
        
        
    
    def printInfo(self):
        print("Agent index {}".format(self.index))
        print("Number reciprocate actions {}".format(self.recip_actions))
        print("Number of collaborations   {}".format(self.collaborations))
        print("Total number of encounters {}".format(self.n_encouters))
        print("Trust towards others       {}".format(self.trusts))
        print("Trust priors a = {}, b = {}".format(self.a, self.b))
        


   
   
#%%
 
net = SocialNet(num_agents=10)    
   


for _ in range(100):
    
    # Pick two agents to interact
    index1 = np.random.randint(net.n)
    # TODO how to scale with reputation - more reputable more chance to be picked
    index2 = np.random.randint(net.n)
    if index2 != index1:
        net.encounter(index1, index2, verbose=0)
    
    net.build_connection_graph()
    print(list(net.agents[0].bfs(net.graph, 1, k=1)))
    print(net.agents[0].propagateRep(net.agents, net.graph, 1, k=1, m=2))
    #net.agents[0].propagateRep(net.agents, net.graph, 1, k=0, m=2)
    
    print()
    
# Remove same sets
    
    
#%%    
    
"""

Reciprocity norm - reciprocity must be high at the end of simulation


+ reputation in ESN(embeded social network) =>  + trust from all agents in ESN



# reciprocive actions in ESN   ~   reputation in ESN
    cumulative dyadic reciprocity that ai engages in with
    other agents in a society should have an influence on
    ai’s reputation as a reciprocating agent in that society.
    
    
Reputation - based on actions towards everyone in ESN - 
            Measures the likelihood that an agent will reciprocate
            
Trust - subjective expectation an agent has about
        another’s future behavior based on the history of
        their encounters.
        The higher the trust level for agent ai, the higher the
        expectation that ai will reciprocate agent aj’s actions.


When estimating trust it is possible to use others information

Estimator = (a + z) / (a + b + N)

STD on the trust:
    (a + z)*(b + N - z)
    (a+b+n-1) * (a+b+n)**2
    
Error - calculation in the paper

Propagation of reputation
"""
    
#%%

a = 100
b = 100

z = 1000
N = 5000

theta = np.arange (0, 1.01, 0.01)
# posterior on reputation given data
p_theta = beta.pdf(theta, a + z, b + N - z)
   
print(np.sum(theta * p_theta))
print((a + z) / (a + b + N))



#%%

graph = {'A': set(['B']),
         'B': set(['A', 'C']),
         'C': set(['B', 'D']),
         'D': set(['C', 'E']),
         'E': set(['D', 'F']),
         'F': set(['E'])}


graph1 = {
        1 : set([2,3,4,5,6]),
        2 : set([1,3,4,5,6]),
        3 : set([1,2,4,5,6]),
        4 : set([1,2,3,5,6]),
        5 : set([1,2,3,4,6]),
        6 : set([1,2,3,4,5])}


graph2 = { p : set(n) for p, n in zip([1,2,3,4], [[2,3], [1,3], [1,2,4], [3]])}



#%%

def bfs_paths(graph, start, goal, k):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        if len(path) > k+1:
            break
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

paths = list(bfs_paths(graph1, 1, 2, k=2)) # [['A', 'C', 'F'], ['A', 'B', 'E', 'F']]


paths
        



