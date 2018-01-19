

Maybe try to make a gif of the society development
and of plotly of the 2 people





#%%
import plotly.plotly as py
import numpy as np
from plotly.offline import iplot 
from scipy.stats import beta

import plotly 
plotly.tools.set_credentials_file(username='BobiSabev', api_key='gmxDImUfXTExACgdXDqo')


#%%


a = 1
b = 1
theta = np.arange (0, 1.001, 0.001)
y = beta.pdf(theta,a,b)

z = np.array([0,1,2,3,3,3,4,5,5,5,6,6,6,7,7,8])
N = np.arange(16)

data = [dict(
        visible = False,
        line=dict(color='00CED1', width=6),
        name = '𝜈 = '+str(step),
        x = theta,
        y = beta.pdf(theta,a+z[step],b+N[step]-z[step])) for step in np.arange(N.shape[0])
        ]


data[0]['visible'] = True



#%%

steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(data)],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = 10,
    currentvalue = {"prefix": "Simulation steps: "},
    pad = {"t": 50},
    steps = steps
)]

layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)


py.iplot(fig, filename='BaysianSubplots')

























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


#%%

def gen_connection_matrix(self, n, p):
        m = np.random.rand(n,n)
        m = np.sqrt(m * m.T) - np.eye(n) # make symetric and the diagonal negative
        return np.where(m > p, 1, 0) # if bigger than p, make connection, else no connection

 def know_indecies(self):
        indecies = []
        for i in range(self.n): 
            i_index = [i for i, connection in 
                             enumerate(self.agents[i].connections) if connection == 1]
            indecies.append(i_index)
        return indecies





#%%
        

#%%
"""
Plot the posterior on reputation for one agent in the eyes of all the rest
"""

agent_index = 2
theta = np.arange(0,1.01,0.01)

for agent in net.agents:
    
    if agent.index == agent_index:
        continue
    z = agent.collaborations[agent_index]
    N = agent.n_encouters[agent_index]
    trust = agent.trusts[agent_index]
    
    dist = beta.pdf(theta, 1+z, 1+N-z)
    plt.plot(theta, dist)
    plt.show()
    
#%%

a=1
b=1
z=6
N=10

theta = np.arange(0,1.01, 0.01)

p_theta = beta.pdf(theta, a+z, b+N-z)
trust = np.round((a + z) / (a + b + N), decimals=2)

plt.plot(theta, p_theta, label='posterior distribution')
plt.axvline(x=trust, color='red', label='expected value: trust')
plt.legend()
plt.title("Posterior P(theta|D), z={}, N={}".format(z,N))
plt.xlabel("P(reciprocate)")
plt.show()
        
        
#%%
        
import imageio

filenames = []

with imageio.get_writer('trust.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
    
#%%
        
import imageio
#images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('trust.gif', images)

    

    
# Remove same sets


#%%

def remove_diag(x):
    new_x = []
    for i in range(10):
        new_x.append(np.append(x[i,:i], x[i,(i+1):]))
        
    return np.array(new_x)
        




