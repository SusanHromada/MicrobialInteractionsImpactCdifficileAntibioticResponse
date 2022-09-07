# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:52:02 2021

@author: hromada
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:40:59 2021

@author: hromada
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import odeint
plt.rcParams.update(plt.rcParamsDefault)
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

def gLV_Model1(x_vector, t=0, u_vector=None, a_matrix=None, antib_param_vector=None, antib_conc=None):
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)):
        new_x_vector[i] = x_vector[i] * (u_vector[i] + np.sum([a_matrix[i][j] * x_vector[j] for j in range(len(x_vector))])+(antib_param_vector[i] * antib_conc))
    return new_x_vector

def simulate_gLV_Model1(x0_vector,Ts,u_vector, a_matrix, antib_param_vector, antib_conc):
    return odeint(gLV_Model1, x0_vector, Ts, args=(u_vector, a_matrix, antib_param_vector, antib_conc))

def simulate_coculture(a_matrix, u_vector, B_vector, initial_condition, conc_range, tmpt):
    conc_range.sort()
    OD48s_i = []
    OD48s_j = []
    my_time = [i for i in range(50)]
    for conc in conc_range:
        my_sim_data = simulate_gLV_Model1(initial_condition, my_time, u_vector, a_matrix, B_vector, conc)
        sim_df = pd.DataFrame(my_sim_data)
        OD48s_i.append(sim_df.loc[tmpt][0])
        OD48s_j.append(sim_df.loc[tmpt][1])
    OD_pair =  [OD48s_i, OD48s_j]
    return OD_pair
    
#Simulate growth enhancment of focal species i in speciesi-speciesj coculture for set aijs and Bjs for 1000 sets of growth rates (u), self interaction terms (aii), Bis, and ajis
df = pd.DataFrame(columns=['category','aij','Bj','aji','aii','ajj','ui','uj','Bi','max_subMIC_fc','no_antib_OD','subMIC_OD'])
myconc = [2/(2**x) for x in range(1,8)]
myconc.append(0)
myconc.sort()
initial_condition = [0.0022,0.0022]
np.random.seed(42)
ux_upper = 1
axx_lower = -1.25
axy_upper = 1.25
Bx_lower = -6
cat = 0
for Bj in [0,-6]:
    for aij in [0,-1.25]:
        cat = cat+1
        catlabel = 'Bj='+str(Bj)+'\n aij='+str(aij)
        print('Working on category '+str(cat))
        for i in range(0,1000): #1000 different param sets
            Bi = np.random.uniform(Bx_lower,0)
            ui = np.random.uniform(0,ux_upper)
            uj = np.random.uniform(0,ux_upper)
            aii = np.random.uniform(axx_lower,0)
            ajj = np.random.uniform(axx_lower,0)
            aji = np.random.uniform(-1*axy_upper,axy_upper)
            a_matrix = [[aii, aij],[aji, ajj]]
            OD_pair=simulate_coculture(a_matrix, [ui,uj], [Bi,Bj],initial_condition, myconc, 48)
            OD_focal = OD_pair[0]
            for myind in range(len(OD_focal)):
                if OD_focal[myind] < 0.001:
                    OD_focal[myind] = 0.001
            # if growth above 0.05 in any concentrations:
            if max(OD_focal) > 0.05:
                df.loc[len(df.index)] = [catlabel, aij, Bj, aji, aii, ajj, ui, uj, Bi, max(OD_focal[1:])/OD_focal[0], OD_focal[0], max(OD_focal[1:])]
#plot
plt.figure(figsize=(3,3))
plt.rcParams['pdf.fonttype'] = 42
plt.yscale('log')
plt.ylim(0.01,10000)
plt.tick_params(axis='x',labelsize=8)
plt.tick_params(axis='y', labelsize=8)
sns.swarmplot(x='category',y='max_subMIC_fc',data=df, color='black', size=3)
plt.plot([-1,4],[1,1], color='black',linewidth=2.5, alpha=0.5)
plt.show()
plt.close()

