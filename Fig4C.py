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
    
#Simulate growth enhancment of focal species i in speciesi-speciesj coculture for different ajis and Bjs
u_vector = [0.25,0.25]
aii = -0.8
ajj = -0.8
aji = 0
B_i = -2
myconc = [2/(2**x) for x in range(1,8)]
myconc.append(0)
myconc.sort()
initial_condition = [0.0022,0.0022]
aij_params =  [0,-0.5,-0.75,-1, -1.25]
Bj_params = [0,-0.5,-1,-2,-4, -6]
max_subMICfc = [[np.nan for b in range(len(Bj_params))] for c in range(len(aij_params))]
for i in range(len(aij_params)):
    aij = aij_params[i]
    a_matrix = [[aii, aij],[aji, ajj]]
    for j in range(len(Bj_params)):
        B_vector = [B_i, Bj_params[j]]
        OD_pair=simulate_coculture(a_matrix, u_vector, B_vector,initial_condition, myconc, 48)
        OD_focal = OD_pair[0]
        subMICfc_focal = []
        for ODconc in OD_focal:
            subMICfc_focal.append(ODconc/OD_focal[0])
        max_subMIC_fc = max(subMICfc_focal)
        max_subMICfc[i][j] = max_subMIC_fc
df = pd.DataFrame(max_subMICfc)
plt.figure(figsize=(3,3))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.tick_params(axis='x',labelsize=8)
plt.tick_params(axis='y', labelsize=8)
ax = sns.heatmap(df, cmap='Greys',vmin=1,norm=LogNorm(),annot=True,linewidth=1, linecolor='black',annot_kws={'size': 7},square = True,cbar=False, yticklabels=aij_params,xticklabels=Bj_params)
plt.ylabel('Species j strength of inhibition (aij)')
plt.xlabel('Species j antibiotic susceptibility (Bj)')
plt.show()
plt.close()



