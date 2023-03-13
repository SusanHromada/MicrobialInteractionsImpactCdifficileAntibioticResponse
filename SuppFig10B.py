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
import ast

def gLV_Model1(x_vector, t=0, u_vector=None, a_matrix=None, antib_param_vector=None, antib_conc=None):
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)):
        new_x_vector[i] = x_vector[i] * (u_vector[i] + np.sum([a_matrix[i][j] * x_vector[j] for j in range(len(x_vector))])+(antib_param_vector[i] * antib_conc))
    return new_x_vector

def simulate_gLV_Model1(x0_vector,Ts,u_vector, a_matrix, antib_param_vector, antib_conc):
    return odeint(gLV_Model1, x0_vector, Ts, args=(u_vector, a_matrix, antib_param_vector, antib_conc))

def simulate_multispecies(a_matrix, u_vector, B_vector, initial_condition, conc_range, tmpt, focal_sp_num, susc_inhib):
    conc_range.sort()
    ODs_focal_sp = []
    ODs_spj = []
    my_time = [i for i in range(50)]
    impossible_ODs = False
    for conc in conc_range:
        my_sim_data = simulate_gLV_Model1(initial_condition, my_time, u_vector, a_matrix, B_vector, conc)
        if max([max(my_sim_data[i]) for i in range(len(my_sim_data))]) > 10:
            impossible_ODs = True
        sim_df = pd.DataFrame(my_sim_data)
        ODs_focal_sp.append(sim_df.loc[tmpt][focal_sp_num])
        ODs_spj.append(sim_df.loc[tmpt][1])
    if impossible_ODs == False:
        return ODs_focal_sp, ODs_spj
    else:
        return -1,-1

#Simulate growth enhancment of focal species i in 5-member community for 1000 sets of growth rates (u), self interaction terms (aii), Bis, Bjs, aijs, and ajis
df = pd.DataFrame(columns=['u_vector','B_vector', 'a_matrix','max_subMIC_fc','no_antib_OD_spi','max_subMIC_OD_spi','no_antib_OD_spj','num_susc_inhibitors', 'num_susc_inhibitors_growth','Susc_Inhibitor'])
myconc = [2/(2**x) for x in range(1,8)]
myconc.append(0)
myconc.sort()
np.random.seed(54)
ux_upper = 1
axx_lower = -1.25
axy_upper = 1.25
Bx_lower = -6
cat = 0
focal_sp_num = 0
commSize = 5
initial_condition = [0.0022 for i in range(0, commSize)]
attempt_num = 1000
param_type = ['equal','less_than'][1]
for myiter in range(attempt_num): #1000 different param sets
    if myiter % 10 == 0:
        print("Iteration "+str(myiter)+' out of '+str(attempt_num))
    #B_list
    Bi = np.random.uniform(Bx_lower,0)
    B_vector = [Bi, np.random.uniform(Bx_lower,-0.01), np.random.uniform(Bx_lower,-0.01), np.random.uniform(Bx_lower,-0.01), np.random.uniform(Bx_lower,-0.01)] #Spi, Spj Spa Spb Spc can be susceptible or not (0)
    #u_list
    u_vector = [np.random.uniform(0,ux_upper) for r in range(0,commSize)]
    #a_matrix
    a_matrix = [[np.nan for r1 in range(0, commSize)] for r2 in range(0,commSize)]
    for r1 in range(0, commSize):
        for r2 in range(0, commSize):
            if r1 == r2: #axx
                a_matrix[r1][r2] = np.random.uniform(axx_lower,0)
            else:
                a_matrix[r1][r2] = np.random.uniform(-1*axy_upper,axy_upper) #Spi, Spj, Spa, Spb, Spc can be inhibitors or enhancers
    #simulate
    OD_focal, OD_spj =simulate_multispecies(a_matrix, u_vector, B_vector,initial_condition, myconc, 48, focal_sp_num)
    num_susc_inhib_growth = 0
    if OD_focal != -1:
        for myind in range(len(OD_focal)):
            if OD_focal[myind] < 0.001:
                OD_focal[myind] = 0.001
        # if growth above 0.05 in any concentrations:
        if max(OD_focal) > 0.05:
            df.loc[len(df.index)] = [u_vector, B_vector, a_matrix, max(OD_focal[1:])/OD_focal[0], OD_focal[0], max(OD_focal[1:]), (OD_spj[0])>0.05]
        else:
            df.loc[len(df.index)] = [u_vector, B_vector, a_matrix, 'no growth', OD_focal[0], max(OD_focal[1:]), (OD_spj[0])>0.05]

# Record number of susceptible inhibitors for each simulation
inhibitor_cutoff = -0.1
susceptible_cutoff = -0.1
for ind in df.index:
    fc = df.at[ind,'max_subMIC_fc']
    if fc != 'no growth':
        fc = float(fc)
    amatrix = ast.literal_eval(df.at[ind,'a_matrix'])
    Bvector = ast.literal_eval(df.at[ind,'B_vector'])
    for s in range(1,commSize):
        if (Bvector[s] < susceptible_cutoff) and (amatrix[0][s] <= inhibitor_cutoff):  
            num_Susc_inhib = True
    df.at[ind,'num_Susc_Inhibitor'] = num_Susc_inhib
            
# Plot results
plt.figure(figsize=(4,2.5))
plt.rcParams['pdf.fonttype'] = 42
plt.yscale('log')
plt.ylim(0.01,10000)
plt.tick_params(axis='x',labelsize=8)
plt.tick_params(axis='y', labelsize=8)
data_df = df[df['max_subMIC_fc']!='no growth']
for ind in data_df.index:
    data_df.at[ind,'max_subMIC_fc'] = float(data_df.at[ind,'max_subMIC_fc'])
mypalette = sns.light_palette("#101D6B", reverse=False,  n_colors=625)
sns.swarmplot(x='num_Susc_Inhibitor',y='max_subMIC_fc',data=data_df, hue="no_antib_OD_spi",palette=mypalette[20:],size=3.5)
plt.plot([-1,4],[1,1], color='black',linewidth=2.5, alpha=0.5)
plt.legend('',frameon=False)
plt.show()
plt.close()

