# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:24:20 2021

@author: hromada
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd
import xlrd
import csv

# Model1 = expanded antibiotic gLV with all aijs=0 (monospecies)
def Model1(x_vector, T, U, AII, my_antib_concs_vector, B):
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)):
        new_x_vector[i] = x_vector[i] * (U + AII * x_vector[i] + B * my_antib_concs_vector[i])
    return new_x_vector
    
# ODEINT SIMULATION
def simulate_Model1(x_vector,T,myu, myaii, my_antib_concs_vector, p):
    X = odeint(Model1, x_vector, T, args=(myu, myaii, my_antib_concs_vector, p))
    return X

# OBJECTIVE FUNCTION
def objective_Model1(p0, x0_vector, T,myu, myaii, my_antib_concs_vector,mylambda, data):
    X = simulate_Model1(x0_vector,T,myu, myaii, my_antib_concs_vector,p0)
    X_transpose = np.transpose(X)
    obj = 0.0
    for i in range(len(x0_vector)):
        obj += (np.mean((np.array(data[i][1:]) - np.array(X_transpose[i][1:]))**2) + (mylambda*abs(p0)))
    return obj

def run_optimization(x0_vector, T, myu, myaii, my_antib_concs_vector, data, mylambda, p0):
    solution = minimize(objective_Model1,p0,args=(x0_vector, T, myu, myaii, my_antib_concs_vector,mylambda, data),method='SLSQP')
    p_optimized = solution.x
    mse = objective_Model1(p_optimized, x0_vector, T,myu, myaii, my_antib_concs_vector, mylambda, data)
    return solution, mse

#load monospecies data
with xlrd.open_workbook('Abundance_data_Fig2A_SuppFig1.xlsx') as wb:
    sh = wb.sheet_by_name('Part1')  # or wb.sheet_by_name('name_of_the_sheet_here')
    with open('part1.csv', 'w', newline="") as f:   # open('a_file.csv', 'w', newline="") for python 3
        c = csv.writer(f)
        for r in range(sh.nrows):
            c.writerow(sh.row_values(r))
    sh = wb.sheet_by_name('Part2')  # or wb.sheet_by_name('name_of_the_sheet_here')
    with open('part2.csv', 'w', newline="") as f:   # open('a_file.csv', 'w', newline="") for python 3
        c = csv.writer(f)
        for r in range(sh.nrows):
            c.writerow(sh.row_values(r))
part1_df = pd.read_csv('part1.csv', index_col=[0])
part2_df = pd.read_csv('part2.csv', index_col=[0])

#load gLV parameters
speciesID={'CD':1,'ER':2,'DP':3,'FP':4,'BH':5,'CA':6,'PC':7,'EL':8,'CH':9,'BO':10,'BT':11,'BU':12,'BV':13,'CS':14}
params_df = pd.read_csv('gLV_parameters_MSB_2021.csv', header=None, index_col=None)
u_vector = params_df[0].tolist()[0::15]
scaling_factor = {'Metronidazole':24,'Vancomycin':12}

#create dataframe to store inferred B parameters in
df = pd.DataFrame(columns=['Species','Antibiotic','B','mse','mse/maxOD', 'scaling factor'])

#find best fit B parameter
for antib in ['Metronidazole','Vancomycin']:
    fig = plt.figure(figsize=(15,15)) 
    ax = [fig.add_subplot(4,4,i+1) for i in range(14)]
    s = 0
    for species in ['DP','FP','BH','CA','PC','CH','ER','EL','BO','BT','BU','BV','CS','CD']:
        spnum = speciesID[species]
        print(species+' '+antib+': ')
        # parse data
        if species in ['DP','FP','BH','CA','PC','CH','CD']:
            subdf = part2_df[(part2_df['sp']==species)&(part2_df['antibiotic']==antib)]
        else:
            subdf = part1_df[(part1_df['sp']==species)&(part1_df['antibiotic']==antib)]  
        subdf=subdf.sort_values(by='concentration')
        U = u_vector[spnum-1]
        AII = -U/np.mean(subdf[subdf['concentration']==0]['OD'])
        antib_concs_vector = []
        x0_vector = []
        data = []
        initial_OD = 0.0022
        for ind in subdf.index:
            antib_concs_vector.append(subdf.at[ind,'concentration'])
            x0_vector.append(initial_OD)
            data.append([initial_OD, subdf.at[ind,'OD']])     
        time = [0,48]
        #scale antibs
        antib_scaled = [antib_concs_vector[a]/scaling_factor[antib] for a in range(len(antib_concs_vector))]

        # do optimization
        best_mse = 1000
        mylambda = 0
        for initial_guess in [-10,-7.5,-5,-2.5,-1,-0.5,-0.1]:
            sol, mse = run_optimization(x0_vector, time, U, AII, antib_scaled, data, mylambda, initial_guess)
            if mse < best_mse:
                best_mse = mse
                best_sol = sol
            print(mse)
        print(best_mse)
        mse_norm = best_mse/(max(data[i][1] for i in range(len(data))))
        df.loc[len(df.index)] = [species, antib, float(best_sol.x[0]),best_mse, mse_norm, scaling_factor[antib]]
        
        #plot the sample data
        ax[s].plot(antib_concs_vector, [data[i][1] for i in range(0,len(antib_scaled))], 'o', color='navy')
        set_antib_concs = list(set(antib_concs_vector))
        set_antib_concs.sort()
        set_antib_scaled = list(set(antib_scaled))
        set_antib_scaled.sort()
        
        avs = []
        for conc in set_antib_scaled:
            rep_list = []
            for ind in range(0, len(antib_scaled)):
                if antib_scaled[ind] == conc:
                    rep_list.append(data[ind][1])
            avs.append(np.mean(rep_list))
        ax[s].plot(set_antib_concs, avs, '-', color='navy', label='data',linewidth=3, alpha=0.6)
        
        #plot the simulation from optimized parameters
        x0_vector = [initial_OD for s in range(len(set_antib_scaled))]
        simulation = simulate_Model1(x0_vector, time, U, AII, set_antib_scaled, best_sol.x)
        simulation_transposed = np.transpose(simulation)
        ax[s].plot(set_antib_concs, [simulation_transposed[i][1] for i in range(0,len(set_antib_scaled))], '.-', color='red', linewidth=3,alpha=0.6, label='simulation')
        ax[s].set_xscale('symlog')
        ax[s].set_title(species+' '+antib+' B: '+str(best_sol.x),fontsize=14)
        ax[s].set_xlabel(antib+' ug/mL', fontsize=12)
        ax[s].set_ylabel('OD600 48 hours', fontsize=12)
        s=s+1
    plt.tight_layout()
    plt.show()
    plt.close()
df.to_csv('Model1_scaled_parameter.csv')