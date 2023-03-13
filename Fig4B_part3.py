# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:03:37 2022

@author: hromada
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from usefulFunctions import COM1_colors, numID
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def gLV_Model1(x_vector, t=0, u_vector=None, a_matrix=None, antib_param_vector=None, antib_conc=None):
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)):
        new_x_vector[i] = x_vector[i] * (u_vector[i] + np.sum([a_matrix[i][j] * x_vector[j] for j in range(len(x_vector))])+(antib_param_vector[i] * antib_conc))
    return new_x_vector

def simulate_gLV_Model1(x0_vector,Ts,u_vector, a_matrix, antib_param_vector, antib_conc):
    return odeint(gLV_Model1, x0_vector, Ts, args=(u_vector, a_matrix, antib_param_vector, antib_conc))

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

# load best fit parameters
speciesID={'CD':1,'ER':2,'DP':3,'FP':4,'BH':5,'CA':6,'PC':7,'EL':8,'CH':9,'BO':10,'BT':11,'BU':12,'BV':13,'CS':14}
gLV_params_df = pd.read_csv('gLV_parameters_MSB_2021.csv', header=None, index_col=None)
u_gLV_vector = gLV_params_df[0].tolist()[0::15]
aijs = gLV_params_df[0].tolist()
del aijs[0::15]
a_matrix = to_matrix(aijs, 14) 
antib_param_df = pd.read_csv('Model1_scaled_parameter.csv', index_col=[0]) 
scaling_factor_dict = {'Metronidazole':24,'Vancomycin':12} 

# load part3 data (communities part1)
E90_communities = {'SEHE90_1':[1],'SEHE90_2':[1,3,4],'SEHE90_3':[1,3,4,6],'SEHE90_4':[1,3,4,9],'SEHE90_5':[1,3,4,11],'SEHE90_6':[1,3,4,7,14],'SEHE90_7':[1,3,4,10],'SEHE90_8':[1,3,4,12],'SEHE90_9':[1,3,4,6,11],'SEHE90_10':[1,3,4,7,9,14],'SEHE90_11':[1,3,4,6,9,11],'SEHE90_12':[1,3,4,7,11,14]}
E90_df = pd.read_csv('Abundance_data_SuppFig3AB_part1.csv', index_col=[0]) #this is the same data as 'Abundance_data_SuppFig3AB' part 1 deposited at https://doi.org/10.5281/zenodo.7626486 , just formatted slightly differently
#duplicate 0 antibiotic condition to be Metr and Vanco
zero_df = E90_df[E90_df['Concentration']==0]
for ind in zero_df.index:
    zero_df.at[ind,'Antibiotic'] = 'Metronidazole'
E90_df = pd.concat([E90_df, zero_df])
modelType_fraction_correct = {'Model1':[-1],'gLV_no_antib':[-1],'gLV_no_int':[-1],'Model1_num':[-1],'gLV_num':[-1],'gLV_no_int_num':[-1]}

#%% Simulate conditions from part2 data with expanded gLV model (Model1) and two null models (gLV_no_antib, gLV_no_int)
# output1: heatmap of prediction (accurate or inaccurate) for each concentration for each condition as .pdf file for Model1
# output2: save the fraction of correct prediction for Model1 and each null model in .csv file
tmpt=48
for modelTypenum in [0,1,2]:
    modelType = ['Model1','gLV_no_antib','gLV_no_int'][modelTypenum] 
    if modelType in ['Model1','gLV_no_antib']:
        my_amatrix = a_matrix
    if modelType in ['gLV_no_antib']:
        my_amatrix = [[] for i in range(len(a_matrix))]
        for r in range(len(a_matrix)):
            for v in range(len(a_matrix)):
                if r==v:
                    my_amatrix[r].append(a_matrix[r][v]) #copy aiis
                else:
                    my_amatrix[r].append(0) #remove all aijs
    count_match = 0
    count_unmatch = 0
    count_nodata = 0
    for antib in ['Metronidazole','Vancomycin']:
        #susceptibility parameters
        if modelType in ['Model1', 'gLV_no_antib']:
            b_list = []
            for i in range(0,14):
                sp = numID[i+1]
                b_val = float(antib_param_df[(antib_param_df['Antibiotic']==antib)&(antib_param_df['Species']==sp)].iloc[0]['B'])
                b_list.append(b_val)
        if modelType in ['gLV_no_antib']:
            b_list = [0 for i in range(0,14)]
        conclist = {'Metronidazole':[0,0.375,0.75,1.5,3,6,12,24,48],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3,6]}[antib]
        d_heatmap = []
        m_heatmap = []
        yticklabels = []
        for s_ind in{'Metronidazole':[0,1,2,3,4,5],'Vancomycin':[0,1,2,3,4,5,10,11]}[antib]:
            mycondition = 'SEHE90_'+str(s_ind+1)
            splist = E90_communities[mycondition]
            spnamelist = [numID[int(spnum)] for spnum in splist]
            yticklabels.append(mycondition)
            initial_condition = [0 for i in range(14)]
            #Data
            heatmap_row = []
            zerodf = E90_df[(E90_df['Condition']==mycondition)&(E90_df['Time']==tmpt)&(E90_df['Antibiotic']==antib)&(E90_df['Concentration']==0)]
            CDODzero = np.mean(zerodf['sp1OD']) 
            for c in range(len(conclist)):
                conc = conclist[c]
                concdf = E90_df[(E90_df['Condition']==mycondition)&(E90_df['Time']==tmpt)&(E90_df['Antibiotic']==antib)&(E90_df['Concentration']==conc)]
                if len(concdf)>0:
                    CDOD = np.mean(concdf['sp1OD'])
                    subMICfc = CDOD/CDODzero
                    if CDOD < 0.05:
                        heatmap_row.append(-1) #no growth
                    elif subMICfc > {'Metronidazole':1.21,'Vancomycin':1.05}[antib]:
                        heatmap_row.append(1)
                    else:
                        heatmap_row.append(0)
                else:
                    heatmap_row.append(0.1)
            d_heatmap.append(heatmap_row)
            #Simulation
            heatmap_row = []
            initial_OD = 0.0022 #This is different than other experiments, but E81 and E90 kept initial OD constant btw species
            for spnum in splist:
                initial_condition[int(spnum)-1]=initial_OD
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, my_amatrix,b_list,0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(conclist)):
                conc = conclist[c]
                spnum = 1
                scaling_factor = scaling_factor_dict[antib]
                scaled_conc = conc/scaling_factor
                #do simulation
                my_sim_data = simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, my_amatrix,b_list,scaled_conc)                    
                sim_df = pd.DataFrame(my_sim_data)
                CDOD = sim_df.loc[1][0]
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > {'Metronidazole':1.21,'Vancomycin':1.05}[antib]:
                    heatmap_row.append(1)
                else:
                    heatmap_row.append(0)
            m_heatmap.append(heatmap_row)
        df1 = pd.DataFrame(m_heatmap) 
        df2 = pd.DataFrame(d_heatmap)
        df3 = df1==df2
        for r in range(len(df2)):
            for v in range(len(df2.iloc[r])):
                if df2.iloc[r][v] == 0.1:
                    count_nodata += 1
                else:
                    if df3.iloc[r][v] == True:
                        count_match += 1
                    else:
                        count_unmatch += 1
        if modelType == 'Model1':
            fig3,ax = plt.subplots()
            ng = np.ma.masked_where(df2.values == 0.1, df3)
            myColors = ['gray','dodgerblue']
            cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))    
            sns.heatmap(df3, cmap=cmap, alpha=0.1, cbar=False, square=True,xticklabels = conclist, yticklabels = yticklabels)
            ax.pcolor(np.arange(len(df3.columns)+1),np.arange(len(df3.index)+1), ng,cmap=cmap,linewidth=1,edgecolor='black', alpha=0.9)
            plt.title(antib)
            plt.show()
            plt.close()        
    total = count_unmatch+count_match
    frac = count_match/total
    modelType_fraction_correct[modelType] = [frac]
    modelType_fraction_correct[modelType+'_num'] = [count_match]
modelType_df = pd.DataFrame.from_dict(modelType_fraction_correct)
modelType_df.to_csv('part3_model_prediction_accuracy.csv')
