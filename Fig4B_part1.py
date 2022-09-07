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

# load part1 data (pairs)
part1_df = pd.read_csv('Abundance_data_Fig2BC_part1.csv', index_col=[0]) #this is the pairs data, the same data as 'Abundance_data_Fig2BC' deposited at https://doi.org/10.5281/zenodo.7049039 , just formatted slightly differently
modelType_fraction_correct = {'Model1':[-1],'gLV_no_antib':[-1],'gLV_no_int':[-1],'Model1_num':[-1],'gLV_num':[-1],'gLV_no_int_num':[-1]}

#%% Simulate conditions from part1 data with expanded gLV model (Model1) and two null models (gLV_no_antib, gLV_no_int)
# output1: heatmap of prediction (accurate or inaccurate) for each concentration for each condition as .pdf file for Model1
# output2: save the fraction of correct prediction for Model1 and each null model in .csv file
for modelTypenum in [0,1,2]:
    modelType = ['Model1','gLV_no_antib','gLV_no_int'][modelTypenum] 
    count_match = 0
    count_unmatch = 0
    count_nodata = 0
    for antib in ['Metronidazole','Vancomycin']:
        data_heatmap = []
        model_heatmap = []
        yticklabels = []
        s = 0
        for cond in ['CD']:
            yticklabels.append(cond)
            u_vector = [u_gLV_vector[0],0]
            a_matrix_CD_sp2 = [[a_matrix[0][0],0],[0,0]]
            subdf = part1_df[(part1_df['Condition']==cond)&(part1_df['Antibiotic']==antib)]
            conclist = {'Metronidazole':[0,0.75,1.5,3,6,12],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3]}[antib]
            scaling_factor = scaling_factor_dict[antib]
            antib_scaled = [conclist[a]/scaling_factor for a in range(len(conclist))]
            heatmap_row = []
            specific_conclist = []
            zerodf = subdf[subdf['Concentration']==0]
            CDODzero = np.mean(zerodf['sp1OD']) 
            for conc in conclist:
                concdf = subdf[subdf['Concentration']==conc]
                CDOD = np.mean(concdf['sp1OD'])
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > {'Metronidazole':1.6114,'Vancomycin':1.112}[antib]:
                    heatmap_row.append(1)
                else:
                    heatmap_row.append(0)
            data_heatmap.append(heatmap_row)
            #Simulation
            if modelType in ['Model1', 'gLV_no_int']:
                CD_b = antib_param_df[(antib_param_df['Antibiotic']==antib)&(antib_param_df['Species']=='CD')].iloc[0]['B']
            if modelType in ['gLV_no_antib']:
                CD_b = 0
            heatmap_row = []
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1([0.00022,0],[0,48],u_vector, a_matrix_CD_sp2,[CD_b,0], 0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(antib_scaled)):
                conc = antib_scaled[c]
                my_sim_data = simulate_gLV_Model1([0.00022,0],[0,48],u_vector, a_matrix_CD_sp2,[CD_b,0], conc)
                sim_df = pd.DataFrame(my_sim_data)
                CDOD = sim_df.loc[1][0]
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > {'Metronidazole':1.6114,'Vancomycin':1.112}[antib]:
                    heatmap_row.append(1)
                else:
                    heatmap_row.append(0)
            model_heatmap.append(heatmap_row)
        for sp2 in ['DP','FP','BH','CA','PC','CH','ER','EL','BO','BT','BU','BV','CS']:
            cond = 'CD-'+sp2
            yticklabels.append(cond)
            sp2num = speciesID[sp2]
            u_vector = [u_gLV_vector[0],u_gLV_vector[sp2num-1]]
            if modelType in ['Model1','gLV_no_antib']:
                a_matrix_CD_sp2 = [[a_matrix[0][0],a_matrix[0][sp2num-1]],[a_matrix[sp2num-1][0],a_matrix[sp2num-1][sp2num-1]]]
            if modelType in ['gLV_no_int']:
                a_matrix_CD_sp2 = [[a_matrix[0][0],0],[0,a_matrix[sp2num-1][sp2num-1]]]
            #Data
            subdf = part1_df[(part1_df['Condition']==cond)&(part1_df['Antibiotic']==antib)]
            conclist = {'Metronidazole':[0,0.75,1.5,3,6,12],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3]}[antib]
            scaling_factor = scaling_factor_dict[antib]
            antib_scaled = [conclist[a]/scaling_factor for a in range(len(conclist))]
            heatmap_row = []
            zerodf = subdf[subdf['Concentration']==0]
            CDODzero = np.mean(zerodf['sp1OD']) 
            for conc in conclist:
                concdf = subdf[subdf['Concentration']==conc]
                if len(concdf) > 0 and len(zerodf) > 0:
                    CDOD = np.mean(concdf['sp1OD'])
                    subMICfc = CDOD/CDODzero
                    if CDOD < 0.05:
                        heatmap_row.append(-1) #no growth
                    elif subMICfc > {'Metronidazole':1.6114,'Vancomycin':1.112}[antib]:
                        heatmap_row.append(1)
                    else:
                        heatmap_row.append(0)
                else:
                    heatmap_row.append(0.1)
            data_heatmap.append(heatmap_row)
            #Simulation
            if modelType in ['Model1', 'gLV_no_int']:
                CD_b = antib_param_df[(antib_param_df['Antibiotic']==antib)&(antib_param_df['Species']=='CD')].iloc[0]['B']
                sp2_b = antib_param_df[(antib_param_df['Antibiotic']==antib)&(antib_param_df['Species']==sp2)].iloc[0]['B']
            if modelType in ['gLV_no_antib']:
                CD_b = 0
                sp2_b = 0
            heatmap_row = []
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1([0.00022,0.00198],[0,48],u_vector, a_matrix_CD_sp2,[CD_b,sp2_b], 0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(antib_scaled)):
                conc = antib_scaled[c]
                my_sim_data = simulate_gLV_Model1([0.00022,0.00198],[0,48],u_vector, a_matrix_CD_sp2,[CD_b,sp2_b], conc)
                sim_df = pd.DataFrame(my_sim_data)
                CDOD = sim_df.loc[1][0]
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > {'Metronidazole':1.6114,'Vancomycin':1.112}[antib]:
                    heatmap_row.append(1)
                else:
                    heatmap_row.append(0)
            model_heatmap.append(heatmap_row)
        df1 = pd.DataFrame(model_heatmap)
        df2 = pd.DataFrame(data_heatmap)
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
    total= count_match + count_unmatch
    frac = count_match/total
    modelType_fraction_correct[modelType] = [frac]
    modelType_fraction_correct[modelType+'_num'] = [count_match]
modelType_df = pd.DataFrame.from_dict(modelType_fraction_correct)
modelType_df.to_csv('part1_model_prediction_accuracy.csv')
    
            