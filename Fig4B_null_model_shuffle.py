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
import random
import ast

def gLV_Model1(x_vector, t=0, u_vector=None, a_matrix=None, antib_param_vector=None, antib_conc=None):
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)):
        new_x_vector[i] = x_vector[i] * (u_vector[i] + np.sum([a_matrix[i][j] * x_vector[j] for j in range(len(x_vector))])+(antib_param_vector[i] * antib_conc))
    return new_x_vector

def simulate_gLV_Model1(x0_vector,Ts,u_vector, a_matrix, antib_param_vector, antib_conc):
    return odeint(gLV_Model1, x0_vector, Ts, args=(u_vector, a_matrix, antib_param_vector, antib_conc))

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

# load data
part1_df = pd.read_csv('Abundance_data_Fig2BC_part1.csv', index_col=[0]) #this is the pairs data, the same data as 'Abundance_data_Fig2BC' deposited at https://doi.org/10.5281/zenodo.7626486 , just formatted slightly differently
E81_df = pd.read_csv('Abundance_data_SuppFig3AB_part2.csv', index_col=[0]) #this is the same data as 'Abundance_data_SuppFig3AB' part 2 deposited at https://doi.org/10.5281/zenodo.7626486 , just formatted slightly differently
E90_df = pd.read_csv('Abundance_data_SuppFig3AB_part1.csv', index_col=[0]) #this is the same data as 'Abundance_data_SuppFig3AB' part 1 deposited at https://doi.org/10.5281/zenodo.7626486 , just formatted slightly differently
E32_df = pd.read_csv('Abundance_data_SuppFig3AB_part3.csv', index_col=[0]) #this is the same data as 'Abundance_data_SuppFig3AB' part 3 deposited at https://doi.org/10.5281/zenodo.7626486 , just formatted slightly differently
#duplicate 0 antibiotic condition to be Metr and Vanco
zero_df = E81_df[E81_df['Concentration']==0]
for ind in zero_df.index:
    zero_df.at[ind,'Antibiotic'] = 'Metronidazole'
E81_df = pd.concat([E81_df, zero_df])
#duplicate 0 antibiotic condition to be Metr and Vanco
zero_df = E90_df[E90_df['Concentration']==0]
for ind in zero_df.index:
    zero_df.at[ind,'Antibiotic'] = 'Metronidazole'
E90_df = pd.concat([E90_df, zero_df])

#load metadata
E81_comms = {'SEHE81_1':[1],'SEHE81_2':[1,3,4,8],'SEHE81_3':[1,3,4,6,8],'SEHE81_4':[1,3,4,8,9],'SEHE81_5':[1,3,4,8,13],'SEHE81_6':[1,3,4,7,8,14]}
E90_communities = {'SEHE90_1':[1],'SEHE90_2':[1,3,4],'SEHE90_3':[1,3,4,6],'SEHE90_4':[1,3,4,9],'SEHE90_5':[1,3,4,11],'SEHE90_6':[1,3,4,7,14],'SEHE90_7':[1,3,4,10],'SEHE90_8':[1,3,4,12],'SEHE90_9':[1,3,4,6,11],'SEHE90_10':[1,3,4,7,9,14],'SEHE90_11':[1,3,4,6,9,11],'SEHE90_12':[1,3,4,7,11,14]}
E32_comm = pd.read_csv('SEHE32_metadata.csv')
E32_splists = []
for ind in E32_comm.index:
    splist = ast.literal_eval(E32_comm.at[ind,'Community'])
    if splist not in [['1','4'],['1','6']]:
        E32_splists.append(splist)

# load best fit parameters
speciesID={'CD':1,'ER':2,'DP':3,'FP':4,'BH':5,'CA':6,'PC':7,'EL':8,'CH':9,'BO':10,'BT':11,'BU':12,'BV':13,'CS':14}
gLV_params_df = pd.read_csv('gLV_parameters_MSB_2021.csv', header=None, index_col=None)
u_gLV_vector = gLV_params_df[0].tolist()[0::15]
aijs = gLV_params_df[0].tolist()
del aijs[0::15]
a_matrix = to_matrix(aijs, 14) 
antib_param_df = pd.read_csv('Model1_scaled_parameter.csv', index_col=[0]) 
scaling_factor_dict = {'Metronidazole':24,'Vancomycin':12} 


#initialize dataframe
b_df = pd.DataFrame(columns = ['Shuffle iteration','Experiment','Antibiotic','Match','Total','Fraction','B_list'])

#%% Simulate conditions from part1-part4 data with null model of the expanded gLV model (Model1) with shuffled antibiotic susceptibility parameters (B)
# output1: save the fraction of correct prediction for Model1 with 1000 shuffled parameter set
# output2: save a plot of the fraction of correct prediction for each of 1000 shuffled parameter set 
b_list_mtz = []
b_list_van = []
for s in range(1,15):
    sp = numID[s]
    sp_b_mtz = antib_param_df[(antib_param_df['Antibiotic']=='Metronidazole')&(antib_param_df['Species']==sp)].iloc[0]['B']
    sp_b_van = antib_param_df[(antib_param_df['Antibiotic']=='Vancomycin')&(antib_param_df['Species']==sp)].iloc[0]['B']
    b_list_mtz.append(sp_b_mtz)
    b_list_van.append(sp_b_van)
b_list_dict = {'Metronidazole':b_list_mtz,'Vancomycin':b_list_van}

for b in range(0,1000):
    print(b)
    for antib in ['Metronidazole','Vancomycin']:
        if b == 0:
            b_list_shuffle = b_list_dict[antib]
        else:
            b_list_shuffle = random.sample(b_list_dict[antib], len(b_list_dict[antib]))
        #part1 (pairs data)
        count_match = 0
        count_unmatch = 0
        count_nodata = 0
        d_heatmap = []
        m_heatmap = []
        for cond in ['CD']:
            u_vector = [u_gLV_vector[0],0]
            a_matrix_CD_sp2 = [[a_matrix[0][0],0],[0,0]]
            subdf = part1_df[(part1_df['Condition']==cond)&(part1_df['Antibiotic']==antib)]
            conclist = {'Metronidazole':[0,0.75,1.5,3,6,12],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3]}[antib]
            scaling_factor = scaling_factor_dict[antib]
            antib_scaled = [conclist[a]/scaling_factor for a in range(len(conclist))]
            #data
            heatmap_row = []
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
            d_heatmap.append(heatmap_row)
            #simulation
            heatmap_row = []
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1([0.00022,0],[0,48],u_vector, a_matrix_CD_sp2,[b_list_shuffle[speciesID['CD']-1],0], 0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(antib_scaled)):
                conc = antib_scaled[c]
                my_sim_data = simulate_gLV_Model1([0.00022,0],[0,48],u_vector, a_matrix_CD_sp2,[b_list_shuffle[speciesID['CD']-1],0], conc)
                sim_df = pd.DataFrame(my_sim_data)
                CDOD = sim_df.loc[1][0]
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > 1:
                    heatmap_row.append(1)
                else:
                    heatmap_row.append(0)
            m_heatmap.append(heatmap_row)
        for sp2 in ['DP','FP','BH','CA','PC','CH','ER','EL','BO','BT','BU','BV','CS']:
            cond = 'CD-'+sp2
            sp2num = speciesID[sp2]
            u_vector = [u_gLV_vector[0],u_gLV_vector[sp2num-1]]
            a_matrix_CD_sp2 = [[a_matrix[0][0],a_matrix[0][sp2num-1]],[a_matrix[sp2num-1][0],a_matrix[sp2num-1][sp2num-1]]]
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
            d_heatmap.append(heatmap_row)
            #Simulation
            heatmap_row = []
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1([0.00022,0.00198],[0,48],u_vector, a_matrix_CD_sp2,[b_list_shuffle[speciesID['CD']-1],b_list_shuffle[speciesID[sp2]-1]], 0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(antib_scaled)):
                conc = antib_scaled[c]
                my_sim_data = simulate_gLV_Model1([0.00022,0.00198],[0,48],u_vector, a_matrix_CD_sp2,[b_list_shuffle[speciesID['CD']-1],b_list_shuffle[speciesID[sp2]-1]], conc)
                sim_df = pd.DataFrame(my_sim_data)
                CDOD = sim_df.loc[1][0]
                subMICfc = CDOD/CDODzero
                if CDOD < 0.05:
                    heatmap_row.append(-1) #no growth
                elif subMICfc > 1:
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
        total= count_match + count_unmatch
        frac = count_match/total
        b_df.loc[(len(b_df.index))] = [b,'E54',antib, count_match,total, frac,b_list_shuffle]
        #E81
        d_heatmap = []
        m_heatmap = []
        tmpt=47
        cutoff = 0.052
        count_match = 0
        count_unmatch = 0
        count_nodata = 0
        conclist = {'Metronidazole':[0,0.375,0.75,1.5,3,6,12,24,48],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3,6]}[antib]
        for s in range(6):
            #Get general information
            comm_name = 'SEHE81_'+str(s+1)
            splist = E81_comms[comm_name]
            initial_condition = [0 for i in range(14)]
            #Data
            heatmap_row = []
            zerodf = E81_df[(E81_df['Condition']==comm_name)&(E81_df['Time']==tmpt)&(E81_df['Antibiotic']==antib)&(E81_df['Concentration']==0)]
            CDODzero = np.mean(zerodf['sp1OD']) 
            for c in range(len(conclist)):
                conc = conclist[c]
                concdf = E81_df[(E81_df['Condition']==comm_name)&(E81_df['Time']==tmpt)&(E81_df['Antibiotic']==antib)&(E81_df['Concentration']==conc)]
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
            initial_OD = 0.0022 
            for spnum in splist:
                initial_condition[int(spnum)-1]=initial_OD
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(conclist)):
                conc = conclist[c]
                spnum = 1
                scaling_factor = scaling_factor_dict[antib]
                scaled_conc = conc/scaling_factor
                #do simulation
                my_sim_data = simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,scaled_conc)                    
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
        total= count_match + count_unmatch
        frac = count_match/total
        b_df.loc[(len(b_df.index))] = [b,'E81',antib, count_match,total, frac,b_list_shuffle]
        #E90
        tmpt = 48
        conclist = {'Metronidazole':[0,0.375,0.75,1.5,3,6,12,24,48],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3,6]}[antib]
        d_heatmap = []
        m_heatmap = []
        yticklabels = []
        count_match = 0
        count_unmatch = 0
        count_nodata = 0
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
            sim_zero_df = pd.DataFrame(simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,0))
            CDODzero = sim_zero_df.loc[1][0]
            for c in range(len(conclist)):
                conc = conclist[c]
                spnum = 1
                scaling_factor = scaling_factor_dict[antib]
                scaled_conc = conc/scaling_factor
                #do simulation
                my_sim_data = simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,scaled_conc)                    
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
        total= count_match + count_unmatch
        frac = count_match/total
        b_df.loc[(len(b_df.index))] = [b,'E90',antib, count_match,total, frac,b_list_shuffle]
        #E32
        if antib == 'Metronidazole':
            conclist = {'Metronidazole':[0,0.375,0.75,1.5,3,6,12,24,48],'Vancomycin':[0,0.09375,0.1875,0.375,0.75,1.5,3,6]}[antib]
            count_match = 0
            count_unmatch = 0
            count_nodata = 0
            d_heatmap = []
            m_heatmap = []
            yticklabels = []
            for mys in [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]: 
                comm_max = 0
                splist = E32_splists[mys]
                spnamelist = [numID[int(spnum)] for spnum in splist]
                commdf = E32_df[(E32_df['splist']==str(splist))]
                mycondition = 'SEHE32_fakenum_'+str(mys)
                yticklabels.append(mycondition)
                initial_condition = [0 for i in range(14)]
                #Data
                heatmap_row = []
                zerodf = E32_df[(E32_df['splist']==mycondition)&(E32_df['Time']==tmpt)&(E32_df['Concentration']==0)]
                CDODzero = np.mean(zerodf['sp1OD']) 
                for c in range(len(conclist)):
                    conc = conclist[c]
                    concdf = E32_df[(E32_df['splist']==str(splist))&(E32_df['Time']==tmpt)&(E32_df['Concentration']==conc)]
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
                sim_zero_df = pd.DataFrame(simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,0))
                CDODzero = sim_zero_df.loc[1][0]
                for c in range(len(conclist)):
                    conc = conclist[c]
                    spnum = 1
                    scaling_factor = scaling_factor_dict[antib]
                    scaled_conc = conc/scaling_factor
                    #do simulation
                    my_sim_data = simulate_gLV_Model1(initial_condition,[0,48],u_gLV_vector, a_matrix,b_list_shuffle,scaled_conc)                    
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
            total= count_match + count_unmatch
            frac = count_match/total
            b_df.loc[(len(b_df.index))] = [b,'E32',antib, count_match,total, frac,b_list_shuffle]
b_df.to_csv('part1_part2_part3_part4_shuffle_null_model_prediction_accuracy.csv') #output1

#%% Output2, plot of prediction accuracy for each of 1000 shuffles
# shade each point with the distance of shuffled C. difficile B parameter from unshuffled correct B parameter
import ast
B_unshuffle = {'Metronidazole':ast.literal_eval(b_df.at[0,'B_list']),'Vancomycin':ast.literal_eval(b_df.at[4,'B_list'])}
barplot_df = pd.DataFrame(columns=['b_fraction','distance','CD_distance','b_num','even/odd','constant'])
b_fractions = []
b_unshuffled = -1
b_num = int(len(b_df)/8)
for b in range(1,b_num): 
    sub_bdf = b_df[b_df['Shuffle iteration']==b]
    match_sum = np.sum(sub_bdf['Match'])
    total_sum = np.sum(sub_bdf['Total'])
    sum_antib_distance = 0
    sum_CD = 0
    for antibiotic in ['Vancomycin','Metronidazole']:
        my_distance = 0
        antibsub_bdf = b_df[(b_df['Shuffle iteration']==b)&(b_df['Antibiotic']==antibiotic)]
        B_shuffle = ast.literal_eval(antibsub_bdf.iloc[0]['B_list'])
        for i in range(len(B_unshuffle)):
            my_distance += abs(B_shuffle[i] - B_unshuffle[antibiotic][i])
        CD_distance = abs(B_shuffle[0]-B_unshuffle[antibiotic][0])
        sum_antib_distance += my_distance
        sum_CD += CD_distance
    if b == 0:
        b_unshuffled = (match_sum/total_sum)
    else:
        b_fractions.append(match_sum/total_sum)
        barplot_df.loc[len(barplot_df)] = [match_sum/total_sum, sum_antib_distance, sum_CD,b, b%2,1]
print('Average of shuffled: '+str(round(np.mean(b_fractions),2)))
print('Min of shuffled: '+str(round(min(b_fractions),2)))
print('Max of shuffled: '+str(round(max(b_fractions),2)))
print('Unshuffled value: '+str(round(b_unshuffled,2)))

#plot
plt.rcParams['pdf.fonttype'] = 42    
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.tick_params(axis='x',labelsize=8)
plt.tick_params(axis='y', labelsize=8)
plt.rcParams['figure.figsize'] = 3.5,1.5
mypalette = sns.dark_palette("#ffffff", reverse=True,  n_colors=100)
plt.barh([0],[np.mean(b_fractions)], alpha=0.5, color='white', edgecolor='black')
ax = sns.stripplot(y="constant",x="b_fraction",data=barplot_df, hue='CD_distance',orient="h",palette=mypalette,size=5,alpha=1, linewidth=1, jitter=0.025)
plt.legend(fontsize=1, bbox_to_anchor=(1.1,1.1))
plt.xlim(0,1)
plt.show()
plt.close()
