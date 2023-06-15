# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:17:29 2020

@author: 1
"""

import structure as strct
import DrawCells2 as drcl
import pc_io as pio
import numpy as np
import Heat_field_Vitya as hf

path = 'C:\Work\PLACE_CELLS\\'   
fnames = ['CA1_22_1D','CA1_22_2D','CA1_22_3D','CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']
days = [[0,3,6,9],[1,4,7,10],[5,8,11]]
mice = [0,1,2,3]
all_cells = [[],[],[]]
all_mu = [[],[],[]]
all_std = [[],[],[]]
match = []
match2 = []
hmap = [[],[],[]]
sortday = 0

ds = [0, 1, 2]
        
def RemoveRepeats(cell_list, mu_list, std_list, n_cells):
#Remove repeats of the fields of the same cell    
    for cn in range(n_cells): 
        ind = np.argwhere(np.array(cell_list) == cn)
        for rep in range(1,len(ind)):
            del cell_list[ind[rep,0]]
            del mu_list[ind[rep,0]]
            del std_list[ind[rep,0]]


for m in mice:
    match.append(np.genfromtxt(path + fnames[m*3][:6] + '_match_new.csv', delimiter = ','))
    
    #Sort match by cells of 2nd day
    match2.append(np.array([match[m][i,:] for i in np.argsort(match[m][:,1])]))



for m in days[sortday]:
    Mouse = pio.LoadMouse(path+fnames[m]+'_light.npz')
    all_cells[sortday].append(Mouse.super_cells)
    all_mu[sortday].append(np.array(Mouse.mu_list)[Mouse.super_in_true].tolist())
    all_std[sortday].append(np.array(Mouse.std_list)[Mouse.super_in_true].tolist())
       
    #in bimodal case let's leave largerst field only
    RemoveRepeats(all_cells[sortday][-1], all_mu[sortday][-1], all_std[sortday][-1], len(Mouse.spikes[0]))


for d in range(3):
    if d == sortday:
        continue
    for m in mice:
        all_cells[d].append((np.zeros(len(all_cells[sortday][m]))*np.nan).tolist())
        all_mu[d].append((np.zeros(len(all_cells[sortday][m]))*np.nan).tolist())
        all_std[d].append((np.zeros(len(all_cells[sortday][m]))*np.nan).tolist())

for d in range(3):
    if d == sortday:
        continue
    for m in days[d]:

        Mouse = pio.LoadMouse(path+fnames[m]+'.npz')
        m_num = int((m-d)/3)
        
        for row, cell in enumerate(all_cells[sortday][m_num]):
            all_cells[d][m_num][row] = match2[m_num][cell,d]-1
            if match2[m_num][cell,d]-1 in Mouse.super_cells: #or np.isnan(match[mn][cell,d]-1):
#                all_cells[d][mn][row] = np.nan;
#                all_mu[d][mn].append(np.nan)
#                all_std[d][mn].append(0)
#            else:
                ind = np.argwhere(np.array(Mouse.super_cells)==match2[m_num][cell,d]-1)
                all_mu[d][m_num][row] = Mouse.mu_list[ind[0,0]]
                all_std[d][m_num][row] = Mouse.std_list[ind[0,0]]
                        
                
fields = np.array([all_cells, all_mu, all_std])



for m in mice:
    sorted_fields = np.argsort(fields[1,sortday,m])
#    hf.DrawAllfields(mu_list = fields[1,0,m], std_list = fields[2,0,m], outfname = path + fnames[m*3] + '_sorted_fields.png', sort = True)
    for d in [0,1,2]:
        mu_list = [fields[1,d,m][fi] for fi in sorted_fields]
        std_list = [fields[2,d,m][fi] for fi in sorted_fields]
        hf.DrawAllfields(mu_list = mu_list, std_list = std_list, outfname = path + fnames[m*3+d] + '_sorted_fields_new.png', sort = False)

glob_mu = [[],[],[]]
glob_std = [[],[],[]] 

for d in [0,1,2]:
    for m in [1,3]:
        glob_mu[d]+=fields[1,d,m]
        glob_std[d]+=fields[2,d,m]
        
sorted_fields = np.argsort(glob_mu[0])        

for d in [0,1,2]:
    glob_mu[d] = [glob_mu[d][fi] for fi in sorted_fields]
    glob_std[d] = [glob_std[d][fi] for fi in sorted_fields] 
    hf.DrawAllfields(mu_list = glob_mu[d], std_list = glob_std[d], outfname = path + '3D_mice_fields_' + str(d+1)+ 'D.png', sort = False)    
        

