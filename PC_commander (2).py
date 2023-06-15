# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:24 2019

@author: VOVA
"""

import structure as strct
import DrawCells2 as drcl
import pc_io as pio
import numpy as np
import Heat_field_Vitya as hf

path = 'M:\PLACE CELLS\\'   
ts = [15.35, 10.7, 14.65, 25.4, 11.9, 23.4, 17.35, 15.3, 13.35, 25.3, 26.05, 9.9]
fnames = ['CA1_22_1D','CA1_22_2D','CA1_22_3D','CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']

non_super = [[2,3,4,7,8,10,13,35,43,44,55,63,83,164,183,185,189,194,201,226,231,234,244,253,260,261,273,274,289,299,302,308,316,323,329,334,356,357],[14,104,116,120,144,206,214,215,219,223,224,228,244,248,250,252,266,274,278,283,304],[],[11,22,24,26,42,49,57,58,77,78,83,86,111,121],[0,1,2,3,4,5,6,7,8,9,10,14,16,17,19,22,23,24,25,31,32,35,37,41],[0,1,2,3,4,5,9,12,16,17,23,25,29,36,38],[12,19,26,28,29,31],[15,22,34,46],[],[14,25,82,85],[9,17,32,48,51,54],[20]]
days = [[0,3,6,9],[1,4,7,10],[5,11]]
first_t = [[],[],[]]
sum_t = [[],[],[]]
aver_neur = [[],[],[]]
max_neur = [[],[],[]]
sum_neur = [[],[],[]]
glob_mu_list = [[],[],[]]
glob_std_list = [[],[],[]]


#for i in range(len(fnames)):
#    Mouse = strct.Digo(fnames[i],  path_track = path + fnames[i] + '_track.csv', path_spikes = path + fnames[i] + '_cnmfe_spikes_filtered.csv', time_shift = ts[i])
#    Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
#    Mouse.get_neuropil(path + fnames[i] + '_cnmfe_raw_neuro.csv')
#    Mouse.get_angle()
#    Mouse.get_direction()
#    
#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.FilterFields(Mouse)
#     
#    
#    Mouse.get_binned_neuro(10)
#    Mouse.get_binned_angle(40)
#    Mouse.get_mi_score()
#    
#    np.savez(path + fnames[i] + '.npz', [Mouse])

#
for ms,f in enumerate(fnames):
    Mouse = pio.LoadMouse(path+f+'.npz')    
##    Mouse.super_ind = [i for i,cell in enumerate(Mouse.true_cells) if np.count_nonzero(Mouse.sp_passes[:][i]) >= 5]
##    Mouse.super_ind = [cell for i,cell in enumerate(Mouse.super_ind) if i not in non_super[ms]]
##            
#
    Mouse.aver_neuro_passes = []
    Mouse.max_neuro_passes = []
    Mouse.sum_neuro_passes = []
    
    for ind,cell in enumerate(Mouse.super_cells):
        aver_pass = []
        max_pass = []
        sum_pass = []
        for i,ti in enumerate(Mouse.times_in[Mouse.super_ind[ind]]):
            to = Mouse.times_out[Mouse.super_ind[ind]][i]
            if to<=ti:
                continue
            aver_pass += [np.mean(Mouse.neur[ti:to, cell])]
            max_pass += [np.max(Mouse.neur[ti:to, cell])]
            sum_pass += [sum(Mouse.neur[ti:to, cell])]             
        Mouse.aver_neuro_passes.append(aver_pass)
        Mouse.max_neuro_passes.append(max_pass)
        Mouse.sum_neuro_passes.append(sum_pass)
        
    np.savez(path + f + '.npz', [Mouse])
    
#min_passes = []    
#for ms,f in enumerate(fnames):
#    Mouse = pio.LoadMouse(path+f+'.npz')
#    min_passes.append(Mouse.min_n_passes)
    
#    
#    
for d in [0,1,2]:
    for ms,f in enumerate(fnames):
        Mouse = pio.LoadMouse(path+f+'.npz')
        if ms in days[d]:
            glob_mu_list[d] += Mouse.super_mu
            glob_std_list[d] += Mouse.super_std
            for i,cell in enumerate(Mouse.super_cells):
                aver_neur[d] += Mouse.aver_neuro_passes[i][0:10]
                max_neur[d] += Mouse.max_neuro_passes[i][0:10]
                sum_neur[d] += Mouse.sum_neuro_passes[i][0:10] 
    
drcl.DrawNeuroPasses(aver_neur, path + 'Aver_neur')    
drcl.DrawNeuroPasses(max_neur, path + 'Max_neur')   
drcl.DrawNeuroPasses(sum_neur, path + 'Sum_neur')     
    
    
    
    
#for d in [0,1,2]:
#    for ms,f in enumerate(fnames):
#        Mouse = pio.LoadMouse(path+f+'.npz')
#        if ms in days[d]:
#            for i in range(len(Mouse.t_spec)):
#                first_t[d] = first_t[d] + [Mouse.t_spec[i][0]]
#                for t in Mouse.t_spec[i]:
#                    sum_t[d] = sum_t[d] + [t]
    
#    Mouse.super_cells = []
#    Mouse.super_mu = []
#    Mouse.super_std = []
#    for i,cell in enumerate(Mouse.true_cells):
#        if np.count_nonzero(Mouse.sp_passes[:][i]) >= 5:
#            Mouse.super_cells.append(cell)
#            Mouse.super_mu.append(Mouse.mu_list[i])
#            Mouse.super_std.append(Mouse.std_list[i])
#    Mouse.super_cells = [x for i,x in enumerate(Mouse.super_cells) if i not in non_super[ms]]
#    Mouse.super_mu = [x for i,x in enumerate(Mouse.super_mu) if i not in non_super[ms]]
#    Mouse.super_std = [x for i,x in enumerate(Mouse.super_std) if i not in non_super[ms]]
#    np.savez(path + f + '.npz', [Mouse])
#    drcl.TraceviewerPC_horizontal(Mouse, path+f+'_flat_map.png')
    #drcl.DrawPassMap(Mouse, path+f+'_pass_map.png')
#    drcl.DrawAllSuper(Mouse, k_word = path +  f)
#    print(f)
#    print(Mouse.super_cells)
#    hf.DrawAllfields(mu_list = Mouse.super_mu, std_list = Mouse.super_std, outfname = path + f + '_super_fields.png')
#for f in ffnames:
#    Mouse = pio.LoadMouse(path+f+'_sp.npz')
#    drcl.PlotNeuroCirclesAllTrue(Mouse)