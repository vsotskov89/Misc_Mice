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
import SpikingPasses as  spps
import mouse_class as mcl

path = 'D:\Work\PLACE_CELLS\\'   
ts = [15.35, 10.7, 14.65, 25.4, 11.9, 23.4, 17.35, 15.3, 13.35, 25.3, 26.05, 9.9]
t_end = [989.55, 947.25, 750.8, 988.25, 748.3, 773.8, 996.6, 776.0, 769.3, 1111.7, 794.3, 763.8]
fnames = ['CA1_22_1D','CA1_22_2D','CA1_22_3D','CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']
#fnames = ['CA1_22_1D','CA1_23_1D','CA1_24_1D','CA1_25_1D']

#non_super = [[2,3,4,7,8,10,13,35,43,44,55,63,83,164,183,185,189,194,201,226,231,234,244,253,260,261,273,274,289,299,302,308,316,323,329,334,356,357],[14,104,116,120,144,206,214,215,219,223,224,228,244,248,250,252,266,274,278,283,304],[],[11,22,24,26,42,49,57,58,77,78,83,86,111,121],[0,1,2,3,4,5,6,7,8,9,10,14,16,17,19,22,23,24,25,31,32,35,37,41],[0,1,2,3,4,5,9,12,16,17,23,25,29,36,38],[12,19,26,28,29,31],[15,22,34,46],[],[14,25,82,85],[9,17,32,48,51,54],[20]]
days = [[0,3,6,9],[1,4,7,10],[5,11]]
#first_t = [[],[],[]]
#sum_t = [[],[],[]]

#for i,f in enumerate(fnames):
#    Mouse = strct.Digo(f,  path_track = path + f + '_track.csv', path_spikes = path + f + '_cnmfe_spikes_filtered.csv', time_shift = ts[i])
#    Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
#    Mouse.get_neuropil(path + fnames[i] + '_cnmfe_raw_neuro.csv')
#    Mouse.get_angle()
#    Mouse.get_direction()
#    
#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.Get_KS_score(Mouse)
#    Mouse = strct.FilterFieldsNew(Mouse, ks_thresh = 5)
#    Mouse.get_binned_neuro(10)
#    Mouse.get_binned_angle(40)      
#    np.savez(path + f + '.npz', [Mouse])
    

#    Mouse.get_mi_score()
#    
#    np.savez(path + fnames[i] + '.npz', [Mouse])

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
    
    
#for f in fnames: 
#     Mouse = pio.LoadMouse(path+f+'.npz')
#     Mouse = strct.Get_KS_score(Mouse)
#     drcl.DrawMostInformative(Mouse, path+ f +'_ks_', len(Mouse.spikes[0]))

#Mouse = strct.Digo(fnames[8],  path_track = path + fnames[8] + '_track.csv', path_spikes = path + fnames[8] + '_cnmfe_spikes_filtered.csv', time_shift = ts[8])
#Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
#Mouse.get_neuropil(path + fnames[8] + '_cnmfe_raw_neuro.csv')
#Mouse.get_angle()
#Mouse.get_direction()
#
#Mouse = strct.GetFields(Mouse)
#Mouse = strct.Get_KS_score(Mouse)
#Mouse = strct.FilterFields(Mouse)
##Mouse = strct.FilterFieldsNew(Mouse, ks_thresh = 10)
#Mouse.super_in_true = [5, 12, 14, 17, 25, 32, 36, 37, 39, 51, 58, 63, 69]
#Mouse.super_cells = np.array(Mouse.true_cells)[Mouse.super_in_true[:]].tolist()
#Mouse.super_mu = np.array(Mouse.mu_list)[Mouse.super_in_true[:]].tolist()
#Mouse.super_std = np.array(Mouse.std_list)[Mouse.super_in_true[:]].tolist()
#
#Mouse.get_binned_neuro(10)
#Mouse.get_binned_angle(40)     
#np.savez(path + fnames[8] + '_old.npz', [Mouse])
#


#for f in fnames: 
#     Mouse = pio.LoadMouse(path+f+'.npz')
#     Mouse = strct.FilterFieldsNew(Mouse, ks_thresh = 10)
#     Mouse = strct.Get_KS_score(Mouse)
#     Mouse.get_binned_neuro(10)
#     Mouse.get_binned_angle(40)     
#     np.savez(path + f + '.npz', [Mouse])
#     drcl.DrawAllCells(Mouse, k_word = path + Mouse.name + '_circle_plot_cell_')
 #    drcl.DrawPlaceCells(Mouse=Mouse, rad = 5, outname = path + Mouse.name + '_all_fields.png')
 
#for f in fnames: 
#     Mouse = pio.LoadMouse(path+f+'.npz') 
#     pio.ExportSuperCells(Mouse, path+f+'_super_cells.csv')
#
#for f in fnames: 
#     Mouse = pio.LoadMouse(path+f+'_light.npz')
#     Mouse =  strct.ReformFields(Mouse)
#     pio.ExportRears(Mouse, path+f+'_rears_from_npz.csv')     
#     pio.ExportTrack(Mouse, path+f+'_track_from_npz.csv') 
 
#runfile('C:/Work/SCRIPTS/Python/untitled4.py', wdir='C:/Work/SCRIPTS/Python') 
#    
 
# 
#for f in fnames: 
#     Mouse = pio.LoadMouse(path+ 'old_npz\\' + f + '.npz') 
#     for num, cell in enumerate(Mouse.true_cells):
#         if Mouse.t_spec[num][0] == 0:
#             sp_passes, sp_times = spps.getSpikingPercent(Mouse, Mouse.times_in[num], Mouse.times_out[num], cell, 0)
#             Mouse.t_spec[num][0] = sp_times[0]
#     np.savez(path+ 'old_npz\\' + f + '.npz', [Mouse])       
    
# 
#for i,f in enumerate(fnames): 
#     Mouse = pio.LoadMouse(path + f + '_light.npz')
#     Mouse = strct.TrimTime(Mouse, t_end[i])
#     Mouse = strct.FilterFieldsNew_Light(Mouse)
#     Mouse = strct.GetRears(Mouse, path + f + '_track.csv_stances.csv')
#     np.savez(path + f + '_light.npz', [Mouse])  
# 
 
#bef = []
#aft = []
#av_bef = []
#av_aft = []
#
#tm = np.linspace(0, 900, num =15)
#rears = np.zeros(15)
#
#for d in days[0]: 
#     Mouse = pio.LoadMouse(path + fnames[d] + '_light.npz')
##     ncells = len(Mouse.super_cells)
##     for nc, cell in enumerate(Mouse.super_in_true):
##         rears, t_bef,t_aft = strct.RearStat(Mouse,nc, cell)
##         bef.append(t_bef)
##         aft.append(t_aft)
##     av_bef.append(np.nansum(bef[-ncells:])/ncells)
##     av_aft.append(np.nansum(aft[-ncells:])/ncells)
#     for i,ti in enumerate(tm):
#         rears[i] += sum(np.abs(Mouse.rear[i*60*20: (i+1)*60*20]))/1200
#all_c = 0
#sup_c = 0 
#for d in days[0]: 
#     Mouse = pio.LoadMouse(path + 'old_npz\\' + fnames[d] + '.npz')
#     all_c += len(Mouse.spikes[0,:])
#     sup_c += len(Mouse.super_cells)


for f in fnames: 
    ms = pio.LoadMouse(path+f+'_light.npz')

    print(len(ms.super_cells))
    np.savetxt(path + f + '_cells.txt', np.array(ms.super_cells), fmt = '%d')

#    Mouse =  strct.ReformFields(Mouse)
#    drcl.DrawKScoreMap(Mouse, filterpath = path+f+'\\filters\\', f_out = path+f+'_kscore_add_map.png')
    