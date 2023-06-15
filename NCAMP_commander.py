# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:39:18 2020

@author: 1
"""

import structure as strct
import DrawCells2 as drcl
import pc_io as pio
import numpy as np
import Heat_field_Vitya as hf

path = 'M:\\NCAMP\\'
#fnames = ['NC_701_1D','NC_701_1D','NC_702_1D']
fnames = ['CA1_22_1D','CA1_23_1D','CA1_24_1D','CA1_25_1D']
#ts = [7.5, 11.7]
ts = [15.35, 25.4, 17.35, 25.3]
cells = [[10,11,12,13,14,16,20,22,25,26,35,36,41,42,48,49,52,56,58,60,63,65,68,70,73,74],[5,6,10,11,14,20,23,40,42], [0,2,6,10,17,18,19,22,26,27,34],[3,4,5,6,8,9,10,14,15,16,17,18,19,20,21,22,23,27,28,30,31,36,38,40,41,48,49,51,52,53,54,55]]
max_neur = np.zeros((50,78))

#for i,f in enumerate(fnames):
#    Mouse = strct.Digo(f,  path_track = path + f + '_track.csv', path_spikes = path + 'spikes_' + f + '_data_processed_traces.csv', time_shift = ts[i])
#    Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
#    Mouse.get_neuropil(path + f + '_data_processed_traces.csv')
#    Mouse.get_angle()
#    Mouse.get_direction()
#    
#    Mouse = strct.GetFieldsNeuro(Mouse)
#    Mouse = strct.FilterFields(Mouse)
#     
#    
#    Mouse.get_binned_neuro(10)
#    Mouse.get_binned_angle(40)
#    Mouse.get_mi_score()
#    
#    np.savez(path + f + '_neur.npz', [Mouse])
#    drcl.DrawAllTrue(Mouse, k_word = path +  f + '_neur_')
#
#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.FilterFields(Mouse)    
#
#    np.savez(path + f + '.npz', [Mouse])
#   drcl.DrawAllTrue(Mouse, k_word = path +  f + '_sp_')
k = 0    
for i,f in enumerate(fnames):
    Mouse = pio.LoadMouse(path+f+'.npz')
#    Mouse = strct.FilterFields(Mouse)
#    np.savez(path + f + '.npz', [Mouse])
    for cell in cells[i]:
        for j,ti in enumerate(Mouse.times_in[cell]):
            to = Mouse.times_out[cell][j]
            if to<=ti:
                continue
            max_neur[j,k] = np.max(Mouse.neur[ti:to, Mouse.true_cells[cell]])
        k+=1
#            
#drcl.DrawNeuroPasses(max_neur, path + 'Max_neur')             
#    drcl.DrawAllTrue(Mouse, k_word = path +  f + '_neur_')