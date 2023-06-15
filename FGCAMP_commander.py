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

path = 'D:\VOVA\\NCaMP\\'
fnames = ['FG_1','FG_2_1D']
ts = [46.1, 8.15]
#cells = [26,38,39,40,42,53,69]
cells = [3,7,8,9,10,12,15]
max_neur = np.zeros((50,7))
#
#for i,f in enumerate(fnames):
#    Mouse = strct.Digo(f,  path_track = path + f + '_track.csv', path_spikes = path + 'spikes_' + f + '_NR_data_processed_traces.csv', time_shift = ts[i])
#    Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
#    Mouse.get_neuropil(path + f + '_NR_data_processed_traces.csv')
#    Mouse.get_angle()
#    Mouse.get_direction()
#    
##    np.savez(path + f + '_neur.npz', [Mouse])   
#    
#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.FilterFields(Mouse)
#     
#    
#    Mouse.get_binned_neuro(10)
#    Mouse.get_binned_angle(40)
#    Mouse.get_mi_score()
#    
#    np.savez(path + f + '_sp.npz', [Mouse])
#    drcl.DrawAllTrue(Mouse, k_word = path +  f + '_sp_')

#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.FilterFields(Mouse)    
#
#    np.savez(path + f + '.npz', [Mouse])
#    drcl.DrawAllTrue(Mouse, k_word = path +  f + '_sp_')
#k = 0    
#for i,f in enumerate(fnames):
#    Mouse = pio.LoadMouse(path+f+'.npz')
##    Mouse = strct.FilterFields(Mouse)
##    np.savez(path + f + '.npz', [Mouse])
#    for cell in cells[i]:
#        for j,ti in enumerate(Mouse.times_in[cell]):
#            to = Mouse.times_out[cell][j]
#            if to<=ti:
#                continue
#            max_neur[j,k] = np.max(Mouse.neur[ti:to, Mouse.true_cells[cell]])
#        k+=1
#            
#drcl.DrawNeuroPasses(max_neur, path + 'Max_neur')             
#    drcl.DrawAllTrue(Mouse, k_word = path +  f + '_neur_')

Mouse = pio.LoadMouse(path+fnames[1]+'_neur.npz')
#drcl.DrawAllTrue(Mouse, k_word = path + fnames[1] + '_neur_')
 
for c,cell in enumerate(cells):
    for j,ti in enumerate(Mouse.times_in[cell]):
        to = Mouse.times_out[cell][j]
        if to<=ti:
            continue
        max_neur[j,c] = np.max(Mouse.neur[ti:to, Mouse.true_cells[cell]])

#            