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

path = 'G:\PLACE CELLS\\'   
ts = [15.35]#, 10.7, 14.65, 25.4, 11.9, 23.4, 17.35, 15.3, 13.35, 25.3, 26.05, 9.9]
fnames = ['CA1_22_1D']#,'CA1_22_2D','CA1_22_3D','CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']


for i in range(len(fnames)):
    Mouse = strct.Digo(fnames[i],  path_track = path + fnames[i] + '_track.csv', path_spikes = path + fnames[i] + '_cnmfe_spikes_filtered.csv', time_shift = ts[i])
    Mouse.get_xy_data(delimiter = ' ', skip_rows = 1, skip_cols = 0)
    Mouse.get_neuropil(path + fnames[i] + '_cnmfe_raw_neuro.csv')
    Mouse.get_angle()
    Mouse.get_direction()
    
#    Mouse = strct.GetFields(Mouse)
#    Mouse = strct.FilterFields(Mouse)
#    Mouse = strct.GetSuperCells(Mouse)
#    strct.WriteCells(Mouse.true_cells, outname = path + fnames[i] + '_true_cells.csv')     
#    strct.WriteCells(Mouse.super_cells, outname = path + fnames[i] + '_super_cells.csv')       
    Mouse.get_binned_neuro(10)
    Mouse.get_binned_angle(40)
    Mouse.get_mi_score()
    Mouse.verify_cells(1000)
    
    
    
    #np.savez(path + fnames[i] + '.npz', [Mouse])

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