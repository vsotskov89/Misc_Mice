# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:44:04 2021

@author: vsots
"""
import numpy as np
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import Heat_field_Vitya as hf
import PlaceFields as pcf

# names = ['NC_701_1D', 'NC_701_2D', 'NC_702_1D', 'NC_702_2D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D', 'FG_2_2D']
names = ['NC_701_1D','NC_702_1D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D']
path = 'D:\Work\\N_FGCAMP\\'
time_shift = [7.5, 28.3, 11.7, 16.5, 21.2, 10.7, 18.05, 8.15, 15.3]


# i = 8
# name = names[8]
# # for i,name in enumerate(names):
# ms = mcl.Mouse(name, time_shift[i], path_track = path + name + '_track_fl.csv', path_neuro = path + name + '_NR_data_processed_traces.csv', xy_delim = ',')
# ms.get_spike_data(path + 'spikes_' + name + '_NR_data_processed_traces.csv')
# ms.get_min1pipe_spikes(path + name + '_NR_data_processed_spikes.csv')
# ms.get_cell_centers(path + name + '_NR_data_processed_filters\\')

# if i>0:
#     ms.get_markers_data(path + name + '_track_fl.csv', delimiter = ',')

# ms.get_angle()
# ms.get_ks_score(min_n_spikes = 3)
# ms.get_place_cells_in_circle(mode = 'spikes')
# np.savez(path + name + '_sp.npz', [ms])

# # for i,name in enumerate(names):    
# ms = pio.LoadMouse(path + name + '_sp.npz')
# sp_cells = np.array([np.count_nonzero(i) for i in ms.spikes.transpose()])
# print(ms.name + '\n' + str(ms.n_cells) + '\n' + str(len(sp_cells[sp_cells>=3])) + '\n' + str(len(ms.pc)) + '\n')    


# ms = mcl.FieldsToList(ms)
# ms.get_binned_neuro(10)    
# drcl.DrawAllCells(ms, k_word = path + name + '_sp_circle_map')


for i,name in enumerate(names):    
    ms = pio.LoadMouse(path + name + '_sp.npz')
    ms = mcl.FieldsToList(ms)
    if hasattr(ms,'markers'):
        drcl.Plot_All_Coord(ms, path + name + '_coord.png')
    drcl.DrawTSpecVsMuDistrib(ms, path + name + '_t_spec_vs_mu_distr.png')    

