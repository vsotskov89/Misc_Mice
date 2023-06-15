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

#names = ['G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_1_2D', 'G7F_2_1D', 'G7F_2_2D']
#names = ['G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_1_2D', 'G7F_2_1D', 'G7F_2_2D', 'NC_701_1D','NC_702_1D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D']

path = 'D:\Work\\NEW_MICE\\'
time_shift = [13.1, 11.55, 11.8, 11.1, 12.35, 9.1]

for i,name in enumerate(names):
    # ms = mcl.Mouse(name, time_shift[i], path_track = path + name + '_track_fl.csv', path_neuro = path + name + '_NR_data_processed_traces.csv', xy_delim = ',')
    # ms.get_spike_data(path + 'spikes_' + name + '_NR_data_processed_traces.csv')
    # ms.get_min1pipe_spikes(path + name + '_NR_data_processed_spikes.csv')
    # ms.get_cell_centers(path + name + '_NR_data_processed_filters\\')
    
    # if i>1:
    #     ms.get_markers_data(path + name + '_track_fl.csv', delimiter = ',')

    # ms.get_angle()
    # ms.get_ks_score(min_n_spikes = 3)
    # ms.get_place_cells_in_circle(mode = 'min_spikes')
    # np.savez(path + name + '_min_sp.npz', [ms])

    
    # ms = pio.LoadMouse(path + name + '_sp.npz')
    # sp_cells = np.array([np.count_nonzero(i) for i in ms.spikes.transpose()])
    # print(ms.name + '\n' + str(ms.n_cells) + '\n' + str(len(sp_cells[sp_cells>=3])) + '\n' + str(len(ms.pc)) + '\n')    


    # ms = mcl.FieldsToList(ms)
    # ms.get_binned_neuro(10)    
    # drcl.DrawAllCells(ms, k_word = path + name + '_min_sp_circle_map')
    
  
    ms = pio.LoadMouse(path + name + '_sp.npz')
    
    # pio.ExportCoordAndTraces(ms, path + name + '_track_and_neur.csv')
    # pio.ExportSpikes(ms, path + name + '_spikes.csv')

    if hasattr(ms, 'markers'):
        pio.Export_Track_and_Markers(ms, path + name + '_track_from_npz.csv')
    else:
        pio.ExportTrack(ms, path + name + '_track_from_npz.csv')
        
    pio.Export_Neurodata(ms, path + name + '_neurodata_from_npz.csv')
    
    print(name, ms.time_shift)
#    ms = mcl.FieldsToList(ms)
    # if hasattr(ms,'markers'):
    #     drcl.Plot_All_Coord(ms, path + name + '_coord.png')
    # drcl.DrawTSpecVsMuDistrib(ms, path + name + '_t_spec_vs_mu_distr.png')  
#    hf.DrawAllfields(ms.mu_list, ms.std_list, path + name + '_pf_distr.png')
#    ms.get_place_cells_in_circle_2d(mode = 'spikes', outpath = path)
