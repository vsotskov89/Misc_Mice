# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:52:11 2020

@author: 1
"""

#import structure as strct
import DrawCells2 as drcl
import pc_io as pio
import numpy as np
import Heat_field_Vitya as hf
import SpikingPasses as  spps
import mouse_class as mcl


path = 'D:\Work\\HOLY_TEST\\'   
t_sync = [34.48, 18.12, 14.24, 54.28, 43.8, 30.8, 13.52, 8.44, 19.4, 16.88, 43.92, 16.88]
fnames = ['CA1_22_HT1','CA1_22_HT2','CA1_22_HT3','CA1_23_HT1','CA1_23_HT2','CA1_23_HT3','CA1_24_HT1','CA1_24_HT2','CA1_24_HT3','CA1_25_HT1','CA1_25_HT2','CA1_25_HT3']

for i,name in enumerate(fnames):
    ms = mcl.Mouse(name, t_sync[i], path_track = path + name + '_track.csv', path_neuro = path + name + '_NR_raw_neuro.csv', xy_delim = '\t', xy_sk_rows = 0)
    ms.get_spike_data(path + 'spikes_' + name + '_NR_raw_neuro.csv')
    
    np.savez(path + name + '.npz', [ms])   
    
    pio.ExportTrack(ms, path + name + '_track_from_npz.csv')
    pio.Export_Neurodata(ms, path + name + '_neurodata_from_npz.csv')
    pio.ExportSpikes(ms, path + name + '_spikes_from_npz.csv') 
    
    
#    Mouse = strct.Digo(f,  path_track = path + f + '_track.csv', path_spikes = path + f + '_NR_spikes.csv', time_shift = t_sync[i])
#    Mouse.get_xy_data(delimiter = '\t', skip_rows = 0, skip_cols = 0)
#    Mouse.get_neuropil(path + fnames[i] + '_NR_raw_neuro.csv')
#    
#    
#    pio.ExportCoordAndTraces(Mouse, path+f+'_coord_traces.csv')
#    np.savez(path + f + '_light.npz', [Mouse])
    
    # Mouse = pio.LoadMouse(path + f + '_light.npz')
    # Mouse.get_holed_bins(step = 64, vid_h = 1080, vid_w = 1440)
    # Mouse.get_binned_neuro(Nbins = 5)
    # Mouse.get_mi_score()
    # for cell in range(len(Mouse.spikes[0])): 
    #     drcl.DrawXYMap(Mouse, cell, f_out = path + f + 'mi_' + str(Mouse.mi_score[cell]) + '_xy_map_cell_' + str(cell) + '.png', line_width = 3)
    #jkvnjgb