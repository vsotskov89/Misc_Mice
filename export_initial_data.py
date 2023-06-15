# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:07:32 2020

@author: vsots
"""

import numpy as np
import mouse_class as mcl
import pc_io as pio
import draw_cells as drcl


names = ['CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']
#'CA1_22_1D','CA1_22_2D','CA1_22_3D',
path = 'D:\\Work\\PLACE_CELLS\\'

time_shift = [25.4, 11.9, 23.4, 17.35, 15.3, 13.35, 25.3, 26.05, 9.9]
#[15.35, 10.7, 14.65, 
#t_end = [989.55, 947.25, 750.8, 988.25, 748.3, 773.8, 996.6, 776.0, 769.3, 1111.7, 794.3, 763.8]

cen_data = np.genfromtxt(path + 'Circle_params.csv', delimiter = ',', skip_header = 4)

for i,name in enumerate(names):
    ms = mcl.Mouse(name, time_shift[i], path_track = path + name + '_track.csv', path_neuro = path + name + '_cnmfe_raw_neuro.csv', cen_data = cen_data[i,1:5])
#    ms.get_spike_data(path + name + '_cnmfe_spikes_filtered.csv')
    ms.get_markers_data(path_track = path + name + '_track.csv', cen_data = cen_data[i,1:5])
#    ms.trim_time(t_end[i])
    drcl.Plot_All_Coord(ms, path + name + '_all_coord.png')
    np.savez(path + name + '_initial_data.npz', [ms])
#    pio.ExportSpikes(ms, path + name + '_initial_data_spikes.csv')
    pio.Export_All_Initial_Data(ms, path + name + '_initial_data.csv')
    