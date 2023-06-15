# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 05:54:09 2020

@author: VOVA
"""
import numpy as np
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio

name = 'CA1_22_HT1' 
path = 'C:\Work\Holes\\'
time_shift = 34.5

ms = mcl.Mouse(name, time_shift, path_track = path + name + '_track.csv', path_neuro = path + name + '_NR_raw_neuro.csv', xy_delim = '\t', xy_sk_rows = 0)
ms.get_spike_data(spike_path = path + name + '_NR_spikes.csv')
#ms.get_angle()
#ms.get_ks_score(min_n_spikes = 3)
#ms.get_place_cells_in_circle(mode = 'spikes')
#np.savez(path + name + '_test.npz', [ms])
#
#
#ms = pio.LoadMouse(path + name + '_test.npz')
#ms = mcl.FieldsToList(ms)
#ms.get_binned_neuro(Nbins = 5)
#
#drcl.DrawAllCells(ms, k_word = 'test_circle_map')