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
import os
import SpikingPasses as spps
import matplotlib.pyplot as plt
import pandas as pd
# import Match_days as md

#names = ['G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_1_2D', 'G7F_2_1D', 'G7F_2_2D']
path = 'D:\Work\\NEW_MICE\\'
#fnames = ['G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_1_2D', 'G7F_2_1D', 'G7F_2_2D', 'NC_701_1D','NC_702_1D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D']
#ts = [15.35, 10.7, 14.65, 25.4, 11.9, 23.4, 17.35, 15.3, 13.35, 25.3, 26.05, 9.9]
#t_end = [989.55, 947.25, 750.8, 988.25, 748.3, 773.8, 996.6, 776.0, 769.3, 1111.7, 794.3, 763.8]
#fnames = ['CA1_22_1D','CA1_22_2D','CA1_22_3D','CA1_23_1D','CA1_23_2D','CA1_23_3D','CA1_24_1D','CA1_24_2D','CA1_24_3D','CA1_25_1D','CA1_25_2D','CA1_25_3D']

#fnames = ['CA1_22_1D', 'CA1_23_1D', 'CA1_24_1D', 'CA1_25_1D', 'G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_2_1D', 'NC_701_1D','NC_702_1D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D']
#fnames = ['CA1_22_2D', 'CA1_23_2D', 'CA1_24_2D', 'CA1_25_2D', 'G7F_2_2D']
fnames = ['FG_2_2D', 'NC_701_2D', 'NC_702_2D']
t_list =[]
n_list = []
glob_spec_score = []

t_spec = []
n_spec = []

#time_shift = [13.1, 11.55, 11.8, 11.1, 12.35, 9.1]
#%% Export data
for i,name in enumerate(fnames):
    ms = pio.LoadMouse(path + name + '_sp_20bins.npz')
    
    if hasattr(ms, 'markers'):
        pio.Export_Track_and_Markers(ms, path + name + '_track_from_npz.csv')
    else:
        pio.ExportTrack(ms, path + name + '_track_from_npz.csv') 
        
    pio.Export_Neurodata(ms, path + name + '_neurodata_from_npz.csv')
    pio.ExportSpikes(ms, path + name + '_spikes_from_npz.csv')     



#%% Mice creation

for i,name in enumerate(fnames):
    ms = pio.LoadMouse(path + name + '_sp.npz')
    ts = ms.time_shift
    ms = mcl.Mouse(name, ts, path_track = path + name + '_track_fl.csv', path_neuro = path + name + '_NR_data_processed_traces.csv', xy_delim = ',')
    ms.get_spike_data(path + 'spikes_' + name + '_NR_data_processed_traces.csv')
    #   ms.get_min1pipe_spikes(path + name + '_data_processed_spikes.csv')
    ms.get_cell_centers(path + name + '_data_processed_filters\\')
    # ms.get_markers_data(path + name + '_track_fl.csv', delimiter = ' ')
    
    ms.get_angle()
    ms.get_ks_score(min_n_spikes = 3)
    ms.get_place_cells_in_circle(mode = 'spikes')
    ms.get_binned_neuro(5)  
    
    # np.savez(path + name + '_sp_20bins.npz', [ms])  

    
    # ms = pio.LoadMouse(path + name + '_sp_20bins.npz') 
    # mu_list = []
    # std_list = []
    # multimodal_fields = []

    # slen = 10
    # spec_score = np.zeros(slen)
    # count =0
    # spec_count = 0

    # if not '1D' in name:
    #     continue
    
    # ms = pio.LoadMouse(path + name + '_sp_20bins.npz')
    # nums = md.DrawPlaceFieldsMap(ms, path + name + 'full_fields.png')
    # print (nums)
    # n_single_cells = 0
    # for i, pc in enumerate(ms.pc):
    #     if len(pc.pf) > 1:
    # # #         multimodal_fields.append(pc.pf)
    # # #         # for pf in pc.pf:
    # # #         #     mu_list.append(pf.mu)
    # # #         #     std_list.append(pf.std)
    #         continue
    #     ms.pc[i].pf[0].t_spec, ms.pc[i].pf[0].n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(ms.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 4)
    # #     if not ms.pc[i].pf[0].t_spec:
    # #         continue
    # # #     mu_list.append(pc.pf[0].mu)
    # # #     std_list.append(pc.pf[0].std)
    #     n_spec.append(ms.pc[i].pf[0].n_spec)
    #     t_spec.append(ms.pc[i].pf[0].t_spec)
    # #     n_list.append(ms.pc[i].pf[0].n_spec)
    # #     t_list.append(ms.pc[i].pf[0].t_spec)
    # #     count += 1
    # #     try:
    # #         spec_score += fil_score[0:slen]
    # #         glob_spec_score.append(fil_score[0:slen])
    # #         spec_count += 1
    # #     except:
    # #         pass
    #     n_single_cells +=1
    # ms.spec_score = spec_score/spec_count
    np.savez(path + name + '_sp_20bins.npz', [ms])     
        
    # print('Place_fields:',len(mu_list))
    # print('Multiple-pf cells:', len(multimodal_fields))
    # print('Average t_spec:', t_spec*0.05/count)
    # print('Average n_spec:', n_spec/count)
    # plt.plot(ms.spec_score)

    # hf.DrawAllfields(mu_list, std_list, outfname = path + name + '_single_fields_sorted.png')
    # hf.DrawMultipleFields(multimodal_fields, outfname = path + name + '_multiple_fields.png')    

#    glob_spec_score.append(ms.spec_score)
    # print(n_single_cells)
# gsc = np.array(glob_spec_score)
# plt.plot(np.mean(gsc, axis = 0))
# pd.DataFrame(gsc).to_csv(path + '2D_mousewize_selective_score.csv')
    
        
    # for cell in range(ms.n_cells):
    #     for n_field in range(5):
    #         drcl.DrawOneCell(ms, cell, pf_n = n_field, k_word = path + name)    



# for i,name in enumerate(fnames):    
#     ms = pio.LoadMouse(path + name + '_sp.npz')
    # sp_cells = np.array([np.count_nonzero(i) for i in ms.spikes.transpose()])
    # print(ms.name + '\n' + str(ms.n_cells) + '\n' + str(len(sp_cells[sp_cells>=3])) + '\n' + str(len(ms.pc)) + '\n')    


    # ms = mcl.FieldsToList(ms)
    # # ms.get_binned_neuro(10)    
    # # drcl.DrawAllCells(ms, k_word = path + name + '_sp_circle_map')
    
    # # ms = mcl.FieldsToList(ms)
    # for pc in ms.pc:
    #     n_first = 100500
    #     t_first = 100500
    #     for pf in pc.pf:
    #         if pf.n_spec < n_first*20:
    #             n_first = pf.n_spec*20
    #         if pf.t_spec < t_first:
    #             t_first = pf.t_spec
    #     t_list.append(t_first)
    #     n_list.append(n_first)   
    
    # if hasattr(ms, 'markers'):
    #     pio.Export_Track_and_Markers(ms, path + name + '_track_from_npz.csv')
    # else:
    #     pio.ExportTrack(ms, path + name + '_track_from_npz.csv')
        
    # pio.Export_Neurodata(ms, path + name + '_neurodata_from_npz.csv')
    # pio.ExportSpikes(ms, path + name + '_spikes_from_npz.csv') 
    
    # print(name, ms.time_shift)
    
    # pio.ExportCoordAndTraces(ms, path + name + '_track_and_neur.csv')
    # pio.ExportSpikes(ms, path + name + '_spikes.csv')
    
    # print(name, ms.time_shift)
#    ms = mcl.FieldsToList(ms)
    # if hasattr(ms,'markers'):
    #     drcl.Plot_All_Coord(ms, path + name + '_coord.png')
    # drcl.DrawTSpecVsMuDistrib(ms, path + name + '_t_spec_vs_mu_distr.png')  
    # hf.DrawAllfields(ms.mu_list, ms.std_list, path + name + '_pf_distr.png')
#    ms.get_place_cells_in_circle_2d(mode = 'spikes', outpath = path)
