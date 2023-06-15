# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:29:01 2021

@author: vsots
"""
import numpy as np
import os
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import Heat_field_Vitya as hf
import PlaceFields as pcf
import SpikingPasses as spps
import Match_days as md

path = 'D:\Work\\NEW_MICE\\'
#fnames = ['CA1_22_1D','CA1_23_1D','CA1_24_1D','CA1_25_1D', 'G6F_01', 'G6F_02', 'G7F_1_1D', 'G7F_2_1D', 'NC_701_1D','NC_702_1D', 'NC_722_1D', 'NC_761_1D', 'FG_1', 'FG_2_1D']
# fnames = ['FG_2_2D', 'NC_701_2D', 'NC_702_2D']
#fnames = ['CA1_22_2D', 'CA1_23_2D', 'CA1_24_2D', 'CA1_25_2D', 'G7F_1_2D', 'G7F_2_2D']
fnames = ['CA1_22_3D' , 'CA1_23_3D', 'CA1_24_3D', 'CA1_25_3D']

t_list =[]
n_list = []

for name in fnames:    
    ms = pio.LoadMouse(path + name + '_sp_20_bins.npz')
    for i, pc in enumerate(ms.pc):
        if len(pc.pf) == 1:
            ms.pc[i].pf[0].t_spec, ms.pc[i].pf[0].n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(ms.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 4)
    # # drcl.DrawSelectivityScoreMap(ms, path + name + '_sel_score_dirty.png',  max_n_visits =15)
    np.savez(path + name + '_sp_20_bins.npz', [ms]) 
    ms=md.PurifyPlaceCells(ms)
    drcl.DrawSelectivityScoreMap(ms, path + name + '_sel_score_mu_sorted.png',  max_n_visits =15, sort_mode = 'mu')
    
    # npc = 0
    # npf = 0
    # for pc in ms.pc:
    #     lnpf = 0
    #     for pf in pc.pf:
    #         if not pf.t_spec:
    #             continue
    #         else:
    #             lnpf +=1
    #     if lnpf:
    #         npc += 1
    #         npf += lnpf
    # print(name + '\tn_cells:'+str(ms.n_cells)+'\tn_pc:' + str(len(ms.pc)) + '\tvalid_pc:' + str(npc) + '\tvalid_pf:' + str(npf))
    # ms.get_binned_neuro(5) 
    # if not os.path.isdir(path + name + '_verified_cells'):
    #     os.mkdir(path + name + '_verified_cells')
    #     os.mkdir(path + name + '_excluded_cells')
    # for cell in range(ms.n_cells):
    #     if nscc.check_place_cell_SI(ms, cell, SI_sigma_thr = 0.5):
    #         drcl.DrawOneCell(ms, cell, pf_n = 0, k_word = path + name + '_verified_cells\\')
    #     else:
    #         drcl.DrawOneCell(ms, cell, pf_n = 0, k_word = path + name + '_excluded_cells\\')

#     sp_cells = np.array([np.count_nonzero(i) for i in ms.spikes.transpose()])
#     print(ms.name + '\n' + str(ms.n_cells) + '\n' + str(len(sp_cells[sp_cells>=3])) + '\n' + str(len(ms.pc)) + '\n')    


#     ms = mcl.FieldsToList(ms)
#     # ms.get_binned_neuro(10)    
#     # drcl.DrawAllCells(ms, k_word = path + name + '_sp_circle_map')
    
#     # ms = mcl.FieldsToList(ms)
#     for pc in ms.pc:
#         n_first = 100500
#         t_first = 100500
#         for pf in pc.pf:
#             if pf.n_spec < n_first:
#                 n_first = pf.n_spec
#             if pf.t_spec/20 < t_first:
#                 t_first = pf.t_spec/20
#         t_list.append(t_first)
#         n_list.append(n_first)
        
# t_list = np.array(t_list)
# n_list = np.array(n_list)

# print(len(t_list[t_list<=180])/len(t_list))
# print(len(n_list))
# print(len(n_list[n_list==1])/len(n_list))
# print(len(n_list[n_list==2])/len(n_list))
# print(len(n_list[n_list==3])/len(n_list))