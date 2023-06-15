# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 05:54:09 2020

@author: VOVA
"""
import numpy as np
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import Heat_field_Vitya as hf
import PlaceFields as pcf

# names = ['CA1_22_1D', 'CA1_23_1D',
names = ['CA1_24_1D', 'CA1_25_1D']
names2 = ['NC_701_1D', 'NC_702_1D', 'FG_2_1D']
# path = 'D:\Work\\NCAMP\old NCAMP data\\'
path = 'D:\Work\PLACE_CELLS\\'
time_shift = [15.35, 25.4, 17.35, 25.3]

mice = []
std = []
n_single_pf = 0
t_list =[]
n_list = []

# for i,name in enumerate(names):
#     # ms = mcl.Mouse(name, time_shift[i], path_track = path + name + '_track.csv', path_neuro = path + name + '_cnmfe_raw_neuro.csv')
#     # ms.get_spike_data(path + name + '_cnmfe_spikes_filtered.csv')
#     # ms.get_angle()
#     # ms.get_ks_score(min_n_spikes = 3)
#     # ms.get_place_cells_in_circle(mode = 'spikes')
    
#     # np.savez(path + name + '_sp_new.npz', [ms])
    
#     mice.append(pio.LoadMouse(path + name + '_sp_new.npz'))
    
    # ms = mcl.FieldsToList(ms)
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
            
#    print(np.mean(ms.speed)*50/(max(ms.x) - min(ms.x)))
    # print('\n')

   
    
    
    # # hf.DrawAllfields(ms.mu_list, ms.std_list, outfname = path + name + '_place_fields.png')
    # drcl.DrawTSpecVsMuDistrib(ms, f_out = path + name + '_mu_vs_t_spec.png')    
    # drcl.DrawSelectivityScore(ms, path + name + '_sel_score.png')
    # mice.append(ms)
    # std.extend(ms.std_list)
    # print(ms.n_cells)

    # for pc in ms.pc:
    #     if len(pc.pf) == 1:
    #         n_single_pf +=1


# [t_list, n_list] = mcl.PoolMice(mice) 
# # print(np.mean(std))
# print('\n\n')
# print(len(t_list[t_list<=180])/len(t_list))

# n_list = np.array(n_list)

# print(len(n_list))
# print(len(n_list[n_list==1]))
# print(len(n_list[n_list==2]))
# print(len(n_list[n_list==3]))

# drcl.DrawTSpecHistogram(t_list, n_bins = 15, f_out = path+'t_spec_distrib.png')
# drcl.DrawTSpecHistogram(n_list, n_bins = 25, f_out = path+'n_spec_distrib.png')

# ms = pio.LoadMouse(path + names[0] + '_sp_new.npz')

# sp_angles = ms.angle[ms.spikes[:,376]!=0]
# mu, std = pcf.CellFields(sp_angles.tolist(), sigma=2.5,angle_len=90, realbin=40) 


# mice[0].get_binned_neuro(10)
# drcl.DrawOneCell(mice[0],376,0,path + name + '_circ_plot')
   




# ms = mcl.FieldsToList(ms)
# hf.DrawAllfields(mu_list = ms.mu_list, std_list = ms.std_list, outfname = path + name + '_sorted_fields_cnmfe_spikes.png', sort = True)
# drcl.Draw_PC_ContourMap(ms, filterpath = 'C:\\Work\\PLACE_CELLS\\PCA_vs_CNMF\\CNMFE_filters_549\\', f_out = path + name + '_pc_contours_cnmfe_neuro_uns.png', sort = False)

#
#
#ms = pio.LoadMouse(path + name + '_test.npz')
#ms = mcl.FieldsToList(ms)
#ms.get_binned_neuro(Nbins = 5)
#
##drcl.DrawAllCells(ms, k_word = 'test_circle_map')
#

for i,name in enumerate(names):    
    ms = pio.LoadMouse(path + name + '_sp_new.npz')
    for pc in ms.pc:
        popt = [pc.pf[0].mu, pc.pf[0].std, pc.pf[0].t_spec/20 + ms.time[0], 1]
        drcl.DrawFlatPlaceField(ms, pc.cell_num, popt, outpath = path)
    # sp_cells = np.array([np.count_nonzero(i) for i in ms.spikes.transpose()])
    # print(ms.name + '\n' + str(len(ms.spikes[0])) + '\n' + str(len(sp_cells[sp_cells>=3])) + '\n' + str(len(ms.true_cells)) + '\n')    


