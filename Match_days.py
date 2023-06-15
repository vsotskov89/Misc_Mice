# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:17:29 2020

@author: 1
"""

import numpy as np
import pandas as pd
import mouse_class as mcl
import pc_io as pio
import scipy.ndimage as sf
from scipy.stats import pearsonr
import scipy.interpolate as interp
from scipy import spatial
import matplotlib.pyplot as plt
import SpikingPasses as spps
import PlaceFields as pcf
import matplotlib.patches as patches


path = 'D:\\Work\\NEW_MICE\\'
path1 = 'D:\\Work\\PLACE_CELLS\\MIN1PIPE\\'

names = ['CA1_23', 'CA1_24', 'CA1_25', 'CA1_22', 'NC_701', 'NC_702', 'G7F_1', 'G7F_2', 'FG_2', 'NC_761', 'G6F_01', 'FG_1']
names2 = names[0:-3]
names3 = names[0:3]

p_names = ['CA1_22', 'CA1_25', 'NC_701', 'G7F_2']
r_names = ['CA1_23', 'CA1_24', 'NC_702', 'G7F_1', 'FG_2']

nik_names = ['CA1_22', 'CA1_23', 'CA1_24', 'CA1_25', 'NC_702', 'NC_701']

def PurifyPlaceCells(ms):

    for ipc, plc in enumerate(ms.pc):
        a = [p for p in plc.pf if p.t_spec]
        ms.pc[ipc].pf = a

    ms.pc = [p for p in ms.pc if p.pf]
    return ms
       
def Get_Place_Cells_In_Circle(Mouse, mode = 'spikes', min_n_siz = 3, n_bins = 20):
    #searching for candidate cellsin circle track (former true_cells) by activity statistics
    #mode = 'spikes' | 'raw_neuro' <==> count spike cell activity or raw neuro traces 
    Mouse.pc = []
    for cell, sp in enumerate(Mouse.spikes[0]):
        
        if mode == 'spikes':
            sp_angles = Mouse.angle[Mouse.spikes[:,cell]!=0]
            if len(sp_angles) < min_n_siz:
                continue
            mu, std = pcf.CellFields(sp_angles.tolist(), sigma=1, angle_len=90, realbin=n_bins) 
        elif mode == 'raw_neuro':
            bin_neuro = pcf.BinRawNeuro(angles = Mouse.angle, neurodata = Mouse.neur[:,cell], Nbin = n_bins)
            mu, std = pcf.CellFieldsNeuro(bin_neuro, sigma=2.5,bin_len=10, realbin=n_bins)
        
        elif mode == 'min_spikes':
            n_data = np.convolve(Mouse.min_spikes[:,cell], np.ones(21)/21, mode='same')
            n_data = n_data[0:Mouse.n_frames]
            bin_neuro = pcf.BinRawNeuro(angles = Mouse.angle, neurodata = n_data, Nbin = n_bins)
            mu, std = pcf.CellFieldsNeuro(bin_neuro, sigma=2.5,bin_len=10, realbin=n_bins) 
            
        else:
            break
        
        if not len(mu):
            continue            
        Mouse.pc.append(mcl.PlaceCell(cell))
        Mouse.pc[-1].pf = []
        for i,m in enumerate(mu):
            #each place cell can have multiple place fields              
            pf = mcl.PlaceField(mu = m, std = std[i])
            pf.times_in, pf.times_out = spps.getPasses(Mouse.angle, m, std[i] + 2*360/n_bins) 
            
            #if there are less than 50% non-spiking passes, discard this field
            siz = []
            for i, tin in enumerate(pf.times_in):
                siz.append(sum(Mouse.spikes[tin:pf.times_out[i], cell]))
            if np.count_nonzero(siz) < min_n_siz: #or np.count_nonzero(siz)/len(siz) < 0.5 
                continue
            
            #t_spec is the first time the cells fires in its zone
            #n_spec is cooresponding number of in-zone visit (when the first in_zone spike occures)
            for i, tin in enumerate(pf.times_in):
                spikes_in_zone = Mouse.spikes[tin:pf.times_out[i], cell]
                if np.count_nonzero(spikes_in_zone):
                     rel_t = np.nonzero(spikes_in_zone)
                     pf.t_spec = tin + rel_t[0][0]
                     pf.n_spec = i + 1
                     break
                 
            #JUST IN CASE
            if not hasattr(pf, 't_spec'):
                pf.t_spec = Mouse.n_frames
                pf.n_spec = len(pf.times_in)
            pf.is_super = False
            Mouse.pc[-1].pf.append(pf)
            
        if not len(Mouse.pc[-1].pf):
            del Mouse.pc[-1]
    return Mouse

def DrawPlaceFieldsMap(ms, outfname, draw = True, sort = True, cell_nums = []):
    
    
    sp_dens_map = []
    peak_mu = []
    
    if sort:
        cell_nums = []
        for pc in ms.pc:
            cell_nums.append(pc.cell_num)
        cell_nums = np.array(cell_nums) 
        
    for c_num in cell_nums: 
        
        if c_num >= 0:
            sp_angles = ms.angle[ms.spikes[:,c_num]!=0]
        else:
            sp_angles = []
            
        sp_dens = np.histogram(sp_angles, bins = 40, range=(0,360))[0].astype(float)
        sp_dens = sf.filters.gaussian_filter1d(sp_dens, sigma = 2, order=0, mode='reflect')
        peak_mu.append(np.argwhere(sp_dens == max(sp_dens))[0][0])
        if not len(sp_dens) or np.isnan(sum(sp_dens)) or not sum(sp_dens):
            sp_dens = np.zeros(40)
        else:
            sp_dens = sp_dens/max(sp_dens)
            
        sp_dens_map.append(sp_dens)
        
    sp_dens_map = np.array(sp_dens_map)
    peak_mu = np.array(peak_mu)

    
    if sort:  #Sorting!!!
        order = np.argsort(peak_mu)
        cell_nums = cell_nums[order]
        sp_dens_map = sp_dens_map[order,:]
        
    if draw:
         fig = plt.figure(figsize=(10,10), dpi=300)
         ax = fig.add_subplot(111) 
         ax.imshow(sp_dens_map, cmap='jet',interpolation = 'nearest', extent = (0,360,0,720))
         plt.savefig(outfname) 
         plt.close()
    
    return cell_nums, sp_dens_map  
    
def TransmitNums(filename, day_from, day_to, nums_in):
    match = np.genfromtxt(filename, delimiter=',')
    true_ind = np.nonzero(match[:, day_from-1])[0]
    D = dict(zip(match[true_ind, day_from-1], match[true_ind, day_to-1]))
    nums_out = []
    for n in nums_in:
        if n + 1 in D:
            nums_out.append(D[n+1] - 1) #!!!Match is done in MATLAB
        else:
            nums_out.append(-1)         #-1 is when there is no match
    return np.array(nums_out, dtype = 'int')

def TransitionMatrix(ms1, ms2, match_fname, d1, d2, n_bins=20):
    TR = np.zeros((n_bins+1, n_bins+1))
    match = np.genfromtxt(match_fname, delimiter=',')
    for row in range(np.size(match,0)):
        pf1 = FindPlaceField(ms1, match[row,d1] - 1)        
        pf2 = FindPlaceField(ms2, match[row,d2] - 1) 
        if pf1[0].mu >= 361 and pf2[0].mu >= 361:
            continue
        for p1 in pf1:
            for p2 in pf2:
                TR[int(p1.mu*n_bins/360), int(p2.mu*n_bins/360)] += 1/(len(pf1)*len(pf2))
    return TR

def CellFateTransitionMatrix(ms1, ms2, match_fname, d1, d2):
    max_n_fields = 4
    CF = np.zeros((max_n_fields+2, max_n_fields+2))
    match = np.genfromtxt(match_fname, delimiter=',')
    for row in range(np.size(match,0)):
        pf1 = FindPlaceField(ms1, match[row,d1] - 1)        
        pf2 = FindPlaceField(ms2, match[row,d2] - 1) 

        n1 = len(pf1)
        n2 = len(pf2)
        if pf1[0].mu == 361:
            n1 = 0
        elif pf1[0].mu == 362:
            n1 = max_n_fields + 1
        if pf2[0].mu == 361:
            n2 = 0
        elif pf2[0].mu == 362:
            n2 = max_n_fields + 1
                
        CF[n1, n2] += 1
    return CF    

def FindPlaceField(ms, cell_num):
    
    if cell_num == -1:                        
        pf = mcl.PlaceField(mu = 362, std = 0)  #NOT A CELL
        return [pf]
    
    for pc in ms.pc:
        if pc.cell_num == cell_num:
            return pc.pf

    pf = mcl.PlaceField(mu = 361, std = 0)      #CELL W/O FIELDS
    return [pf]

def TransitionHistogram(ms1, ms2, match_fname, d1, d2, mode = 'both'):
    shifts = []
    match = np.genfromtxt(match_fname, delimiter=',')
    for row in range(np.size(match,0)):    
        pf1 = FindPlaceField(ms1, match[row,d1] - 1)        
        pf2 = FindPlaceField(ms2, match[row,d2] - 1)  
        
        if mode == 'both':
            if len(pf1) > 1 or len(pf2) > 1 or pf1[0].mu > 360 or pf2[0].mu > 360:
                continue
            
        elif mode == 'first':
            if len(pf1) > 1 or pf1[0].mu > 360:
                continue 
            if len(pf2) > 1 or pf2[0].mu > 360: 
                pf2[0].mu = pf1[0].mu - 541
                
        elif mode == 'second':
            if len(pf2) > 1 or pf2[0].mu > 360:
                continue
            if len(pf1) > 1 or pf1[0].mu > 360: 
                pf1[0].mu = pf2[0].mu + 541
                
        delta = pf1[0].mu - pf2[0].mu
        if delta >= 180:
            delta -= 360
        elif delta<= -180:
            delta += 360
            
        shifts.append(delta)
    return shifts

def ShiftsVsDNspec(ms1, ms2, match_fname, d1, d2, mu_bins = 10, nsp_bins = 50):
    shifts = []
    dnspec = []
    hmap = np.zeros((mu_bins+1, nsp_bins+1))
    match = np.genfromtxt(match_fname, delimiter=',')

    for row in range(np.size(match,0)):    
        pf1 = FindPlaceField(ms1, match[row,d1] - 1)        
        pf2 = FindPlaceField(ms2, match[row,d2] - 1)  

        if len(pf2) > 1 or pf2[0].mu > 360:
            continue
        if len(pf1) > 1 or pf1[0].mu > 360: 
            # pf1[0].mu = pf2[0].mu + 541
            # ns1 = 0
            continue
        else:
            ts1, ns1, ssc, fsc = spps.Get_Selectivity_Score(ms1.spikes[:,int(match[row,d1] - 1)], pf1[0].times_in, pf1[0].times_out, min_sp_len = 3)
        ts2, ns2, ssc, fsc = spps.Get_Selectivity_Score(ms2.spikes[:,int(match[row,d2] - 1)], pf2[0].times_in, pf2[0].times_out, min_sp_len = 3)        
        
        d_mu = pf2[0].mu - pf1[0].mu
        if d_mu >= 180:
            d_mu -= 360
        elif d_mu<= -180:
            d_mu += 360
            
        dnspec.append(ns2-ns1)    
        shifts.append(np.fabs(d_mu))
        hmap[int(shifts[-1]*mu_bins/180), int((dnspec[-1] + 25))] += 1
        
    # plt.scatter(dnspec,shifts, label = ms1.name)

    # plt.imshow(hmap, extent = (-25, 10, 0, 10))
    # plt.title(ms1.name + '_days_' + str(d1+1) + str(d2+1))    
    return shifts, dnspec, hmap   
   
def DrawSelectivityScoreMap(Mouse, f_out, max_n_visits =15, aspect_r = 3, mode = 'n_spec', sort_mode = 'none', shifts = []):
    sel_score_map = []
    mu_list = []
    n_spec_list = []
    inv_rear_list = []
    out_rear_list = []
    turn_list = []
   
    fig = plt.figure(figsize=(5,15), dpi=100)
    ax = fig.add_subplot(111) 
    
    
    for i, pc in enumerate(Mouse.pc):
        if len(pc.pf) > 1:
            continue
        t_spec, n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(Mouse.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 3)
        if not t_spec:
            n_spec = max_n_visits
        if len(fil_score) < max_n_visits:       #extend list with zeros
            if not isinstance(fil_score, list):
                fil_score = fil_score.tolist()
            fil_score.extend(np.zeros(max_n_visits - len(fil_score)).tolist())

        if mode == 'rears_turns':
            inv_rears = []
            out_rears = []
            turns = []
            for i, (tin, tout) in enumerate(zip(pc.pf[0].times_in, pc.pf[0].times_out)):
                inv_rears.append((np.count_nonzero(Mouse.rear[tin:tout]) - np.sum(Mouse.rear[tin:tout]))/2)
                out_rears.append((np.count_nonzero(Mouse.rear[tin:tout]) + np.sum(Mouse.rear[tin:tout]))/2)
                turns.append(np.fabs(Mouse.direction[tout] - Mouse.direction[tin]))
            inv_rear_list.append(inv_rears)
            out_rear_list.append(out_rears)
            turn_list.append(turns)            
                
        sel_score_map.append(fil_score[0:max_n_visits-1])
        mu_list.append(pc.pf[0].mu)
        n_spec_list.append(n_spec)
    
    if sort_mode == 'mu':
        ind = np.argsort(mu_list) 
    elif sort_mode == 'shift':
        ind = np.argsort(shifts)
    elif sort_mode == 'n_spec':
        ind = np.argsort(n_spec_list)
    else:
        ind = ((np.linspace(0, len(mu_list)-1,len(mu_list))).astype(int)).tolist()
        
    sel_score_map = [sel_score_map[i] for i in ind]
    fact = aspect_r*max_n_visits/len(sel_score_map)


    
    if mode == 'n_spec':
        n_spec_list = [n_spec_list[i] for i in ind]    
        for i, n_spec in enumerate(n_spec_list):    
            ax.add_patch(patches.Polygon([[n_spec+0.2, i*fact],[n_spec+0.2, (i+1)*fact],[n_spec+1, (i+0.5)*fact]], facecolor = 'red', edgecolor = 'none'))
    elif mode == 'rears_turns':
        inv_rear_list = [inv_rear_list[i] for i in ind]
        out_rear_list = [out_rear_list[i] for i in ind]
        turn_list = [turn_list[i] for i in ind]
        for i in range(len(sel_score_map)):       
            for j, (i_r, o_r, tn) in enumerate(zip(inv_rear_list[i], out_rear_list[i], turn_list[i])):
                if i_r:
                    ax.add_patch(patches.Rectangle((j, i*fact), 0.33, fact, facecolor = [1, 0.6, 0], edgecolor = 'none'))
                if o_r:
                    ax.add_patch(patches.Rectangle((j + 0.66, i*fact), 0.33, fact, facecolor = [1, 0, 1], edgecolor = 'none'))
                if tn:
                    ax.add_patch(patches.Rectangle((j + 0.33, i*fact), 0.33, fact, facecolor = [1, 1, 1], edgecolor = 'none'))                    
    plt.imshow(sel_score_map, extent = (0, max_n_visits, max_n_visits*aspect_r, 0))
    plt.savefig(f_out)  
    return sel_score_map


def DrawSelectivityScoreBehavMap(Mouse, f_out, max_n_visits =15, aspect_r = 3, mode = 'n_spec', sort_mode = 'none', shifts = []):
    sel_score_map = []
    mu_list = []
    n_spec_list = []
    inv_rear_list = []
    out_rear_list = []
    turn_list = []
   
    fig = plt.figure(figsize=(10,5), dpi=300)
    ax = fig.add_subplot(111) 
    
    
    for i, pc in enumerate(Mouse.pc):
        if len(pc.pf) > 1:
            continue
        t_spec, n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(Mouse.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 3)
        if len(fil_score) < max_n_visits:       #extend list with zeros
            if not isinstance(fil_score, list):
                fil_score = fil_score.tolist()
            fil_score.extend(np.zeros(max_n_visits - len(fil_score)).tolist())

        if mode == 'rears_turns':
            inv_rears = []
            out_rears = []
            turns = []
            for i, (tin, tout) in enumerate(zip(pc.pf[0].times_in, pc.pf[0].times_out)):
                inv_rears.append((np.count_nonzero(Mouse.rear[tin:tout]) - np.sum(Mouse.rear[tin:tout]))/2)
                out_rears.append((np.count_nonzero(Mouse.rear[tin:tout]) + np.sum(Mouse.rear[tin:tout]))/2)
                turns.append(np.fabs(Mouse.direction[tout] - Mouse.direction[tin]))
            inv_rear_list.append(inv_rears)
            out_rear_list.append(out_rears)
            turn_list.append(turns)            
                
        sel_score_map.append(fil_score[0:max_n_visits-1])
        mu_list.append(pc.pf[0].mu)
        n_spec_list.append(n_spec)
    
    if sort_mode == 'mu':
        ind = np.argsort(mu_list) 
    elif sort_mode == 'shift':
        ind = np.argsort(shifts)
    elif sort_mode == 'n_spec':
        ind = np.argsort(n_spec_list)
    else:
        ind = ((np.linspace(0, len(mu_list)-1,len(mu_list))).astype(int)).tolist()
        
    sel_score_map = [sel_score_map[i] for i in ind]
    fact = aspect_r*max_n_visits/len(sel_score_map)


    
    if mode == 'n_spec':
        n_spec_list = [n_spec_list[i] for i in ind]    
        for i, n_spec in enumerate(n_spec_list):    
            ax.add_patch(patches.Polygon([[n_spec+0.2, i*fact],[n_spec+0.2, (i+1)*fact],[n_spec+1, (i+0.5)*fact]], facecolor = 'red', edgecolor = 'none'))
    elif mode == 'rears_turns':
        inv_rear_list = [inv_rear_list[i] for i in ind]
        out_rear_list = [out_rear_list[i] for i in ind]
        turn_list = [turn_list[i] for i in ind]
        for i in range(len(sel_score_map)):       
            for j, (i_r, o_r, tn) in enumerate(zip(inv_rear_list[i], out_rear_list[i], turn_list[i])):
                if i_r:
                    ax.add_patch(patches.Rectangle((j, i*fact), 0.33, fact, facecolor = [1, 0.6, 0], edgecolor = 'none'))
                if o_r:
                    ax.add_patch(patches.Rectangle((j + 0.66, i*fact), 0.33, fact, facecolor = [1, 0, 1], edgecolor = 'none'))
                if tn:
                    ax.add_patch(patches.Rectangle((j + 0.33, i*fact), 0.33, fact, facecolor = [1, 1, 1], edgecolor = 'none'))                    
    plt.imshow(sel_score_map, extent = (0, max_n_visits, max_n_visits*aspect_r, 0))
    plt.savefig(f_out)  
    return sel_score_map
    pass


#%% Mouse reformation 
min_sel_len = 10

nik_score = []
for d in [1,2,3]:
    for name in names:
        if name == 'CA1_22' and d==3:
            continue
        try:
            ms = pio.LoadMouse(path1 + name + '_' + str(d) + 'D_sp_20_bins.npz')
        except:
            continue
        
        ms = Get_Place_Cells_In_Circle(ms, mode = 'spikes', min_n_siz = 3, n_bins = 20)
        sum_score = []
        for i in range(min_sel_len):
            sum_score.append([])
        av_nspec = 0
        n_single_pf = len([pc for pc in ms.pc if len(pc.pf) == 1])
        for i, pc in enumerate(ms.pc):
            if len(pc.pf)>1:
                continue
            t_spec, n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(ms.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 3)  
            if t_spec:
                av_nspec += n_spec
            ms.pc[i].pf[0].n_spec = n_spec
            ms.pc[i].pf[0].t_spec = t_spec
            for i in range(min_sel_len):
                try:
                    sum_score[i].append(fil_score[i])
                except:
                    break 
                
        ms.av_nspec = av_nspec/n_single_pf
        ms.spec_score = []
        for s_sc in sum_score:
            ms.spec_score.append(np.mean(s_sc))
            
        ms = PurifyPlaceCells(ms)
        np.savez(path + name + '_' + str(d) + 'D_new.npz', [ms])
      
#%% Sel score in t-domain
aspect_r = 0.35
sort = 'mu'

for name in names:
    
    if name in names3:
        nd = 3
    elif name in names2:
        nd = 2
    else:
        nd = 1
    
    fig = plt.figure(figsize = (nd*10, 10), dpi = 100)
    plt.axis('off')
    plt.title(name)

        
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
        except:
            continue 
        ms.ts_score = []
        t_spec_list = []
        mu_list = []
        ms.get_direction()
        timeline = np.linspace(0, ms.n_frames, int(ms.n_frames/200)) #every 10s
        for i, pc in enumerate(ms.pc):
            if len(pc.pf)>1:
                continue
            t_spec, n_spec, sel_score, fil_score = spps.Get_Selectivity_Score(ms.spikes[:,pc.cell_num], pc.pf[0].times_in, pc.pf[0].times_out, min_sp_len = 3)
            score_fn= interp.interp1d((np.array(pc.pf[0].times_in) + np.array(pc.pf[0].times_in))/2, fil_score)
            tsc = []
            for t in timeline:
                try:
                    tsc.append(score_fn(t))
                except:
                    tsc.append(np.nan)
            ms.ts_score.append(tsc)
            t_spec_list.append(t_spec)
            mu_list.append(pc.pf[0].mu)
            
        if not len(ms.ts_score):
            continue
        
        if sort =='t_spec':
            ind = np.argsort(t_spec_list)
            ms.ts_score = [ms.ts_score[i] for i in ind]
        elif sort =='mu':
            ind = np.argsort(mu_list)
            ms.ts_score = [ms.ts_score[i] for i in ind]   
        
        np.savez(path + name + '_' + str(d) + 'D_new.npz', [ms])            
        # fig = plt.figure()
        # plt.axis('off')
        # plt.title(name + '_day_' + str(d))
        
        ax1 = fig.add_subplot(200 + nd*10 + d, yticks=[])
        ax1.imshow(ms.ts_score, extent = (0, ms.n_frames, ms.n_frames*aspect_r, 0))
                
        ax2 = fig.add_subplot(200 + nd*11 + d, xlim = (0, ms.n_frames))
        ax2.set_box_aspect = 0.25
        ax2.plot(timeline, np.sum(ms.ts_score, axis = 0)*2/len(ms.ts_score), label = 'av_sel_score', linewidth = 1.5, zorder = 1)
        ax2.plot(ms.speed*ms.direction[0:len(ms.speed)]/np.max(ms.speed), label = 'speed', zorder = 0)
        ax2.legend()
    
    plt.show()
    fig.savefig(path+name+'_timed_sel_score.png')
    plt.close()
         
#%% Nikita-like plots
sscore = []
esscore = []
vscore = []
mean_sscore = []
df = pd.read_csv(path+ 'resvar\\resvar.csv')
df2 = pd.read_csv(path+ 'resvar\\resvar_std.csv')

cols = df.columns.values
mice_names = cols[1:-2]
start_frames = df[cols[-2]].values.T
dists = df[mice_names].to_numpy().T
edists = df2[mice_names].to_numpy().T

for name in nik_names:
    ms = pio.LoadMouse(path + name + '_1D_new.npz')
    sscore.append(1 - np.mean(ms.ts_score, axis = 0))
    esscore.append(np.std(ms.ts_score, axis = 0))

fig = plt.figure(figsize = (10, 10), dpi = 100)
plt.axis('off')
ax = plt.subplot(211)
ax.title.set_text('1 - selectivity score')
for i, score in enumerate(sscore):
    timeline1 = np.linspace(0, len(score)*200, len(score))
    ax.plot(timeline1, score, label = nik_names[i] + '_1D')
    ax.fill_between(timeline1, score - esscore[i], score + esscore[i], color='gray', alpha=0.2)
plt.xlim(5000, 15000)
plt.ylim(0.4, 0.8)
# plt.legend()

ax = plt.subplot(212)
ax.title.set_text('Reconstruction error')
for i, score in enumerate(dists[0:6]):
    timeline2 = np.linspace(5000, 15000, len(score))
    ax.plot(timeline2, score, label = nik_names[i] + '_1D')
    ax.fill_between(timeline2, score - edists[i], score + edists[i], color='gray', alpha=0.2)
plt.xlim(5000, 15000)
plt.legend()
plt.show()

cmap = np.zeros((6,6))
for i in range(6):
    slen = int(len(sscore[i])/2)
    score = np.mean(np.reshape(sscore[i][0:slen*2], (slen, 2)), axis =1)
    timeline1 = np.linspace(0, slen*400, len(score))
    score_fn = interp.interp1d(timeline1, score)
    score = [score_fn(t) for t in timeline2]
    for j in range(6):                                                                                                                                                                                                                                                                                                                                                               
        cmap[i,j], _ = pearsonr(score, dists[j])   
    mean_sscore.append(np.mean(score))
    vscore.append(score)    
mean_dscore = np.mean(dists[0:6], axis =1)

cossim = [1 - spatial.distance.cosine(s,d) for s, d in zip(np.array(vscore).T, dists[0:6].T) ]

plt.figure()
plt.plot(timeline2, cossim)
        
plt.figure()
plt.ylabel('1-sel_score')
plt.xlabel('Rec error')
plt.imshow(cmap)

plt.figure()
plt.plot(mean_sscore, label = '1-sel score')
plt.plot(mean_dscore[0:6], label = 'Rec error')
plt.legend()
plt.show()

print('Cosine similarity:\t' + str(1 - spatial.distance.cosine(mean_sscore, mean_dscore[0:6])))
    
  #%% Statistics   

for d in [1,2,3]:
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
        except:
            continue          
        n_single_pf = len([pc for pc in ms.pc if len(pc.pf) == 1])  
        av_nspec = 0
        av_tspec = 0
        for i, pc in enumerate(ms.pc):
            if len(pc.pf)>1:
                continue
            av_nspec += ms.pc[i].pf[0].n_spec
            av_tspec += ms.pc[i].pf[0].t_spec/20
        if n_single_pf:                                     
            print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_single_pf) + '\t' + str(av_nspec/n_single_pf) + '\t' + str(av_tspec/n_single_pf) + '\t' + str(ms.spec_score[0]) + '\t' + str(ms.spec_score[-1]))
        else:
            print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_single_pf))
  
        
  
#%% T_spec and N_spec distribution
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for d in [1,2,3]:
    n_spec = []
    t_spec = []
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
        except:
            continue
        for pc in ms.pc:
            if len(pc.pf)>1:
                continue
            n_spec.append(pc.pf[0].n_spec)
            t_spec.append(pc.pf[0].t_spec/20)                 
    ax1.hist(n_spec)
    ax2.hist(t_spec)
    
    
    print('Day ' + str(d) + '\tAv_n_spec = ' + str(np.mean(n_spec)) + '\tAv_t_spec = ' + str(np.mean(t_spec)))

plt.show()

 

         
#%% Scatter maps 
fig = plt.figure()

hmap = []
d1 = []
for name in p_names:
    
    ms1 = pio.LoadMouse(path + name + '_1D_new.npz')
    ms2 = pio.LoadMouse(path + name + '_2D_new.npz')
    s, d, h = ShiftsVsDNspec(ms1, ms2, path + name + '_match.csv', 0, 1, mu_bins = 10)
    hmap.append(h)
    d1.append(d)

hmap1 = np.sum(hmap, axis = 0)
hmap1[hmap1>0]+=2    
plt.imshow(hmap1, extent = (-25, 25, 19, 0), cmap = 'jet')

fig = plt.figure()
hmap2 = []
d2 = []
for name in r_names:
    
    ms1 = pio.LoadMouse(path + name + '_1D_new.npz')
    ms2 = pio.LoadMouse(path + name + '_2D_new.npz')
    s, d, h = ShiftsVsDNspec(ms1, ms2, path + name + '_match.csv', 0, 1, mu_bins = 10)
    hmap2.append(h)
    d2.append(d)    

hmap1 = np.sum(hmap2, axis = 0)
hmap1[hmap1>0]+=2    
plt.imshow(hmap1, extent = (-25, 25, 19, 0), cmap = 'jet')

fig = plt.figure()
hmap = np.sum(np.sum(hmap, axis = 0), axis = 0)
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'remap')

hmap = np.sum(np.sum(hmap2, axis = 0), axis = 0)
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'remap')
plt.legend()

print(np.sum(d1, axis = 0))
print(np.sum(d2, axis = 0))   
 
    
#%% Selectivity score maps
    
for name in names:
    for d in [1,2,3]:

        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
            DrawSelectivityScoreMap(ms, path + name + '_' + str(d) + 'D_sel_score_non_normed.png',  max_n_visits =20, mode = 'n_spec', sort_mode = 'n_spec')
        except:
            continue
        


#%% MEAN Selectivity score calculation

fig = plt.figure(figsize=(15,5), dpi=100)
for d in [1,2,3]:
    ax = fig.add_subplot(130+d)
    ax.set_ylim(ymin = 0.0, ymax = 0.6)
    p_sscore = []
    # r_sscore = []
    for name in names:
        if name == 'CA1_22' and d==3:
            continue
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
            p_sscore.append(ms.spec_score)  
        except:
            continue
        # s_score = DrawSelectivityScoreMap(ms, path + name  + '_' + str(d) + 'D_sel_score_n_spec_sorted.png',  max_n_visits =12, mode = 'n_spec', sort_mode = 'n_spec')    
        # s_score = np.mean(s_score, axis = 0)
      
        if d > 1 and  name in p_names:
            color = [0, .7, .7]
        elif d > 1 and name in r_names:
            color = [1, 0.6, 0]
        else:
            color = [.6, .6, .6]
        ax.plot(ms.spec_score, color = color, linewidth = 1)
    ax.plot(np.mean(p_sscore, axis = 0), color = [0, 0, 1], linewidth = 2.5, label = 'persist')     
    # ax.plot(np.mean(r_sscore, axis = 0), color = [1, 0.6, 0], linewidth = 2, label = 'remap')     
    
# ax.legend()    
plt.show()
plt.savefig(path + 'Mean_sel_score.png') 

#%% PLACE FIELD MAPS
days = [1,2,3]

for sd in days:
    for name in names:
        fig = plt.figure()        
        for d in [sd]+days[:sd-1]+days[sd:]:
    
            try:
                ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
                ax = fig.add_subplot(130+d)
                plt.setp(ax, xticks=[0,360], xticklabels=['0', '360'], yticks=[])
                if d == sd:
                    nums0, hmap = DrawPlaceFieldsMap(ms, 'none', draw = False, sort = True)
                else:
                    nums = TransmitNums(path + name + '_match.csv', sd, d, nums0)
                    _, hmap = DrawPlaceFieldsMap(ms, 'none', draw = False, sort = False, cell_nums = nums)
                ax.imshow(hmap, cmap='jet',interpolation = 'nearest', extent = (0,360,0,720))
     
            except:
                continue  
        plt.title(name + ' sorted at day ' + str(sd))
        plt.savefig(path + name + '_fields_sorted' + str(sd) + '.png')
        plt.close()
        
        
#%% Cell fate matrices

for name in names2:
    ms = []
    for d in [1,2]:
        ms.append(pio.LoadMouse(path + name + '_' + str(d) + '_new.npz'))
    plt.figure()
    TR = CellFateTransitionMatrix(ms[0], ms[1], path + name + '_match.csv', 0, 1) 
    plt.imshow(TR, cmap = 'jet')
    for (j,i),label in np.ndenumerate(TR):
        plt.text(i,j,int(label),ha='center',va='center', size = 'xx-large')    
    plt.savefig(path + name + '_pf_count_12.png')

for name in names3:
    ms = []
    for d in [1,2,3]:
        ms.append(pio.LoadMouse(path + name + '_' + str(d) + '_new.npz'))

    plt.figure()
    TR = CellFateTransitionMatrix(ms[0], ms[2], path + name + '_match.csv', 0, 2) 
    plt.imshow(TR, cmap = 'jet')
    for (j,i),label in np.ndenumerate(TR):
        plt.text(i,j,int(label),ha='center',va='center', size = 'xx-large')
    plt.savefig(path + name + '_pf_count_13.png')    
    
    plt.figure() 
    TR = CellFateTransitionMatrix(ms[1], ms[2], path + name + '_match.csv', 1, 2)
    plt.imshow(TR, cmap = 'jet')
    for (j,i),label in np.ndenumerate(TR):
        plt.text(i,j,int(label),ha='center',va='center', size = 'xx-large')
    plt.savefig(path + name + '_pf_count_23.png') 

#%% PF pie plots
labels = '0 PF', '1 PF', '2 PF', '3 PF'

for name in names:
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
            sizes = [ms.n_cells]
            for npf in range(1,4):
                sizes.append(len([pc for pc in ms.pc if len(pc.pf) == npf]))
            sizes[0] -= np.sum(sizes[1:3]) 
            plt.figure()
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.savefig(path + name + '_' + str(d) + 'D_pf_count.png')
            plt.close()
        except:
            break

#%% Transition histograms
bins = np.linspace(-180,180, 21)

for name in names2:
    ms = []
    for d in [1,2,3]:
        try:
            ms.append(pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz'))
        except:
            continue 
        
    fig = plt.figure()
    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        try:
            TR = TransitionMatrix(ms[dd[0]], ms[dd[1]], path + name + '_match.csv', dd[0], dd[1])
            TR[TR>0]+=2
           
            ax = fig.add_subplot(131+i)
            plt.setp(ax, xticks=[], yticks=[])
            ax.imshow(TR, cmap = 'jet')
        except:
            break
    plt.savefig(path + name + '_trans_matrices.png')
    plt.close()
        
    fig = plt.figure()
    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        try:
            SH = TransitionHistogram(ms[dd[0]], ms[dd[1]], path + name + '_match.csv', dd[0], dd[1], mode = 'second')
            ax = fig.add_subplot(131+i)
            plt.setp(ax, xticks=[-180,0,180], xticklabels=['-180','0','180'])
            ax.hist(SH, bins = bins)
        except:
            break
    plt.savefig(path + name + '_trans_hist.png')
    plt.close()        
        

