# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:17:29 2020

@author: 1
"""

import numpy as np
import pandas as pd
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import scipy.ndimage as sf
from scipy.stats import pearsonr, chisquare
import scipy.interpolate as interp
from scipy import spatial
import matplotlib.pyplot as plt
import SpikingPasses as spps
import PlaceFields as pcf
import matplotlib.patches as patches
import os


path = 'D:\\Work\\NEW_MMICE\\'
path1 = 'D:\\Work\\PLACE_CELLS\\MIN1PIPE\\'

names = ['CA1_23', 'CA1_24', 'CA1_25', 'CA1_22', 'NC_701', 'NC_702', 'G7F_1', 'G7F_2', 'FG_2', 'NC_761', 'G6F_01', 'FG_1']
names2 = names[0:-3]
names3 = names[0:3]

nd = []
for name in names:
    if name in names3:
        nd.append(3)
    elif name in names2:
        nd.append(2)
    else:
        nd.append(1)
nd = dict(zip(names, nd))

p_names = ['CA1_22', 'CA1_25', 'NC_701', 'G7F_2']
r_names = ['CA1_23', 'CA1_24', 'NC_702', 'G7F_1', 'FG_2']

nik_names = ['CA1_22', 'CA1_23', 'CA1_24', 'CA1_25', 'NC_702', 'NC_701']

plt.rcParams['font.size'] = '16'

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
   
def GetSelectivityScoreMap(Mouse, f_out, max_n_visits =15, sort_mode = 'none', shifts = []):
    sel_score_map = []
    mu_list = []
    n_spec_list = []
       
    for i, pc in enumerate(Mouse.pc):
        for pf in pc.pf:
            fscore = pf.fil_score.tolist()
            while len(fscore) < max_n_visits:       #extend list with zeros 
                fscore.append(np.nan)
            sel_score_map.append(fscore[0:max_n_visits-1])
            mu_list.append(pf.mu)
            n_spec_list.append(pf.n_spec)
    
    if sort_mode == 'mu':
        ind = np.argsort(mu_list) 
    elif sort_mode == 'shift':
        ind = np.argsort(shifts)
    elif sort_mode == 'n_spec':
        ind = np.argsort(n_spec_list)
        n_spec_list = [n_spec_list[i] for i in ind] 
    else:
        ind = ((np.linspace(0, len(mu_list)-1,len(mu_list))).astype(int)).tolist()
        
    sel_score_map = [sel_score_map[i] for i in ind]

    return sel_score_map, n_spec_list


def wide_diag_sum(A, wd = 2):
    S = 0
    for w in range(-wd, wd+1):
        S += np.trace(A, offset = w)
    return S
            


def measure_matrix_diagonality(M0):
    M = M0.astype(int)
    if not np.allclose(M-M0, np.zeros((M0.shape))):
        raise Exception('Non-integer entries are not allowed')

    x, y = np.nonzero(M)
    vals = M[x,y]

    all_x = []
    all_y = []
    for i, val in enumerate(vals): # could have been done much more elegantly but who the fuck cares?
        currx = x[i]
        curry = y[i]
        all_x.extend([currx for _ in range(val)])
        all_y.extend([curry for _ in range(val)])

    r = np.round(np.corrcoef(np.array(all_x), np.array(all_y))[0,1], 4)
    return r


def cosine_dist_with_errors(x, y, xerr, yerr):
    d = spatial.distance.cosine(x,y)

    sqx = sum([v**2 for v in x])
    sqy = sum([v**2 for v in y])
    halfsqx = np.sqrt(sqx)
    halfsqy = np.sqrt(sqy)
    sumxy = sum([vx*vy for vx,vy in zip(x,y)])
    
    pdx = [(vy*sqx-vx*sumxy)/(halfsqx**3*halfsqy) for vx, vy in zip(x,y)]
    pdy = [(vx*sqy-vy*sumxy)/(halfsqx*halfsqy**3) for vx, vy in zip(x,y)]

    derr = np.sqrt(sum([(vpdx*vxerr)**2 for vpdx, vxerr in zip(pdx,xerr)]) + sum([(vpdy*vyerr)**2 for vpdy, vyerr in zip(pdy,yerr)]))
    return np.round(d,4), np.round(derr, 4)


#%% Stop here
a = []
print (a[1])
#%% Mouse reformation 
min_sel_len = 10

for d in [1,2,3]:
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_new.npz')
        except:
            continue
        n_pf = 1
        while n_pf: #refining cells until toltal number of place fields stabilizes
            n_pf = np.sum([len(pc.pf) for pc in ms.pc])
            for i, pc in enumerate(ms.pc):
                ms.pc[i] = spps.Get_Selectivity_Score_MultiPF(ms.spikes[:,pc.cell_num], pc, min_sp_len = 3)         
            ms = PurifyPlaceCells(ms)
            n_pf -= np.sum([len(pc.pf) for pc in ms.pc])
        
        sum_score = []
        for i in range(min_sel_len):
            sum_score.append([])
        for pc in ms.pc:
            for pf in pc.pf:
                for i in range(min_sel_len):
                    try:
                        sum_score[i].append(pf.fil_score[i])
                    except:
                        break 

        ms.spec_score = []
        for s_sc in sum_score:
            ms.spec_score.append(np.mean(s_sc))
           

        np.savez(path + name + '_' + str(d) + 'D_nnew.npz', [ms])

#%%

ms = pio.LoadMouse(path + 'CA1_22_2D_nnew.npz')
drcl.Draw_All_Fields_of_OneCell(ms, 478, k_word = path + 'CA1_22_2D')
drcl.Draw_All_Fields_of_OneCell(ms, 228, k_word = path + 'CA1_22_2D')
drcl.Draw_All_Fields_of_OneCell(ms, 485, k_word = path + 'CA1_22_2D')
        
#%% Draw all fields

try:
    os.mkdir(path + 'circular_plots\\')
except:
    pass

for name in names[1:-1]:
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue
        for cell in range(ms.n_cells):
            drcl.Draw_All_Fields_of_OneCell(ms, cell, k_word = path + 'circular_plots\\' + name + '_' + str(d) + 'D')

#%% Sel score in t-domain
aspect_r = 0.35
sort = 'mu'

for name in ['CA1_24']:

    fig = plt.figure(figsize = (nd[name]*10, 10), dpi = 100)
    plt.axis('off')
    plt.title(name)
        
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue 
        ms.ts_score = []
        t_spec_list = []
        mu_list = []
        ms.get_direction()
        timeline = np.linspace(0, ms.n_frames, int(ms.n_frames/200)) #every 10s
        for pc in ms.pc:
            for pf in pc.pf:
                score_fn= interp.interp1d((np.array(pf.times_in) + np.array(pf.times_in))/2, pf.fil_score)
                tsc = []
                for t in timeline:
                    try:
                        tsc.append(score_fn(t))
                    except:
                        tsc.append(np.nan)
                ms.ts_score.append(tsc)
                t_spec_list.append(pf.t_spec)
                mu_list.append(pf.mu)
                
        if not len(ms.ts_score):
            continue
        
        if sort =='t_spec':
            ind = np.argsort(t_spec_list)
            ms.ts_score = [ms.ts_score[i] for i in ind]
        elif sort =='mu':
            ind = np.argsort(mu_list)
            ms.ts_score = [ms.ts_score[i] for i in ind]   
        
                   
        # fig = plt.figure()
        # plt.axis('off')
        # plt.title(name + '_day_' + str(d))
        
        ax1 = fig.add_subplot(200 + nd[name]*10 + d, yticks=[])
        ax1.imshow(np.array(ms.ts_score)[:,0:-6], extent = (0, (ms.n_frames-1000)/20, (ms.n_frames-1000)*aspect_r/20, 0))
                
        ax2 = fig.add_subplot(200 + nd[name]*11 + d, xlim = (0, (ms.n_frames-1000)/20))
        ax2.set_box_aspect = 0.25
        ax2.plot(timeline/20, np.sum(ms.ts_score, axis = 0)*2/len(ms.ts_score), label = 'av_sel_score', linewidth = 1.5, zorder = 1)
        ax2.plot(np.linspace(0,len(ms.speed)/20,len(ms.speed)), ms.speed*ms.direction[0:len(ms.speed)]/np.max(ms.speed), label = 'speed', zorder = 0.5)
        try:
            ms = mcl.GetRears(ms, path+name+'_' + str(d) + 'D_track_stances.csv')
            ax2.plot(np.linspace(0,len(ms.speed)/20,len(ms.speed)), np.array(ms.rear)/2, label = 'rears', zorder = 0)
        except:
            continue
        ax2.legend()
    
    plt.show()
    fig.savefig(path+name+'_timed_sel_score.png')
    plt.close()
         
#%% Nikita-like plots
sscore = []
esscore = []
vscore = []
w = 25
mean_sscore = []
df = pd.read_csv(path+ 'resvar\\resvar.csv')
df2 = pd.read_csv(path+ 'resvar\\resvar_std.csv')

cols = df.columns.values
mice_names = cols[1:-2]
start_frames = df[cols[-2]].values.T
dists = df[mice_names].to_numpy().T
edists = df2[mice_names].to_numpy().T

for i, d in enumerate(dists):
    dists[i] = sf.filters.gaussian_filter1d(dists[i], sigma = 2, order=0, mode='reflect')
for name in nik_names:
    ms = pio.LoadMouse(path + name + '_1D_nnew.npz')
    sscore.append(1 - np.convolve(np.mean(ms.ts_score, axis = 0), np.ones(w)/w, mode = 'valid'))
    esscore.append(np.std(ms.ts_score, axis = 0))

fig = plt.figure(figsize = (10, 15), dpi = 100)
plt.rcParams['font.size'] = '16'
plt.axis('off')
ax = plt.subplot(211)
ax.title.set_text('1 - selectivity score')
for i, score in enumerate(sscore):
    timeline1 = np.linspace(0, len(score)*200, len(score))
    ax.plot(timeline1/20, score, label = nik_names[i] + '_1D')
    # ax.fill_between(timeline1, score - esscore[i], score + esscore[i], color='gray', alpha=0.2)
plt.xlim(0, 15000/20)
plt.ylim(0.4, 0.8)
# plt.legend()

ax = plt.subplot(212)
ax.title.set_text('Reconstruction error')
for i, score in enumerate(dists[0:6]):
    # timeline1 = np.linspace(0, len(sscore[i])*200, len(sscore[i]))
    # ax.plot(timeline1/20, sscore[i], linewidth = 3, linestyle = 'dashed')
    timeline2 = np.linspace(0, 15000, len(score))
    ax.plot(timeline2/20, score, linewidth = 3, label = nik_names[i] + '_1D')

#    ax.fill_between(timeline2/20, score - edists[i], score + edists[i], color='gray', alpha=0.2)
plt.xlim(0, 15000/20)
plt.legend(prop={"size":10})
plt.show()

cmap = np.zeros((6,6))
dists_cr = []
for i in range(6):
    sscore[i] = sscore[i][15:51]
    dists_cr.append(dists[i][4:14])
    
timeline2= np.linspace(0,10,10)    
for i in range(6):    
    slen = len(sscore[i])
    # score = np.mean(np.reshape(sscore[i][0:slen*2], (slen, 2)), axis =1)
    timeline1 = np.linspace(0, slen*200, slen)
    score_fn = interp.interp1d(timeline1, sscore[i])
    score = [score_fn(t) for t in timeline2]
    for j in range(6): 
        # cmap[i,j], _= pearsonr(score, dists_cr[j])                                                                                                                                                                                                                                                                                                                                                              
        cmap[i,j]= spatial.distance.cosine(score, dists_cr[j])   
    mean_sscore.append(np.mean(score))
    vscore.append(score)    
mean_dscore = np.mean(dists_cr[0:6], axis =1)

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
#%% Aver rec error

rvar = np.array([0.27821079, 0.76232735, 0.66930664, 0.49516429, 0.95152621, 0.75089925])
ervar = np.array([0.02768165, 0.0822305,  0.15826364, 0.05522812, 0.01332941, 0.10093822])
sscore = []
esscore = []

for name in nik_names:
    ms = pio.LoadMouse(path + name + '_1D_nnew.npz')
    sscore.append(1 - np.mean(ms.ts_score, axis = 0)[25:75])
    esscore.append(np.std(ms.ts_score, axis = 0)[25:75])

sscore = np.mean(sscore, axis = 1)
esscore = np.mean(esscore, axis = 1)
plt.figure()
plt.rcParams['font.size'] = '16'
# plt.plot(sscore, label = '1-sel score')
# plt.plot(rvar, label = 'Rec error')

plt.errorbar(np.linspace(0,6,6),
                sscore,
                yerr=esscore,
                fmt='o',
                capsize = 5,
                barsabove = 1,  label = '1-sel score')
plt.errorbar(np.linspace(0,6,6),
                rvar,
                ervar,
                fmt='o',
                capsize = 5,
                barsabove = 1,  label = 'Rec error')

# plt.fill_between(np.linspace(0,5,6),sscore - esscore, sscore + esscore, color='gray', alpha=0.2)
# plt.fill_between(np.linspace(0,5,6),rvar - ervar, rvar + ervar, color='gray', alpha=0.2)
plt.legend(prop={"size":10})
plt.show()    
    
dist, error = cosine_dist_with_errors(sscore, rvar, esscore, ervar)
print('cosine similarity is {} +- {}'.format(1- dist,error))    
 #%% Statistics   
ttlen = []
n_first = 0
av_nspec = 0
av_tspec = 0
for d in [1]:#,2,3]:
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue          
        n_single_pf = len([pc for pc in ms.pc if len(pc.pf) == 1]) 
        n_pf = np.sum([len(pc.pf) for pc in ms.pc])
        

        tlen=[]
        for pc in ms.pc:
            for pf in pc.pf:
                tlen.append(len(pf.times_in))
                av_nspec += pf.n_spec
                av_tspec += pf.t_spec/20
                if pf.t_spec <=1200:
                    n_first += 1
        ttlen.append(np.min(tlen))       
        # if n_pf:                                     
        #     print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_pf) + '\t' + str(n_single_pf) + '\t' + str(av_nspec/n_pf) + '\t' + str(av_tspec/n_pf) + '\t' + str(ms.spec_score[0]) + '\t' + str(ms.spec_score[-1]))
        # else:
        #     print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_pf) + '\t' + str(n_single_pf))
#print(np.mean(ttlen))
# print(np.mean(av_nspec))
# print(np.mean(av_tspec))
print(n_first)

#%% T_spec and N_spec distribution
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.title.set_text('N spec distribution')
ax2.title.set_text('T spec distribution')
for d in [1,2,3]:
    n_spec = []
    t_spec = []
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue
        for pc in ms.pc:
            for pf in pc.pf:
                n_spec.append(pf.n_spec)
                t_spec.append(pf.t_spec/20)                 
    ax1.hist(n_spec, label = str(d)+' day')
    ax2.hist(t_spec, label = str(d)+' day')

    print('Day ' + str(d) + '\tAv_n_spec = ' + str(np.mean(n_spec)) + '\tAv_t_spec = ' + str(np.mean(t_spec)))
plt.legend()
plt.show()

 

         
#%% Scatter maps 
fig = plt.figure()

hmap = []
d1 = []
for name in p_names:
    
    ms1 = pio.LoadMouse(path + name + '_1D_nnew.npz')
    ms2 = pio.LoadMouse(path + name + '_2D_nnew.npz')
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
    
    ms1 = pio.LoadMouse(path + name + '_1D_nnew.npz')
    ms2 = pio.LoadMouse(path + name + '_2D_nnew.npz')
    s, d, h = ShiftsVsDNspec(ms1, ms2, path + name + '_match.csv', 0, 1, mu_bins = 10)
    hmap2.append(h)
    d2.append(d)    

hmap1 = np.sum(hmap2, axis = 0)
hmap1[hmap1>0]+=2    
plt.imshow(hmap1, extent = (-25, 25, 19, 0), cmap = 'jet')

fig = plt.figure()
hmap = np.sum(np.sum(hmap, axis = 0), axis = 0)
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'persist')

hmap = np.sum(np.sum(hmap2, axis = 0), axis = 0)
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'remap')
plt.legend()

print(np.sum(d1, axis = 0))
print(np.sum(d2, axis = 0))   
 
    
#%% Selectivity score maps
   
 
for name in names:
    fig = plt.figure(figsize=(nd[name]*3,5), dpi=100)
    plt.rcParams['font.size'] = '16'
    plt.title(name+'\n')
    if nd[name] >1:
        plt.axis('off')
        
    for d in [1,2,3]:
        max_n_visits = 40# - (d-1)*5
        aspect_r = 2#*(1 +(d-1)/2)

        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue
        sel_score_map, n_spec_list = GetSelectivityScoreMap(ms, 'none',  max_n_visits = max_n_visits, sort_mode = 'n_spec')
        fact = aspect_r*max_n_visits/len(sel_score_map)
        ax = fig.add_subplot(100+nd[name]*10+d)
        if nd[name]>1:
            ax.title.set_text(f'Day {d}')
        ax.set_yticks([])
        for i, n_spec in enumerate(n_spec_list):    
            ax.add_patch(patches.Polygon([[n_spec+0.2, i*fact],[n_spec+0.2, (i+1)*fact],[n_spec+1, (i+0.5)*fact]], facecolor = 'red', edgecolor = 'none'))
        ax.imshow(sel_score_map, extent = (0, max_n_visits, max_n_visits*aspect_r, 0))
    plt.savefig(path + name + '_sel_score_40.png')  


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
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
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

for sd in [1]:#days:
    for name in names:

        fig = plt.figure(figsize=(nd[name]*3,5), dpi=100)
        plt.rcParams['font.size'] = '16'
        plt.title(name+'\n')
        if nd[name] >1:
            plt.axis('off')
     
        for d in [sd]+days[:sd-1]+days[sd:]:
    
            try:
                ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
            except:
                continue
            ax = fig.add_subplot(100+nd[name]*10+d)
            plt.setp(ax, xticks=[0,360], xticklabels=['0', '360'], yticks=[])
            if d == sd:
                nums0, hmap = DrawPlaceFieldsMap(ms, 'none', draw = False, sort = True)
            else:
                nums = TransmitNums(path + name + '_match.csv', sd, d, nums0)
                _, hmap = DrawPlaceFieldsMap(ms, 'none', draw = False, sort = False, cell_nums = nums)
            ax.imshow(hmap, cmap='jet',interpolation = 'nearest', extent = (0,360,0,720))
            if nd[name]>1:
                ax.title.set_text(f'Day {d}')                

        # plt.title(name + ' sorted at day ' + str(sd))
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


for name in names:
    fig = plt.figure(figsize = (nd[name]*5,5), dpi = 100)
    plt.rcParams['font.size'] = '16'
    plt.axis('off')
    plt.title(name + '\n')
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            break
        sizes = [ms.n_cells]
        for npf in range(1,4):
            sizes.append(len([pc for pc in ms.pc if len(pc.pf) == npf]))

        sizes[0] -= np.sum(sizes[1:3]) 
        labels = []
        for npf in range(4):
            labels.append(f'{npf} PF ({sizes[npf]*100/np.sum(sizes):.1f}%)')            
        ax = fig.add_subplot(100+nd[name]*10+d)
        if nd[name]>1:
            ax.title.set_text(f'Day {d}')
        patches,_ = ax.pie(sizes, startangle=90)
        ax.axis('equal')
        plt.legend(patches, labels, loc="lower left")

    plt.savefig(path + name + '_pf_count.png')
    plt.close()

#%% Transition histograms
bins = np.linspace(-180,180, 21)
m = 0.12
k = 1.1

for name in names2:
    fact = 3 + (nd[name]-3)*2 #[2,3] to [1,3]
    fact2 = nd[name] - 1 #[1,2]
    ms = []
    for d in [1,2,3]:
        try:
            ms.append(pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz'))
        except:
            break 
        
    fig = plt.figure(figsize = (fact*5,5*k), dpi = 100)
    fig.subplots_adjust(bottom = m, left = m*fact2/fact, top = 2-m-k, right = 1-m*fact2/fact, wspace = m*2/((1-m*2)*fact2))
    plt.rcParams['font.size'] = '22'
    plt.title(name+'\n')
    if nd[name]>2:
        plt.axis('off')

    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        try:
            TR = TransitionMatrix(ms[dd[0]], ms[dd[1]], path + name + '_match.csv', dd[0], dd[1])
            n_pf = np.sum([len(pc.pf) for pc in ms[dd[0]].pc])
            print(name, i, wide_diag_sum(TR[:-2,:-2])/n_pf)
            # diag = measure_matrix_diagonality(TR[:-2,:-2]*24)
            TR[TR>0]+=2
        except:
            break           
        ax = fig.add_subplot(101+ fact*10+i)
        if nd[name]==2:
            ax.title.set_text(f'{name}  day {dd[0]+1} vs day {dd[1]+1}')
        else:
            ax.title.set_text(f'day {dd[0]+1} vs day {dd[1]+1}')          
        plt.setp(ax, xticks=[0.5,17.5,20.5], xticklabels = ['0','360','N'], yticks=[0.5,18.5,20.5], yticklabels = ['0','360','N'])

        ax.imshow(TR, extent = (0,21,21,0), cmap = 'jet')

    #plt.savefig(path + name + '_trans_matrices.png')
    plt.close()
        
    fig = plt.figure(figsize = (fact*5,5*k), dpi = 100)
    fig.subplots_adjust(bottom = m, left = m*fact2/fact, top = 2-m-k, right = 1-m*fact2/fact, wspace = m*2/((1-m*2)*fact2))
    plt.rcParams['font.size'] = '22'
    plt.title(name+ '\n')
    if fact>1:
        plt.axis('off')

    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        try:
            SH = TransitionHistogram(ms[dd[0]], ms[dd[1]], path + name + '_match.csv', dd[0], dd[1], mode = 'second')
        except:
            break
        shist,_ = np.histogram(SH, bins=bins)
        f_exp,_ = np.histogram(np.random.normal(0, 1, 1000), bins=bins)
        rem_rate, pr = chisquare(shist)#, f_exp = f_exp) 
        #print(name + f' {dd[0]}{dd[1]} {rem_rate:.1f} {pr:.3f}')
        ax = fig.add_subplot(101+ fact*10+i)
        if nd[name]==2:
            ax.title.set_text(f'{name}  day {dd[0]+1} vs day {dd[1]+1}')
        else:
            ax.title.set_text(f'day {dd[0]+1} vs day {dd[1]+1}')
        plt.setp(ax, yticks = [0, max(shist)], xticks=[-180,0,180], xticklabels=['-180','0','180'])
        ax.hist(SH, bins = bins)

    #plt.savefig(path + name + '_trans_hist.png')
    plt.close()        
        
#%% Draw cell hist
ms = pio.LoadMouse(path + 'CA1_22_2D_nnew.npz')
cell = 228
sp_angles = ms.angle[ms.spikes[:,cell]!=0]
mu, std = pcf.CellFields(sp_angles.tolist(), sigma=1, angle_len=90, realbin=20) 
