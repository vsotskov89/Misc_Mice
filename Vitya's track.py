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
from scipy.stats import pearsonr, chisquare, kstest
import scipy.interpolate as interp
from scipy import spatial
import matplotlib.pyplot as plt
import SpikingPasses as spps
import PlaceFields as pcf
import matplotlib.patches as patches
import os
import cmath
from glob import glob
from math import dist

path = 'D:\\Work\\Fields_STFP\\'
path_px = 'D:\\Work\\Fields_STFP\\pics\\'

track_names = glob(path + 'STFP*track.csv')
spike_names = glob(path + 'STFP*spikes.csv')
field_names = glob(path + 'STFP*fields.csv')

nums = [1,3,4,5,6,7,9]
names = ['STFP_' + str(n) for n in nums]
days = ['_D' + str(d) for d in range(1,8)]

class Mesh:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        
def PurifyPlaceCells(ms):

    for ipc, plc in enumerate(ms.pc):
        a = [p for p in plc.pf if p.t_spec]
        ms.pc[ipc].pf = a

    ms.pc = [p for p in ms.pc if p.pf]
    return ms


def Get_STFP_PC_from_Vitya(ms, filename):
    ms.pc = []
    table = np.genfromtxt(filename, delimiter = ',', skip_header =1)
    if len(np.size(table)) == 1:
        table = [table]
    for i in range(ms.n_cells):
        fields = [tab for tab in table if tab[0] == i+1]  
        if len(fields):
            ms.pc.append(mcl.PlaceCell(i))
            ms.pc[-1].pf = []
            for fl in fields:
                ms.pc[-1].pf.append(mcl.PlaceField(mu = 0, std = 1))
                ms.pc[-1].pf[-1].group = fl[2]
                ms.pc[-1].pf[-1].x = fl[3]
                ms.pc[-1].pf[-1].y = fl[4]
                ms.pc[-1].pf[-1].a = fl[5]
                ms.pc[-1].pf[-1].b = fl[6]
                ms.pc[-1].pf[-1].ang = fl[7]
                times = [int(tm) - 1 for tm in fl[8:] if not np.isnan(tm)]
                ms.pc[-1].pf[-1].times_in = [t for t in times[::2] if t]
                ms.pc[-1].pf[-1].times_out = [t for t in times[1::2] if t < ms.n_frames-1]
    return ms
          
       
def Get_Place_Cells_In_Circle(Mouse, mode = 'spikes', min_n_siz = 3, n_bins = 20):
    #searching for candidate cells in circle track (former true_cells) by activity statistics
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

def DrawPlaceFieldsMap(ms, outfname, draw = True, sort = True, nbins = 40, cell_nums = []):
    
    
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
            
        sp_dens = np.histogram(sp_angles, bins = nbins, range=(0,360))[0].astype(float)
        sp_dens = sf.filters.gaussian_filter1d(sp_dens, sigma = 2, order=0, mode='reflect')
        peak_mu.append(np.argwhere(sp_dens == max(sp_dens))[0][0])
        if not len(sp_dens) or np.isnan(sum(sp_dens)) or not sum(sp_dens):
            sp_dens = np.zeros(nbins)
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

def FindPlaceFieldsSTFP(ms, cell_num):
    for pc in ms.pc:
        if pc.cell_num == cell_num:
            return pc.pf
    return []

def TransitionHistogram_HT(ms1, ms2, match_fname, d1, d2, mode = 'both'):
    shifts = []
    match = np.genfromtxt(match_fname, delimiter=',')
    for row in range(np.size(match,0)):    
        pf1 = FindPlaceField(ms1, match[row,d1] - 1)        
        pf2 = FindPlaceField(ms2, match[row,d2] - 1)  
        dist = 0
        
        if mode == 'both':
            if len(pf1) > 1 or len(pf2) > 1 or pf1[0].mu > 360 or pf2[0].mu > 360:
                continue
            
        elif mode == 'first':
            if len(pf1) > 1 or pf1[0].mu > 360:
                continue 
            if len(pf2) > 1 or pf2[0].mu > 360: 
                dist = 1000
                
        elif mode == 'second':
            if len(pf2) > 1 or pf2[0].mu > 360:
                continue
            if len(pf1) > 1 or pf1[0].mu > 360:  
                dist = 1000
        if not dist:        
            dist = dist([pf1[0].x, pf1[0].y], [pf2[0].x, pf2[0].y])
        shifts.append(dist)
        
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
    group_list = []
       
    for i, pc in enumerate(Mouse.pc):
        for pf in pc.pf:
            fscore = pf.fil_score.tolist()
            while len(fscore) < max_n_visits:       #extend list with zeros 
                fscore.append(np.nan)
            sel_score_map.append(fscore[0:max_n_visits-1])
            mu_list.append(pf.mu)
            n_spec_list.append(pf.n_spec)
            group_list.append(pf.group)
    
    if sort_mode == 'mu':
        ind = np.argsort(mu_list) 
    elif sort_mode == 'shift':
        ind = np.argsort(shifts)
    elif sort_mode == 'n_spec':
        ind = np.argsort(n_spec_list)
        n_spec_list = [n_spec_list[i] for i in ind] 
    elif sort_mode == 'group':
        ind = np.argsort(group_list)
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

#%% Construction of new STFP npz

for tr, sp, fl in zip(track_names, spike_names, field_names):
    st = tr.find('STFP_')
    name = tr[st:st+9]
    if name[5] in ['1','3','4'] or name[5] == '5' and name[-1] in ['1','2','3','4']:
        continue
    try:    
        ms = mcl.Mouse(name, 0, path_track = tr, path_neuro = sp, xy_delim = ',', xy_sk_rows = 0, skip_cols = -1)
        ms.get_spike_data(sp)
        
        ms = Get_STFP_PC_from_Vitya(ms, fl)
        n_pf = 1
        while n_pf: #refining cells until toltal number of place fields stabilizes
            n_pf = np.sum([len(pc.pf) for pc in ms.pc])
            for i, pc in enumerate(ms.pc):
                ms.pc[i] = spps.Get_Selectivity_Score_MultiPF(ms.spikes[:,pc.cell_num], pc, min_sp_len = 3)         
            ms = PurifyPlaceCells(ms)
            n_pf -= np.sum([len(pc.pf) for pc in ms.pc])     
              
        np.savez(path + name + '.npz', [ms]) 
    except:
        pass


#%% HT and RT matching statistics
for name in names:
    ms = pio.LoadMouse(path_rt + name + '_1D_nnew.npz')
    match = np.genfromtxt(path_ht + name + '_match.csv', delimiter = ',')
    pc_nums = [pc.cell_num for pc in ms.pc]
    # pcs = [pc for pc in ms.pc]
    ipc_nums = [pc.cell_num for pc in ms.pc if len([pf.n_spec for pf in pc.pf if not pf.n_spec])]
    print(name)
    print(ms.n_cells, len(pc_nums), len(ipc_nums))
    for d in range(3):
        print(len([row for row in match if row[3+d]]))
        print(len([row for row in match if row[0] and row[3+d]]))
        print(len([num for num in pc_nums if match[num][3+d]]))
        print(len([num for num in ipc_nums if match[num][3+d]]), '\n')

#%% Sel score in t-domain HT
aspect_r = 0.35
sort = 'mu'

for name in names:

    fig = plt.figure(figsize = (30, 10), dpi = 100)
    plt.axis('off')
    plt.title(name)
        
    for d, day in enumerate(days[3:]):
        ms = pio.LoadMouse(path_ht + name + day + '.npz')

        ms.ts_score = []
        t_spec_list = []

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
                
        if not len(ms.ts_score):
            continue
        
        # if sort =='t_spec':
        ind = np.argsort(t_spec_list)
        ms.ts_score = [ms.ts_score[i] for i in ind]
        # elif sort =='mu':
        #     ind = np.argsort(mu_list)
        #     ms.ts_score = [ms.ts_score[i] for i in ind]   
        # fig = plt.figure()
        # plt.axis('off')
        # plt.title(name + '_day_' + str(d))
        
        ax1 = fig.add_subplot(231 + d, yticks=[])
        ax1.imshow(np.array(ms.ts_score)[:,0:-6], extent = (0, (ms.n_frames-1000)/20, (ms.n_frames-1000)*aspect_r/20, 0))
                
        ax2 = fig.add_subplot(234 + d, xlim = (0, (ms.n_frames-1000)/20))
        ax2.set_box_aspect = 0.25
        ax2.plot(timeline/20, np.sum(ms.ts_score, axis = 0)*2/len(ms.ts_score), label = 'av_sel_score', linewidth = 1.5, zorder = 1)
        ax2.plot(np.linspace(0,len(ms.speed)/20,len(ms.speed)), ms.speed/np.max(ms.speed), label = 'speed', zorder = 0.5)
        # try:
        #     ms = mcl.GetRears(ms, path+name+'_' + str(d) + 'D_track_stances.csv')
        #     ax2.plot(np.linspace(0,len(ms.speed)/20,len(ms.speed)), np.array(ms.rear)/2, label = 'rears', zorder = 0)
        # except:
        #     continue
        ax2.legend()
    
    plt.show()
    fig.savefig(path_px + name + '_timed_sel_score.png')
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


for name in names:
    for d in [1,2,3]:

        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue          
        n_single_pf = len([pc for pc in ms.pc if len(pc.pf) == 1]) 
        n_pf = np.sum([len(pc.pf) for pc in ms.pc])
        
        av_nspec = 0
        av_tspec = 0
        
        tlen=[]
        for pc in ms.pc:
            for pf in pc.pf:
                tlen.append(len(pf.times_in))
                av_nspec += pf.n_spec
                av_tspec += pf.t_spec/20
                if pf.t_spec <=1200:
                    n_first += 1
        ttlen.append(np.min(tlen))       
        if n_pf:                                     
            print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_pf) + '\t' + str(n_single_pf) + '\t' + str(av_nspec/n_pf) + '\t' + str(av_tspec/n_pf) + '\t' + str(ms.spec_score[0]) + '\t' + str(ms.spec_score[-1]))
        else:
            print(name + '_' + str(d) + 'D\t'+ str(ms.n_cells) + '\t' + str(len(ms.pc))  + '\t' + str(n_pf) + '\t' + str(n_single_pf))
# print(np.mean(ttlen))
# print(np.mean(av_nspec))
# print(np.mean(av_tspec))
# print(n_first)

#%% T_spec and N_spec distribution
fig = plt.figure(figsize=(8,5), dpi=100)
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

    print('Day ' + str(d) + '\tAv_n_spec = ' + str(np.mean(n_spec)) + '\tAv_t_spec = ' + str(np.mean(t_spec)) + '\t' + str(len([t for t in t_spec if t < 60])/len(t_spec)))
    
plt.legend()
plt.show()

#%% T_spec and N_spec STFP distribution
fig = plt.figure(figsize=(8,5), dpi=100)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.title.set_text('N spec HT distribution')
ax2.title.set_text('T spec HT distribution')
for day in days:
    n_spec = []
    t_spec = []
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + day + '.npz')
        except:
            continue
        for pc in ms.pc:
            for pf in pc.pf:
                n_spec.append(pf.n_spec)
                t_spec.append(pf.t_spec/20)                 
    ax1.hist(n_spec, label = day[1:])
    ax2.hist(t_spec, label = day[1:])

#    print(day + '\tAv_n_spec = ' + str(np.mean(n_spec)) + '\tAv_t_spec = ' + str(np.mean(t_spec)) + '\t' + str(len([t for t in t_spec if t < 60])/len(t_spec)))
    print(len([n for n in n_spec if not n])/len(n_spec))
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
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'retention')

hmap = np.sum(np.sum(hmap2, axis = 0), axis = 0)
plt.plot(np.linspace(-25,25,51), hmap/np.sum(hmap), label = 'remapping')
plt.legend()

print(np.sum(d1, axis = 0))
print(np.sum(d2, axis = 0))   
 
#%% Selectivity score STFP maps
   
for name in names:
    fig = plt.figure(figsize=(20,5), dpi=100)
    plt.rcParams['font.size'] = '16'
    plt.title(name+'\n')
    plt.axis('off')
    max_n_visits = 50
    aspect_r = 2
       
    for d, day in enumerate(days):

        try:
            ms = pio.LoadMouse(path + name + day + '.npz')
            sel_score_map, n_spec_list = GetSelectivityScoreMap(ms, 'none',  max_n_visits = max_n_visits, sort_mode = 'group')
            fact = aspect_r*max_n_visits/len(sel_score_map)
            ax = fig.add_subplot(170+d+1)
            ax.title.set_text(day)
            ax.set_yticks([])
            for i, n_spec in enumerate(n_spec_list):    
                ax.add_patch(patches.Polygon([[n_spec+0.2, i*fact],[n_spec+0.2, (i+1)*fact],[n_spec+1, (i+0.5)*fact]], facecolor = 'red', edgecolor = 'none'))
            ax.imshow(sel_score_map, extent = (0, max_n_visits, max_n_visits*aspect_r, 0))
        except:
            pass
    plt.savefig(path_px + name + '_sel_score_daily_group_sorted.png') 
    
#%% MEAN Selectivity score STFP calculation
min_sel_len = 40
fig = plt.figure(figsize=(15,5), dpi=100)
for d, day in enumerate(days):
    ax = fig.add_subplot(170+d+1)
    ax.set_ylim(ymin = 0.0, ymax = 0.6)
    p_sscore = []
    for name in names:
        try:
            ms = pio.LoadMouse(path + name + day + '.npz')
        except:
            continue
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
        p_sscore.append(ms.spec_score)
        ax.plot(ms.spec_score, color = [.6, .6, .6], linewidth = 1)
    ax.plot(np.mean(p_sscore, axis = 0), color = [0, 0, 1], linewidth = 2.5, label = 'persist') 

#%% PLACE FIELD MAPS
days = [1,2,3]

for sd in days:
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

        plt.title(name + ' sorted at day ' + str(sd))
        plt.savefig(path + name + '_fields_sorted' + str(sd) + '.png')
        plt.close()
        
#%% Spike density distribution 
gmap = []
mmap = []
pfcenters = []  
 
for name in names: #['CA1_23', 'CA1_24', 'CA1_25', 'NC_701', 'NC_702', 'G7F_2', 'FG_2', 'NC_761']:
    for d in [1,2,3]:
        try:
            ms = pio.LoadMouse(path + name + '_' + str(d) + 'D_nnew.npz')
        except:
            continue
        
        _, hmap = DrawPlaceFieldsMap(ms, 'none', draw = False, sort = True)
        gmap.append(hmap)
        mmap.append(np.sum(hmap, axis = 0)/np.max(np.sum(hmap, axis = 0)))
        for pc in ms.pc:
            for pf in pc.pf:
                pfcenters.append(pf.mu)

gmap = np.concatenate(tuple(gmap))
order = np.argsort(np.argmax(gmap, axis = 1))
gmap = gmap[order,:]


fig, axs = plt.subplots(1, 2, figsize=(6, 5), dpi=100)
plt.title('Overall spiking map')
axs[0].imshow(gmap, cmap='jet',interpolation = 'nearest', extent = (0,360,0,720))
axs[1].plot(np.sum(gmap, axis = 0)) 
plt.ylim(0, np.max(np.sum(gmap, axis = 0)))


mmap = np.array(mmap)
fig, axs = plt.subplots(1, 2, figsize=(6, 5), dpi=100)
plt.title('Mousewise spiking map')
axs[0].imshow(mmap, cmap='jet',interpolation = 'nearest', extent = (0,360,0,720))
for m in mmap:
    axs[1].plot(m, color = [0.7, 0.7, 0.7])
axs[1].plot(np.mean(mmap, axis = 0), 'b', linewidth = 3) 
plt.ylim(0, 1)

fig = plt.figure(figsize=(6, 5), dpi=100)
plt.title('Place field location distribution')
plt.hist(pfcenters, bins = range(0,360,20))
              
    
        
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

#%% Transition HT histograms
bins = np.linspace(-180,180, 21)
m = 0.12
k = 1.1

for name in names:
    fact = 3 
    fact2 = 2
    ms = []
    for d, day in enumerate(days):
        ms.append(pio.LoadMouse(path_ht + name + day + '.npz')) 
        
    fig = plt.figure(figsize = (fact*5,5*k), dpi = 100)
    fig.subplots_adjust(bottom = m, left = m*fact2/fact, top = 2-m-k, right = 1-m*fact2/fact, wspace = m*2/((1-m*2)*fact2))
    plt.rcParams['font.size'] = '22'
    plt.title(name+ '\n')
    plt.axis('off')

    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        SH = TransitionHistogram(ms[dd[0]], ms[dd[1]], path + name + '_match.csv', dd[0], dd[1], mode = 'second')

        shist,_ = np.histogram(SH, bins=bins)
        f_exp,_ = np.histogram(np.random.normal(0, 1, 1000), bins=bins)
        rem_rate, pr = chisquare(shist)#, f_exp = f_exp) 
        #rem_rate, pr = kstest(SH, 'norm')
        print(name + f' {dd[0]}{dd[1]} {rem_rate:.1f} {pr:.3f}')
        ax = fig.add_subplot(101+ fact*10+i)
        if nd[name]==2:
            ax.title.set_text(f'{name}  day {dd[0]+1} vs day {dd[1]+1}')
        else:
            ax.title.set_text(f'day {dd[0]+1} vs day {dd[1]+1}')
        plt.setp(ax, yticks = [0, max(shist)], xticks=[-180,0,180], xticklabels=['-180','0','180'])
        ax.hist(SH, bins = bins)

    plt.savefig(path + name + '_trans_hist.png')
    plt.close()  
    
#%% Matching statistics
for name in names3:
    match = np.genfromtxt('D:\Work\PC_paper\matching\\' + name + '_match.csv', delimiter = ',')
    for i, dd in enumerate([[0,1],[1,2],[0,2]]):
        print(len([row for row in match if row[dd[0]] and row[dd[1]]]))     

for name in names2:
    if name in names3:
        continue
    match = np.genfromtxt('D:\Work\PC_paper\matching\\' + name + '_match.csv', delimiter = ',')
    print(len([row for row in match if row[0] and row[1]]))
    
#%% Draw cell hist
ms = pio.LoadMouse(path + 'CA1_22_2D_nnew.npz')
cell = 228
sp_angles = ms.angle[ms.spikes[:,cell]!=0]
mu, std = pcf.CellFields(sp_angles.tolist(), sigma=1, angle_len=90, realbin=20) 

#%% STFP place cells matching
root = 'D:\Work\Fields_STFP'
mnames = ['STFP_1','STFP_4','STFP_9']
mdays = [['_D6', '_D7'],['_D3','_D4'],['_D6','_D7']]
all_sh = []

for name, days in zip(mnames, mdays):
    ms = []
    shifts = []
    for d in days:
        ms.append(pio.LoadMouse(root + name + d + '.npz'))
    match = np.genfromtxt(root + name + '_match.csv', delimiter = ',') 
    for m_row in match:
        if m_row[0]*m_row[1]:
            pfs = [FindPlaceFieldsSTFP(ms[i], m_row[i]) for i in range(2)]
            if len(pfs[0]) == 1 and len(pfs[1]) == 1:
                shifts.append(np.linalg.norm([pfs[0][0].x - pfs[1][0].x, pfs[0][0].y - pfs[1][0].y]))
    all_sh.append(shifts)       