# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:31:13 2021

@author: vsots
"""
import numpy as np


def get_spike_space(ms, i, size = 1000, shift = 0):
#Nikita's routine        
    spspace = np.zeros((size, size))
    scx, scy = ms.scx, ms.scy

    if shift != 0:
        spikes = np.roll(ms.spikes[:,i], shift = shift)
    else:
        spikes = ms.spikes[:,i]

    spiking_frames = np.where(spikes!=0)[0]
    xcoords = [int(xc) for xc in scx[spiking_frames]*size]
    ycoords = [int(yc) for yc in scy[spiking_frames]*size]
    for (x,y) in zip(xcoords, ycoords):
        spspace[x,y] += 1.0

    return spspace

def define_space(ms, mouse_radius, size = 1000):
#Nikita's routine        
    space = np.zeros((size, size))
    mr = mouse_radius

    scx, scy = ms.scx, ms.scy
    coords = list(zip(scx, scy))
    for c in coords:
        xgrid, ygrid = int(c[0]*size), int(c[1]*size)
        '''
        for xc in range(xgrid - mouse_radius, xgrid + mouse_radius):
            for yc in range(ygrid - mouse_radius, ygrid + mouse_radius):
                #if (xc-xgrid)**2 + (yc-ygrid)**2 <= mouse_radius**2:
        '''
        space[xgrid-mr:xgrid+mr, ygrid-mr:ygrid+mr] += 1.0
    
    return space    

def get_SI_score(nspikes, space, spspace):
#Nikita's routine for spatial info computation  
    if nspikes <= 1:
        #print('this cell has only %s spikes'%nspikes)
        return None

    firing_rates = np.divide(spspace, space)
    np.nan_to_num(firing_rates, copy = False)    
    mean_firing_rate = nspikes/np.count_nonzero(space)

    logs = np.nan_to_num(np.log2(firing_rates/mean_firing_rate), copy = False)
    info_space = np.multiply(np.multiply(space/np.count_nonzero(space), firing_rates), logs)
    I = np.sum(info_space)/mean_firing_rate
    return I


def check_place_cell_SI(ms, cell, SI_sigma_thr, nsim = 1000, size = 100, recalculate = 0, silent = 1):
#Nikita's routine based on spatial info
    res = False
            
    ms.scx = (ms.x-min(ms.x))/(max(ms.x)-min(ms.x))-0.000001
    ms.scy = (ms.y-min(ms.y))/(max(ms.y)-min(ms.y))-0.000001 

    # if not hasattr(ms, 'spatial_info') or np.isnan(ms.spatial_info[cell]) or np.isnan(ms.rand_spatial_info[cell]) \
    #                                 or np.isnan(ms.rand_spatial_info_std[cell]) or recalculate:

    spikes = ms.spikes[:,cell]
    nspikes = np.count_nonzero(spikes)
    space = define_space(ms, 2, size = size)
    spspace = get_spike_space(ms, cell, size = size)
    #show(spspace)
    SI = get_SI_score(nspikes, space, spspace)
    #print('SI =', SI)
        
    if not SI is None:
            rand_SI = []
            #for i in tqdm.tqdm(range(nsim), position = 0, leave = True):
            for i in range(nsim):
                rand_shift = np.random.randint(100, len(spikes) - 100)
                r_spspace = get_spike_space(ms, cell, size = size, shift = rand_shift)
                rSI = get_SI_score(nspikes, space, r_spspace)
                #print('rSI =', rSI)
                rand_SI.append(rSI)
            
            #fig, ax = plt.subplots(figsize = (10,10))
            #ax.hist(rand_SI, bins = 50)
            #thr = np.percentile(np.array(rand_SI), 98)
            thr = np.mean(np.array(rand_SI)) + SI_sigma_thr*np.std(np.array(rand_SI))
            # ms.spatial_info[cell] = SI
            # ms.rand_spatial_info[cell] = np.mean(np.array(rand_SI))
            # ms.rand_spatial_info_std[cell] = np.std(np.array(rand_SI))
            #ax.axvline(x=SI, c = 'r')
            #ax.axvline(x=thr, c = 'y')
            if SI > thr:
                res = True

    # elif ms.spatial_info[cell] > ms.rand_spatial_info[cell] + SI_sigma_thr*ms.rand_spatial_info_std[cell]:
    #     res = True

    return res