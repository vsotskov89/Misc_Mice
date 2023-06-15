# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 14:06:18 2021

@author: vsots
"""

import scipy.ndimage as sf
import numpy as np
from copy import copy

class PlaceCell():
    def __init__(self, cell_num):
        self.cell_num = cell_num
        
class PlaceField():
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        
        
def get_place_cells_in_circle(mouse, mode = 'sel_score', min_n_spikes = 5):
    #searching for selective  cells in circle track

    mouse.pc = [] #list of place CELLS, each place cell may have several (but at least one) of non-overlapping  place FIELDS
    for cell, sp in enumerate(mouse.spikes[0]):
        
        sp_angles = mouse.angle[mouse.spikes[:,cell]!=0]  #array of angles(in degrees) where spikes of this cell were cetected
        if len(sp_angles) < min_n_spikes: #discard cells with <=  min_n_spikes
            continue
        mu, std = CellFields(sp_angles.tolist(), sigma=1, angle_len=90, realbin=20) # mu is centroid angle of place field (in degrees)

        if not len(mu):
            continue     
        
        mouse.pc.append(PlaceCell(cell))
        mouse.pc[-1].pf = []                  #list of place FIELDS of a given place CELL
        
        for i,m in enumerate(mu):
            #each place cell can have multiple place fields              
            pf = PlaceField(mu = m, std = std[i])
            pf.times_in, pf.times_out = getPasses(mouse.angle, m, std[i]+30) #30 is a reserve to include near-in-zone spikes
            

            siz = []
            for i, tin in enumerate(pf.times_in):
                siz.append(sum(mouse.spikes[tin:pf.times_out[i], cell]))
            if np.count_nonzero(siz) < min_n_spikes: #discard fields with <=  min_n_spikes of IN-ZONE spikes
                continue
            
            
            
            #t_spec is the first time the cells fires SELECTIVELY in its zone
            #n_spec is corresponding number of in-zone visit (when the first in_zone spike occures)
            if mode == 'sel_score':
                pf[0].t_spec, pf[0].n_spec = Get_Selectivity_Score(mouse.spikes[:,cell], pf.times_in, pf.times_out, min_sp_len = 4)
                #if t_spec = 0 than GetSelectivity score failed;             
                if not pf[0].t_spec:
                    continue
            elif mode == 'first_time' and not pf[0].t_spec: #more simple version, t_spec is the first time the cells JUST fires in its zone
                for i, tin in enumerate(pf.times_in):
                    spikes_in_zone = mouse.spikes[tin:pf.times_out[i], cell]
                    if np.count_nonzero(spikes_in_zone):
                         rel_t = np.nonzero(spikes_in_zone)
                         pf.t_spec = tin + rel_t[0][0]
                         pf.n_spec = i + 1
                         break
            else:
                continue
                 
            # #JUST IN CASE
            # if not hasattr(pf, 't_spec'):
            #     pf.t_spec = mouse.n_frames
            #     pf.n_spec = len(pf.times_in)

            mouse.pc[-1].pf.append(pf)
            
            
        if not len(mouse.pc[-1].pf):
            del mouse.pc[-1]
    return




def Get_Selectivity_Score(spikes, t_in, t_out, min_sp_len = 5):
    sel_score = []
    for i, tin in enumerate(t_in):
        if not i:
            i_start = 1
        else:
            i_start = t_out[i-1]
        if i == len(t_in)-1:
            i_end = len(spikes)-1
        else:
            i_end = t_in[i+1]
        selective_sum = np.count_nonzero(spikes[tin:t_out[i]])
        overall_sum = np.count_nonzero(spikes[i_start:i_end])
        
        if overall_sum:
            sel_score.append(selective_sum/overall_sum)
        else:
            sel_score.append(0)
    
    fil_score = sf.filters.gaussian_filter1d(sel_score, sigma = 1, order=0, mode='reflect')
    fil_score = fil_score/max(fil_score)
    high_score = (fil_score >= 0.5)*fil_score
    isl, num = sf.measurements.label(high_score)
    for i in range(1,num+1):
        if np.count_nonzero(isl == i) >= min_sp_len:
            n_spec = np.nonzero(isl == 1)[0][0]
            if np.count_nonzero(spikes[t_in[n_spec]:t_out[n_spec]]):
                t_spec = np.nonzero(spikes[t_in[n_spec]:t_out[n_spec]])[0][0] + t_in[n_spec] #first spiking time in visit #n_spec
            else:
                t_spec = int((t_in[n_spec] + t_out[n_spec])/2)
            break
    if 'n_spec' in locals():
        return t_spec, n_spec
    else:
        return 0,0
    
    
    

def IsInZone(ang, alpha, d_alpha):
    for super_ang in [ang, ang - 360, ang +360]:
        if alpha-d_alpha/2 <= super_ang <= alpha+d_alpha/2:
            return True
    return False

def getPasses(track, p_alpha, p_wid):
    times_in = []
    times_out = []
    for i in range(1,len(track)):
        if (i==1 or not IsInZone(track[i-1], p_alpha, p_wid)) and IsInZone(track[i], p_alpha, p_wid) :
            times_in.append(i)
        if IsInZone(track[i], p_alpha, p_wid) and (i+1 == len(track) or not IsInZone(track[i+1], p_alpha, p_wid)) :
            times_out.append(i)
    return times_in, times_out 

    
def CellFields(spikes,sigma,angle_len,realbin):
    extended_bin = int(realbin+angle_len*realbin/360)
    append_bin = int((extended_bin - realbin)/2)
    spikes_append = analfitAppend(spikes, angle_len)
    div_spikes = BinDiv(spikes_append,extended_bin,angle_len)
    print(div_spikes)
    smooth_spikes = sf.filters.gaussian_filter1d(div_spikes, sigma =sigma, order=0, mode='reflect')
    print(smooth_spikes)
    norm_spikes = smooth_spikes/max(smooth_spikes[append_bin:realbin+append_bin])
    print(norm_spikes)
    pik_spikes = (norm_spikes>=0.5)*norm_spikes
    print(pik_spikes)
    mu_list, std_list = FindPlace(pik_spikes, realbin)
    std_list, mu_list = RefineFields(std_list, mu_list)
    print(mu_list)
    print(std_list)

    # PlotHist(spikes_append,extended_bin,mu_list,std_list)
    return mu_list, std_list


#divide array of spikes on "Nbin" bins 
def BinDiv(spikes, Nbin, angle):
    ydata = np.zeros(Nbin, 'i')
    for ang in spikes:
        ydata[int(ang//((360+angle)/Nbin))] += 1
    return ydata

def FindPlace(pik_spikes, Nbin):
    count = 0
    std_list = []
    mass = []
    mu_list = []
    centr = 0
    angle_mu = 0
    for ix in range(len(pik_spikes)-1):
        if pik_spikes[ix] != 0:
            count += 1
            mass.append(pik_spikes[ix])
        if pik_spikes[ix+1] == 0 and pik_spikes[ix] != 0:
            std_list.append(count*360/Nbin)
            centr = sf.measurements.center_of_mass(np.array(mass))
            angle_mu = (ix-count+1.5+centr[0])*360/Nbin
            mu_list.append(round(angle_mu))
            count = 0
            mass.clear()
    return mu_list, std_list

def RefineFields(std_list,mu_list, max_field_size = 180):
    # std_list = [std_list[i] for i in range(len(mu_list)) if mu_list[i] >= 30 and mu_list[i]<= 390]
    # mu_list = [mu_list[i] for i in range(len(mu_list)) if mu_list[i] >= 30 and mu_list[i]<= 390]
    mu_list = [mu_list[i] for i in range(len(std_list)) if std_list[i] <= max_field_size]
    std_list = [std_list[i] for i in range(len(std_list)) if std_list[i] <= max_field_size]
    mu_list = (np.array(mu_list)%360).tolist()
    n_fields = len(mu_list)

    for i in range(n_fields):
        if i and i<len(mu_list):
            for k in range(1,i+1):
                if IsInZone(mu_list[i]-std_list[i]/2, mu_list[i-k],std_list[i-k]) or IsInZone(mu_list[i]+std_list[i]/2, mu_list[i-k],std_list[i-k]) or IsInZone(mu_list[i-k]-std_list[i-k]/2, mu_list[i],std_list[i]) or IsInZone(mu_list[i-k]+std_list[i-k]/2, mu_list[i],std_list[i]):
                   #check if the current place field intersects with k-previous, delete the thinnest of two
                   if std_list[i-k] >= std_list[i]:
                       del mu_list[i]
                       del std_list[i]
                   else:
                       del mu_list[i-k]
                       del std_list[i-k]               
                   break  #it is supposed that it could not be more than 1 intersection
        
    return std_list, mu_list

def analfitAppend(spikes,angle):
    spikess = copy(spikes)
    for ang in range(len(spikess)):
        if int(spikess[ang]) < angle:
            spikess.append(spikess[ang]+360)
    return spikess[:]


    
    
    
    

    


