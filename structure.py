##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:53:06 2018

2019 Nov 21 Vova: get_xy_data fixed, for .csv track files header_lines_number = 1 delimiter = ' '
2018 Dec 12 Vova: get_xy_data fixed, if first string is 0,0,0.. then header_lines_number = 1 delimiter = ','
2018 Jul 03 Vova: get_mult_behavior fixed, works normally
2018 Jun 22 Vova: get_mult_behavior added to score several acts simultaneously; get_angle added
2018 Apr 25 Vova: get_behavior revised, arbitrary behavior act sequence is now allowed
2018 Apr xz Vova: get_neuropil, get_behavior added
2018 Apr 1 Vova: get_xy_data method added, get_data retained intact
2018 mar 6 Vova: Digo.time is original VT time

@author: daniel
"""


import numpy as np
from itertools import islice
from scipy import interpolate, signal
import math
import random
import os, xlrd
import SpikingPasses as  spps
import PlaceFields as pcf
import random
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from scipy.stats import kstest


class PlaceCell():
    def __init__(self, cell_num):
        self.cell_num = cell_num
        
class PlaceField():
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

class Digo():
    def __init__(self, name1, **kwargs):
        self.name = name1
        self.dtypef = np.float32        
        if 'path_track' in kwargs and 'time_shift' in kwargs and 'path_spikes' in kwargs:
            self.path_track = kwargs['path_track']
            self.path_spikes = kwargs['path_spikes']
            self.time_shift = kwargs['time_shift']
        else:
            self.path_track = 'C:\Work\SCRIPTS\O-maze Python\cells\outputs\CA1_6_1day_20161116 output _new.csv'
            self.path_spikes = 'C:\Work\SCRIPTS\O-maze Python\cells\spikes\spikes_CA1_6_1day_conc_recording_20161116_121433_corrected_neuropil_30.csv'
            self.time_shift = 13.75

        
    def get_data(self):
        #old version, usable only for output files!!!!
         self.time, self.angle, self.speed = np.genfromtxt(self.path_track, delimiter=',', usecols=[0,1,2], dtype=self.dtypef, comments='None', unpack=True)
         self.min_time_trace = self.time[0]
         shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
         t_size = self.time.size
         sp = np.genfromtxt(self.path_spikes, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
         self.spikes = sp[:, 1:]
         self.angle = self.angle[0:len(self.spikes)]
         for i in range(len(self.angle)):
             while self.angle[i]>=360:
                 self.angle[i]-=360
             while self.angle[i]<0:
                 self.angle[i]+=360 
         self.speed = self.speed[0:len(self.spikes)]
         self.time = self.time[0:len(self.spikes)]
         self.get_direction()
         self.get_acceleration()
         return
     
        
    def get_xy_data(self, delimiter, skip_rows, skip_cols):
        w = 21
        time_massive = []
        x_massive = []
        y_massive = []
        self.x = []
        self.y = []
        self.speed = []
       
        #reading track file        
#        tr_file = open(self.path_track, 'r')
#        first_line = tr_file.readline()
#        
#        if first_line[0] =='0':
#            header_lines_number = 1
#            delimeter = ','
#            col = 0
#        elif self.path_track[-3:] == 'csv':
#            header_lines_number = 1
#            delimeter = ' '
#            col = 0        
#        else:
#            header_lines_number = int(first_line[-5:-3])
#            delimeter = first_line[-2]
#            col = 1
        
        tr = np.genfromtxt(self.path_track, delimiter=delimiter, skip_header=skip_rows)
        time_massive = tr[:,0]
        x_massive = tr[:,1+skip_cols]
        y_massive = tr[:,2+skip_cols]
        
        #pre-filtering
        x_massive = signal.medfilt(x_massive, 11)
        y_massive = signal.medfilt(y_massive, 11)
        
        #removing NaN values
        valid_indices = ~np.isnan(x_massive) * ~np.isnan(y_massive)
        time_massive = time_massive[valid_indices]
        x_massive = x_massive[valid_indices]
        y_massive = y_massive[valid_indices] 

        #forsing to 20 fps        
        fx = interpolate.interp1d(time_massive, x_massive)
        fy = interpolate.interp1d(time_massive, y_massive)
        self.time = list(range(int(time_massive[0]*20)+1, int(time_massive[-1]*20)))
        for i in range(len(self.time)):
            self.time[i] *= 0.05
            self.x.append(fx(self.time[i]))
            self.y.append(fy(self.time[i]))
        
        #smoothing with 1-s window average filter
        self.x = np.convolve(self.x, np.ones(w)/w, mode='same')
        self.y = np.convolve(self.y, np.ones(w)/w, mode='same')
        
        for i in range(1, len(self.x)):
            self.speed.append(round(math.sqrt((self.x[i] - self.x[i-1])**2 + (self.y[i] - self.y[i-1])**2)/(self.time[i]-self.time[i-1]), 3))

        self.min_time_trace = self.time[0]
        shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
        t_size = len(self.time)
        sp = np.genfromtxt(self.path_spikes, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
        self.spikes = sp[:, 1:]
        self.x = self.x[0:len(self.spikes)]
        self.y = self.y[0:len(self.spikes)] 
        self.speed = np.array(self.speed[0:len(self.spikes)])
        self.time = np.array(self.time[0:len(self.spikes)])
        self.get_acceleration()
        return
    
    def get_neuropil(self, n_path):
        shift_sp_rows = int(round((self.min_time_trace - self.time_shift)*20))
        t_size = len(self.time)
        neur = np.genfromtxt(n_path, delimiter=',', skip_header=shift_sp_rows, max_rows=t_size, dtype=self.dtypef, comments='None')
        self.neur = neur[:, 1:]
        return

    def get_obj_behavior(self, b_path, b_skip_rows, b_col):
        #scores acts as they are in behav file, even if they are overlapping; recognizes native START and STOP marks
        self.behav = []
#        book = xlrd.open_workbook(b_path, on_demand = True)
#        sheet = book.sheet_by_index(0)
#        data = [[str(c.value) for c in sheet.row(j)] for j in range(b_skip_rows,sheet.nrows)]
#        book.release_resources()
#        del book
        data =  np.genfromtxt(b_path, delimiter=',', skip_header=b_skip_rows)
        btype = 0 #archive of behavior types
        
        if float(data[0][0]) <= self.time[0]: #initial behavior act
            btype = int(CheckObj(data[0][b_col]))
        else:
            btype = 0
        
        #loop for strings in behavior file
        for j in range(len(data)):
            while len(self.behav) < len(self.time) and self.time[len(self.behav)] < float(data[j][0]):
                self.behav.append(btype)
            if ~CheckObj(data[j][b_col]):
                continue
            if float(data[j][0]) > self.time[-1]:
                break
            if data[j][b_col+3] == 'START':
                btype = 1
            if data[j][b_col+3] == 'STOP':
                btype = 0
        #the last interval    
        while len(self.behav) < len(self.time):
            self.behav.append(btype)
        
    def get_mult_behavior(self, b_path, b_skip_rows, b_col, default_behav_code):
        #scores acts as they are in behav file, even if they are overlapping; recognizes native START and STOP marks
        self.behav = []
        book = xlrd.open_workbook(b_path, on_demand = True)
        sheet = book.sheet_by_index(0)
        data = [[str(c.value) for c in sheet.row(j)] for j in range(b_skip_rows,sheet.nrows)]
        book.release_resources()
        del book
    
        btype = [] #archive of behavior types
        
        if float(data[0][0]) <= self.time[0]: #initial behavior act
            btype.append(float(data[0][b_col]))
        else:
            btype.append(default_behav_code)
        
        #loop for strings in behavior file
        for j in range(len(data)):
            while len(self.behav) < len(self.time) and self.time[len(self.behav)] < float(data[j][0]):
                self.behav.append(btype[:])
            if float(data[j][b_col]) ==  default_behav_code:
                continue
            if float(data[j][0]) > self.time[-1]:
                break
            if data[j][b_col+3] == 'START' and float(data[j][b_col]) not in btype:
                btype.append(float(data[j][b_col]))
                while default_behav_code in btype:
                    btype.remove(default_behav_code)
            if data[j][b_col+3] == 'STOP' and float(data[j][b_col]) in btype:
                self.behav.append(btype[:])
                btype.remove(float(data[j][b_col]))
                if not len(btype):
                    btype.append(default_behav_code)
        #the last interval    
        while len(self.behav) < len(self.time):
            self.behav.append(btype[:])
            
    def get_behavior(self, b_path, b_skip_rows, b_col):
        #scores consequent acts, in case of overlapping with respect to priority; positive figures = start, negative = stop
        self.behav = np.zeros(len(self.time))
        book = xlrd.open_workbook(b_path, on_demand = True)
        sheet = book.sheet_by_index(0)
        data = [[str(c.value) for c in sheet.row(j)] for j in range(b_skip_rows,sheet.nrows)]
        book.release_resources()
        del book
    
        btype = [0] #archive of behavior types
        for j in range(len(data)):
            if float(data[j][b_col]) > 0:
                btype.append(float(data[j][b_col]))
            else:
                btype.remove(-float(data[j][b_col]))
            self.behav[np.array(self.time) >= float(data[j][0])] = max(btype)
        return
            
    def get_angle(self):
        self.angle = []
        xc = (min(self.x)+max(self.x))/2
        yc = (min(self.y)+max(self.y))/2
        for i in range(len(self.x)):
            self.angle.append(180 + round((math.degrees(math.atan2(self.y[i] - yc, self.x[i] - xc))),3))
        self.angle = np.array(self.angle)  
        return

    def get_direction(self):
         der = np.ediff1d(self.angle, 0)
         der[der <= -90] += 360
         der[der >= 90] -= 360
         self.direction = np.sign(der)
         return

    def get_acceleration(self):
        w = 35
        sp = np.convolve(self.speed, np.ones(w)/w, mode='same')
        self.acceleration = np.gradient(sp, 0.05, edge_order=2)
        return
        
    def get_binned_neuro(self, Nbins):
        self.neuro_bin = np.zeros(self.neur.shape, dtype = int)
        for i in range(self.neur.shape[1]):
            ma = np.max(self.neur[:,i])
            mi = np.min(self.neur[:,i])
            self.neuro_bin[:,i] = (self.neur[:,i] - mi)*Nbins/(ma-mi)
        self.neuro_bin[self.neuro_bin == Nbins] -= 1
        return

    def get_binned_angle(self, Nbins):
        self.spatial_bin = np.zeros(len(self.angle), dtype = int)
        self.spatial_bin = self.angle*Nbins/360
        return
    
    def get_holed_bins(self, step, vid_h, vid_w):
        #step - size of bin in px; #vid_h/w - height/width of video in px
        Nx = np.ceil(vid_w/step)
        Ny = np.ceil(vid_h/step)
        self.spatial_bin = np.zeros(len(self.time), dtype = int)     #initialization
        self.spatial_bin = np.fix(self.x/step) + np.fix(self.y/step)*Nx     #binarization
        occup = np.histogram2d(np.fix(self.x/step), np.fix(self.y/step), bins = [Nx, Ny], range=[[0,Nx], [0,Ny]])
        true_ind = np.flatnonzero(occup[0]).tolist()           
#        self.spatial_bin = [true_ind.index(spb) for spb in self.spatial_bin.tolist()]

 
       
    def get_mi_score(self):
        self.mi_score = np.zeros(self.neur.shape[1], dtype = np.float32)
        for i in range(self.neur.shape[1]):
            nbi = self.neuro_bin[:,i]
#            self.mi_score[i] = adjusted_mutual_info_score(self.neuro_bin[:,i], self.spatial_bin)
            self.mi_score[i] = mutual_info_score(nbi[nbi>0], self.spatial_bin[nbi > 0])
        return
            
    def verify_cell(self, cell, N_shuff):
        count = 0
        ang_shuffled = self.spatial_bin
        for i in range(N_shuff):
            np.random.shuffle(ang_shuffled)
            mi = adjusted_mutual_info_score(self.neuro_bin[:,cell], ang_shuffled)
            if mi > self.mi_score[cell]:
                count += 1
        return count <= 0.05*N_shuff
    
    def verify_cells(self, N_shuff):  
        self.super_cells = []
        for i in range(self.neur.shape[1]):
            if self.verify_cell(cell = i, N_shuff = N_shuff):
                self.super_cells.append(i)
        return
            
	
def GetFields(Mouse):
    Mouse.mu_list = []
    Mouse.std_list = []
    Mouse.true_cells = []
    for cell in range(len(Mouse.spikes[0])):
        sp_angles = Mouse.angle[Mouse.spikes[:,cell]!=0]
        if len(sp_angles) < 3:
            continue
        mu, std = pcf.CellFields(sp_angles.tolist(), sigma=2.5,angle_len=90, realbin=40)
        for i in range(len(mu)):
            Mouse.mu_list.append(mu[i])
            Mouse.std_list.append(std[i])
            Mouse.true_cells.append(cell)
    return Mouse

def Get_KS_score(Mouse):
    Mouse.ks_stat = []
    Mouse.ks_p_score = []
    for cell in range(len(Mouse.spikes[0])):
        sp_angles = Mouse.angle[Mouse.spikes[:,cell]!=0]
        #sp_distrib, edges = np.histogram(sp_angles.tolist(), bins = 40, range = (0,360))
        if len(sp_angles) >= 3:
            stat, p = kstest(sp_angles, 'uniform', N = len(sp_angles), args = (0.0, 360.0))
        else:
            stat = p = 0
        Mouse.ks_stat.append(stat)
        Mouse.ks_p_score.append(-np.log(p))
    return Mouse

def GetFieldsNeuro(Mouse):
    Mouse.mu_list = []
    Mouse.std_list = []
    Mouse.true_cells = []
    for cell in range(len(Mouse.spikes[0])):
        bin_neuro = pcf.BinRawNeuro(angles = Mouse.angle, neurodata = Mouse.neur[:,cell], Nbin = 40)
        mu, std = pcf.CellFieldsNeuro(bin_neuro, sigma=2.5,bin_len=10, realbin=40)
        for i in range(len(mu)):
            Mouse.mu_list.append(mu[i])
            Mouse.std_list.append(std[i])
            Mouse.true_cells.append(cell)
    return Mouse


def FilterFields(Mouse):
    Mouse.t_spec = []
    Mouse.spec_len = []
    Mouse.field_dir = []
    Mouse.times_in = []
    Mouse.times_out = []
    Mouse.sp_passes = []
    Mouse.sp_intervals = []
    Mouse.all_t_spec = []
    Mouse.all_spec_len = []
    
    for cell in range(len(Mouse.true_cells)):
        times_in, times_out = spps.getPasses(Mouse.angle, Mouse.mu_list[cell],  Mouse.std_list[cell])
        sp_passes, sp_times = spps.getSpikingPercent(Mouse, times_in, times_out, Mouse.true_cells[cell], 0)
        sp_intervals, bricks = spps.GetSpikingIntervals(sp_passes, minlen = 3)
               
        if np.count_nonzero(sp_intervals):
            Mouse.t_spec.append([sp_times[bricks[i,0]] for i in range(len(bricks[:,0]))])
            Mouse.spec_len.append(bricks[:,1])
            for i in range(len(bricks[:,0])):
                Mouse.all_t_spec.append(sp_times[bricks[i,0]])
                Mouse.all_spec_len.append(bricks[i,1])

        Mouse.sp_passes.append(sp_passes)
        Mouse.sp_intervals.append(sp_intervals)
        Mouse.times_in.append(times_in)
        Mouse.times_out.append(times_out) 
              
    return Mouse

def FilterFieldsNew(Mouse, ks_thresh):
    Mouse.super_cells = [] 
    Mouse.super_in_true = []    #numbers of true_cells which also are super
    Mouse.t_spec = []
    Mouse.n_spec = []
    Mouse.spec_len = []
    Mouse.times_in = []
    Mouse.times_out = []
    Mouse.sp_passes = []
    Mouse.sp_intervals = []

    
    for cell, trcl in enumerate(Mouse.true_cells):
        times_in, times_out = spps.getPasses(Mouse.angle, Mouse.mu_list[cell],  Mouse.std_list[cell])
        sp_passes, sp_times = spps.getSpikingPercent(Mouse, times_in, times_out, trcl, 0)
        sp_intervals, bricks = spps.GetSpikingIntervals(signal.medfilt(sp_passes,5), minlen = 5)
               
        if np.count_nonzero(sp_intervals) and np.count_nonzero(sp_passes) >= 0.5*(len(sp_passes) - sp_passes.index(1)) and Mouse.ks_p_score[trcl] > ks_thresh:
            Mouse.super_cells.append(trcl)
            Mouse.super_in_true.append(cell)
            Mouse.t_spec.append([sp_times[bricks[i,0]] for i in range(len(bricks[:,0]))])
            Mouse.n_spec.append([bricks[i,0] for i in range(len(bricks[:,0]))])
           
            if Mouse.t_spec[-1][0] == 0:
                Mouse.t_spec[-1][0] = sp_times[0]
            Mouse.spec_len.append(bricks[:,1])
            
        Mouse.sp_passes.append(sp_passes)
        Mouse.sp_intervals.append(sp_intervals)
        Mouse.times_in.append(times_in)
        Mouse.times_out.append(times_out) 
        
    return Mouse    


def FilterFieldsNew_Light(Mouse):
    Mouse.super_cells = [] 
    Mouse.super_in_true = []    #numbers of true_cells which also are super
    Mouse.t_spec = []
    Mouse.n_spec = []
    Mouse.spec_len = []
    Mouse.times_in = []
    Mouse.times_out = []
    Mouse.sp_passes = []
    Mouse.sp_intervals = []

    
    for cell, trcl in enumerate(Mouse.true_cells):
        times_in, times_out = spps.getPasses(Mouse.angle, Mouse.mu_list[cell],  Mouse.std_list[cell])
        sp_passes, sp_times = spps.getSpikingPercent(Mouse, times_in, times_out, trcl, 0)
        sp_intervals, bricks = spps.GetSpikingIntervals(signal.medfilt(sp_passes,5), minlen = 3)
               
        if np.count_nonzero(sp_intervals) and np.count_nonzero(sp_passes) >= 0.5*(len(sp_passes) - sp_passes.index(1)):
            Mouse.super_cells.append(trcl)
            Mouse.super_in_true.append(cell)
            Mouse.t_spec.append([sp_times[bricks[i,0]] for i in range(len(bricks[:,0]))])
            Mouse.n_spec.append([bricks[i,0] for i in range(len(bricks[:,0]))])
           
            if Mouse.t_spec[-1][0] == 0:
                Mouse.t_spec[-1][0] = sp_times[0]
            Mouse.spec_len.append(bricks[:,1])
            
        Mouse.sp_passes.append(sp_passes)
        Mouse.sp_intervals.append(sp_intervals)
        Mouse.times_in.append(times_in)
        Mouse.times_out.append(times_out) 
        
    return Mouse 

def GetRears(Mouse, rear_fname):
    rr = np.genfromtxt(rear_fname, delimiter=',', skip_header=0)
    fr = interpolate.interp1d(rr[:,0], rr[:,1])
    Mouse.rear = [np.sign(fr(t)) for t in Mouse.time]
    return Mouse
    
def TrimTime(Mouse, t_end):
    fr_end = int((t_end - Mouse.time[0])*20) - 1
    Mouse.time = Mouse.time[0:fr_end]
    Mouse.angle = Mouse.angle[0:fr_end]
    Mouse.spikes = Mouse.spikes[0:fr_end,:]
    Mouse.neur = Mouse.neur[0:fr_end,:]
    Mouse.neuro_bin = Mouse.neuro_bin[0:fr_end,:]
    return Mouse

def RearStat(Mouse, n_sup, cell):
    rear_t_inz = []
    n_spec = 0
    for i, ti in enumerate(Mouse.times_in[cell]):
        if (i==0 or i>=1 and Mouse.t_spec[n_sup][0] >= Mouse.times_out[cell][i-1])  and Mouse.t_spec[n_sup][0] <= Mouse.times_out[cell][i]:
            n_spec = i
        rear_t_inz.append(np.sum(np.abs(Mouse.rear[ti:Mouse.times_out[cell][i]]))/(Mouse.times_out[cell][i]-ti))
    if n_spec:
        rear_t_bef = sum(rear_t_inz[:n_spec])/n_spec
    else:
        rear_t_bef = 0   
    rear_t_aft = sum(rear_t_inz[n_spec:])/(len(rear_t_inz) - n_spec)
    return rear_t_inz, rear_t_bef, rear_t_aft
    

def ReformFields(Mouse):
    Mouse.pc = []
    for cell in range(len(Mouse.spikes[0])):
        pf_pos = [i for i, c in enumerate(Mouse.true_cells) if c==cell]
        if not len(pf_pos):
            continue
        Mouse.pc.append(PlaceCell(cell))
        Mouse.pc[-1].pf = []
                
        for n_pos, pos in enumerate(pf_pos):
            pf = PlaceField(mu = Mouse.mu_list[pos], std = Mouse.std_list[pos])
            pf.is_super = pos in Mouse.super_in_true
            pf.times_in = Mouse.times_in[pos]
            pf.times_out = Mouse.times_out[pos]
            if pf.is_super:
                pf.t_spec = Mouse.t_spec[Mouse.super_in_true.index(pos)]
            Mouse.pc[-1].pf.append(pf)
    return Mouse

        

#def GetBinnedNeuro(Mouse, Nbins):
#    Mouse.neuro_bin = np.zeros(Mouse.neur.shape, dtype = int)
#    for i in range(Mouse.neur.shape[1]):
#        ma = np.max(Mouse.neur[:,i])
#        mi = np.min(Mouse.neur[:,i])
#        Mouse.neuro_bin[:,i] = (Mouse.neur[:,i] - mi)*Nbins/(ma-mi)
#    Mouse.neuro_bin[Mouse.neuro_bin == Nbins] -= 1
#    
#def GetBinnedAngle(Mouse, Nbins):
#    Mouse.spatial_bin = np.zeros(len(Mouse.angle), dtype = int)
#    Mouse.spatial_bin = Mouse.angle*Nbins/360    
#    
#    
#def GetMIScore(Mouse):
#    Mouse.mi_score = np.zeros(Mouse.neur.shape[1], dtype = np.float32)
#    for i in range(Mouse.neur.shape[1]):
#        Mouse.mi_score[i] = mutual_info_score(Mouse.spatial_bin, Mouse.neuro_bin[:,i])
#    
#    
#    
def CheckObj(strg):
    return strg.find("obj") > 0 or strg.find("1") > 0 or strg.find("Obj") > 0
#

