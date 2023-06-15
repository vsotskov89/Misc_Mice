# -*- coding: utf-8 -*-
'''
5-6 Apr Vova: getSpikeBrickLen() added, getSpikingPercent() modifyed: spiking_times are back
8 Apr Vova: added GetSpikingIntervals() - adapted from brics.py by Daniel
 '''   
    
import numpy as np
import scipy.signal as scps
import scipy.ndimage as sf

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
        
        
def getSpikingPercent(Mouse, times_in, times_out, cell, direction):
    n_passes = len(times_in)
    spiking_passes = []
    spiking_times = []
    for i in range(n_passes):
        if direction !=0 and Mouse.direction[times_in[i]] == direction and Mouse.direction[times_out[i]] == direction or direction==0:
            t = 0
            for j in range(times_in[i], times_out[i]):
                if (direction !=0 and Mouse.direction[j] == direction or direction==0) and Mouse.spikes[j,cell]:
                    t = j
                    break
            spiking_passes.append(np.sign(t))
            spiking_times.append(t) #0 if there are no proper spikes in zone
    return spiking_passes, spiking_times
'''
    spiking_percent = scps.medfilt(spiking_passes,3)
    i = len(spiking_percent)-1
    while i>=0 and spiking_percent[i]:
        i-=1
    if  i>=-1 and i < len(spiking_times)-1:
        t_spec = spiking_times[i+1]
        if i<len(spiking_times)-2 and not spiking_times[i+1]:
            t_spec = spiking_times[i+2]
    else:
        t_spec = 0'''
    
#    return spiking_passes#, len(spiking_percent)-1-i, t_spec


def getSpikeBrickLen(a, med, min_size):
	a = np.asarray(a)
	if med:
		a = scps.medfilt(a)
	#res_list = []
	res_size = min_size
	size = 0
	res_begin = 0
	inbrick = False
	for i, x in enumerate(a):
		if x:
			if not inbrick:
				begin = i
				inbrick = True
			size += 1
		else:
			if inbrick:
				if (res_size < size):
					res_size = size
					res_begin = begin
				size = 0
				begin = 0
			inbrick = False
	if inbrick and (res_size < size):
		res_size = size
		res_begin = begin
		size = 0
		begin = 0
		inbrick = False
	if res_size == 2:
		res_size = 0	
	return res_begin, res_size

def GetSpikingIntervals(a, minlen):
	a = np.asarray(a)
	bricks = []
	inbrick = False
	for i, x in enumerate(a):
		if x:
			if not inbrick:
				begin = i
				inbrick = True
		else:
			if inbrick:
				size = i - begin
				if size >= minlen:
					bricks.append((begin, size))
				else:
					a[begin:i+1] = 0
				inbrick = False
	if inbrick:
		size = i + 1 - begin
		if size >= minlen:
			bricks.append((begin, size))
		else:
			a[begin:i+1] = 0
	return a, np.array(bricks)

def FindTSpec(spiking_percent):
#    t_spec,pcov=sc.curve_fit(hvsd_step,range(len(spiking_percent)-1),spiking_percent,bounds=([0],[len(spiking_percent)-1]))
    opt = 100500
    t_spec = 0
    lens = len(spiking_percent)
    for i in range(lens):
        opt2 = sum([(spiking_percent[x] - hvsd_step(x,i))**2 for x in range(lens)])
        if opt2<opt:
            opt = opt2
            t_spec = i
    return t_spec + 1
        
        #separate massive for spikes in zone is needed (?)


def hvsd_step(x,t): 
    return 0.5 * (np.sign(x-t) + 1)


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
    # fil_score = fil_score/max(fil_score)
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
        return t_spec, n_spec, sel_score, fil_score
    else:
        return 0,0,sel_score, fil_score


def Get_Selectivity_Score_MultiPF(spikes, pc, min_sp_len = 5): 
    times_in = [0, len(spikes)-1]
    times_out = [0, len(spikes)-1]
    for pf in pc.pf:
            times_in += pf.times_in
            times_out += pf.times_out 
    
    for i, pf in enumerate(pc.pf):            
        pf.sel_score = []
        for tin, tout in zip(pf.times_in, pf.times_out):
            if not tin:
                tin += 1
            if tout == len(spikes)-1:
                tout -= 1
            touts = [t for t in times_out if t < tin]
            tins = [t for t in times_in if t > tout]
            i_start = np.max(touts)  #max of touts less than tin
            i_end = np.min(tins)  #min of tins more than tout
            selective_sum = np.count_nonzero(spikes[tin:tout])
            overall_sum = np.count_nonzero(spikes[i_start:i_end])
            
            if overall_sum:
                pf.sel_score.append(selective_sum/overall_sum)
            else:
                pf.sel_score.append(0)       
            
        pf.fil_score = sf.filters.gaussian_filter1d(pf.sel_score, sigma = 1, order=0, mode='reflect')
        high_score = (pf.fil_score >= 0.5)*pf.fil_score
        isl, num = sf.measurements.label(high_score)
        flag = 0
        for j in range(1,num+1):
            if np.count_nonzero(isl == j) >= min_sp_len:
                pf.n_spec = np.nonzero(isl == 1)[0][0]
                if pf.sel_score[pf.n_spec]:
                    pf.t_spec = np.nonzero(spikes[pf.times_in[pf.n_spec]:pf.times_out[pf.n_spec]])[0][0] + pf.times_in[pf.n_spec] #first spiking time in visit #n_spec
                else:
                    pf.t_spec = int((pf.times_in[pf.n_spec] + pf.times_out[pf.n_spec])/2)
                flag = 1
                break
        if not flag:
            pf = 'none'

        pc.pf[i] = pf
        
    while 1: #delete all dropped place fields
        try:
            pc.pf.remove('none')
        except:
            break        
    return pc


