# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 02:12:40 2018

@author: Vova

1 Apr 2018 Vova: plot(..) modified: xy plot is shown when polar=False
4 May 2018 Vova: colornum_Metro modified!!!! 0-salad 1-red 2-green etc

"""

import numpy as np
import matplotlib.pyplot as plt
import pc_io as pio
import matplotlib.patches as patches
import matplotlib.cm as cm
import structure as strct
import glob
from PIL import Image
#import behavior_plot as bplt

def DrawField(mu, std, t_spec, t_end, **kwargs):
   if 'ax' in kwargs:
       ax = kwargs['ax']
   else:
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar')
   plt.setp(ax.get_yticklabels(), visible=False)
   plt.setp(ax.get_xticklabels(), visible=False)
   bins = np.arange(mu-std/2, mu+std/2, 5)*np.pi/180 
   for b in bins: 
       ax.add_patch(patches.Rectangle((b,t_spec), 5*np.pi/180, t_end-t_spec, facecolor = 'None', edgecolor = 'm', linewidth = 2, alpha=0.5))
   return ax
	

def colornum_Metro(num):
    return {
    1:[1,0,0],        #red
    2:[0,0.7,0],      #green
    3:[0,0,1],        #blue
    4:[0,1,1],        #cyan
    5:[0.5,0.25,0.2], #brown
    6:[1,0.5,0.1],    #orange
    7:[0.5,0,1],      #violet
    8:[0.8,0.7,0],    #yellow
    9:[0.5,0.5,0.5],  #grey
    0:[0.5,1, 0.1]}.get(num%10)   #salad



def Traceviewer(Mouse, out_fname):
    fig = plt.figure(figsize=(50,50), dpi=300)
    ax = fig.add_subplot(111)
    
    n_cells = len(Mouse.spikes[0])
    absmax = np.amax(Mouse.neur)
    if hasattr(Mouse, 'behav'):
        for i in range(len(Mouse.time)):
            if Mouse.behav[i] > 0:
                ax.add_patch(patches.Rectangle((Mouse.time[i], 0), 0.05, n_cells+1, facecolor = colornum_Metro(int(Mouse.behav[i])), edgecolor = 'none', alpha=0.3))
    for cell in range(n_cells):
        ax.plot(Mouse.time, Mouse.neur[:,cell]/absmax + cell , c = colornum_Metro(cell), linewidth=1)
    plt.savefig(out_fname, dpi = 300)
    plt.close()


def TraceviewerPC(Mouse, out_fname):
    #Draws traces, patches for in zone periods and decelerations
    n_cells = len(Mouse.true_cells)
    absmax = np.amax(Mouse.neur[:,Mouse.true_cells])/2
    thr = np.mean(Mouse.acceleration) - 2*np.std(Mouse.acceleration)
    
    fig = plt.figure(figsize=(int(Mouse.time[-1]/10),n_cells), dpi=200)
    ax = fig.add_subplot(111)
    ind = np.argsort(Mouse.mu_list)    
#    for i in range(len(Mouse.times_in)):
#        ax.add_patch(patches.Rectangle((Mouse.times_in[i], 0), Mouse.times_out[i], n_cells+1, facecolor = 'r', edgecolor = 'none', alpha=0.2))
    for i in range(len(Mouse.time)):
        if Mouse.acceleration[i] < thr:
            ax.add_patch(patches.Rectangle((Mouse.time[i], 0), 0.05, n_cells+1, facecolor = 'g', edgecolor = 'none', alpha=0.2))
    for cell in range(n_cells):
        ax.plot(Mouse.time, Mouse.neur[:,Mouse.true_cells[ind[cell]]]/absmax + cell , c = colornum_Metro(cell), linewidth=0.5)
        sps = Mouse.spikes[:,Mouse.true_cells[ind[cell]]]
        ax.scatter(Mouse.time[sps>0], sps[sps>0]/absmax + cell, marker='^', s=10 , edgecolors = 'black',  c = colornum_Metro(cell))
        for i in range(len(Mouse.times_in[ind[cell]])):
            ax.add_patch(patches.Rectangle((Mouse.time[Mouse.times_in[ind[cell]][i]], cell), Mouse.time[Mouse.times_out[ind[cell]][i]] - Mouse.time[Mouse.times_in[ind[cell]][i]], 1, facecolor = 'r', edgecolor = 'none', alpha=0.2))
    plt.savefig(out_fname, dpi = 200)
    plt.close()        

def TraceviewerPC_horizontal(Mouse, out_fname):
    #Draws traces, patches for in zone periods and decelerations
    n_cells = len(Mouse.true_cells)
    absmax = np.amax(Mouse.neur[:,Mouse.true_cells])
    absmin = np.amin(Mouse.neur[:,Mouse.true_cells])
    
    fig = plt.figure(figsize=(int(Mouse.time[-1]/10),n_cells*2), dpi=100)
    ax = fig.add_subplot(111)
    ind = np.argsort(Mouse.mu_list)  
    
    '''pr_behav = get_priority(Mouse.behav) #ranging behavior acts
    
    for i in range(len(pr_behav)):
        if pr_behav[i] < 10:
            continue
        if pr_behav[i]%10: #inner acts depicted in green, outer in blue
            color = 'g'
        else:
            color = 'b'   
        ax.add_patch(patches.Rectangle((Mouse.time[i], 0), 0.05, n_cells*2, facecolor = color, edgecolor = 'none', alpha=pr_behav[i]/100))'''    
            
                
    for cell in range(n_cells):
        absmax = np.amax(Mouse.neur[:,Mouse.true_cells[ind[cell]]])
        absmin = np.amin(Mouse.neur[:,Mouse.true_cells[ind[cell]]])        
        
        
        ax.plot(Mouse.time, Mouse.angle/360 + cell*2 + 1 , c = [0.7, 0.7, 0.7], linewidth=1.5)
        ax.axhline(cell*2, c=[0.2, 0.2, 0.2], lw=1.5)
        ax.axhline(cell*2 + 1, c=[0.5, 0.5, 0.5], lw=0.75)
        ax.axhline(cell*2 + 1 + (Mouse.mu_list[ind[cell]] - Mouse.std_list[ind[cell]]/2)/360, c=[0.7, 0.7, 0.7], ls=':', lw=0.75)
        ax.axhline(cell*2 + 1 + (Mouse.mu_list[ind[cell]] + Mouse.std_list[ind[cell]]/2)/360, c=[0.7, 0.7, 0.7], ls=':', lw=0.75)    

        ax.plot(Mouse.time, (Mouse.neur[:,Mouse.true_cells[ind[cell]]]-absmin)/(absmax-absmin) + cell*2, c = colornum_Metro(cell), linewidth=1)
        sps = Mouse.spikes[:,Mouse.true_cells[ind[cell]]]
        ax.scatter(Mouse.time[sps>0], (sps[sps>0]-absmin)/(absmax-absmin) + cell*2, marker='^', s=10 , edgecolors = 'black',  c = colornum_Metro(cell))

        for i in range(len(Mouse.times_in[ind[cell]])):
            if Mouse.sp_intervals[ind[cell]][i]:
                alp = 0.4
            else:
                alp = 0.2
            ax.add_patch(patches.Rectangle((Mouse.time[Mouse.times_in[ind[cell]][i]], cell*2 + 1 + (Mouse.mu_list[ind[cell]] - Mouse.std_list[ind[cell]]/2)/360), Mouse.time[Mouse.times_out[ind[cell]][i]] - Mouse.time[Mouse.times_in[ind[cell]][i]], Mouse.std_list[ind[cell]]/360, facecolor = 'r', edgecolor = 'none', alpha=alp))
            ax.vlines(Mouse.time[Mouse.times_in[ind[cell]]], cell*2, cell*2 + 1 + Mouse.mu_list[ind[cell]]/360, colors=[0.7, 0.7, 0.7], linestyles=':', lw=0.3)
            ax.vlines(Mouse.time[Mouse.times_out[ind[cell]]], cell*2, cell*2 + 1 + Mouse.mu_list[ind[cell]]/360, colors=[0.7, 0.7, 0.7], linestyles=':', lw=0.3)
    plt.savefig(out_fname, dpi = 100)
    plt.close()

def get_priority(behav):
    #returns list of thr most important behaviors
    behavior_codes = [6,0,1,9,4,7,8,2,5,3,12,20,21,30,31,40,41,50,51,10,11,99]
    priorities = [6,0.1,1,9,4,40,41,2,5,3,1.2,10,11,20,21,0.1,0.1,50,51,30,31,0]
    descriptions = ['running','quiet','shortening','stretching','rear','rear support out','rear support in','head down',
'grooming','eat/gnaw','head up','head turn out','head turn in','body turn out','body turn in',
'turn around out','turn around in','hang out out','hang out in','body up out','body up in','X factor']
    new_behav = []
    for i in range(len(behav)):
        new_b = []
        for b in behav[i]:
           new_b.append(priorities[behavior_codes.index(b)]) 
        new_behav.append(max(new_b))
    return new_behav

def plot(Mouse, cell, mode, k_word, polar, save, ax, **kwargs):
    if 'line_width' in kwargs:
        lnwd = kwargs['line_width']
    else:
        lnwd = 1.5  
    markers_amplitude = Mouse.spikes[Mouse.spikes[:,cell]!=0, cell]// 0.01
    markers_amplitude[markers_amplitude >= 10] = 10
    markers_amplitude = (np.around(markers_amplitude) + 5)**2

    if polar:  
        #Mouse.time = np.array(list(range(1,len(Mouse.angle)+1)))/20
        
        if mode == "None":
            pass
        
        elif mode == "Plain":
            ax.plot(Mouse.angle*np.pi/180, Mouse.time, color = 'black', linewidth=lnwd, zorder = 0.0)
        
        elif mode == "Speed":
            for x in range(len(Mouse.angle)-1):
                if Mouse.speed[x] < 0.1:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0.7, 1., 0.7), linewidth=lnwd, zorder = 0.0)
                elif 0.1 <= Mouse.speed[x] < 0.5:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0., 1., 0.), linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.angle[x]*np.pi/180, Mouse.angle[x+1]*np.pi/180],[Mouse.time[x], Mouse.time[x+1]], c = (0., 0.5, 0.), linewidth=lnwd, zorder = 0.0)
                                
        elif mode == "Direction":
            for x in range(1, len(Mouse.angle)):
                if Mouse.direction[x] >=0: 
                    ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], c = 'b', linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], c = 'c', linewidth=lnwd, zorder = 0.0)
                    
        elif mode == "Neuro":
            maxn = max(Mouse.neur[:, cell])
            minn = min(Mouse.neur[:, cell])
            for x in range(1, len(Mouse.angle)):
                color = cm.jet((Mouse.neur[x, cell]-minn)/(maxn-minn))
                ax.plot([Mouse.angle[x-1]*np.pi/180, Mouse.angle[x]*np.pi/180],[Mouse.time[x-1], Mouse.time[x]], c = color, linewidth=lnwd, zorder = 0.0) 
                
        elif mode == "Neuro_bin":
            Nbins = np.max(Mouse.neuro_bin[:,cell])
            for n_bin in range(Nbins):
                color = cm.jet(n_bin/Nbins)                
                ax.plot(Mouse.angle[Mouse.neuro_bin[:,cell] == n_bin]*np.pi/180, Mouse.time[Mouse.neuro_bin[:,cell] == n_bin], c = color, ls = ' ', ms=5, marker='.', zorder = 0.0) 
                 
        ax.scatter(Mouse.angle[Mouse.spikes[:,cell]!=0]*np.pi/180, Mouse.time[Mouse.spikes[:,cell]!=0], marker='^', s=markers_amplitude*lnwd,  edgecolors = 'black', c = (1., 0., 0.), zorder = 1.0)
    

    else:
        if mode == "None":
            ax.plot(list(Mouse.x), list(Mouse.y), color = 'black', linewidth=lnwd, zorder = 0.0)
        elif mode == "Speed":
            ms = max(Mouse.speed)
            for i in range(len(Mouse.x)-1):
                if Mouse.speed[i] < ms/3:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0., 0.5, 0.), linewidth=lnwd, zorder = 0.0)
                elif Mouse.speed[i] < 2*ms/3:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0., 1., 0.), linewidth=lnwd, zorder = 0.0)
                else:
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0.7, 1., 0.7), linewidth=lnwd, zorder = 0.0)
                                
        elif mode == "Acceleration":
            for i in range(1, len(Mouse.x)-1):
                meda = np.median(Mouse.acceleration)
                stda = np.std(Mouse.acceleration)
                if Mouse.acceleration[i] >= meda + 2*stda: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = 'm', linewidth=lnwd, zorder = 0.0)
                elif Mouse.acceleration[i] <= meda - 2*stda: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = 'c', linewidth=lnwd, zorder = 0.0)
                else: 
                    ax.plot([Mouse.x[i], Mouse.x[i+1]],[Mouse.y[i], Mouse.y[i+1]], c = (0.7, 0.7, 0.7), linewidth=lnwd, zorder = 0.0)  
                    
        ax.scatter(Mouse.x[Mouse.spikes[:,cell]!=0], Mouse.y[Mouse.spikes[:,cell]!=0], marker='^', s=markers_amplitude*1.5,  edgecolors = 'black', c = (1., 0., 0.), zorder = 1.0)


    if save:
        path = Mouse.path_track
        plt.savefig(path + '_' + str(cell) + '_plot_' + k_word + '.png', dpi = 300)
        plt.close()
		
def DrawPassMap(Mouse, out_fname):
	fig = plt.figure(figsize=(8,16), dpi=300)
	ax = fig.add_subplot(111)
	plt.setp(ax.get_yticklabels(), visible=False)
	plt.setp(ax.get_xticklabels(), visible=False)
	ind = np.argsort(Mouse.mu_list)
	for cell in range(len(Mouse.true_cells)): 
		for j in range(len(Mouse.times_in[ind[cell]])):
			wd = 100/len(Mouse.times_in[ind[cell]])
			ht = 100
			if Mouse.sp_intervals[ind[cell]][j]:
			     alp = 0.4
			else:
			     alp = 0.2
			if Mouse.direction[Mouse.times_in[ind[cell]][j]] == 1 and Mouse.direction[Mouse.times_out[ind[cell]][j]] == 1:
				ax.add_patch(patches.Rectangle((j*wd, cell*ht), wd, ht, facecolor = 'b', edgecolor = 'black', alpha=alp))
			elif Mouse.direction[Mouse.times_in[ind[cell]][j]] == -1 and Mouse.direction[Mouse.times_out[ind[cell]][j]] == -1:
				ax.add_patch(patches.Rectangle((j*wd, cell*ht), wd, ht, facecolor = 'c', edgecolor = 'black', alpha=alp))
			else:
				ax.add_patch(patches.Rectangle((j*wd, cell*ht), wd, ht, facecolor = 'm', edgecolor = 'black', alpha=alp))
			if np.count_nonzero(Mouse.spikes[Mouse.times_in[ind[cell]][j]:Mouse.times_out[ind[cell]][j],Mouse.true_cells[ind[cell]]]):
				 ax.scatter(j*wd + wd/2, cell*ht + ht/2, marker='^',  edgecolors = 'black', c = (1., 0., 0.), zorder = 1.0)
	plt.savefig(out_fname, dpi = 300)
	plt.close()
    
    

def DrawLdgMap(Mouse):
    fig = plt.figure(figsize=(10,5), dpi=300)
    tlen = int(len(Mouse.time)/10)
    n_cells = len(Mouse.spikes[0])
    bigmap = np.zeros(shape = (3*n_cells,tlen,3))
    msp = max(Mouse.speed)
    meda = np.median(Mouse.acceleration)
    stda = np.std(Mouse.acceleration)
    for t in range(tlen-2):    
        if np.mean(Mouse.speed[t*10:(t+1)*10] <msp/3):
                bigmap[0:n_cells-1, t:t+1] = [0, 0.5, 0]
        elif np.mean(Mouse.speed[t*10:(t+1)*10] <2*msp/3):
                bigmap[0:n_cells-1, t:t+1] = [0, 1, 0]
        else:
                bigmap[0:n_cells-1, t:t+1] = [0.7, 1, 0.7]
                
        if np.mean(Mouse.acceleration[t*10:(t+1)*10] >= meda + 2*stda):
                bigmap[n_cells:2*n_cells-1, t:t+1] = [1, 0, 1]
        elif np.mean(Mouse.acceleration[t*10:(t+1)*10] <= meda - 2*stda):
                bigmap[n_cells:2*n_cells-1, t:t+1] = [0, 1, 1]
        else:
                bigmap[n_cells:2*n_cells-1, t:t+1] = [0.7, 0.7, 0.7] 

        if np.mean(Mouse.behav[t*10:(t+1)*10] > 1):
                bigmap[2*n_cells:3*n_cells-1, t:t+1] = [1, 0.7, 0.2]
        elif np.mean(Mouse.behav[t*10:(t+1)*10] > 0):
                bigmap[2*n_cells:3*n_cells-1, t:t+1] = [1, 1, 0]
        else:
                bigmap[2*n_cells:3*n_cells-1, t:t+1] = [0.7, 0.7, 0.7]   
                           
        for cell in range(n_cells-1):
            if np.count_nonzero(Mouse.spikes[t*10:(t+1)*10, cell]):
                bigmap[cell*3+1, t:t+2] = [1, 0, 0]
                bigmap[cell*3:cell*3+2, t+1] = [1, 0, 0]
    plt.imshow(bigmap)
    plt.savefig(Mouse.path_track + '_ldg_map' + '.png', dpi = 300)


def DrawLdgInterestinPlaces(Mouse, out_fname):
    fig = plt.figure(figsize=(320,5), dpi=200)
#    ax1 = fig.add_subplot(111)
#    fig, (ax1,ax2) = plt.subplots(nrows = 2, figsize = (320,6))
    ax1 = plt.subplot(111)
#    ax2 = plt.subplot(211)
    tbin= 4
    ncells = len(Mouse.neur[0])
    nbins = int(len(Mouse.neur[:,0])/tbin) 
    neur_map = np.zeros((ncells,nbins-1))
    for cell in range(ncells):
        neur_map[cell,:] = np.array([Mouse.neur[i*4:(i+1)*4, cell].mean() for i in range(nbins-1)])
    sm_behav = np.array([max(Mouse.behav[i*4:(i+1)*4]) for i in range(nbins-1)])
    sm_speed = np.array([Mouse.speed[i*4:(i+1)*4].mean() for i in range(nbins-1)])
#    sm_accel = np.array([Mouse.acceleration[i*4:(i+1)*4].mean() for i in range(nbins-1)])

    for i in range(nbins-1):
        if sm_behav[i]:
            ax1.add_patch(patches.Rectangle((i, 0), 1, ncells, fc = 'black', ec = 'none', alpha=0.1*(sm_behav[i]-1)))
    ax1.imshow(neur_map, cmap='jet',interpolation = 'nearest')
    ax1.plot(np.linspace(0, nbins-2, num=nbins-1), -sm_speed, color = 'r', scalex = nbins-1, )
#    ax3.plot(np.linspace(0, nbins-2, num=nbins-1), sm_accel, color = 'r')
    plt.savefig(out_fname, bbox_inches='tight', dpi = 200, pad_inches=0)    

		
def DrawAllSuper(Mouse):
	Mouse.mu_list = np.delete(Mouse.mu_list, Mouse.false_cells).tolist()
	Mouse.std_list = np.delete(Mouse.std_list, Mouse.false_cells).tolist()	
	for i in range(len(Mouse.super_cells)):
		ax, fig = DrawField(Mouse.mu_list[i], Mouse.std_list[i], Mouse.t_spec[i]*0.05, len(Mouse.time)/20)
		plot(Mouse, Mouse.super_cells[i], mode='Direction', k_word = 'super', polar=True, save=True, ax = ax)

def DrawAllTrue(Mouse, k_word):
   for i in range(len(Mouse.true_cells)):
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(Mouse.name + ' cell_' + str(Mouse.true_cells[i]) + ' MI =' + str(Mouse.mi_score[i]), fontsize=16)
       ax = DrawField(Mouse.mu_list[i], Mouse.std_list[i], Mouse.time[0], Mouse.time[len(Mouse.angle)-1], ax = ax)
       plot(Mouse, Mouse.true_cells[i], mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 0.5)
       plt.savefig(k_word + '_' + str(i) + '_' + str(Mouse.true_cells[i]) + '_circ.png')
       plt.close()  
   
def DrawAllTruePlain(Mouse, k_word):
   for i in range(len(Mouse.true_cells)):
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar') 

       ax = DrawField(Mouse.mu_list[i], Mouse.std_list[i], Mouse.time[0], Mouse.time[len(Mouse.angle)-1], ax = ax)
       plot(Mouse, Mouse.true_cells[i], mode='Plain', k_word = k_word, polar=True, save=False, ax = ax, line_width = 0.5)
       plt.savefig(k_word + '_' + str(Mouse.true_cells[i]) + '_circ.png')  
       plt.close()


def PlotNeuroCirclesAllTrue(Mouse):
    for i in range(len(Mouse.true_cells)):
#        if np.count_nonzero(Mouse.spikes[:,i]) < 3:
#            continue;
        fig = plt.figure(figsize=(10,10), dpi=300)
        ax = fig.add_subplot(111, projection='polar')  
        ax = DrawField(Mouse.mu_list[i], Mouse.std_list[i], Mouse.time[0], Mouse.time[-1], ax = ax)
        plot(Mouse, Mouse.true_cells[i], mode='Neuro', k_word = '_neuro_circle_', polar=True, save=True, ax = ax, line_width = 0.5)   


def DrawBrickDistrib(Mouse):
    pass


def PlotAverageSpeed(path, files, step, fps):
    fig = plt.figure(figsize=(20,6), dpi=300)
    ax = fig.add_subplot(111)
    fi = 1
    for f in files:
        ms = pio.LoadMouse(path+f+'.npz')
        t = np.arange(step,ms.time[-1],step)
        scale_fact = 55/(max(ms.x)-min(ms.x))
        av_speed = []
        for i in range(len(t)):
            av_speed.append(np.mean(ms.speed[max(0,int((t[i]-step)*fps)):min(int((t[i]+step)*fps),len(ms.speed)-1)])*scale_fact)  
        ax.plot(t, av_speed, c =np.array(colornum_Metro(fi)))
        fi+=1

def DrawMostInformative(Mouse, k_word, Ncells):
   sorted_mi = np.argsort(Mouse.mi_score)[::-1][:Ncells]
   for i in range(Ncells):
       plt.clf()
       fig = plt.figure(figsize=(10,10), dpi=300)
       ax = fig.add_subplot(111, projection='polar') 
       fig.suptitle(ms.name + ' cell_' + str(sorted_mi[i]) + ' MI =' + str(Mouse.mi_score[sorted_mi[i]]), fontsize=16)
       
       plot(Mouse, sorted_mi[i], mode='Neuro_bin', k_word = k_word, polar=True, save=False, ax = ax, line_width = 0.5)
       plt.savefig(k_word + '_' + str(i) + '_' + str(sorted_mi[i]) + '_circ.png')  
       plt.close()    


'''
#DrawAllTrue(ca1_6)
#DrawAllSuper(ca1_6)	
trs = pio.LoadMice('C:\Work\LDG\LDG_neurop.npz') 
       
        
#fig = plt.figure(figsize=(8,6), dpi=50)
#ax = fig.add_subplot(111)
for i in [9,10,12,13]:
    Mouse = strct.Digo(trs[i].name, path_track = trs[i].path_track, path_spikes = trs[i].path_spikes, time_shift = trs[i].time_shift)
    Mouse.get_xy_data()
    Mouse.get_neuropil(trs[i].path_spikes.replace('spikes_',''))
    Mouse.get_behavior(glob.glob('C:\Work\LDG\segmentation\\trial'+str(i+1)+'*')[0], 16, 10)
    Traceviewer(Mouse,'C:\Work\LDG\LDG_tr_'+ str(i)+'_traces.png')
    trs[i] = Mouse
pio.SaveMice('C:\Work\LDG\LDG_neurop.npz',trs)
#    DrawLdgInterestinPlaces(trs[i], word = str(i))


#for i in range(len(ldg1.spikes[0])):
#    fig = plt.figure(figsize=(8,6), dpi=50)
#    ax = fig.add_subplot(111)
#    plot(ldg1, i, 'None', '_none_cell_'+ str(i), False, True, ax, line_width = 0.5)
#plot(ldg1, 1, 'Speed', '_speed_cell_1_', False, True, ax, line_width = 0.5)
#plot(ldg1, 21, 'Acceleration', '_accel_cell_1_', False, True, ax, line_width = 0.5)
#plot(trs[12], 35, 'None', '_none_cell_35_', False, True, ax, line_width = 0.5)		
'''
#bla = pio.LoadMouse('D:\VOVA\BLA\BLA_12_2D_tr 1.npz')
#DrawLdgInterestinPlaces(bla, 'C:\Work\BLA\BLA_12_2D_1tr_heatmap.png')
#Traceviewer(bla ,'D:\VOVA\BLA\BLA_12_2D_1tr_traces.png')
#file_name = "C:\Work\PLACE_CELLS\data\\behavior\CA1_10_1D_5min.xls"
#data = bplt.get_data(file_name)


#b_name = "C:\Work\PLACE_CELLS\data\\behavior\CA1_13_1D_5min.xls"
#TraceviewerPC(ms,'D:\VOVA\PLACE_CELLS\CA1_15_1D_traces_zones_decel_sorted.png')
#DrawPassMap(ms,'D:\VOVA\PLACE_CELLS\CA1_15_1D_passes_sorted.png')


#path = 'D:\VOVA\PLACE_CELLS\\'
#files = ['CA1_16_2D_nr','CA1_18_2D_sp','CA1_20_2D_sp','CA1_20_3D_sp', 'CA1_16_1D_main_nr', 'CA1_17_1D_main_nr','CA1_18_1D_main_nr','CA1_19_1D_main_nr','CA1_20_1D_main_nr','CA1_18_2D_nr','CA1_20_2D_nr','CA1_20_3D_nr']
#for f in files:
#    ms = pio.LoadMouse(path+f+'.npz')
#    ms.get_direction()
#    DrawAllTrue(ms, k_word = path + f)
#TraceviewerPC_horizontal(ms, 'D:\VOVA\PLACE_CELLS\CA1_16_1D_centroid_timed.csv.png')


#path = 'D:\VOVA\PLACE_CELLS\\'
#files = ['CA1_16_1D_main_nr','CA1_17_1D_main_nr','CA1_18_1D_main_nr','CA1_19_1D_main_nr','CA1_20_1D_main_nr','CA1_16_2D_nr', 'CA1_18_2D_nr', 'CA1_20_2D_nr', 'CA1_20_3D_nr']
#for f in files:
ms = pio.LoadMouse('C:\Work\CA1_22_1D_track.csv.npz')
ms.get_direction()
ms.get_binned_neuro(4)
ms.get_binned_angle(40)
ms.get_mi_score()
#ms = strct.GetFieldsNeuro(ms)
#DrawAllTrue(ms,'C:\Work\CA1_22_1D_neuro_pcs\CA1_22_1D_')

DrawMostInformative(ms, 'C:\Work\CA1_22_1D_40_4_neuro_sect\CA1_22_1D_', 100)
#DrawAllTruePlain(ms, 'E:\Sotskov\CA1_22\\1D\CA1_22_1D')
#    print(ms.time_shift)
#step = 60
#fps = 20

#PlotAverageSpeed(path, files, step, fps)
     
