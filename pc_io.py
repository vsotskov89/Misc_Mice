# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:28:11 2018

@author: VOVA
"""
import numpy as np
import pandas as pd

def SaveMice(path, mice_list):
    np.savez(path, mice_list)
    
def LoadMice(path_npz):
    mc = np.load(path_npz, allow_pickle=True)
    return mc['arr_0']

def SaveMouse(path_npz, Mouse):
    np.savez(path_npz,[Mouse])
    
def LoadMouse(path_npz):
    mc = np.load(path_npz, allow_pickle=True)
    return mc['arr_0'][0]

def Get_spiking_num(Mouse, lim):
	j=0
	for i in range(len(Mouse.spikes[0])):
		if np.count_nonzero(Mouse.spikes[:,i])>=lim:
			j+=1
	return j

def Get_spiking_cells(Mouse, lim):
	j=[]
	for i in range(len(Mouse.spikes[0])):
		if np.count_nonzero(Mouse.spikes[:,i])>=lim:
			j.append(i)
	return j

def ExportSuperCells(Mouse, outfname):
    pd.DataFrame(Mouse.neur[:,Mouse.super_cells]).to_csv(outfname, header = Mouse.super_cells)


def ExportTrack(Mouse, outfname):
    pd.DataFrame(np.transpose([Mouse.time, Mouse.x, Mouse.y])).to_csv(outfname, header = ['time_s','x','y'])

def ExportRears(Mouse, outfname):
    pd.DataFrame(np.transpose([Mouse.time, Mouse.rear])).to_csv(outfname, header = ['time_s','rear_1_outward_-1_inward'])

def ExportCoordAndTraces(Mouse, outfname):
    n_cells = len(Mouse.neur[0,:])
    pd.DataFrame(np.transpose([Mouse.time, Mouse.x, Mouse.y, *np.transpose(Mouse.neur)])).to_csv(outfname, header = ['time_s','x','y', *np.linspace(0,n_cells-1,n_cells)])

def ExportSpikes(Mouse, outfname):
    n_cells = len(Mouse.neur[0,:])
    pd.DataFrame(np.transpose([Mouse.time, *np.transpose(Mouse.spikes)])).to_csv(outfname, header = ['time_s', *np.linspace(0,n_cells-1,n_cells)])


def Export_All_Initial_Data(Mouse, outfname):
    #realy all data, including markers
    n_cells = len(Mouse.neur[0,:])
    pd.DataFrame(np.transpose([Mouse.time,
                               Mouse.x, 
                               Mouse.y, 
                               *Mouse.markers,
                               *np.transpose(Mouse.neur)])).to_csv(outfname, header = ['time_s',
                                                                              'x',
                                                                              'y', 
                                                                              'x_green',
                                                                              'y_green',
                                                                              'x_red',
                                                                              'y_red',                                                                              
                                                                              *np.linspace(0,n_cells-1,n_cells)])

def Export_Track_and_Markers(Mouse, outfname):
    pd.DataFrame(np.transpose([Mouse.time,
                               Mouse.x, 
                               Mouse.y, 
                               *Mouse.markers])).to_csv(outfname, header = ['time_s',
                                                                              'x',
                                                                              'y', 
                                                                              'x_green',
                                                                              'y_green',
                                                                              'x_red',
                                                                              'y_red']) 

def Export_Neurodata(Mouse, outfname):
    n_cells = len(Mouse.neur[0,:])
    pd.DataFrame(np.transpose([Mouse.time,
                               *np.transpose(Mouse.neur)])).to_csv(outfname, header = ['time_s',
                                                                                       *np.linspace(0,n_cells-1,n_cells)])                                                                                    
                                                                                       
'''
path = 'C:\Work\LDG\LDG1_TR'
ms = []
for i in range(14):
    ms.append(LoadMouse(path+str(i+1)+'.npz'))
SaveMice(path+'_all.npz', ms)

#list, not dictionary!!!
'''