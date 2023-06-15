# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:41:17 2021

@author: vsots
"""

import numpy as np
import pyinform as pif
import matplotlib.pyplot as plt

def Binarize_Data(data, n_bins, mode = 'equipopular'):
    #Binarizes 2D array in different modes (equal_width, equipopular)
    data2 = np.zeros(np.shape(data))
    for i in range(1, np.size(data,1)):
        tr = data[:,i]
        
        if mode == 'equipopular':
            quant = np.linspace(0, 1, n_bins+1)
            qa = np.quantile(tr, quant)
            
        elif mode == 'equal_width':
            qa = np.linspace(np.min(tr), np.max(tr), n_bins+1)
            
        for b in range(1, n_bins+1):
            mask = np.zeros(np.size(data,0))
            mask[tr >= qa[b-1]] = 1
            mask[tr > qa[b]] = 0
            data2[mask == 1,i] = b-1
            
    return data2


def Adj_Matrix_TE(filename, n_bins = 6, mode = 'equipopular'):
    #Calculaces adjasency matrix from OASYS data contained in filename
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    data = Binarize_Data(data, n_bins, mode)
    
    n_sources = np.size(data,1) - 1
    TE = np.zeros((n_sources, n_sources))
    
    for i in range(n_sources):
        for j in range(n_sources):
            TE[i,j] = pif.transfer_entropy(data[:,i+1], data[:,j+1], k = 2)
            
    return TE


def Plot_Mean_TE_Of_Nbins(filename, max_nbins = 15, mode = 'equipopular'):
    #Plots the mean value of adjasency matrix as function of n_bins
    mean_te = []
    for n_bins in range(1,max_nbins):
        TE = Adj_Matrix_TE(filename, n_bins, mode)
        mean_te.append(np.mean(TE))
    plt.plot(mean_te)  
    

path = 'D:\Work\\Sirius_entropy\\ts_extracted\\controls\\AAL\\'
fname = 'sub-OAS30001_ses-d0129_task-rest_run-2_atlas-AAL.csv'

Plot_Mean_TE_Of_Nbins(path + fname, max_nbins = 15, mode = 'equipopular')
Plot_Mean_TE_Of_Nbins(path + fname, max_nbins = 15, mode = 'equal_width')
plt.legend(['equipopular','equal_width'])



  
    