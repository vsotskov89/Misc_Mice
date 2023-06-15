# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 16:51:12 2018
@author: VVP
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, random
import scipy.ndimage as sf
import scipy.stats as st
from copy import copy
from scipy.signal import argrelextrema
import matplotlib.patches as patches

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

def refine_neuro(std_list,mu_list, bin_neuro, pik_spikes):
    std_list = [std_list[i] for i in range(len(mu_list)) if mu_list[i] >= 30 and mu_list[i]<= 390]
    mu_list = [mu_list[i] for i in range(len(mu_list)) if mu_list[i] >= 30 and mu_list[i]<= 390]
    mu_list = [mu_list[i] for i in range(len(std_list)) if std_list[i] <=90]
    std_list = [std_list[i] for i in range(len(std_list)) if std_list[i] <=90]
    for i in range(len(mu_list)):
        mu_list[i] = mu_list[i]%360
    noize = [bin_neuro[i] for i in range(len(bin_neuro)) if pik_spikes[i] ==0]
    if sum(noize) > 0.25*len(noize):
        std_list = []
        mu_list = []
    return std_list, mu_list


def analfitAppend(spikes,angle):
    spikess = copy(spikes)
    for ang in range(len(spikess)):
        if int(spikess[ang]) < angle:
            spikess.append(spikess[ang]+360)
    return spikess[:]

def AnalExtension(arr, nbins):
    #extends given array by nbins at both sides
    new_arr = np.zeros(len(arr)+2*nbins)
    new_arr[0:nbins-1] = arr[len(arr)-nbins:-1]
    new_arr[nbins:len(arr)+nbins-1] = arr[0:len(arr)-1]
    new_arr[len(arr)+nbins:len(arr)+2*nbins-1] = arr[0:nbins-1]
    return new_arr
    
    
    
    
def PlotHist(spikes,binss,mu,std):
    plt.clf()
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111)
    ax.hist(spikes, bins=binss, density=True, alpha=0.6, color='g', zorder = 0.5)
    xmin, xmax = plt.xlim()
    x = np.linspace(0, 360, 1000)
    for i in range(len(mu)):
        p = st.norm.pdf(x, mu[i], std[i]/2) + st.norm.pdf(x, mu[i] - 360, std[i]/2) + st.norm.pdf(x, mu[i] + 360, std[i]/2)
        ax.plot(x, p, 'k', linewidth=2, zorder = 1)
        ax.add_patch(patches.Rectangle((mu[i] - std[i]/1.7, 0), std[i]/0.85, max(p)/2, facecolor = [0.7,0.7,0.7], edgecolor = 'None', zorder = 0))
    title = "Fit results: mu = %s,  std = %s" % (mu, std)
    plt.title(title)
    plt.show()
    plt.savefig('D:\Work\PLACE_CELLS\\test.png') 
    plt.close(fig)
#    plt.clf()
    
def PlotFields(bin_neuro,realbin,mu_list,std_list):
    bins = np.linspace(0, 360, realbin)
    x = np.linspace(0, 360, 1000)
    plt.plot(bins, bin_neuro, color='green',linewidth=2)
    for i in range(len(mu_list)):
        p = st.norm.pdf(x, mu_list[i], std_list[i])
        plt.plot(x, p/max(p), 'k', linewidth=2)
    title = "Fit results: mu = %s,  std = %s" % (mu_list, std_list)
    plt.title(title)
    plt.show()
    plt.clf()

def PlotHist2(spikes,binss,ydata):

    xmin, xmax = plt.xlim()
    xdata = [x for x in range(0, binss)]
    plt.plot(xdata, ydata, 'k', linewidth=2)
    plt.show()
    plt.clf()

def IsInZone(ang, alpha, d_alpha): 
    for super_ang in [ang, ang - 360, ang +360]:
        if alpha-d_alpha/2 <= super_ang <= alpha+d_alpha/2:
            return True
    return False

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

    PlotHist(spikes,realbin,mu_list,std_list)
    return mu_list, std_list



def BinRawNeuro(angles, neurodata, Nbin):
    bin_neuro = np.zeros(Nbin)
    t_spent = np.zeros(Nbin)
    for i in range(len(angles)):
        b = int(angles[i]*Nbin/360)
        bin_neuro[b] += neurodata[i]
        t_spent[b] += 1
    return bin_neuro/t_spent


def CellFieldsNeuro(bin_neuro ,sigma, bin_len, realbin):
    norm_bin_neuro = (bin_neuro-min(bin_neuro))/(max(bin_neuro)-min(bin_neuro))
    bin_neuro_append = AnalExtension(bin_neuro.tolist(), bin_len)
    smooth_spikes = sf.filters.gaussian_filter1d(bin_neuro_append, sigma, order=0, mode='reflect')
    norm_spikes = (smooth_spikes-min(smooth_spikes))/(max(smooth_spikes)-min(smooth_spikes))
    pik_spikes = (norm_spikes>0.9)*norm_spikes
    mu_list, std_list = FindPlace(pik_spikes[bin_len:realbin+bin_len], realbin)
    std_list, mu_list = refine_neuro(std_list,mu_list, norm_bin_neuro, pik_spikes[bin_len:realbin+bin_len])
    PlotFields(norm_bin_neuro,realbin,mu_list,std_list)
    return mu_list, std_list

