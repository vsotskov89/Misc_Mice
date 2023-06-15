# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:16:46 2022

@author: vsots
"""

from datetime import datetime
import scipy.io as sio
import re
import os
import h5py
import csv
import tensorflow as tf
import time
import logging
import zipfile
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import numpy as np
from moviepy.editor import *
import smtplib
from glob import glob
import cowsay

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import peakutils
   
#%%Motion correction
# dataset dependent parameters
frate = 30                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

# motion correction parameters
motion_correct = True    # flag for performing motion correction
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (5, 5)       # size of high pass spatial filtering, used in 1p data
max_shifts = (10, 10)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 5  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries
use_cuda = True          # Set to True in order to use GPU
only_init_patch = True
memory_fact = 1          # How much memory to allocate. 1 works for 16Gb, so 0.8 show be optimized for 12Gb.

mc_dict = {
    #'fnames': fnames,
    'fr': frate,
    'niter_rig': 1,
    'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
    # if none all the splits are processed and the movie is saved
    'num_splits_to_process_rig': None, # intervals at which patches are laid out for motion correction            
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan,
    'use_cuda' : use_cuda,
    'only_init_patch' : only_init_patch,
    'memory_fact': memory_fact
}

# Parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = (5, 5)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (15, 15)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .85     # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 1            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 2             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .8      # min peak value from correlation image
min_pnr = 8        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

cnmf_dict = {'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'memory_fact': memory_fact,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': 3}                       # number of pixels to not consider in the borders)

min_SNR = 2             # adaptive way to set threshold on the transient size
r_values_min = 0.5      # threshold on space consistency (if you lower more components will be accepted, potentially with worst quality)               

#%% filenames parsing

root = 'F:\\VOVA\\FAD_mice\\'
timestamps = glob(root + '*\\*\\*\\*\\Miniscope\\timeStamps.csv')
video_fnames = []

for tm in timestamps:
    video_fnames.append(glob(os.path.dirname(tm) + '\\*.avi'))
    
    opts = params.CNMFParams(params_dict=mc_dict)
    try:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    except:
        pass
        
    start = time.time() # This is to keep track of how long the analysis is running
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),np.max(np.abs(mc.y_shifts_els)))).astype(np.int) if pw_rigid else  bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        bord_px = 0 if border_nan is 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)
    else:
        fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C', border_to_0=0, dview=dview)    
    
    print(f'{time.time() - start:.1f}s\t{tm}\tmotion corrected (not) done, mapped to memory')

    opts = params.CNMFParams(params_dict=cnmf_dict) 
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    #restart server
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    # Compute some summary images (correlation and peak to noise) while downsampling temporally 5x to speedup the process and avoid memory overflow
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    true_ind = cnm.estimates.idx_components

    print(f'{time.time() - start:.1f}s\t{tm}\tcnmf done')
    print(f'Accepted {len(true_ind)} of {len(cnm.estimates.C)} components')




    
    



#c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
#fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C', border_to_0=0, dview=dview)

#%% restart cluster to clean up memory
plt.plot(mc.shifts_rig)
#to plot: cn_filter, pnr
#Plot the results of the correlation/PNR projection
plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1); plt.imshow(cn_filter); plt.colorbar(); plt.title('Correlation projection')
plt.subplot(2, 2, 2); plt.imshow(pnr); plt.colorbar(); plt.title('PNR')



#%% plot traces
%matplotlib inline

#How many neurons to plot
neuronsToPlot = 25
tlen = 5000

true_ind = cnm.estimates.idx_components
DeconvTraces = cnm.estimates.S[true_ind]
RawTraces = cnm.estimates.C[true_ind]
SFP = cnm.estimates.A[:,true_ind]
SFP_dims = list(dims)
SFP_dims.append(SFP.shape[1]) 
print('Spatial foootprints dimensions (height x width x neurons): ' + str(SFP_dims))

numNeurons = SFP_dims[2]

SFP_resh = np.reshape(SFP.toarray(), SFP_dims, order='F')

maxRawTraces = np.amax(RawTraces)

plt.figure(figsize=(30,15))
plt.subplot(341);
plt.subplot(345); #plt.plot(mc.shifts_rig); plt.title('Motion corrected shifts')
plt.subplot(3,4,9);
plt.subplot(3,4,2); plt.imshow(cn_filter); plt.colorbar(); plt.title('Correlation projection')
plt.subplot(3,4,6); plt.imshow(pnr); plt.colorbar(); plt.title('PNR')
plt.subplot(3,4,10); plt.imshow(np.amax(SFP_resh,axis=2)); plt.colorbar(); plt.title('Spatial footprints')

plt.subplot(2,2,2); plt.figure; plt.title('Example traces (first 50 cells)')
plot_gain = 5 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
  for i in range(neuronsToPlot):
    if i == 0:
      plt.plot(RawTraces[i,:tlen],'k')
    else:
      trace = RawTraces[i,:tlen] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')
else:
  for i in range(numNeurons):
    if i == 0:
      plt.plot(RawTraces[i,:tlen],'k')
    else:
      trace = RawTraces[i,:tlen] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')

plt.subplot(2,2,4); plt.figure; plt.title('Deconvolved traces (first 50 cells)')
plot_gain = 20 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
  for i in range(neuronsToPlot):
    if i == 0:
      plt.plot(DeconvTraces[i,:tlen],'k')
    else:
      trace = DeconvTraces[i,:tlen] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')
else:
  for i in range(numNeurons):
    if i == 0:
      plt.plot(DeconvTraces[i,:tlen],'k')
    else:
      trace = DeconvTraces[i,:tlen] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k') 
      
#%% Export filters and traces as .mat file
from scipy.io import savemat

mdic = {"sigfn": RawTraces, "spkfn": DeconvTraces, "pixh": SFP_dims[0], "pixw": SFP_dims[1], "imax": pnr, "roifn": SFP.todense()}
#traces:n_cells X n_frames, roifn: n_pixels X n_cells

savemat(fnames[0][:-4] + '_cnmf_results.mat', mdic)

cm.stop_server(dview=dview)
dview.terminate()
  