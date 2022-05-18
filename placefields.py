#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:39:50 2022

@author: dhruv
"""
import numpy as np
import pandas as pd
import scipy.io
from pylab import *
import os, sys
import pynapple as nap
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.backends.backend_pdf import PdfPages    

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import filtfilt
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data)
	return y


# data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
data_directory = '/media/DataDhruv/Recordings/B1600/B1603'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_CA1.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    name = s.split('/')[-1]
    print(name)
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
          
    spikes = data.spikes   
    epochs = data.epochs
    position = data.position
    
    lfp = nap.load_eeg(path + '/' + name + '.eeg' , channel = 1 , n_channels = 32, frequency = 1250, precision ='int16', bytes_size = 2) 
    lfp_filt_theta = nap.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250, 2))
    power_theta = nap.Tsd(lfp_filt_theta.index.values, np.abs(hilbert(lfp_filt_theta.values)))
    power_theta = power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)    
    
    
    spikes_by_location = spikes.getby_category('location')
    spikes_ca1 = spikes_by_location['CA1']
    ca1_neurons = pd.DataFrame(data = spikes_ca1.keys())
    
    if name == 'B1603-220305':
    
        position_spike = {}
        
        tcurves2d_free, binsxy_free = nap.compute_2d_tuning_curves(group = spikes_ca1, feature = position[['x', 'z']], ep = position.time_support.loc[[0]] , nb_bins=20)                                               
        tcurves2d_rand, binsxy_rand = nap.compute_2d_tuning_curves(group = spikes_ca1, feature = position[['x', 'z']], ep = position.time_support.loc[[1]] , nb_bins=20)                                               
        tcurves2d_OF, binsxy_OF = nap.compute_2d_tuning_curves(group = spikes_ca1, feature = position[['x', 'z']], ep = position.time_support.loc[[2]] , nb_bins=20)                                               
       
        for i in spikes_ca1.keys(): 
            position_spike[i] = position.realign(spikes[i].restrict(epochs['wake'].loc[[0]]))
            
            tcurves2d_free[i][np.isnan(tcurves2d_free[i])] = 0
            tcurves2d_free[i] = scipy.ndimage.gaussian_filter(tcurves2d_free[i], 1)
            
            tcurves2d_rand[i][np.isnan(tcurves2d_rand[i])] = 0
            tcurves2d_rand[i] = scipy.ndimage.gaussian_filter(tcurves2d_rand[i], 1)
            
            tcurves2d_OF[i][np.isnan(tcurves2d_OF[i])] = 0
            tcurves2d_OF[i] = scipy.ndimage.gaussian_filter(tcurves2d_OF[i], 1)
            
            
                                    
        plt.figure()
        for n in range(len(ca1_neurons)):
            plt.suptitle('Free alternation')
            plt.subplot(7,6,n+1)
            # plt.plot(position['x'].restrict(epochs['wake'].loc[[0]]), position['z'].restrict(epochs['wake'].loc[[0]]), color = 'grey', zorder = 1)
            # plt.scatter(position_spike[n]['x'], position_spike[n]['z'], zorder = 2, s = 1)
            plt.imshow(tcurves2d_free[ca1_neurons.values[n][0]], extent=(binsxy_free[1][0],binsxy_free[1][-1],binsxy_free[0][0],binsxy_free[0][-1]), cmap = 'jet')        
            plt.colorbar()
            plt.subplot_tool()
                     
        plt.figure()
        for n in range(len(ca1_neurons)):
              plt.suptitle('Random exploration')
              plt.subplot(7,6,n+1)
              plt.imshow(tcurves2d_rand[ca1_neurons.values[n][0]], extent=(binsxy_rand[1][0],binsxy_rand[1][-1],binsxy_rand[0][0],binsxy_rand[0][-1]), cmap = 'jet')        
              plt.colorbar()
              plt.subplot_tool()
              
     
        plt.figure()
        for n in range(len(ca1_neurons)):
              plt.suptitle('Open field')
              plt.subplot(7,6,n+1)
              plt.imshow(tcurves2d_OF[ca1_neurons.values[n][0]], extent=(binsxy_OF[1][0],binsxy_OF[1][-1],binsxy_OF[0][0],binsxy_OF[0][-1]), cmap = 'jet')        
              plt.colorbar()
              plt.subplot_tool()
              
     
        for n in range(len(ca1_neurons)):
              plt.figure()
              plt.suptitle('Cell ' + str(n))
              plt.subplot(1,3,1)
              plt.title('Free')
              tmp = imshow(tcurves2d_free[ca1_neurons.values[n][0]], extent=(binsxy_free[1][0],binsxy_free[1][-1],binsxy_free[0][0],binsxy_free[0][-1]), cmap = 'jet')
              plt.colorbar(tmp,fraction=0.046, pad=0.04)
                      
              plt.subplot(1,3,2)
              plt.title('Random')
              tmp = imshow(tcurves2d_rand[ca1_neurons.values[n][0]], extent=(binsxy_rand[1][0],binsxy_rand[1][-1],binsxy_rand[0][0],binsxy_rand[0][-1]), cmap = 'jet')        
              plt.colorbar(tmp,fraction=0.046, pad=0.04)
             
              plt.subplot(1,3,3)
              plt.title('OF')
              tmp = imshow(tcurves2d_OF[ca1_neurons.values[n][0]], extent=(binsxy_OF[1][0],binsxy_OF[1][-1],binsxy_OF[0][0],binsxy_OF[0][-1]), cmap = 'jet')        
              plt.colorbar(tmp,fraction=0.046, pad=0.04)
              plt.tight_layout()
            
     
        multipage(data_directory + '/' + 'Allcells.pdf', dpi=250)
    
    
    # plt.figure()
    # plt.title('LFP trace')
    # plt.plot(lfp.restrict(position.time_support.loc[[0]]))
    # plt.plot(lfp_filt_theta.restrict(position.time_support.loc[[0]]), color = 'orange')
    
    
    # fig, ax = plt.subplots()
    # [plot(spikes[n].restrict(position.time_support.loc[[0]]).as_units('s').fillna(n), '|') for n in spikes.keys()]
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)