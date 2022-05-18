#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:44:26 2021

@author: dhruv
"""
import numpy as np
import pandas as pd
import scipy.io
import neuroseries as nts
from pylab import *
import os, sys
from wrappers import loadSpikeData
from wrappers import loadXML
from wrappers import loadPosition
from wrappers import loadEpoch
from functions import *
import sys
import matplotlib.pyplot as plt

data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    name = s.split('/')[-1]
    print(name)
    path = os.path.join(data_directory, s)
    
    files = os.listdir(data_directory) 
    episodes = ['sleep', 'wake', 'sleep','wake', 'sleep']
    events = ['1','3']
   
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(path)
    position = loadPosition(path, events, episodes)
    
    wake_ep = loadEpoch(path, 'wake', episodes)
    sleep_ep = loadEpoch(path, 'sleep')   

    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'trials' in f]
    freetrials = pd.read_csv(os.path.join(filepath,file[1]))
    
    fwdpass = nts.IntervalSet(start = freetrials['fw_dep_start'], end = freetrials['fw_dep_end'])
    freepos = position.restrict(fwdpass)
    
    spatial_curves, extent = computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[1]], 40)
    
    plt.figure()
    for i in spikes:
        plt.title('Spatial tuning')
        plt.subplot(7,6,i+1)
        tmp = scipy.ndimage.gaussian_filter(spatial_curves[i].values, 1)
        plt.imshow(tmp, extent = extent, interpolation = 'bilinear', cmap = 'jet')
        plt.colorbar()
        plt.subplots_adjust(wspace=0.2, hspace=1, top = 0.85)
        
    dur = np.zeros(len(fwdpass))
    for i in fwdpass.index.values:
        dur[i] = (fwdpass.iloc[i]['end'] - fwdpass.iloc[i]['start']) / 1e6
        
    count = 0 
    for s in fwdpass['start'].values:
        for j in spikes.keys():
            t = spikes[j].index.values - s
            t2 = t[(t >= -5e6) & (t <= 5e6)]
            t3 = t2.fillna(count)
        
        