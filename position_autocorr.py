#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:07:50 2022

@author: dhruv
"""
import numpy as np
import pandas as pd
import scipy.io
from pylab import *
import os, sys
import pynapple as nap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages    

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    

#data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
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
    
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'trials' in f]
    freetrials = pd.read_csv(os.path.join(filepath,file[1]))
    
    fwdpass = nap.IntervalSet(start = freetrials['fw_dep_start'], end = freetrials['fw_dep_end'], time_units= 'us')
    revpass = nap.IntervalSet(start = freetrials['bw_arm_start'], end = freetrials['bw_arm_end'], time_units = 'us')
    
    neuron_location = spikes.get_info('location')
    spikes_by_location = spikes.getby_category('location')
    spikes_adn = spikes_by_location['ADn']
    spikes_ca1 = spikes_by_location['CA1']
    
      
    cc_choice = nap.compute_autocorrelogram(group=spikes_ca1, 
                                       ep=fwdpass, 
                                       binsize=5, # ms
                                       windowsize=500, # ms
                                       norm=True)
    
    cc_rev = nap.compute_autocorrelogram(group=spikes_ca1, 
                                       ep=revpass, 
                                       binsize=5, # ms
                                       windowsize=500, # ms
                                       norm=True)
    
    xtchoice = cc_choice.index.values
    xtrev = cc_rev.index.values
    
    fig = plt.figure(figsize = (12, 9))
    fig.suptitle(name)      
    
    for i,n in enumerate(cc_choice.columns):
        plt.subplot(8,5,i+1)
        plt.plot(xtchoice,cc_choice[n], label = 'choice')
        plt.plot(xtrev, cc_rev[n], label = 'rev')
        plt.title(neuron_location[n] + '-' + str(n), fontsize = 12)
        plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        handles, labels = fig.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'upper right')
    
multipage(data_directory + '/' + 'position_autocorr.pdf', dpi=250)


    
    
    
   
    
