#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:31:53 2022

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
    
def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
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
    lfp = lfp.restrict(epochs['sleep'].loc[0])
    
    frequency = 1250.0
    low_cut = 100
    high_cut = 300
    windowLength = 51
    low_thresFactor = 1.5
    high_thresFactor = 2
    minRipLen = 10 # ms
    maxRipLen = 200 # ms
    minInterRippleInterval = 20 # ms
    limit_peak = 20


signal = butter_bandpass_filter(lfp.values, low_cut, high_cut, frequency, order = 4)

squared_signal = np.square(signal)

window = np.ones(windowLength)/windowLength

nSS = scipy.signal.filtfilt(window, 1, squared_signal)



x = np.linspace(0, 1, 1250)
y = np.sin(2*np.pi*8*x) +  np.random.normal(scale=0.1, size=len(x))
z = butter_bandpass_filter(y, 4, 12, 1250, 5)

y2 = np.sin(2*np.pi*8*x) + np.sin(2*np.pi*50*x)    
z2 = butter_bandpass_filter(y2, 4, 12, 1250, 5)

y3 = np.sin(2*np.pi*8*x) + np.sin(2*np.pi*2*x)    
z3 = butter_bandpass_filter(y3, 4, 12, 1250, 5)


plt.subplot(311)
plt.plot(x,y)
plt.plot(x,z)
plt.subplot(312)
plt.plot(x,y2)
plt.plot(x,z2)
plt.subplot(313)
plt.plot(x,y3)
plt.plot(x,z3)