#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:05:47 2021

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

allz1 = []
alltrialnumbers = []

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
    passes = pd.read_csv(path + '/' + name +'_vte.csv')
        
    pass_ep = nts.IntervalSet(start = passes['start'], end = passes['end'])
    
    file = [f for f in listdir if 'VTE_fwd' in f]
    vtedata = scipy.io.loadmat(os.path.join(filepath,file[0]))

    xpos = vtedata['x']
    ypos = vtedata['y']

    x1_py = vtedata['x1_py']
    y1_py = vtedata['y1_py']
    v_py = vtedata['v_py']
    r_py = vtedata['r_py']

    z1 = vtedata['z1']
    
    trialnumber = []
    
    for i in range(len(z1[0])):
        allz1.append(z1[0][i])
        trialnumber.append(i)
        alltrialnumbers.append(i)

    v = {}
    x = {}
    y = {}
    r = {}

    for i in range(size(v_py)): 
        v_array = []
        x_array = []
        y_array = []
        r_array = []
    
        for j in range(size(v_py[0][i])):
            v_array.append(v_py[0][i][j][0])
            x_array.append(x1_py[0][i][j][0])
            y_array.append(y1_py[0][i][j][0])
            r_array.append(r_py[0][i][j][0])
    
        v[i] = v_array
        x[i] = x_array
        y[i] = y_array
        r[i] = r_array

    vmax = [] 
    vmin = []

    for i in range(len(v)): 
        vmax.append(max(v[i]))
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
    passes = pd.read_csv(path + '/' + name +'_vte.csv')
        
    pass_ep = nts.IntervalSet(start = passes['start'], end = passes['end'])
    
    file = [f for f in listdir if 'VTE_fwd' in f]
    vtedata = scipy.io.loadmat(os.path.join(filepath,file[0]))

    xpos = vtedata['x']
    ypos = vtedata['y']

    x1_py = vtedata['x1_py']
    y1_py = vtedata['y1_py']
    v_py = vtedata['v_py']
    r_py = vtedata['r_py']

    z1 = vtedata['z1']
    
    trialnumber = []
    
    for i in range(len(z1[0])):
        allz1.append(z1[0][i])
        trialnumber.append(i)
        alltrialnumbers.append(i)

    v = {}
    x = {}
    y = {}
    r = {}

    for i in range(size(v_py)): 
        v_array = []
        x_array = []
        y_array = []
        r_array = []
    
        for j in range(size(v_py[0][i])):
            v_array.append(v_py[0][i][j][0])
            x_array.append(x1_py[0][i][j][0])
            y_array.append(y1_py[0][i][j][0])
            r_array.append(r_py[0][i][j][0])
    
        v[i] = v_array
        x[i] = x_array
        y[i] = y_array
        r[i] = r_array

    vmax = [] 
    vmin = []

    for i in range(len(v)): 
        vmax.append(max(v[i]))
        vmin.append(min(v[i]))

    miv = min(vmin)
    mav = max(vmax)    
      
    
    #Duration of trials

    dur = np.zeros(len(pass_ep))
    for i in pass_ep.index.values:
        dur[i] = (pass_ep.iloc[i]['end'] - pass_ep.iloc[i]['start']) / 1e6
    
    #select which cells you want to plot raster for 
    
    cells = {}
    
    for j in spikes.keys():
        spk = []
        count = 0
        for s in range(len(pass_ep)):
            t = spikes[j].index.values - pass_ep['start'][s]
            t2 = t[ (t > 0) & (t <= (dur[s])*1e6)]
            t3 = nts.Ts(t = t2).fillna(count).as_units('s')
            count += 1 
            spk.append(t3)
            
        cells[j] = spk
    
    n = len(z1)
    tmp = np.argsort(z1)
    desc = tmp[::-1][:n]
    
    for j in spikes.keys():
        plt.figure()
        plt.title('cell ' + str(j))
        for i in desc[0]:
            if len(cells[j][i]) > 0:
                plt.plot(cells[j][i],'|', color = 'k')
                plt.plot(cells[j][i].index.values[-1] + 1e-3, cells[j][i].values[-1],'o', color = 'r')
       
            