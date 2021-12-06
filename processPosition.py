#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:28:06 2021

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

fwdtot = []
revtot = []
meandur_rev = []

session = ['A2908-190611', 'A2913-190710', 'A2925-200806', 'A2926-200312', 'A2929-200711', 'A2930-201015']
xparam = [0.06, 0.065, 0.03, 0.055, 0.03, 0.03]
yparam = [0.07, 0.085, 0.2, 0.07, 0.2, 0.2]
rth = 0.2

sessdata = pd.DataFrame(index = ['x', 'y'], columns = session, data = [xparam, yparam])
sessdata = sessdata.T

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
       
    for j in sessdata.index: 
        if j == name: 
                xth = sessdata.loc[j,'x']
                yth = sessdata.loc[j,'y']
                
                print(xth, yth)
                
                circle1 = plt.Circle((xth, yth), rth, color='k', fill = False)
                
                freepos = position.restrict(wake_ep.iloc[[1]])
                d = np.sqrt((freepos['x'].values - xth)**2 + (freepos['z'].values - yth)**2)
                dy = freepos['z'].values - yth
                dint = pd.DataFrame(index = freepos.index.values, columns = ['dist'], data = d)
                dint['dy'] = dy
                r1 = dint[dint['dist'] < rth]
                
                start = [r1.index[0]]
                stvals = [r1['dy'].values[0]]
                end = []
                evals = []
                
                for i in range(len(r1)): 
                    if (r1.index.values[i] - r1.index.values[i-1]) > 1e6:
                        end.append(r1.index.values[i-1])
                        start.append(r1.index.values[i])
                        stvals.append(r1['dy'].values[i])
                        evals.append(r1['dy'].values[i-1])
                        
                        
                start = np.array(start[0:-1])
                stvals = np.array(stvals[0:-1])
                end = np.array(end)
                evals = np.array(evals)
                
                tokeep = []
                
                for i in range(len(stvals)):
                    if (stvals[i] < 0 and evals[i] > 0):  #forward trials only 
                      # if (stvals[i] > 0 and evals[i] < 0) :
                        tokeep.append(i)
                       
                # plt.figure()
                # plt.plot(r1['dy'],'o-')
                # plt.plot(start[tokeep], stvals[tokeep], "x")
                # plt.plot(end[tokeep], evals[tokeep], "*")
                # plt.show()
                
    fwd = nts.IntervalSet(start = start[tokeep], end = end[tokeep])
    # rev = nts.IntervalSet(start = start[tokeep], end = end[tokeep])    
        
    plt.figure()
    plt.plot(position['x'],position['z'],'silver', zorder = 1)
    plt.scatter(position['x'].restrict(fwd), position['z'].restrict(fwd), zorder = 2)
    # plt.scatter(position['x'].restrict(rev), position['z'].restrict(rev), zorder = 2)
    plt.gca().add_patch(circle1)
       
    fwd = (fwd - wake_ep.iloc[1]['start']) / 1e6
    # rev = (rev - wake_ep.iloc[1]['start']) / 1e6
    
    fwd.to_csv(path + '/' + name + '_fwd.csv', index=False)    
    # rev.to_csv(path + '/' + name + '_rev.csv', index=False)
           
               
               
            