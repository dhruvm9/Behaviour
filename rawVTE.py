#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:12:07 2021

@author: dhruv
"""
import numpy as np
import pandas as pd
import neuroseries as nts
import scipy.io
import scipy.stats
from pylab import *
import os
from wrappers import loadSpikeData
from wrappers import loadXML
from wrappers import loadPosition
from wrappers import loadEpoch
from functions import *
from matplotlib.colors import hsv_to_rgb
import sys
from functions import computeAngularVelocityTuningCurves
import seaborn as sns 
from scipy.stats import pearsonr 

data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

BIGGER_SIZE = 14

allz1 = []
alltrialnumbers = []

for s in datasets:
    name = s.split('/')[-1]
    print(name)
    path = os.path.join(data_directory, s)
    
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
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
    normalize = matplotlib.colors.Normalize(vmin=miv, vmax=mav)
    
    
    for i in range(len(v)):
        if z1[0][i] > 0.5:
            plt.figure()
            plt.axis('off')
            plt.plot(xpos,ypos,'silver', zorder = 1)
            plt.scatter(x[i],y[i], c = v[i], zorder = 2,label = 'zIdPhi =' + str(round(z1[0][i],ndigits = 4)),norm = normalize, cmap = 'hot')
            plt.colorbar(label = 'velocity (cm/s)')
            plt.rc('font', size=BIGGER_SIZE)  
            plt.legend(loc = 'upper right')

#     corr, pvalue = pearsonr(trialnumber,z1[0].T)    
#     plt.figure()
#     plt.title('Correlation between trial number and zIdPhi_' + s)
#     plt.xlabel('Trial number')
#     plt.ylabel('zIdPhi')
#     plt.axhline(0.5,color = 'k')
#     plt.rc('font', size=BIGGER_SIZE)          
#     plt.scatter(trialnumber,z1,label = 'Pearson R =  ' +  str(round(corr,4)))
#     plt.legend(loc = 'upper right')

    

# bins = np.arange(round(min(allz1),1),round(max(allz1),1)+0.1,0.1)
# plt.figure()
# sns.histplot(data = allz1, bins = bins)
# plt.title('zIdPhi Histogram for all sessions')
# plt.xticks(bins[::5])
# plt.xlabel('zIdPhi value')
# plt.rc('font', size=BIGGER_SIZE)  
# plt.axvline(0.5, c = 'r', label = 'VTE threshold', linewidth=4)
# plt.legend(loc = 'upper right')

# a = np.histogram(allz1, bins)[0]
# cumpercentage = np.cumsum(a)/np.sum(a)*100
# absval = np.abs(bins - 0.5)
# closest = absval.argmin()

# plt.figure()
# plt.title('Cumulative percentile of zIdPhi values')
# plt.plot(bins[0:-1],cumpercentage)
# plt.xlabel('zIdPhi value')
# plt.ylabel('Percentile')
# plt.axvline(0.5,label = 'Percentile = ' + str(round(cumpercentage[closest],4)),c='k')
# plt.legend(loc = 'lower right')

# corr, pvalue = pearsonr(trialnumber,z1[0].T)    
# plt.figure()
# plt.title('Correlation between trial number and zIdPhi (pooled)')
# plt.xlabel('Trial number')
# plt.ylabel('zIdPhi')
# plt.axhline(0.5,color = 'k')
# plt.rc('font', size=BIGGER_SIZE)          # co
# plt.scatter(alltrialnumbers,allz1,label = 'Pearson R =  ' +  str(round(corr,4)))
# plt.legend(loc = 'upper right')

# multipage(path + '/' + 'VTE_fwd.pdf', dpi=250)
