#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:04:47 2021

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
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu 
from functions import computeSpeedTuningCurves
    

data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

BIGGER_SIZE = 14

allz1 = []
alltrialnumbers = []
allcorrs = []
allp = []
allwilxp = []
allpcomp = []
alldurcorr = []
alldurp = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    episodes = ['sleep', 'wake']
    events = ['1']

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(path)
    position = loadPosition(path, events, episodes)
    wake_ep = loadEpoch(path, 'wake', episodes)
    sleep_ep = loadEpoch(path, 'sleep')        
    
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)  
    passes = pd.read_csv(path + '/' + name +'_vte.csv')
    
    # if s == 'A8504-210707':
    #     sys.exit()
    
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
    z2 = z1[0].copy()
    
    #COMPUTE MEAN FIRING RATE
              
    rates = computeMeanFiringRate(spikes, [pass_ep, wake_ep.loc[[1]]], ['pass','wake'])
    corr, pvalue = pearsonr(rates['wake'],rates['pass'])
    
    dur = np.zeros(len(pass_ep))
    for i in range(len(pass_ep)):
        dur[i] = pass_ep.loc[[i]].tot_length('s')
    
    # plt.figure()
    # plt.scatter(rates['wake'],rates['pass'], label  = 'Pearson R =  ' +  str(round(corr,4)))
    # plt.xlabel('Wake FR')
    # plt.ylabel('Pass FR')
    # plt.legend(loc = 'upper right')
    
    # #SHOW HD TUNING CURVES AND STABILITY 
    # from functions import computeAngularTuningCurves
    # if s == 'A8504-210706':
    #     tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], 60, 120)
        
    # else: tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60, 120)

    # from functions import smoothAngularTuningCurves
    # tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)
    
    # wake2_ep = splitWake(wake_ep.loc[[0]])
    # tokeep2 = []
    # stats2 = []
    # tcurves2 = []
    # for i in range(2):
    #     tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
    #     tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
             
        
    #     tokeep, stat = findHDCells(tcurves_half, z = 10, p = 0.05 , m = 1) 
    #     tokeep2.append(tokeep)
    #     stats2.append(stat)
    #     tcurves2.append(tcurves_half)
        
        # figure()
        
        # for j, n in enumerate(tuning_curves.columns):
        #     title('Neuron' + ' ' + str(j) + ' shank_' +str(shank[n]) + ' portion' + str(i+1), loc ='center', pad=25)   
        #     subplot(7,6,j+1, projection = 'polar')
        #     plot(tcurves_half[n])
        #     subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
        # show()
        
    # tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    
    # figure()
    # for i, n in enumerate(tuning_curves.columns):
    #     title('Neuron' + ' ' + str(i) + ' shank_' +str(shank[n]) + ' full session', loc ='center', pad=25)   
    #     subplot(7,6,i+1, projection = 'polar')
    #     plt.plot(tuning_curves[n])
    #     # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    #     subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    
    # from functions import computeSpeedTuningCurves
    # speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]], bin_size = 0.1, nb_bins = 20, speed_max = 0.4)
    # speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
    
    # figure()
    # for i in spikes:
    #     title('Speed tuning')
    #     subplot(7,6,i+1)
    #     plot(speed_curves[i], label = str(shank[i]))
    #     plt.xlabel('speed (m/s)')
    #     plt.ylabel('Firing rate (Hz)')
    #     legend()    
    #     subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    
    #COMPUTE FIRING RATE DURING A PASS FOR EACH CELL
    passrate = pd.DataFrame(index = range(len(pass_ep)) , columns = spikes.keys() )
          
    
    for i in range(len(pass_ep)):
        for j in spikes.keys() :
    #     # plt.figure()
    #     # plt.axis('off')
    #     # plt.plot(xpos,ypos,'silver', zorder = 1)
    #     # plt.plot(position['x'].restrict(pass_ep.loc[[i]]), position['z'].restrict(pass_ep.loc[[i]]))
            r2 = len(spikes[j].restrict(pass_ep.loc[[i]]))/pass_ep.loc[[i]].tot_length('s')
            passrate[j][i] = r2
            
    #CORRELATION BETWEEN zIdPhi and Pass FR (+ shuffle) for each cell
    sessioncorrs = []
    sessionp = []
    sessiondurcorr = []
    sessiondurp = []
    
    for i in spikes.keys():
        if sum(passrate[i]) > 0:
            corr, pvalue = pearsonr(passrate[i],z1[0])
            shu_c = []
            shu_p = []
            for k in range(1000):
                np.random.shuffle(z2)
                corr_shu, p_shu = pearsonr(passrate[i],z2)
                shu_c.append(corr_shu)
                shu_p.append(p_shu)
            plt.figure()
            plt.title('Corr between pass FR v/s zIdPhi_Neuron_' + str(i+1) +'_' + s)
            plt.scatter(z1[0],passrate[i], label = 'R =  ' +  str(round(corr,4)))
            plt.legend(loc = 'upper right')
            plt.ylabel('Pass FR (Hz)')
            plt.xlabel('zIdPhi')
        # plt.xticks([-1, 0, 1, 2])
            sessioncorrs.append(corr)
            sessionp.append(pvalue)
            allcorrs.append(corr)
            allp.append(pvalue)
        
            c = np.where(shu_c > abs(corr))
            p_comp = len(c[0])/1000
            allpcomp.append(p_comp)
        
        
        #CORRELATION BETWEEN FR AND TRIAL DURATION
        # corr,pvalue = pearsonr(passrate[i],dur)
        # sessiondurcorr.append(corr)
        # sessiondurp.append(pvalue)
        # alldurcorr.append(corr)
        # alldurp.append(pvalue)
        
        # plt.figure()
        # plt.title('Corr between pass FR v/s Pass duration_' + str(i+1) +'_' + s)
        # plt.scatter(passrate[i], dur, label = 'R =  ' +  str(round(corr,4)))
        # plt.legend(loc = 'upper right')
        # plt.ylabel('Pass FR (Hz)')
        # plt.xlabel('Pass duration (s)')
        
        
        # plt.figure()
        # plt.title('Histogram of shuffled FR v/s zIdPhi corr_Neuron_ ' + str(i+1) +'_' + s)
        # plt.hist(shu_c, label = 'p = ' + str(round(p_comp,4)))
        # plt.axvline(abs(corr),color = 'k')
        # plt.ylabel('Number of iterations')
        # plt.xlabel('Pearson R')
        # plt.legend(loc = 'upper right')
    
        
    
    
    #zIdPhi v/s FR corr distributions for all sessions
    
    # plt.figure()
    # plt.hist(sessioncorrs, label = 'Mean =' + str(round(np.mean(sessioncorrs),4)))
    # plt.title('Distribution of zIdPhi v/s FR corr_' + s )
    # plt.xlabel('Pearson R')
    # plt.ylabel('Number of cells')
    # plt.legend(loc = 'upper right')
    
    z_statistic, p_value = wilcoxon(np.array(sessioncorrs).flatten()-0)
    allwilxp.append(p_value)

#HOW MANY CELLS POSITIVELY OR NEGATIVELY MODULATED BY zIdPhi?     
pos_mod = []
neg_mod = []

for i in range(len(allpcomp)):
        if allpcomp[i] < 0.05 and allcorrs[i] > 0:
            pos_mod.append(i)
        elif allpcomp[i] < 0.05 and allcorrs[i] < 0:
            neg_mod.append(i)
    
print(len(pos_mod), len(neg_mod))
    







   
    