#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:55:30 2021

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
from scipy.stats import pearsonr, wilcoxon 
from functions import computeSpeedTuningCurves
import statsmodels.api as sm
    

data_directory = '/media/DataDhruv/SandyReplayAnalysis/Data'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

BIGGER_SIZE = 14

allz1 = []
alltrialnumbers = []
allcorrs = []
allpvals = []
allvcorrs = []
allvp = []
allwilxp = []

allbeta_speed = []
allbeta_vte = []
allbeta_dur = []

sesscorr = []
sessp = []
# allpassrates = {}

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
    normalize = matplotlib.colors.Normalize(vmin=miv, vmax=mav)
     
    #COMPUTE MEAN FIRING RATE AND AVERAGE SPEED DURING A PASS
    
    rates = computeMeanFiringRate(spikes, [pass_ep, wake_ep], ['pass','wake'])
    corr, pvalue = pearsonr(rates['wake'],rates['pass'])
    
    avg_v = np.zeros(len(v))
    for i in range(len(v)):
        avg_v[i] = np.mean(v[i])
   
    #COMPUTE FIRING RATE  DURING A PASS FOR EACH CELL
    passrate = pd.DataFrame(index = range(len(pass_ep)) , columns = spikes.keys() )
    # allpassrates[i] = passrate
      
    for i in range(len(pass_ep)):
        for j in spikes.keys() :
            # plt.figure()
            # plt.axis('off')
            # plt.plot(xpos,ypos,'silver', zorder = 1)
            # plt.plot(position['x'].restrict(pass_ep.loc[[i]]), position['z'].restrict(pass_ep.loc[[i]]))
            r2 = len(spikes[j].restrict(pass_ep.loc[[i]]))/pass_ep.loc[[i]].tot_length('s')
            passrate[j][i] = r2
            
    #CORRELATION BETWEEN PASS FR AND zIdPhi
    sessioncorrs = []
    sessionvcorrs = []
    sessionp = []
    sessionvp = []
    
    for i in spikes.keys():
        if sum(passrate[i]) > 0:
            corr, pvalue = pearsonr(passrate[i],z1[0])
            # plt.figure()
            # plt.title('Corr between pass FR v/s zIdPhi_Neuron_' + str(i+1) +'_' + s)
            # plt.scatter(z1[0],passrate[i], label = 'R =  ' +  str(round(corr,4)))
            # plt.legend(loc = 'upper right')
            # plt.xlabel('Pass FR')
            # plt.ylabel('zIdPhi')
            sessioncorrs.append(corr)
            sessionp.append(pvalue)
            allcorrs.append(corr)
            allpvals.append(pvalue)
            
            vcorr, vp = pearsonr(passrate[i],avg_v)
            sessionvcorrs.append(vcorr)
            sessionvp.append(vp)
            allvcorrs.append(vcorr)
            allvp.append(vp)
            
        # plt.figure()
        # plt.title('Corr between pass speed v/s pass FR_Neuron_' + str(i+1) +'_' + s)
        # plt.scatter(passrate[i],avg_v, label = 'R =  ' +  str(round(vcorr,4)))
        # plt.legend(loc = 'upper right')
        # plt.ylabel('Pass FR (Hz)')
        # plt.xlabel('Velocity (cm/s)')
        
        dur = np.zeros(len(pass_ep))
        for i in pass_ep.index.values:
            dur[i] = (pass_ep.iloc[i]['end'] - pass_ep.iloc[i]['start']) / 1e6
    
    corr, p = pearsonr(dur, z1[0])
    sesscorr.append(corr)
    sessp.append(p)
        
###############################################################################
####BEGIN GLM
# ###############################################################################
    x1 = z1[0]
    x2 = avg_v
    x3 = dur
    
    x_glm = pd.DataFrame(list(zip(x1, x2)), columns = ['vte', 'speed'])
    # x_glm = pd.DataFrame(list(zip(x1, x3)), columns = ['vte', 'dur'])
    
    beta_vte = []
    beta_speed = []
    beta_dur = []
    
    for i in spikes.keys():
        if sum(passrate[i]) > 0:
            y_glm = pd.DataFrame(passrate[i].values, columns = ['PassFR'])
            model= sm.GLM(y_glm.astype(float), x_glm.astype(float), family=sm.families.Poisson())
            r = model.fit(link=sm.genmod.families.links.Log)
            beta_vte.append(r.params[0])
            allbeta_vte.append(r.params[0])
            beta_speed.append(r.params[1])
            allbeta_speed.append(r.params[1])
            # beta_dur.append(r.params[1])
            # allbeta_dur.append(r.params[1])
    
abs_beta_vte = [abs(ele) for ele in allbeta_vte]
abs_beta_speed = [abs(ele) for ele in allbeta_speed]
# abs_beta_dur = [abs(ele) for ele in allbeta_dur]
        
z,p_glm = wilcoxon(abs_beta_speed, abs_beta_vte)
# z,p2_glm = wilcoxon(abs_beta_dur, abs_beta_vte)
means_speed = np.nanmean(abs_beta_speed)
means_vte = np.nanmean(abs_beta_vte)
# means_dur = np.nanmean(abs_beta_dur)

#figure 1 
label = ['All units over 4 sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_vte, width, label='zIdPhi')
rects2 = ax.bar(x + width/2, means_speed, width, label='speed')

pval = np.vstack([(abs_beta_vte), (abs_beta_speed)])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('GLM |beta| value')
ax.set_title('GLM for speed v/s zIdPhi')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')

fig.tight_layout()        
    
# #figure 2
# label = ['All units over 4 sessions']
# x = np.arange(len(label))  # the label locations
# width = 0.35  # the width of the bars


# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, means_vte, width, label='zIdPhi')
# rects2 = ax.bar(x + width/2, means_dur, width, label='trial duration')

# pval = np.vstack([(abs_beta_vte), (abs_beta_dur)])

# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('GLM |beta| value')
# ax.set_title('GLM for trial duration v/s zIdPhi')
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# ax.legend(loc = 'upper right')

# fig.tight_layout()        
    
    