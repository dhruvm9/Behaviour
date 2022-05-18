#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:28:36 2022

@author: dhruv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys 
from typing import List
from pathlib import Path
import scipy
import pynapple as nap

plt.rcParams["figure.figsize"] = (8, 8)

def get_init_points() -> List:
    """Extract initial points from the plot of the trajectories
    Returns:
        List: [departure, left, right, center]
    """      
    points = []
    helper_texts = ['Please click on the departing point',
                    'Please click on the left reward',
                    'Please click on the right reward',
                    'Please click on the center'
                    ]
    for text in helper_texts:
        plt.scatter(x, y, s=0.5, alpha=0.06, c='black')
        plt.title(text)
        point = None
        
        while not point:
            point = plt.ginput(1)
            print(point)
        points.append(np.array(point[0]))
        plt.close()
    print("""
          Departure: {0}
          Left: {1}
          Right: {2}
          Center: {3}
          """.format(*points))
    return points

data_directory = '/media/DataDhruv/Recordings/B1600/B1603'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_CA1.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    name = s.split('/')[-1]
    print(name)
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    
    epochs = data.epochs
    position = data.position
    
    freepos = position.restrict(position.time_support.loc[[0]])
    pos_data = freepos.values       
    pos_time = freepos.index
    pos_var = freepos.columns
    
    x = freepos['x'].values
    y  = freepos['z'].values             
    pos = np.array([x, y]).T
    
    departure, left, right, center = get_init_points()
       
    # First, we express everything as the distance from the center
    c_dist = np.array([pos - center]).T
    c_dist = np.linalg.norm(c_dist, axis=0).reshape((1,-1))
    
    # We normalize the distance by the longest arm (all supposed to be more or
# less the same anyway...)
    points = np.array([departure, left, right]) - center
    arm_length = np.linalg.norm(points.T, axis=0)
    arm_length = max(arm_length)
    c_dist = c_dist / arm_length
    
    c_dist = scipy.ndimage.gaussian_filter(c_dist, 10)
    c_dist = c_dist[0]
    
    # When is the animal on the left or right  arm?
    left_idx = (pos[:, 0].T < center[0]) & (pos[:, 1].T > center[1])
    right_idx = (pos[:, 0].T > center[0]) & (pos[:, 1].T > center[1])
    
    # by default, everything else is departure arm
    dep_idx = ~ (left_idx | right_idx)
    # linPos is the linearized position.
    lin_pos = np.zeros(len(c_dist))
    # On Dep arm, we want it between 0 and 1
    lin_pos[dep_idx] = max(c_dist) - c_dist[dep_idx]
    # We want left between 1 and 2. 
    lin_pos[left_idx] = c_dist[left_idx] + 1
    # Same for right, but we want it between 2 and 3
    lin_pos[right_idx] = c_dist[right_idx] + 2
    
    plt.scatter(x=range(len(lin_pos)), y=lin_pos, s=0.2, c='black')
    plt.show()
    
    df = pd.DataFrame(data=lin_pos, columns=[ 'position'])
    df['Time (s)'] = pos_time.values
    df.set_index('Time (s)', inplace=True)
    df.to_csv(data_directory + '/' + 'LinearPos_' + s)
    