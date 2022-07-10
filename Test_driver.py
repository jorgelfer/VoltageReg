# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import SLP_LP_scheduling
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import time
import seaborn as sns
import shutil
from functools import reduce

ext = '.png'
dispatch = 'SLP'
metric = np.inf# 1,2,np.inf
plot = True 
h = 6 
w = 4 
script_path = os.path.dirname(os.path.abspath(__file__))

# output directory
# time stamp 
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)  
# create directory to store results
today = time.strftime('%b-%d-%Y', t)
directory = "Results3_manualTaps_" + today
output_dir = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
else:
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
# voltage limits
vmin = 0.95
vmax = 1.05

# transformers taps [-16, 16] 
step = 2

for tap1 in range(0, 9, step):
    for tap2 in range(0,9, step):
        for tap3 in range(0,9, step):
            
            print(f"T1_{tap1}_T2_{tap2}_T3_{tap3}")
            tap_dir = pathlib.Path(output_dir).joinpath(f"T1_{tap1}_T2_{tap2}_T3_{tap3}")
            if not os.path.isdir(tap_dir):
                os.mkdir(tap_dir)
                
            # initial scheduling
            SLP_LP_scheduling(tap1, tap2, tap3, tap_dir, vmin, vmax, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)

