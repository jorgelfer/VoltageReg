# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:22:51 2021

@author: tefav
"""
# required for processing
import pathlib
import os

import py_dss_interface
from Methods.dssDriver import dssDriver
#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def SLP_LP_scheduling(tap1, tap2, tap3, output_dir, vmin, vmax, userDemand=None, plot=False, freq="15min", dispatchType='SLP'):

    # initialization
    case = '13bus' # 123bus
    file = 'IEEE13Nodeckt.dss'#"IEEE123Master.dss" 

    # execute the DSS model
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("EV_data", case, file) 
    
    dss = py_dss_interface.DSSDLL()
    dss.text(f"Compile [{dss_file}]")
    
    #compute sensitivities for the test case
    compute = False
    if compute:
        computeSensitivity(script_path, case, dss, dss_file, plot)
    
    # get init load
    loadNames, dfDemand, dfDemandQ = getInitDemand(script_path, dss, freq)
    
    # correct native load by user demand
    if userDemand is not None:
        dfDemand.loc[loadNames.index,:] = userDemand
        
    #Dss driver function
    Pg_0, v_0, Pjk_0, v_base, Pjk_lim = dssDriver(tap1, tap2, tap3, output_dir, 'InitDSS', script_path, case, dss, dss_file, loadNames, dfDemand, dfDemandQ, dispatchType, vmin, vmax, plot=plot)
    outDSS = save_initDSS(script_path, Pg_0, v_0, Pjk_0, v_base, Pjk_lim, loadNames, dfDemand, dfDemandQ)


