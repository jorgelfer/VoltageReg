# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
import shutil
import time
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import py_dss_interface


# set control mode off
def set_baseline(dss):
    dss.text("Set Maxiterations=100")
    dss.text("Set controlmode=OFF")
    

# extract voltages from dss
def extract_voltages(dss, dssFile, trafo, tap):
    dss.text(f"Compile [{dssFile}]")
    set_baseline(dss)
    # set the trafo as the active element
    dss.transformers_write_name(trafo)
    # set the second winding active
    dss.transformers_write_wdg(2)
    readprevtap = dss.transformers_read_tap()  # debug
    newtap = 1.0 + tap*0.00625
    dss.text(f"Transformer.{trafo}.Taps=[1.0, {newtap}]")
    readnewtap = dss.transformers_read_tap()  # debug
    dss.text("solve")
    voltages = dss.circuit_all_bus_vmag()
    return voltages


def get_nodeBaseVolts(dss):
    # create a directory to store voltages
    node_base_voltage = dict()

    # get all buses
    buses = dss.circuit_all_bus_names()
    # iterate over buses
    for i, bus in enumerate(buses):
        dss.circuit_set_active_bus(bus)
        bus_voltage = dss.bus_kv_base()  # VLN for 1phase buses
        bus_nodes = dss.bus_nodes()
        for n in bus_nodes:
            node = bus + f'.{n}'
            node_base_voltage[node] = bus_voltage
    
    return node_base_voltage

ext = '.png'
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

# initialization
dssCase = '123bus'  # 123bus
file = "IEEE123Master.dss"  # 'IEEE13Nodeckt.dss', "IEEE123Master.dss"

# get file path
script_path = os.path.dirname(os.path.abspath(__file__))
dssFile = pathlib.Path(script_path).joinpath("EV_data", dssCase, file)

# execute the DSS model
dss = py_dss_interface.DSSDLL()
dss.text(f"Compile [{dssFile}]")

# get node names
nodeNames = dss.circuit_all_node_names()
nodeBaseVolts_dict = get_nodeBaseVolts(dss)
nodeBaseVolts = [nodeBaseVolts_dict[n] for n in nodeNames]

# list dss elements
trafos = dss.transformers_all_Names()
regs = [tr for tr in trafos if "reg" in tr]

outputFile = pathlib.Path(output_dir).joinpath("TapVoltage.npy")
if not os.path.isfile(outputFile):
    # prelocate
    numTaps = 32
    numReg = 7
    volts = np.zeros((len(nodeNames), numTaps + 1, numReg))
    # main loop
    for r, reg in enumerate(regs):
        print(f"{reg}")
        ntap = 0
        for tap in range(-16, 17):
            volts[:, ntap, r] = extract_voltages(dss, dssFile, reg, tap)
            ntap += 1

    # save output file
    with open(outputFile, 'wb') as f:
        np.save(f, volts)
else:
    with open(outputFile, 'rb') as f:
        volts = np.load(f)

for n, node in enumerate(nodeNames):
    print(f"{node}")
    # create a new directory for each nod
    output_dirNode = pathlib.Path(output_dir).joinpath(f"{node}")
    if not os.path.isdir(output_dirNode):
        os.mkdir(output_dirNode)

    for r, reg in enumerate(regs):
        # compute pu
        vpu = volts[n, :, r] / (1000*nodeBaseVolts[n])
        vpu = np.expand_dims(vpu, axis=1)
        vmin_vec = vmin * np.zeros((len(vpu), 1))
        vmax_vec = vmax * np.zeros((len(vpu), 1))
        concat = np.hstack((vpu, vmin_vec, vmax_vec))
        # create plot
        plt.clf()
        fig, ax = plt.subplots()  # figsize=(h,w)
        tapRange = np.arange(0.9, 1.1, 0.00625)
        plt.plot(tapRange, concat)
        plt.ylim(0.99*vmin, 1.01*vmax)
        plt.title(f"{node}_{reg}")
        fig.tight_layout()
        output_img = pathlib.Path(output_dirNode).joinpath(f"voltage_{node}_{reg}.png")
        plt.savefig(output_img)
        plt.close('all')
