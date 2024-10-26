"""
Data Preparation for Neural Network Training on JUNO Simulation Data

CREATED BY : LUKAS PERON
LAST UPDATE : 26/10/2024
PURPOSE : CREATE THE DATA FILES USED FOR THE TRAINING OF THE NN

This script processes JUNO simulation data, preparing data files with selected event and charge information from both 
detector and electronics simulation files. The data files are structured for neural network training and testing, with 
entries detailing energy deposits, charge sums, and charge timings across Large PMTs (LPMTs).

Overview:
---------
The script:
1. Selects and loads events from both electronics simulation (elecsim) and detector simulation (detsim) files.
2. Processes data by computing integrated charge deposits and timestamps for each event.
3. Structures the processed data for NN training, saving it to `.txt` files.

Error Handling:
---------------
- Assumes that input files exist and are correctly formatted for ROOT TTree processing.
- Monitors cases of repeated true event IDs, summing charge information when detected.

Dependencies:
-------------
- ROOT (for TTree access)
- my_package: detsim_branches, elecsim_branches, useful_function
- numpy

Instructions:
-------------
1. Ensure all required input files are accessible in specified directories.
2. Run the script and provide a file number when prompted.
"""

from ROOT import *
from ..Tools import detsim_branches as detsim
from ..Tools import elec_branches as elecsim
from ..Tools.useful_function import create_adc_count
import numpy as np

# FILE AND ENTRY SELECTION AND BRANCHES LOADING
num_file = int(input("Enter the file number to process: "))

detsim_file = detsim.load_file(num_file)
elecsim_file = elecsim.load_file(num_file)

detsim_tree_evt = detsim_file.Get("evt")
elec_tree_evt = elecsim_file.Get("evt")

edep, edepX, edepY, edepZ = detsim.load_branches(detsim_tree_evt, "edep", "edepX", "edepY", "edepZ")
NGenEvts, TrueEvtID, WaveformQ, PmtID_WF = elecsim.load_branches(elec_tree_evt, "NGenEvts", "TrueEvtID", "WaveformQ", "PmtID_WF")

true_evt_passed = []
elecsim_data = []
detsim_data = []
num_entry = 0

print(f"Total number of entries : {elec_tree_evt.GetEntries()}")
for entry in elec_tree_evt:
    # Load attributes for the current entry
    NGenEvts, TrueEvtID, WaveformQ, PmtID_WF = elecsim.load_attributes(entry, "NGenEvts", "TrueEvtID", "WaveformQ", "PmtID_WF")

    if NGenEvts == 1:  # Check for non-pileup condition
        if TrueEvtID[0] in true_evt_passed:  # Summation for repeated true events
            adc_count = create_adc_count(WaveformQ)
            init_entry = true_evt_passed.index(TrueEvtID[0])
            for i in range(len(PmtID_WF)):
                lst_loc = int(3 * PmtID_WF[i])
                elecsim_data[init_entry][lst_loc + 1] += np.sum(adc_count[i])
        else:
            true_evt_passed.append(TrueEvtID[0])
            detsim_tree_evt.GetEntry(TrueEvtID[0])
            edep, edepX, edepY, edepZ = detsim.load_attributes(detsim_tree_evt, "edep", "edepX", "edepY", "edepZ")
            detsim_data.append([edep, edepX, edepY, edepZ])
            temp_data = np.zeros(3 * 17612)

            adc_count = create_adc_count(WaveformQ)
            for i in range(len(PmtID_WF)):
                lst_loc = int(3 * PmtID_WF[i])
                temp_data[lst_loc] = 1  # Mark PMT as triggered
                temp_data[lst_loc + 1] = np.sum(adc_count[i])  # Compute integrated charge deposit
                temp_data[lst_loc + 2] = np.argmax(adc_count[i])  # Time of charge deposits
            elecsim_data.append(temp_data)

    print(f"{(num_entry + 1) / elec_tree_evt.GetEntries() * 100:.2f}% processed")
    num_entry += 1

# Convert collected data to arrays for saving
elecsim_data = np.array(elecsim_data)
detsim_data = np.array(detsim_data)

# Save processed data
np.savetxt(f"/sps/l2it/lperon/JUNO/txt/data_profiling/detsim_data_file{num_file}.txt", detsim_data)
np.savetxt(f"/sps/l2it/lperon/JUNO/txt/data_profiling/elecsim_data_file{num_file}.txt", elecsim_data)
