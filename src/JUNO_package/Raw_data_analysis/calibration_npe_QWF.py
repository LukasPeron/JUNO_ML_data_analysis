"""
Calibration of Total Waveform Charge (Q) vs Number of Photoelectrons (NPE)

This script is designed to perform the calibration of the total waveform charge (Q) as a function of the number of 
photoelectrons (NPE) using data from ELECSIM and DETSIM simulations.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load and process ELECSIM data, summing up the waveform charge for each true event ID (TrueEvtID).
2. Save the processed ELECSIM data to a text file for further analysis.
3. (Optional) Load and process DETSIM data, correlating the ELECSIM events with the corresponding DETSIM events based 
   on event IDs.
4. Save the processed DETSIM data to a text file for further analysis.

Notes:
------
- If the script is working with ADC peak values instead of integrated waveform charge, modifications are noted in the comments.
- The section for DETSIM data processing can be commented out if the peak calibration has already been completed.

Dependencies:
-------------
- ROOT
- Tools (contains `elec_branches`, `detsim_branches`, and `useful_function`)
- numpy
- matplotlib
- scipy

Instructions:
-------------
1. Ensure that ROOT and the required Python packages are installed.
2. Provide the ELECSIM file number when prompted.
3. The script will automatically process the data and save the results in the specified directory.
"""

from ..Tools import elec_branches as elec
from ..Tools import detsim_branches as detsim
from ..Tools.useful_function import create_adc_count
import numpy as np

# Input: Specify the ELECSIM file number
num_file_elec = int(input("Enter ELECSIM file number: "))

# Load ELECSIM data and extract relevant branches
file = elec.load_file(num_file_elec)
tree = file.Get("evt")
NGenEvts, TrueEvtID, WaveformQ = elec.load_branches(tree, "NGenEvts", "TrueEvtID", "WaveformQ")

# Convert tree data to numpy arrays for vectorized processing
NGenEvts_array = np.array([entry.NGenEvts for entry in tree])
TrueEvtID_array = np.array([entry.TrueEvtID[0] for entry in tree if entry.NGenEvts == 1])
WaveformQ_array = np.array([entry.WaveformQ for entry in tree if entry.NGenEvts == 1])

# Create ADC counts and sum them using vectorized operations
adc_counts = create_adc_count(WaveformQ_array)
sum_q = np.sum(adc_counts, axis=(1, 2))  # Sum over both axes for each event
# If working with ADC peak, use: sum_q = np.max(adc_counts, axis=2)

# Aggregate summed ADC counts by TrueEvtID using vectorized operations
unique_evt_ids, indices = np.unique(TrueEvtID_array, return_inverse=True)
all_info_elec = np.zeros((len(unique_evt_ids), 3))

all_info_elec[:, 0] = num_file_elec
all_info_elec[:, 1] = unique_evt_ids
all_info_elec[:, 2] = np.bincount(indices, weights=sum_q)

# Save the processed ELECSIM data
np.savetxt(f"/pbs/home/l/lperon/work_JUNO/code/txt/elec/elec_info_numfile{num_file_elec}_integrated.txt", all_info_elec)
# If working with ADC peak, remove "_integrated" from the filename

#! Comment out the following DETSIM processing section if peak calibration has already been done

# Load DETSIM data and extract relevant branches
file = detsim.load_file(num_file_elec)
tree = file.Get("evt")
evtID, totalPE = detsim.load_branches(tree, "evtID", "totalPE")
evtID_array = np.array([entry.evtID for entry in tree])
totalPE_array = np.array([entry.totalPE for entry in tree])

# Use vectorized operations to correlate ELECSIM and DETSIM events by event IDs
det_indices = np.in1d(evtID_array, unique_evt_ids)
correlated_info_det = np.zeros((np.sum(det_indices), 3))

correlated_info_det[:, 0] = num_file_elec
correlated_info_det[:, 1] = evtID_array[det_indices]
correlated_info_det[:, 2] = totalPE_array[det_indices]

# Save the processed DETSIM data
np.savetxt(f"/pbs/home/l/lperon/work_JUNO/code/txt/detsim/detsim_info_numfile{num_file_elec}.txt", correlated_info_det)
