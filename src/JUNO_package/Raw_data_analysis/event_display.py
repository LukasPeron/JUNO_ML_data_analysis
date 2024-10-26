"""
Event Display of JUNO Events

This script visualizes events from the JUNO experiment data by generating animations and plots of the detector's response. 
It includes loading data from multiple branches, extracting relevant attributes, processing event information, and creating visualizations.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load ELECSIM and DETSIM data files and extract relevant branches and attributes.
2. Convert waveform charge data into ADC counts.
3. Transform the true vertex coordinates from Cartesian to spherical.
4. Create and save animations and plots of the JUNO event data.

Error Handling:
---------------
- The file loading process includes error handling to ensure that the ROOT files are successfully opened. 
  If a file cannot be opened or is corrupted, an error is raised with a descriptive message.
- The branch and attribute loading functions include basic checks to ensure that the requested branches or attributes are available.
- The plotting and animation functions assume valid input data; additional error handling may be added as needed.

Dependencies:
-------------
- Tools (contains `elec_branches`, `detsim_branches`, and `useful_function`)

Instructions:
-------------
1. Ensure that numpy, matplotlib, ROOT, and my_package are installed and configured in your environment.
2. Run the script to load the event data, create visualizations, and save them to the specified directory.
"""

from ..Tools import elec_branches as elec
from ..Tools import detsim_branches as detsim
from ..Tools import useful_function as utility

num_file = 0 #int(input())
entry_num = 0 #int(input())

# Elecfile and branches loading

file = elec.load_file(num_file)
tree = file.Get("evt")
ElecEvtID, TrueEvtID, NGenEvts, WaveformQ, PmtID_WF = elec.load_branches(tree, "ElecEvtID", "TrueEvtID", "NGenEvts", "WaveformQ", "PmtID_WF")
tree.GetEntry(entry_num)
ElecEvtID, TrueEvtID, NGenEvts, WaveformQ, PmtID_WF = elec.load_attributes(tree, "ElecEvtID", "TrueEvtID", "NGenEvts", "WaveformQ", "PmtID_WF")

# ADC data loading and processing

lst_pmt_on = PmtID_WF
TrueEvtID = TrueEvtID[0]
adc_count = utility.create_adc_count(WaveformQ, no_pedestal=False)

# Correspondance to detsim data

detsim_file = detsim.load_file(num_file)
tree = detsim_file.Get("evt")
true_x, true_y, true_z = detsim.load_branches(tree, "edepX", "edepY", "edepZ")
tree.GetEntry(TrueEvtID)
true_x, true_y, true_z = detsim.load_attributes(tree, "edepX", "edepY", "edepZ")

# Detsim data processing

true_x, true_y, true_z = true_x / 1000, true_y / 1000, true_z / 1000
true_theta, true_phi = utility.cartesian_to_spherical(true_x, true_y, true_z)
true_vertex = (true_x, true_y, true_z, true_theta, true_phi)

# Creation and saving of plots
utility.make_animation(num_file, entry_num, lst_pmt_on, adc_count, true_vertex, display=False)
utility.plot_colormesh_waveform(adc_count, num_file, entry_num, display=False)