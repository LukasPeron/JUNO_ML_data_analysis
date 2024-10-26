"""
Calibration of Energy Deposition (Edep) vs Number of Photoelectrons (NPE)

This script performs the calibration of energy deposition (Edep) versus the number of photoelectrons (NPE) 
using simulation data. It includes loading data from multiple files, extracting relevant branches and attributes, 
performing a linear fit, and generating plots.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load data from multiple simulation files, extracting the Edep and NPE branches.
2. Store the extracted data into lists.
3. Perform a linear fit of Edep vs NPE using the `linear_fit` function.
4. Generate and save histograms and calibration plots.

Error Handling:
---------------
- The file loading process includes error handling to ensure that the ROOT files are successfully opened. 
  If a file cannot be opened or is corrupted, an error is raised with a descriptive message.
- The `curve_fit` function includes basic checks to ensure valid input data. 
  If the input data is not in the expected format, a `ValueError` will be raised by scipy.
- The plotting functions assume that the input data is valid; additional error handling may be added as needed.

Dependencies:
-------------
- ROOT
- Tools (contains `detsim_branches` and `useful_function`)
- numpy
- matplotlib
- scipy

Instructions:
-------------
1. Ensure that ROOT and the required Python packages are installed.
2. Run the script to perform the calibration and generate the plots.
3. The plots will be saved in the specified directory.
"""

from ..Tools import detsim_branches as detsim
from ..Tools.useful_function import linear_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# FILE AND ENTRY SELECTION AND BRANCHES LOADING

edep_ls = []
NPe_ls = []
for num_file in range(120):
    file = detsim.load_file(num_file)
    tree_evt = file.Get("evt")
    edep, totalPE = detsim.load_branches(tree_evt, "edep", "totalPE")
    for entry in tree_evt:
        edep, totalPE = detsim.load_attributes(entry, "edep", "totalPE")
        edep_ls.append(edep)
        NPe_ls.append(totalPE)

# CREATION AND SAVING OF THE PLOT AND FIT

edep_ls = np.array(edep_ls)
NPe_ls = np.array(NPe_ls)
popt, cormat = curve_fit(linear_fit, edep_ls, NPe_ls, (1300))
x = np.linspace(np.min(edep_ls),np.max(edep_ls),1000)
y = linear_fit(x,*popt)
# Plots

plt.figure(0)
plt.hist(edep_ls,25,zorder=2)
plt.xlabel("edep [MeV]", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid(zorder=1)
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/svg/energy_spectrum.svg")
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/png/energy_spectrum.png")

plt.figure(1)
plt.plot(edep_ls, NPe_ls, 'bo', label="Simulated data")
plt.plot(x,y,'-r',label=f"Linear fit: p={popt[0]:.0f}")
plt.xlabel("edep [MeV]", fontsize=14)
plt.ylabel("Total PE", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/svg/calib_edep_NPE.svg")
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/png/calib_edep_NPE.png")
