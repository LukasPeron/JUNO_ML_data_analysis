"""
Generate Edge Indices for Graph Neural Network Training on JUNO Simulation Data

This script loads data on the JUNO Large PMTs (LPMTs) and computes spatial edge connections for each event. 
The edges are determined based on the nearest neighbors of each PMT in terms of both spatial and temporal proximity.
The generated edge indices are saved in `.txt` files for each event and can be used to construct spatial graphs 
for graph neural network (GNN) training.

Created by: Lukas Peron
Last Update: 30/10/2024

Overview:
---------
1. Loads PMT location data and event data for a specified file.
2. For each PMT activated during an event, computes the six nearest neighbor PMTs based on distance.
3. Saves the computed edge indices in a separate `.txt` file for each event.

Error Handling:
---------------
- Assumes PMT location and event data files exist and are accessible.
- Handles cases with zero distances by setting the distance to itself as infinity to avoid self-loops.

Dependencies:
-------------
- numpy

Instructions:
-------------
1. Ensure that the PMT location and event data files are accessible in the specified directories.
2. Run the script and provide a file number when prompted.

Notes:
------
- The output edge files are saved in the `graph_edge` directory, with file names indicating the event number and file number.
"""

import numpy as np

pwd = "/sps/l2it/lperon/JUNO/txt/data_profiling/"

# Load PMT locations
pmt_loc = np.genfromtxt("/pbs/home/l/lperon/work_JUNO/JUNO_info/PMTPos_CD_LPMT.csv", usecols=(0, 1, 2, 3))

# Prompt for file number
num_file = int(input())

# Load event data for specified file
X_train = np.loadtxt(pwd + f"elecsim_data_file{num_file}.txt")
for num_event, event in enumerate(X_train):
    # Identify PMTs triggered in the event
    lst_pmt_on = np.where(event[::3] == 1)[0]
    positions = pmt_loc[lst_pmt_on, 1:]  # Get spatial coordinates of active PMTs
    lst_time = event[lst_pmt_on * 3 + 2]  # Retrieve time of activation for each PMT

    edge_index = [[], []]  # Initialize edge index for storing connections

    # Loop through each activated PMT and compute nearest neighbors
    for i, loc in enumerate(positions):
        true_pmt1 = lst_pmt_on[i]
        pmt1 = np.array([lst_time[i], loc[0], loc[1], loc[2]])
        
        # Define PMT positions and times for all neighbors
        true_pmt2 = np.delete(lst_pmt_on, i)
        pmt2 = np.array([lst_time, positions.T[0], positions.T[1], positions.T[2]]).T

        # Calculate distances from the current PMT to all other active PMTs
        temp = pmt2 - pmt1
        lst_dist = np.linalg.norm(temp, axis=1)
        lst_dist[np.argmin(lst_dist)] = np.inf  # Set self-distance to infinity to avoid self-loops

        # Identify six nearest neighbors
        nearest_neighbour_index = np.argpartition(lst_dist, 6)[:6]
        nearest_neighbour = lst_pmt_on[nearest_neighbour_index]

        # Append edges to edge index
        edge_index = np.concatenate((edge_index, np.array([nearest_neighbour, true_pmt1 * np.ones(6)])), axis=1)

    # Save the computed edge index for the event
    np.savetxt("graph_edge/" + f"graph_edge_spat_file{num_file}_event{num_event}.txt", edge_index)
