from ..Tools import detsim_branches as detsim
from ..Tools import elec_branches as elec
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
import matplotlib.pyplot as plt

# Update matplotlib settings
matplotlib.rcParams.update({'font.size': 14})

def plot_kl_heatmap():
    lst_energy = []
    lst_radial = []
    lst_kl = []
    # Iterate over all files except file 62
    for num_file in range(0, 119):
        kl_entry=0
        if num_file == 62:  # Skip file number 62
            continue
        print(f"Processing file {num_file}")
        # Load electrical data
        file = elec.load_file(num_file)
        tree_elec = file.Get("evt")
        kl_list = np.loadtxt(f"/sps/l2it/lperon/JUNO/data_nf/kl_div/kl_list_file_{num_file}.txt")
        # Iterate over entries in the event tree
        for num_entry, entry in enumerate(tree_elec):
            tree_elec.GetEntry(num_entry)
            TrueEvtID = elec.load_attributes(entry, "TrueEvtID")
            if len(TrueEvtID) == 1:
                TrueEvtID = TrueEvtID[0]
                # Load corresponding detector simulation data
                detsim_file = detsim.load_file(num_file)
                tree_det = detsim_file.Get("evt")
                edep, true_x, true_y, true_z = detsim.load_branches(tree_det, "edep", "edepX", "edepY", "edepZ")
                tree_det.GetEntry(TrueEvtID)
                edep, true_x, true_y, true_z = detsim.load_attributes(tree_det, "edep", "edepX", "edepY", "edepZ")
                # Convert positions to meters and calculate radial distance
                true_x, true_y, true_z = true_x / 1000, true_y / 1000, true_z / 1000
                true_pos = np.array([true_x, true_y, true_z])
                radial_distance = np.linalg.norm(true_pos)
                # Append data for heatmap
                lst_energy.append(edep)
                lst_radial.append(radial_distance)
                lst_kl.append(kl_list[kl_entry])   # Use corresponding KL value
                kl_entry+=1

    # Convert to numpy arrays
    lst_energy = np.array(lst_energy)
    lst_radial = np.array(lst_radial)
    lst_kl = np.array(lst_kl)

    # Create a 2D histogram for the heatmap
    bins_energy = np.linspace(min(lst_energy), max(lst_energy), 150)  # Energy bins
    bins_radial = np.linspace(min(lst_radial), max(lst_radial), 150)  # Radial distance bins

    # Compute 2D histogram with KL values
    heatmap, xedges, yedges = np.histogram2d(lst_radial, lst_energy, bins=[bins_radial, bins_energy], weights=lst_kl)
    counts, _, _ = np.histogram2d(lst_radial, lst_energy, bins=[bins_radial, bins_energy])

    # Normalize heatmap by counts to get average KL per bin
    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)

    # Set empty bins (where counts == 0) to NaN
    heatmap[counts == 0] = np.nan

    # Plot heatmap with 'plasma' colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.T, 
        extent=[bins_radial[0], bins_radial[-1], bins_energy[0], bins_energy[-1]],
        origin='lower',
        aspect='auto',
        cmap='plasma',
        vmin=np.nanmin(heatmap),  # Set min color scale to ignore NaNs
        vmax=np.nanmax(heatmap)   # Set max color scale to ignore NaNs
    )
    plt.colorbar(label="KL Divergence")
    plt.xlabel("Radial distance [m]")
    plt.ylabel("Energy [MeV]")
    plt.title("KL Divergence Heatmap (All Files Except 62)")
    plt.tight_layout()
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_heatmap_all_files_plasma.png")
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_heatmap_all_files_plasma.svg")
    plt.close()

    # Plot heatmap with 'viridis' colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.T, 
        extent=[bins_radial[0], bins_radial[-1], bins_energy[0], bins_energy[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        vmin=np.nanmin(heatmap),  # Set min color scale to ignore NaNs
        vmax=np.nanmax(heatmap)   # Set max color scale to ignore NaNs
    )
    plt.colorbar(label="KL Divergence")
    plt.xlabel("Radial distance [m]")
    plt.ylabel("Energy [MeV]")
    plt.title("KL Divergence Heatmap (All Files Except 62)")
    plt.tight_layout()
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_heatmap_all_files_viridis.png")
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_heatmap_all_files_viridis.svg")
    plt.close()

    return 0