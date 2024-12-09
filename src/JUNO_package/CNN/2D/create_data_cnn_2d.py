from ...Tools import detsim_branches as detsim
from ...Tools import elec_branches as elec
from ...Tools.useful_function import cartesian_to_spherical, create_adc_count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
    
def cnn_2d_image(num_file, entry_num, adc_count, lst_pmt_on):
    pwd_saving = "/sps/l2it/lperon/JUNO/data_cnn/2d/"
    vertices = np.genfromtxt("/pbs/home/l/lperon/work_JUNO/JUNO_info/PMTPos_CD_LPMT.csv", usecols=(0, 1, 2, 3))
    height = 9
    fig = plt.figure(figsize=(height * 1.618, height))
    ax_2d = fig.add_subplot(111, projection='hammer')
    row_sums = np.sum(adc_count, axis=1)
    max_signal = np.max(row_sums)
    min_signal = np.min(row_sums[row_sums > 0])
    norm = LogNorm(vmin=min_signal, vmax=max_signal)
    cmap = plt.cm.hot
    colors = []  
    theta_vals = []
    phi_vals = []
    for j, id_on in enumerate(lst_pmt_on):
        pos = vertices[id_on, :] / 1000
        q = sum(adc_count[j])
        if q > 0: 
            color = cmap(norm(q))
            colors.append(color)
            theta, phi = cartesian_to_spherical(pos[1], pos[2], pos[3])
            theta_vals.append(theta)
            phi_vals.append(phi)
    if theta_vals and phi_vals:
        ax_2d.scatter(phi_vals, theta_vals, c=colors, cmap=cmap, norm=norm, s=7, zorder=0)
    ax_2d.set_xticks([])
    ax_2d.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        pwd_saving + f"cnn_2d_data_file_{num_file}_entry_{entry_num}.png", 
        bbox_inches='tight', 
        pad_inches=0, 
        dpi=300)
    plt.close(fig)

def create_data_cnn_2d(num_file):
    file = elec.load_file(num_file)
    tree_elec = file.Get("evt")
    TrueEvtID, WaveformQ, PmtID_WF = elec.load_branches(tree_elec, "TrueEvtID", "WaveformQ", "PmtID_WF")
    true_positions = []
    true_energy = []
    for entry_num, entry in enumerate(tree_elec):
        TrueEvtID, WaveformQ, PmtID_WF = elec.load_attributes(entry, "TrueEvtID", "WaveformQ", "PmtID_WF")
        if len(TrueEvtID)>0:
            print(entry_num)
            lst_pmt_on = PmtID_WF
            TrueEvtID = TrueEvtID[0]
            adc_count = create_adc_count(WaveformQ, no_pedestal=False)
            detsim_file = detsim.load_file(num_file)
            tree_det = detsim_file.Get("evt")
            edep, true_x, true_y, true_z = detsim.load_branches(tree_det, "edep", "edepX", "edepY", "edepZ")
            tree_det.GetEntry(TrueEvtID)
            edep, true_x, true_y, true_z = detsim.load_attributes(tree_det, "edep", "edepX", "edepY", "edepZ")
            true_energy.append(edep)
            true_positions.append([true_x, true_y, true_z])
            true_x, true_y, true_z = true_x / 1000, true_y / 1000, true_z / 1000
            cnn_2d_image(num_file, entry_num, adc_count, lst_pmt_on)
    true_energy=np.array(true_energy)
    true_positions=np.array(true_positions)
    pwd_saving = "/sps/l2it/lperon/JUNO/data_cnn/2d/"
    np.savetxt(pwd_saving+f"true_energy_file_{num_file}.txt", true_energy)
    np.savetxt(pwd_saving+f"true_positions_file_{num_file}.txt", true_positions)