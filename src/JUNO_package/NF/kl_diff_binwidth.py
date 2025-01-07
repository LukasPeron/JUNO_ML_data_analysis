from ..Tools import elec_branches as elec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
from scipy.special import rel_entr

def plot_kl_vs_binwidth():
    num_pmt = 17612
    kl_list = []
    err_list = []
    # Loop over binwidths first
    for binwidth in np.linspace(1, 100, 100, endpoint=True, dtype=int):
        temp_kl = []
        print(f"Processing binwidth {binwidth}")
        # Iterate over all files except for file number 62
        for num_file in range(0, 1):
            if num_file == 62:  # Skip file number 62
                continue
            file = elec.load_file(num_file)
            tree_elec = file.Get("evt")
            for num_entry, entry in enumerate(tree_elec):
                tree_elec.GetEntry(num_entry)
                TrueEvtID = elec.load_attributes(entry, "TrueEvtID")
                if len(TrueEvtID)==1:
                    TrueEvtID = TrueEvtID[0]
                    JUNO = np.loadtxt(f"/sps/l2it/lperon/JUNO/data_nf/JUNO_distrib/JUNO_distrib_file_{num_file}_entry_{num_entry}.txt")
                    th = np.loadtxt(f"/sps/l2it/lperon/JUNO/data_nf/th_distrib/TH_distrib_file_{num_file}_entry_{num_entry}.txt")
                    bins = range(0, num_pmt + binwidth, binwidth)
                    JUNO_hist, _ = np.histogram(range(num_pmt), bins=bins, weights=JUNO)
                    th_hist, _ = np.histogram(range(num_pmt), bins=bins, weights=th)
                    kl_hist = sum(rel_entr(JUNO_hist, th_hist))
                    temp_kl.append(kl_hist)
        # Compute average and error for the current binwidth
        kl_list.append(np.mean(temp_kl))
        err_list.append(2 * np.std(temp_kl))

    kl_list = np.array(kl_list)
    err_list = np.array(err_list)

    # Plotting
    plt.plot(np.linspace(1, 100, 100, endpoint=True), kl_list, 'k-')
    plt.fill_between(np.linspace(1, 100, 100, endpoint=True), kl_list - err_list, kl_list + err_list, facecolor="grey", alpha=0.7)
    plt.xlabel("Binwidths (# of PMT merged)")
    plt.ylabel(r"KL(JUNO||TH) (95% error)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_vs_binwidth_all_files.png")
    plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/kl_div/KL_vs_binwidth_all_files.svg")
    plt.close()
    return 0