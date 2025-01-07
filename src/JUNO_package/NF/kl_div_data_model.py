from ..Tools import detsim_branches as detsim
from ..Tools import elec_branches as elec
from ..Tools.useful_function import create_adc_count
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
from scipy.spatial import ConvexHull
from scipy.special import rel_entr

# Parameters
r_disk = 0.508 / 2  # Disk radius
R_JUNO = 35.4 / 2  # Radius of the CD
pwd_juno = "/pbs/home/l/lperon/work_JUNO/JUNO_info/PMTPos_CD_LPMT.csv"
# Load PMT positions (assuming CSV file already read into variables)
x_sphere = np.genfromtxt(pwd_juno, usecols=(1)) / 1000
y_sphere = np.genfromtxt(pwd_juno, usecols=(2)) / 1000
z_sphere = np.genfromtxt(pwd_juno, usecols=(3)) / 1000
sphere_points = np.column_stack([x_sphere, y_sphere, z_sphere])

def PMT_tangent_vectors(PMT_points):
    R = np.linalg.norm(PMT_points, axis=1, keepdims=True)
    x, y, z = PMT_points[:, 0], PMT_points[:, 1], PMT_points[:, 2]
    norm_fac = 1 / np.sqrt(x**2 + y**2)
    e_1 = np.column_stack([-y * norm_fac, x * norm_fac, np.zeros_like(x)])
    e_2 = np.column_stack([-x * z * norm_fac, -y * z * norm_fac, (x**2 + y**2) * norm_fac]) / R
    return e_1, e_2

def find_sphere_intersections(pos, P, R):
    v = P - pos
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / v_norm  # Normalize v
    a = np.sum(v**2, axis=1)
    b = 2 * np.sum(v * pos, axis=1)
    c = np.dot(pos, pos) - R**2
    discriminant = b**2 - 4 * a * c
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)
    t = np.maximum(t1, t2)
    intersections = pos + t[:, None] * v
    normals = intersections / R
    return intersections, normals, t

def compute_projected_area_3D(intersections):
    if intersections.shape[0] < 3:
        return 0
    hull = ConvexHull(intersections)
    return hull.area / 2

def true_data_distribution(num_file, WaveformQ, PmtID_WF, TrueEvtID):
    adc_count = create_adc_count(WaveformQ, no_pedestal=False)
    row_sums = np.sum(adc_count, axis=1)
    true_row_sums = np.zeros(17612)
    for i, pmt_on in enumerate(PmtID_WF):
        true_row_sums[pmt_on] = row_sums[i]
    detsim_file = detsim.load_file(num_file)
    tree_det = detsim_file.Get("evt")
    edep, true_x, true_y, true_z = detsim.load_branches(tree_det, "edep", "edepX", "edepY", "edepZ")
    tree_det.GetEntry(TrueEvtID)
    edep, true_x, true_y, true_z = detsim.load_attributes(tree_det, "edep", "edepX", "edepY", "edepZ")
    true_x, true_y, true_z = true_x/1000, true_y/1000, true_z/1000
    true_pos = [true_x, true_y, true_z]
    data_distribution = true_row_sums/sum(true_row_sums)
    return edep, true_pos, data_distribution

def model_MC(energy, pos):
    vertex = np.array(pos)
    norm_sphere_points = (sphere_points - vertex) / np.linalg.norm(sphere_points - vertex, axis=1, keepdims=True)
    n_simu = 1  # Number of simulations
    nb_photon = int(energy * 1e4)
    batch_size = 100
    nb_batch = nb_photon // batch_size
    for _ in range(n_simu):
        scores = np.column_stack((np.arange(17612), np.zeros(17612, dtype=int)))
        for batch in range(nb_batch):
            theta = np.random.uniform(0, 2 * np.pi, batch_size)
            phi = np.arccos(2 * np.random.uniform(0, 1, batch_size) - 1)
            U = np.sin(phi) * np.cos(theta)
            V = np.sin(phi) * np.sin(theta)
            W = np.cos(phi)
            random_photons = np.column_stack((U, V, W))  # Unit vectors in random directions
            random_photons *= 1  # If needed, scale by any distance (set to 1 here for unit distance)
            random_photons = random_photons + vertex
            distrib = np.arccos(np.dot(random_photons - vertex, norm_sphere_points.T))
            num_pmt = np.argmin(distrib, axis=1)
            occurrences = np.bincount(num_pmt, minlength=17612)
            scores[num_pmt, 1] += occurrences[num_pmt]
    MC_dsitribution = scores[:,1]/sum(scores[:,1])
    return MC_dsitribution

def model_TH(pos):
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, r_disk, 50)
    r, theta = np.meshgrid(r, theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    solid_angle_of_projection = []
    e_1, e_2 = PMT_tangent_vectors(sphere_points)
    for n_PMT in range(len(x_sphere)):
        PMT = sphere_points[n_PMT]
        disk_x = PMT[0] + r * (cos_theta * e_1[n_PMT, 0] + sin_theta * e_2[n_PMT, 0])
        disk_y = PMT[1] + r * (cos_theta * e_1[n_PMT, 1] + sin_theta * e_2[n_PMT, 1])
        disk_z = PMT[2] + r * (cos_theta * e_1[n_PMT, 2] + sin_theta * e_2[n_PMT, 2])
        disk_points = np.column_stack([disk_x.ravel(), disk_y.ravel(), disk_z.ravel()])
        intersections, normals, t_values = find_sphere_intersections(pos, disk_points, R_JUNO)
        valid_mask = t_values >= 0
        valid_intersections = intersections[valid_mask]
        valid_normals = normals[valid_mask]
        solid_angle = 0
        if valid_intersections.shape[0] > 2:
            area_projected = compute_projected_area_3D(valid_intersections)
            N = valid_intersections.shape[0]
            dS = area_projected / N if N > 0 else 0
            r_vecs = valid_intersections - pos
            r_norms = np.linalg.norm(r_vecs, axis=1)
            r_dot_n = np.einsum('ij,ij->i', r_vecs, valid_normals)
            solid_angle = np.sum((r_dot_n / r_norms**3) * dS)
        solid_angle_of_projection.append(solid_angle)
    TH_distrib = np.array(solid_angle_of_projection)/(4*np.pi)
    TH_distrib /= sum(TH_distrib)
    return TH_distrib

def plot_distributions(num_file, num_entry, data_distribution, MC_distribution, TH_distribution, num_pmt, true_pos, edep, kl):
    pwd_saving = "/sps/l2it/lperon/JUNO/data_nf/figures/distributions/"
    plt.plot(num_pmt, data_distribution, "-b", label="True JUNO")
    plt.plot(num_pmt, MC_distribution, "-r", label="MC model")
    plt.plot(num_pmt, TH_distribution, "-k", label="TH proba", alpha=0.5)
    plt.xlabel("PMT Index")
    plt.ylabel("Hit or energy distribution by PMT")
    plt.title(f"file : {num_file}, entry : {num_entry},\n vertex = [{true_pos[0]:.2f},{true_pos[1]:.2f},{true_pos[2]:.2f}],\n energy = {edep:.2f} MeV, KL(JUNO||TH) = {kl:.2f}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(pwd_saving+f"file_{num_file}_entry_{num_entry}_distributions.svg")
    plt.savefig(pwd_saving+f"file_{num_file}_entry_{num_entry}_distributions.png")
    plt.close()

def plot_kl(kl_list, num_file):
    pwd_saving = "/sps/l2it/lperon/JUNO/data_nf/figures/kl_div/"
    plt.plot(range(len(kl_list)), kl_list, "kd")
    plt.xlabel(f"Entry number in file {num_file}")
    plt.ylabel("KL(JUNO||TH)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(pwd_saving+f"KL_divergence_file_{num_file}.svg")
    plt.savefig(pwd_saving+f"KL_divergence_file_{num_file}.png")
    plt.close()

def kl_div_data_model(num_file):
    file = elec.load_file(num_file)
    tree_elec = file.Get("evt")
    TrueEvtID, WaveformQ, PmtID_WF = elec.load_branches(tree_elec, "TrueEvtID", "WaveformQ", "PmtID_WF")
    kl_list = []
    for num_entry, entry in enumerate(tree_elec):
        print(num_entry)
        TrueEvtID, WaveformQ, PmtID_WF = elec.load_attributes(entry, "TrueEvtID", "WaveformQ", "PmtID_WF")
        if len(TrueEvtID)==1:
            TrueEvtID = TrueEvtID[0]
            edep, true_pos, data_distribution = true_data_distribution(num_file, WaveformQ, PmtID_WF, TrueEvtID)
            TH_distribution = model_TH(true_pos)
            num_pmt = range(len(TH_distribution))
            kl = sum(rel_entr(data_distribution, TH_distribution))
            kl_list.append(kl)
            MC_distribution = model_MC(edep, true_pos)
            np.savetxt(f"/sps/l2it/lperon/JUNO/data_nf/mc_distrib/MC_distrib_file_{num_file}_entry_{num_entry}.txt", MC_distribution)
            np.savetxt(f"/sps/l2it/lperon/JUNO/data_nf/th_distrib/TH_distrib_file_{num_file}_entry_{num_entry}.txt", TH_distribution)
            np.savetxt(f"/sps/l2it/lperon/JUNO/data_nf/JUNO_distrib/JUNO_distrib_file_{num_file}_entry_{num_entry}.txt", data_distribution)
            plot_distributions(num_file, num_entry, data_distribution, MC_distribution, TH_distribution, num_pmt, true_pos, edep, kl)
    kl_list = np.array(kl_list)
    np.savetxt(f"/sps/l2it/lperon/JUNO/data_nf/kl_div/kl_list_file_{num_file}.txt", kl_list)
    
    plot_kl(kl_list, num_file)
    return 0