import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm  # For Gaussian fitting
import gc
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
matplotlib.rcParams.update({'font.size': 18})

# Clear unnecessary variables and empty the GPU cache
gc.collect()
torch.cuda.empty_cache()

print("All necessary imports and clearing done.")

# Define the Real NVP coupling layer
class RealNVPCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(RealNVPCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask = mask  # Checkerboard mask (binary tensor)

        # Networks for scale and translation
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, reverse=False):
        # Apply mask to split x into active and passive components
        self.mask = self.mask.to(x.device)
        x_active = x * self.mask  # Active part controlled by the mask
        x_passive = x * (1 - self.mask)  # Unaffected part

        if reverse:
            s = self.scale_net(x_active)
            t = self.translation_net(x_active)
            x_passive = (x_passive - t) * torch.exp(-s)  # Inverse transformation
        else:
            s = self.scale_net(x_active)
            t = self.translation_net(x_active)
            x_passive = x_passive * torch.exp(s) + t  # Forward transformation

        # Combine active and passive components
        x = x_active + x_passive
        return x, (s * (1 - self.mask)).sum(dim=1)

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_coupling_layers):
        super(RealNVP, self).__init__()
        self.output_dim = output_dim
        self.coupling_layers = nn.ModuleList()

        # Alternating checkerboard masks for the coupling layers
        for i in range(num_coupling_layers):
            mask = self._create_checkerboard_mask(input_dim, invert=(i % 2 == 1))
            self.coupling_layers.append(RealNVPCouplingLayer(input_dim, hidden_dim, mask))

        self.fc_out = nn.Linear(input_dim, output_dim)  # Output layer to reduce to latent dimension

    @staticmethod
    def _create_checkerboard_mask(input_dim, invert=False):
        """
        Create a checkerboard mask for the given input dimension.
        Alternates between 0 and 1 for even and odd indices.
        """
        mask = torch.arange(input_dim) % 2  # 0 for even indices, 1 for odd indices
        if invert:
            mask = 1 - mask  # Flip the mask
        return mask.float()

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.coupling_layers:
            x, log_det = layer(x)
            log_det_jacobian += log_det
        z = self.fc_out(x)  # Map to latent dimension
        return z, log_det_jacobian

    def inverse(self, z):
        # Compute the inverse of the fully connected layer
        if self.fc_out.bias is not None:
            z = z - self.fc_out.bias
        weight_inv = torch.linalg.pinv(self.fc_out.weight)  # Pseudo-inverse
        x = torch.matmul(z, weight_inv.T)

        # Reverse the Real NVP coupling layers
        for layer in reversed(self.coupling_layers):
            x, _ = layer(x, reverse=True)
        return x

# Step 1: Load data files, matching each data file with the corresponding true position
def load_data_from_files(label_type, data_dir, num_file_min, num_file_max):
    """
    Load data from multiple sets of files with the prefix f"JUNO_distrib_file_{num_file}_entry",
    and match each set with the corresponding true positions from f"true_positions_file_{num_file}.txt".
    """
    
    all_labels = []
    all_data_points = []

    for num_file in range(num_file_min, num_file_max + 1):
        if num_file == 62:  # Skip file number 62
            continue
        if label_type == "positions":
            # Construct the true positions file path
            true_label_file = f"/sps/l2it/lperon/JUNO/data_cnn/2d/true_positions_file_{num_file}.txt"
        if label_type == "energy":
            # Construct the true energy file path
            true_label_file = f"/sps/l2it/lperon/JUNO/data_cnn/2d/true_energy_file_{num_file}.txt"
        # Load the true positions for this num_file
        true_label = np.loadtxt(true_label_file)
        
        # List data files for the current num_file
        file_prefix = f"JUNO_distrib_file_{num_file}_entry"
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(file_prefix)]
        
        # Sort the files by the num_entry (i.e., by their suffix number)
        file_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Load data from each file and match with the corresponding true position
        for i, file_path in enumerate(file_paths):
            print(file_path)
            # Load the data points from the file
            data = np.loadtxt(file_path)
            
            # Append true position (scaled if needed) and data points
            if label_type == "positions":
                all_labels.append(true_label[i]) # true position in m
            if label_type == "energy":
                all_labels.append(true_label[i]) # energy in MeV
            all_data_points.append(data)
    
        print(f"File {num_file}/{num_file_max} loaded.")        

    return np.array(all_labels), np.array(all_data_points)

def evaluate_best_model(model, train_loader, test_loader):
    """
    Evaluate the saved best model and calculate the prediction errors.
    """
    train_pred = []
    train_real = []

    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            z, _ = model(x)
            train_pred.append(z.cpu())
            train_real.append(y.cpu())

    train_pred = torch.cat(train_pred).numpy()
    train_real = torch.cat(train_real).numpy()

    # Ensure shapes match before subtraction
    if train_pred.ndim == 2 and train_real.ndim == 1:
        train_real = train_real[:, np.newaxis]
        

    train_errors = train_real - train_pred

    model.eval()
    test_pred = []
    test_real = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            z, _ = model(x)
            test_pred.append(z.cpu())
            test_real.append(y.cpu())

    test_pred = torch.cat(test_pred).numpy()
    test_real = torch.cat(test_real).numpy()

    # Ensure shapes match before subtraction
    if test_pred.ndim == 2 and test_real.ndim == 1:
        test_real = test_real[:, np.newaxis]
        

    test_errors = test_real - test_pred
    
    return train_errors, test_errors, train_pred, test_pred, train_real, test_real

# Update dynamic file saving based on model parameters and label type
def generate_file_name(base_path, label_type, input_dim, hidden_dim, output_dim, num_coupling_layers):
    """
    Generate a file name that includes model details and the label type.
    """
    return (
        f"{base_path}_{label_type}_dim{input_dim}_hidden{hidden_dim}_latent{output_dim}_layers{num_coupling_layers}"
    )

data_dir = "/sps/l2it/lperon/JUNO/data_nf/JUNO_distrib/"  # Set this to the directory containing your data files
num_file_min = 0  # Starting num_file
num_file_max = 118  # Ending num_file
label_type = "energy"#input("Enter the label type (positions or energy): ")  # Choose "positions" or "energy"
assert label_type in ["positions", "energy"], "Label type must be 'positions' or 'energy'."

# Check for GPU availability and clear VRAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
true_label, data = load_data_from_files(label_type, data_dir, num_file_min, num_file_max)

print(f"Loaded {len(data)} data points with {len(data[0])} features each.")

# Split data
train_data, test_data, train_label, test_label = train_test_split(
    data, true_label, test_size=0.3)

# Convert to PyTorch tensors (CPU for now)
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.float32)

# Create data loaders
train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
input_dim = 17612
hidden_dim = 512
if label_type == "positions":
    output_dim = 3
if label_type == "energy":
    output_dim = 1
num_coupling_layers = 25

# Directory paths
model_save_dir = "/sps/l2it/lperon/JUNO/models/NF/"
figure_save_dir = "/pbs/home/l/lperon/work_JUNO/figures/NF/"

# File name templates
model_file_base = os.path.join(model_save_dir, "model")
plot_file_base_loss = os.path.join(figure_save_dir, "training_loss")
plot_file_base_hist = os.path.join(figure_save_dir, "histograms")

# Generate specific file names
model_file_name = generate_file_name(
    model_file_base, label_type, input_dim, hidden_dim, output_dim, num_coupling_layers)
plot_file_name_loss = generate_file_name(
    plot_file_base_loss, label_type, input_dim, hidden_dim, output_dim, num_coupling_layers)
plt_file_name_hist = generate_file_name(
    plot_file_base_hist, label_type, input_dim, hidden_dim, output_dim, num_coupling_layers)

print(model_file_name)

del train_data, test_data, train_label, test_label

model = RealNVP(input_dim, hidden_dim, output_dim, num_coupling_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("Model, optimizer, and loss function initialized.")

# Train loop adjustment
def train(model, train_loader, test_loader, num_epochs=100):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z, _ = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                z, _ = model(x)
                loss = criterion(z, y)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_file_name+".pth")
            print(f"Epoch {epoch + 1}: New best test loss {test_loss:.6f}, model saved to {model_file_name+".pth"}")

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    return train_losses, test_losses, best_epoch

# Train the model
# train_losses, test_losses, best_epoch = train(model, train_loader, test_loader, num_epochs=100)

# Visualization for losses
# plt.figure(figsize=(12, 6))
# plt.plot(train_losses, color="blue", label="Train Loss", linewidth=2)
# plt.plot(test_losses, color="red", label="Test Loss", linewidth=2)
# plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch + 1})", linewidth=2)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Testing Loss")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(plot_file_name_loss+"loss.png")
# plt.savefig(plot_file_name_loss+"loss.svg")
# plt.close()

# plt.figure(figsize=(12, 6))
# plt.loglog(train_losses, color="blue", label="Train Loss", linewidth=2)
# plt.loglog(test_losses, color="red", label="Test Loss", linewidth=2)
# plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch + 1})", linewidth=2)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Testing Loss")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(plot_file_name_loss+"loss_loglog.png")
# plt.savefig(plot_file_name_loss+"loss_loglog.svg")
# plt.close()

# print(f"Loss plots saved.")

# Load best model
model.load_state_dict(torch.load(model_file_name+".pth", weights_only=True))

train_errors, test_errors, train_pred, test_pred, train_real, test_real = evaluate_best_model(model, train_loader, test_loader)

nb_test_event = len(test_errors)
nb_train_event = len(train_errors)

# Combine histograms for training and testing errors
if label_type == "positions":
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    coords = ["x", "y", "z"]
    for i in range(3):
        bins = 50
        # Plot histogram for training errors
        train_hist_data = train_errors[:, i]*1000  # Convert to mm
        train_counts, train_bin_edges, _ = axes[i].hist(
            train_hist_data, bins=bins, color="b", alpha=0.7, density=True, label=f"Train\n $\langle \Delta {coords[i]}\\rangle = {np.mean(train_hist_data):.3f}$\n $\sigma_{coords[i]}={np.std(train_hist_data):.3f}$"
        )
        # Plot histogram for test errors
        test_hist_data = test_errors[:, i]*1000  # Convert to mm
        test_counts, test_bin_edges, _ = axes[i].hist(
            test_hist_data, bins=bins, color="r", alpha=0.7, density=True, label=f"Test\n $\langle \Delta {coords[i]}\\rangle = {np.mean(test_hist_data):.3f}$\n $\sigma_{coords[i]}={np.std(test_hist_data):.3f}$"
        )

        # Add title and labels
        axes[i].set_xlabel(f"${coords[i]}"+r"_{true}-"+f"{coords[i]}"+r"_{model}$ [mm]")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.savefig(plt_file_name_hist+".png")
    plt.savefig(plt_file_name_hist+".svg")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    coords = ["x", "y", "z"]
    for i in range(3):
        # Plot histogram for test errors
        test_real_data = test_real[:, i]*1000  # Convert to mm
        test_pred_data = test_pred[:, i]*1000  # Convert to mm
        axes[i].plot(range(nb_test_event), test_real_data, color="darkblue", label="Test Real")
        axes[i].plot(range(nb_test_event), test_pred_data, color="deepskyblue", label="Test Pred")
        axes[i].set_xlabel("Event")
        axes[i].set_ylabel(f"{coords[i]} [mm]")
        axes[i].legend()
        axes[i].grid()
    plt.tight_layout()
    plt.savefig(plt_file_name_hist+"test_real_pred.png")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    coords = ["x", "y", "z"]
    for i in range(3):
        # Plot histogram for test errors
        train_real_data = train_real[:, i]*1000  # Convert to mm
        train_pred_data = train_pred[:, i]*1000 # Convert to mm
        axes[i].plot(range(nb_train_event), train_real_data, color="darkred", label="Train Real")
        axes[i].plot(range(nb_train_event), train_pred_data, color="indianred", label="Train Pred")
        axes[i].set_xlabel("Event")
        axes[i].set_ylabel(f"{coords[i]} [mm]")
        axes[i].legend()
        axes[i].grid()
    plt.tight_layout()
    plt.savefig(plt_file_name_hist+"train_real_pred.png")
    plt.close()

if label_type == "energy":
    fig, axes = plt.subplots(1, 1, figsize=(12,6))
    bins = 50
    # Plot histogram for training errors
    train_counts, train_bin_edges, _ = axes.hist(
        train_errors, bins=bins, color="b", alpha=0.7, density=True, label=f"Train\n $\langle \Delta E \\rangle = {np.mean(train_errors):.3f}$\n $\sigma_E={np.std(train_errors):.3f}$"
    )
    # Plot histogram for test errors
    test_counts, test_bin_edges, _ = axes.hist(
        test_errors, bins=bins, color="r", alpha=0.7, density=True, label=f"Test\n $\langle \Delta E \\rangle = {np.mean(test_errors):.3f}$\n $\sigma_E={np.std(test_errors):.3f}$"
    )

    # Add title and labels
    axes.set_xlabel(r"$E_{true}-E_{model}$ [MeV]")
    axes.set_ylabel("Density")
    axes.legend()
    axes.grid()

    plt.tight_layout()
    plt.savefig(plt_file_name_hist+".png")
    plt.savefig(plt_file_name_hist+".svg")
    plt.close()
