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

# Define the Real NVP coupling layer
class RealNVPCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RealNVPCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh(),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if reverse:
            s = self.scale_net(x1)
            t = self.translation_net(x1)
            x2 = (x2 - t) * torch.exp(-s)
        else:
            s = self.scale_net(x1)
            t = self.translation_net(x1)
            x2 = x2 * torch.exp(s) + t
        return torch.cat([x1, x2], dim=1), s.sum(dim=1)

# Define the Real NVP model
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_coupling_layers):
        super(RealNVP, self).__init__()
        self.latent_dim = latent_dim
        self.coupling_layers = nn.ModuleList([
            RealNVPCouplingLayer(input_dim, hidden_dim) for _ in range(num_coupling_layers)
        ])
        self.fc_out = nn.Linear(input_dim, latent_dim)  # Output layer to reduce to latent dimension

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.coupling_layers:
            x, log_det = layer(x)
            log_det_jacobian += log_det
        z = self.fc_out(x)  # Map to latent dimension
        return z, log_det_jacobian

    def inverse(self, z):
        x = self.fc_out.inverse(z)  # Inverse operation
        for layer in reversed(self.coupling_layers):
            x, _ = layer(x, reverse=True)
        return x

# Step 1: Load true positions from a file
def load_true_positions(true_positions_file):
    """Load true positions from a given file."""
    return np.loadtxt(true_positions_file)

# Step 2: Load data files, matching each data file with the corresponding true position
def load_data_from_files(data_dir, num_file_min, num_file_max):
    """
    Load data from multiple sets of files with the prefix f"JUNO_distrib_file_{num_file}_entry",
    and match each set with the corresponding true positions from f"true_positions_file_{num_file}.txt".
    """
    all_positions = []
    all_data_points = []

    for num_file in range(num_file_min, num_file_max + 1):
        if num_file == 62:  # Skip file number 62
            continue
        # Construct the true positions file path
        true_positions_file = f"/sps/l2it/lperon/JUNO/data_cnn/2d/true_positions_file_{num_file}.txt"
        # Load the true positions for this num_file
        true_positions = load_true_positions(true_positions_file)
        
        # List data files for the current num_file
        file_prefix = f"JUNO_distrib_file_{num_file}_entry"
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(file_prefix)]
        
        # Sort the files by the num_entry (i.e., by their suffix number)
        file_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Load data from each file and match with the corresponding true position
        for i, file_path in enumerate(file_paths):
            # Load the data points from the file
            data = np.loadtxt(file_path)
            
            # Append true position (scaled if needed) and data points
            all_positions.append(true_positions[i]/1000)  # Normalize true positions
            all_data_points.append(data)
    
    return np.array(all_positions), np.array(all_data_points)

data_dir = "/sps/l2it/lperon/JUNO/data_nf/JUNO_distrib/"  # Set this to the directory containing your data files
num_file_min = 0  # Starting num_file
num_file_max = 118  # Ending num_file

# Check for GPU availability and clear VRAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
true_positions, data = load_data_from_files(data_dir, num_file_min, num_file_max)

# Split data
train_data, test_data, train_positions, test_positions = train_test_split(
    data, true_positions, test_size=0.3)

# Convert to PyTorch tensors (CPU for now)
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_positions = torch.tensor(train_positions, dtype=torch.float32)
test_positions = torch.tensor(test_positions, dtype=torch.float32)

# Create data loaders
train_loader = DataLoader(TensorDataset(train_data, train_positions), batch_size=512, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_positions), batch_size=512, shuffle=False)

# Initialize model, optimizer, and loss function
input_dim = train_data.shape[1]
hidden_dim = 512
latent_dim = 3
num_coupling_layers = 25

del train_data, test_data, train_positions, test_positions

model = RealNVP(input_dim, hidden_dim, latent_dim, num_coupling_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
def train(model, train_loader, test_loader, num_epochs=500):
    save_path = "/pbs/home/l/lperon/work_JUNO/models/NF/best_model.pth"
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    total_batches = num_epochs * len(train_loader)  # Total number of batches

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            # Move batch to GPU if available
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z, _ = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Free intermediate tensors to optimize memory usage
            del x, y, z, loss
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                # Move batch to GPU if available
                x, y = x.to(device), y.to(device)
                z, _ = model(x)
                loss = criterion(z, y)
                test_loss += loss.item()

                # Free intermediate tensors
                del x, y, z, loss
                torch.cuda.empty_cache()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # Save model if the current test loss is the best so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: New best test loss {test_loss:.6f}, model saved to {save_path}")

    return train_losses, test_losses

# Train the model
train_losses, test_losses = train(model, train_loader, test_loader)

# Visualize training progress
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/training_loss.png")
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/training_loss.svg")
plt.close()

plt.figure(figsize=(12, 6))
plt.loglog(train_losses, label="Train Loss")
plt.loglog(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss (Log-Log)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/training_loss_loglog.png")
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/training_loss_loglog.svg")
plt.close()

# Evaluate the model and visualize prediction errors
train_pred_positions = []
train_real_positions = []

with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        z, _ = model(x)
        train_pred_positions.append(z.cpu())
        train_real_positions.append(y.cpu())

train_pred_positions = torch.cat(train_pred_positions).numpy()
train_real_positions = torch.cat(train_real_positions).numpy()

train_errors = train_pred_positions - train_real_positions

model.eval()
pred_positions = []
real_positions = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        z, _ = model(x)
        pred_positions.append(z.cpu())
        real_positions.append(y.cpu())

pred_positions = torch.cat(pred_positions).numpy()
real_positions = torch.cat(real_positions).numpy()

errors = pred_positions - real_positions

# Combine histograms for training and testing errors
fig, axes = plt.subplots(1, 3, figsize=(25, 9))
coords = ["x", "y", "z"]
for i in range(3):
    # Plot histogram for test errors
    test_hist_data = errors[:, i]
    bins = 50
    test_counts, test_bin_edges, _ = axes[i].hist(
        test_hist_data, bins=bins, color="red", alpha=0.5, density=True, label="Test Errors"
    )

    # Plot histogram for training errors
    train_hist_data = train_errors[:, i]
    train_counts, train_bin_edges, _ = axes[i].hist(
        train_hist_data, bins=bins, color="blue", alpha=0.5, density=True, label="Train Errors"
    )

    # Fit a Gaussian for test errors
    test_mean, test_std = norm.fit(test_hist_data)
    test_x = np.linspace(test_bin_edges[0], test_bin_edges[-1], 1000)
    test_gaussian = norm.pdf(test_x, test_mean, test_std)

    # Fit a Gaussian for train errors
    train_mean, train_std = norm.fit(train_hist_data)
    train_x = np.linspace(train_bin_edges[0], train_bin_edges[-1], 1000)
    train_gaussian = norm.pdf(train_x, train_mean, train_std)

    # Plot the Gaussian curves
    axes[i].plot(test_x, test_gaussian, '-', color="cyan", label=f"Test Gaussian ($\mu$={test_mean:.3f}, $\sigma$={test_std:.3f})")
    axes[i].plot(train_x, train_gaussian, '-', color="orange", label=f"Train Gaussian ($\mu$={train_mean:.3f}, $\sigma$={train_std:.3f})")

    # Add title and labels
    axes[i].set_title(f"Error in {coords[i]} coordinate")
    axes[i].set_xlabel("Error [m]")
    axes[i].set_ylabel("Density")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/error_histograms_with_train.png")
plt.savefig("/pbs/home/l/lperon/work_JUNO/figures/NF/error_histograms_with_train.svg")
plt.close()