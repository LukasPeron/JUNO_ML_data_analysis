import numpy as np
import torch

# Define the paths for data loading and saving
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pwd = "/sps/l2it/lperon/JUNO/txt/data_profiling/"

# Load ELECSIM and DETSIM data
def load_data(num_file_min=0, num_file_max=118):
    X = np.loadtxt(pwd + f"elecsim_data_file{num_file_min}.txt")
    y = np.loadtxt(pwd + f"detsim_data_file{num_file_min}.txt")
    print(f"Loaded file : {num_file_min}")
    for i in range(num_file_min+1, num_file_max+1):
        X = np.concatenate((X, np.loadtxt(pwd + f"elecsim_data_file{i}.txt")), axis=0)
        y = np.concatenate((y, np.loadtxt(pwd + f"detsim_data_file{i}.txt")), axis=0)
        print(f"Loaded file : {i}")
    return X, y

def scale_data(X, y):
    # Data preprocessing (scaling)
    y = y.T
    y[0] = y[0] / 100  # Scale energy
    y[1:4] = y[1:4] / 17015  # Scale spatial coordinates
    y = y.T

    # Reshape and scale the input features (PMT data)
    X_temp = X.reshape(X.shape[0], -1, 3)
    X_temp[:, :, 1] /= 1e5  # Scale second feature
    X_temp[:, :, 2] = X_temp[:, :, 1] / 100  # Scale third feature
    X = X_temp.reshape(X.shape)

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    return X_tensor, y_tensor