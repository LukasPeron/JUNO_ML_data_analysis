"""
Data Loading, Splitting, and Scaling for JUNO Simulation

This script contains utility functions to load, split, and scale ELECSIM and DETSIM data for the JUNO experiment.
The `load_data` function loads event data from multiple files and converts it to PyTorch tensors.
The `create_dataset` function splits the data into training and testing sets, and the `scale_data` function
applies standard scaling to each set to prepare the data for neural network training.

Overview:
---------
The script performs the following steps:
1. Load ELECSIM and DETSIM data from specified file ranges.
2. Split the dataset into training and testing sets.
3. Scale the features and labels in the dataset using standard scaling for neural network training.
4. Return the scaled datasets as PyTorch `TensorDataset` objects.

Error Handling:
---------------
- Assumes that each file exists and is well-formed. Missing or corrupted files will raise an `IOError`.
- If data dimensions are incorrect (mismatched shapes), numpy or PyTorch may raise a `ValueError` or `RuntimeError`.

Dependencies:
-------------
- numpy
- torch
- sklearn (for StandardScaler)

Instructions:
-------------
1. Ensure that numpy, torch, and sklearn are installed and configured in your environment.
2. Use the `load_data` function to load the ELECSIM and DETSIM data, specifying the file range if needed.
3. Pass the loaded data to `create_dataset` to split, scale, and convert it to PyTorch tensors.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# Define the device and paths for data loading and saving
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pwd = "/sps/l2it/lperon/JUNO/txt/data_profiling/"

def load_data(num_file_min=0, num_file_max=118):
    """
    Load ELECSIM and DETSIM data from a specified range of files and convert to PyTorch tensors.

    Parameters:
    -----------
    num_file_min : int, optional
        The starting file number (default is 0).
    num_file_max : int, optional
        The ending file number (default is 118).

    Returns:
    --------
    tuple
        A tuple (X, y) where X is the ELECSIM data and y is the DETSIM data as PyTorch tensors.
    
    Raises:
    -------
    IOError
        If a file in the specified range is missing or cannot be opened.
    """
    X = np.loadtxt(pwd + f"elecsim_data_file{num_file_min}.txt")
    y = np.loadtxt(pwd + f"detsim_data_file{num_file_min}.txt")
    print(f"Loaded file : {num_file_min}")
    for i in range(num_file_min + 1, num_file_max + 1):
        X = np.concatenate((X, np.loadtxt(pwd + f"elecsim_data_file{i}.txt")), axis=0)
        y = np.concatenate((y, np.loadtxt(pwd + f"detsim_data_file{i}.txt")), axis=0)
        print(f"Loaded file : {i}")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    return X, y

def create_dataset(X, y):
    """
    Split ELECSIM and DETSIM data into training and testing sets and apply scaling.

    Parameters:
    -----------
    X : torch.Tensor
        ELECSIM data as a PyTorch tensor.
    y : torch.Tensor
        DETSIM data as a PyTorch tensor.

    Returns:
    --------
    tuple
        A tuple (train_dataset, test_dataset, scaler_y_train, scaler_y_test) where train_dataset and test_dataset
        are scaled PyTorch `TensorDataset` objects, and scaler_y_train and scaler_y_test are `StandardScaler` objects
        for inverse transformation.
    """
    # Split the dataset into training and testing sets
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Convert the train and test data to numpy for scaling
    X_train = train_dataset[:][0].cpu().numpy()
    y_train = train_dataset[:][1].cpu().numpy()
    X_test = test_dataset[:][0].cpu().numpy()
    y_test = test_dataset[:][1].cpu().numpy()

    # Scale the data and return scaled datasets
    train_dataset, test_dataset, scaler_y_train, scaler_y_test = scale_data(X_train, y_train, X_test, y_test)
    return train_dataset, test_dataset, scaler_y_train, scaler_y_test

def scale_data(X_train, y_train, X_test, y_test):
    """
    Apply standard scaling to the training and testing sets.

    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data for ELECSIM (features).
    y_train : numpy.ndarray
        Training data for DETSIM (labels).
    X_test : numpy.ndarray
        Testing data for ELECSIM (features).
    y_test : numpy.ndarray
        Testing data for DETSIM (labels).

    Returns:
    --------
    tuple
        A tuple (train_dataset, test_dataset, scaler_y_train, scaler_y_test) where train_dataset and test_dataset
        are scaled PyTorch `TensorDataset` objects, and scaler_y_train and scaler_y_test are `StandardScaler` objects
        for inverse transformation.
    
    Notes:
    ------
    Standard scaling is applied separately to X_train, y_train, X_test, and y_test using sklearn's `StandardScaler`,
    which normalizes the data to have mean 0 and standard deviation 1.
    """
    # Standardize X and y using StandardScaler
    scaler_X_train = StandardScaler()
    scaler_y_train = StandardScaler()
    scaler_X_test = StandardScaler()
    scaler_y_test = StandardScaler()
    
    X_train = scaler_X_train.fit_transform(X_train)
    y_train = scaler_y_train.fit_transform(y_train)
    X_test = scaler_X_test.fit_transform(X_test)
    y_test = scaler_y_test.fit_transform(y_test)
    
    # Convert numpy arrays back to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32))

    return train_dataset, test_dataset, scaler_y_train, scaler_y_test
