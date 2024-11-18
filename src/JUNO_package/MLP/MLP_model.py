"""
Define and Initialize a Multilayer Perceptron (MLP) Model for JUNO Simulation Data

This script defines a fully connected Multilayer Perceptron (MLP) neural network architecture and initializes it
for training on JUNO simulation data. The model uses the PyTorch framework and performs regression to predict 
primary vertex information (energy and coordinates) based on PMT data.

Created by: Lukas Peron
Last Update: 18/11/2024

Overview:
---------
The script performs the following steps:
1. Define the `MLP_JUNO` model class, a sequential MLP with dropout and layer normalization for robust training.
2. Create data loaders for training and testing batches.
3. Initialize the model with the specified dropout, batch size, and learning rate, and configure the optimizer and loss function.

Error Handling:
---------------
- Assumes that the provided datasets are well-formed and compatible with the network architecture.
- If data dimensions do not match model expectations, PyTorch may raise `RuntimeError` during forward passes.

Dependencies:
-------------
- torch

Instructions:
-------------
1. Ensure that torch is installed and configured in your environment.
2. Use the `MLP_JUNO` class to define the model architecture.
3. Pass training and testing datasets to the `load_model` function to initialize data loaders, the model, 
   loss function, and optimizer for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import gc

# Clear unnecessary variables and empty the GPU cache
gc.collect()
torch.cuda.empty_cache()

# Set the device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP_JUNO(nn.Module):
    """
    Multilayer Perceptron (MLP) model for JUNO data regression.

    This MLP architecture uses several hidden layers with ReLU activation, dropout, and layer normalization.
    The model predicts the primary vertex information: either energy (E) or coordinates (x, y, z) based on input data
    from PMTs (Photomultiplier Tubes) in the JUNO experiment.

    Parameters:
    -----------
    dropout_prop : float
        Dropout probability to prevent overfitting.

    Layers:
    -------
    Input (17612 * 3) -> Linear(1000) -> LayerNorm -> ReLU -> Dropout -> Linear(500) -> LayerNorm -> ReLU -> Dropout -> 
    Linear(250) -> LayerNorm -> ReLU -> Dropout -> Linear(125) -> LayerNorm -> ReLU -> Dropout -> Linear(64) -> LayerNorm -> 
    ReLU -> Output(4D)

    Output:
    -------
    The model outputs a tensor with 1 or 3 values, corresponding to energy (E) or the coordinates (x, y, z).
    """
    def __init__(self, dropout_prop, data_type):
        super(MLP_JUNO, self).__init__()
        if data_type=="both":
            in_dim = 3*17612
            out_dim = 4
        elif data_type=="energy":
            in_dim = 2*17612
            out_dim = 1
        elif data_type=="spatial":
            in_dim = 2*17612
            out_dim = 3
        else:
            raise AssertionError('You must provide a valid data_type : "energy" or "spatial".')
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(1000, 500),
            nn.LayerNorm(500),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(500, 250),
            nn.LayerNorm(250),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(250, 125),
            nn.LayerNorm(125),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(125, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, out_dim)  # Output 4 (E, x, y, z)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, 3 * 17612) representing PMT data.

        Returns:
        --------
        torch.Tensor
            Output tensor with shape (batch_size, 4) containing the predictions for energy and coordinates.
        """
        return self.layers(x)

def load_model(data_type, train_dataset, test_dataset, batch_size=64, dropout_prop=0.2, lr=1e-5):
    """
    Initialize data loaders, the MLP model, loss function, and optimizer for training.

    Parameters:
    -----------
    train_dataset : torch.utils.data.Dataset
        Training dataset containing input and label tensors.
    test_dataset : torch.utils.data.Dataset
        Testing dataset containing input and label tensors.
    batch_size : int, optional
        Batch size for the DataLoader (default is 64).
    dropout_prop : float, optional
        Dropout probability for the MLP model (default is 0.2).
    lr : float, optional
        Learning rate for the Adam optimizer (default is 1e-5).

    Returns:
    --------
    tuple
        A tuple (train_loader, test_loader, model, criterion, optimizer) where:
        - train_loader : DataLoader for training data.
        - test_loader : DataLoader for testing data.
        - model : MLP_JUNO model instance with specified dropout.
        - criterion : Loss function (Mean Squared Error).
        - optimizer : Optimizer (Adam) configured with specified learning rate.
    """
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = MLP_JUNO(dropout_prop=dropout_prop, data_type=data_type).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # Define the loss function (MSE) and optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return train_loader, test_loader, model, criterion, optimizer