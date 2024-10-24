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

# Define the neural network model
class MLP_JUNO(nn.Module):
    """
    Fully connected feedforward neural network for regression.

    The network consists of several linear layers with ReLU activations. The model predicts 
    the energy (E) and coordinates (x, y, z) of the primary vertex based on the input PMT data.

    Layers:
    -------
    Input -> Linear(1000) -> ReLU -> Linear(500) -> ReLU -> Linear(250) -> ReLU ->
    Linear(125) -> ReLU -> Linear(64) -> ReLU -> Output(4D)
    Between each Linear hidden layer LayerNorm and Dropout is added.
    Parameters:
    -----------
    X_train.shape[1]: int
        The input dimension based on the reshaped ELECSIM data.
    y_train.shape[1]: int
        The output dimension (4 for energy and spatial coordinates).
    """
    def __init__(self, input_shape, dropout_prop):
        super(MLP_JUNO, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 1000),
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
            nn.Linear(64, 4)  # Output 4 (E, x, y, z)
        )

    def forward(self, x):
        return self.layers(x)

def load_model(X_tensor, y_tensor, batch_size=50, dropout_prop=0.2, lr=1e-5):
    # Split the dataset into training and testing sets
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = MLP_JUNO(X_tensor.shape[1],dropout_prop=dropout_prop).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # Define the loss function (MSE) and optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return train_loader, test_loader, model, criterion, optimizer
