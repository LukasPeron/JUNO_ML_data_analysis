"""
Deep Learning Models for JUNO 2D CNN Data

CREATED BY   : LUKAS PERON  
LAST UPDATED : 14/01/2025  
PURPOSE      : Define CNN models and provide functionality for processing JUNO simulation data.

Description:
------------
This script implements various Convolutional Neural Network (CNN) architectures and utilities for processing
2D CNN data generated from JUNO simulations. The main functionalities include:

1. `SimpleCNN`:  
   - A custom CNN architecture designed for large image data (400x200).
   - Contains convolutional layers, batch normalization, ReLU activations, pooling layers, and fully connected layers.  
   - Suitable for regression tasks (e.g., energy prediction or 3D position estimation).

2. `load_model`:  
   - Initializes the `SimpleCNN` model, prepares data loaders, and sets up the optimizer and loss function for training.

3. `create_non_pretrained_resnet`:  
   - Generates a ResNet50 model (non-pretrained) adapted for JUNO data.  

4. `create_non_pretrained_vgg`:  
   - Generates a VGG16 model (non-pretrained) adapted for JUNO data.

Dependencies:
-------------
- PyTorch  
- torchvision (for pre-trained models)  
- numpy  
- gc (garbage collection for memory management)

Features:
---------
- Supports GPU acceleration (if available).  
- Implements memory optimization techniques by clearing unused variables and caches.  
- Provides a flexible framework for CNN-based JUNO data analysis.  

"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

# Clear unnecessary variables and empty the GPU cache
gc.collect()
torch.cuda.empty_cache()

# Set the device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleCNN(nn.Module):
    """
    Custom Convolutional Neural Network (CNN) for JUNO Simulation Data.

    Parameters:
    -----------
    input_channels : int, optional (default=3)
        Number of input channels in the image (e.g., 3 for RGB images).
    num_outputs : int, optional (default=1)
        Number of output nodes (e.g., 1 for energy, 3 for positions).
    image_size : tuple, optional (default=(400, 200))
        Dimensions of the input image.

    Notes:
    ------
    - Designed for processing large (400 x 200) images.
    - Includes multiple convolutional, batch normalization, ReLU, and pooling layers.
    - Flattened convolutional features are passed through fully connected layers for regression tasks.
    """

    def __init__(self, input_channels=3, num_outputs=1, image_size=(400, 200)):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Compute the flattened size after convolutional layers
        self._flattened_size = self._get_flattened_size(image_size, input_channels)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self._flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, num_outputs)
        )

    def _get_flattened_size(self, image_size, input_channels):
        """
        Compute the flattened size of the output from the convolutional layers.

        Parameters:
        -----------
        image_size : tuple
            Input image dimensions (height, width).
        input_channels : int
            Number of input channels.

        Returns:
        --------
        int
            Flattened size after passing through the convolutional layers.
        """
        with torch.no_grad():
            x = torch.zeros(1, input_channels, *image_size)
            for layer in self.conv_layers:
                x = layer(x)
            return x.reshape(x.size(0), -1).size(1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, height, width)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_outputs)`.
        """
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def load_model(label_type, train_dataset, test_dataset, batch_size, lr):
    """
    Prepare data loaders, initialize the SimpleCNN model, and configure training components.

    Parameters:
    -----------
    label_type : str
        Specifies the output type ("energy" for 1D output or "positions" for 3D output).
    train_dataset : torch.utils.data.Dataset
        Training dataset.
    test_dataset : torch.utils.data.Dataset
        Testing dataset.
    batch_size : int
        Batch size for training and testing.
    lr : float
        Learning rate for the optimizer.

    Returns:
    --------
    tuple
        (train_loader, test_loader, model, criterion, optimizer)
        - train_loader : DataLoader for the training set.
        - test_loader : DataLoader for the testing set.
        - model : Initialized SimpleCNN model.
        - criterion : Loss function (MSELoss).
        - optimizer : Optimizer (Adam).
    """
    if label_type == "energy":
        num_outputs = 1
    elif label_type == "positions":
        num_outputs = 3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cnn = SimpleCNN(num_outputs=num_outputs, image_size=(400, 200)).to(device)
    num_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    return train_loader, test_loader, cnn, criterion, optimizer

def create_non_pretrained_resnet(num_outputs):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)  # Adjust the final layer for your output
    return model

def create_non_pretrained_vgg(num_outputs):
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_outputs)  # Adjust the final layer for your output
    return model