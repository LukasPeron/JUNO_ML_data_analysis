"""
Deep Learning Models for JUNO 2D CNN Data

CREATED BY : LUKAS PERON  
LAST UPDATE : [DATE]  
PURPOSE : DEFINE CNN MODELS AND LOAD FUNCTIONALITY FOR JUNO SIMULATION DATA  

This script defines various CNN architectures for processing 2D CNN data generated from JUNO simulations.  
It also includes utilities for loading models and creating instances of standard architectures such as ResNet and VGG.

Overview:
---------
1. **`SimpleCNN`**:
   - A custom CNN architecture for processing large (4368 × 2184) image data.
   - Features convolutional, batch normalization, and fully connected layers.

2. **`load_model`**:
   - Prepares data loaders, initializes the `SimpleCNN` model, and sets up the optimizer and loss function.

3. **`create_resnet`**:
   - Creates a pre-trained ResNet50 model adapted for JUNO data.

4. **`create_vgg`**:
   - Creates a pre-trained VGG16 model adapted for JUNO data.

Dependencies:
-------------
- PyTorch
- torchvision (for pre-trained models)
- numpy
- gc (garbage collection for memory optimization)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
from torchvision import models

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
    image_size : tuple, optional (default=(4368, 2184))
        Dimensions of the input image.

    Notes:
    ------
    - Designed for processing large (4368 × 2184) images.
    - Includes multiple convolutional, batch normalization, ReLU, and pooling layers.
    - Flattened convolutional features are passed through fully connected layers for regression tasks.
    """

    def __init__(self, input_channels=3, num_outputs=1, image_size=(4368, 2184)):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=32, stride=5, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=32, stride=5, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 2048, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 4096, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(4096),
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
            nn.Tanh(),
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
            x = self.conv_layers(x)
            return x.view(x.size(0), -1).size(1)

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
        x = x.view(x.size(0), -1)
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
    num_outputs = 1 if label_type == "energy" else 3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cnn = SimpleCNN(num_outputs=num_outputs, image_size=(4368, 2184)).to(device)
    num_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    return train_loader, test_loader, cnn, criterion, optimizer

def create_resnet(num_outputs=1):
    """
    Create a pre-trained ResNet50 model for JUNO data.

    Parameters:
    -----------
    num_outputs : int, optional (default=1)
        Number of output nodes.

    Returns:
    --------
    torch.nn.Module
        ResNet50 model with updated output layer.
    """
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_outputs)
    return resnet

def create_vgg(num_outputs=1):
    """
    Create a pre-trained VGG16 model for JUNO data.

    Parameters:
    -----------
    num_outputs : int, optional (default=1)
        Number of output nodes.

    Returns:
    --------
    torch.nn.Module
        VGG16 model with updated output layer.
    """
    vgg = models.vgg16(pretrained=True)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_outputs)
    return vgg
