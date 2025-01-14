"""
Data Preprocessing for 2D CNN Using JUNO Simulation Data

CREATED BY   : Lukas Péron  
LAST UPDATED : 14/01/2025  
PURPOSE      : Prepare JUNO simulation data for 2D CNN training, including loading, scaling, and structuring datasets.

Overview:
---------
This script provides functions to preprocess JUNO simulation data for training a 2D CNN. 
It includes functionalities to:
1. Load image data and corresponding labels.
2. Split the dataset into training and testing subsets.
3. Scale labels and input data for efficient training.

Key Functionalities:
---------------------
1. **`load_data`**:  
   - Loads high-resolution (4368 × 2184) image data and labels from disk.  
   - Converts image data to NumPy arrays and labels to continuous arrays for processing.

2. **`createDataset`**:  
   - Splits the data into training (80%) and testing (20%) subsets.  
   - Optionally applies scaling to the labels for better model convergence.

3. **`scale_data`**:  
   - Scales input data and labels using `StandardScaler` from `sklearn`.

4. **`create_dataloaders`**:  
   - Creates PyTorch DataLoader objects for efficient batch processing during training and testing.

Dependencies:
-------------
- numpy  
- pathlib  
- PIL (Pillow)  
- torch (PyTorch)  
- sklearn.preprocessing (StandardScaler)  

Parameters:
-----------
- **`label_type` (str)**: Specifies the type of labels (`"energy"` or `"positions"`).  
- **`num_file_min` (int, optional)**: First file number to load (default=0).  
- **`num_file_max` (int, optional)**: Last file number to load (default=118).  
- **`scaler` (bool, optional)**: Whether to apply scaling to the labels (default=True).  
- **`batch_size` (int)**: Number of samples per batch for DataLoader.  

Returns:
--------
The preprocessing functions return:
1. **`load_data`**:  
   - `X` (torch.Tensor): Array of input images with shape `(N, H, W, C)`.  
   - `y` (torch.Tensor): Array of labels.  

2. **`createDataset`**:  
   - `train_dataset` / `test_dataset`: TensorDatasets for training and testing data.  
   - `scaler_y_train` / `scaler_y_test` (if `scaler=True`): StandardScaler objects for labels.  

3. **`scale_data`**:  
   - Scaled training/testing datasets and label scalers.  

4. **`create_dataloaders`**:  
   - PyTorch DataLoader objects for training and testing datasets.  

Instructions:
-------------
1. Place image and label files in the specified directory (`data_dir`).  
2. Adjust `label_type` to `"energy"` or `"positions"` based on the prediction task.  
3. Call the functions in the following order:  
   - Use `load_data` to load images and labels.  
   - Use `createDataset` to split and optionally scale the dataset.  
   - Use `create_dataloaders` to generate DataLoader objects for training/testing.  
4. Use the DataLoaders in your CNN training loop.  
"""

from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = Path("/sps/l2it/lperon/JUNO/data_cnn/2d/")

def load_data(label_type, num_file_min=0, num_file_max=118):
    """
    Load 2D image data and corresponding labels from disk with optional downscaling.

    Parameters:
    -----------
    label_type : str
        Must be "energy" or "positions".
    num_file_min : int, optional
        First file number to load.
    num_file_max : int, optional
        Last file number to load.
    downscale_factor : int, optional
        Factor by which to downscale images to reduce memory usage.

    Returns:
    --------
    X : np.ndarray
        (N, H, W, C) array of downscaled images.
    y : np.ndarray
        Flattened array of labels.
    """
    if label_type not in {"energy", "positions"}:
        raise ValueError('label_type must be "energy" or "positions"')

    X_list = []
    y_list = []

    for num_file in range(num_file_min, num_file_max + 1):
        y_file = np.loadtxt(data_dir / f"true_{label_type}_file_{num_file}.txt")
        y_list.append(y_file)

        entry_files = sorted(
            data_dir.glob(f"cnn_2d_data_file_{num_file}_entry_*.png"),
            key=lambda x: int(x.stem.split("entry_")[1])
        )
        for entry_file in entry_files:
            image = Image.open(entry_file).convert("RGB")
            # w, h = image.size
            # image = image.resize((w // downscale_factor, h // downscale_factor))
            X_list.append(np.array(image, dtype=np.float32))
            print(f"Loaded image {entry_file}")
        print(f"Loaded file {num_file}")
    X = np.array(X_list, dtype=np.float32)
    y = np.concatenate(y_list, axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def createDataset(X, y, scaler=False):
    """
    Split data into training and testing sets, with optional scaling.

    Parameters:
    -----------
    X : np.ndarray
        The input data array (e.g., image data).
    y : np.ndarray
        The labels array.
    scaler : bool, optional
        Whether to apply scaling to the labels (default is True).

    Returns:
    --------
    train_dataset : torch.utils.data.TensorDataset
        Dataset containing the training data.
    test_dataset : torch.utils.data.TensorDataset
        Dataset containing the testing data.
    scaler_y_train : sklearn.preprocessing.StandardScaler, optional
        Scaler for training labels (if `scaler=True`).
    scaler_y_test : sklearn.preprocessing.StandardScaler, optional
        Scaler for testing labels (if `scaler=True`).

    Example:
    --------
    train_dataset, test_dataset, scaler_y_train, scaler_y_test = createDataset(X, y, scaler=True)
    """
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
    if scaler:
        train_dataset, test_dataset, scaler_y_train, scaler_y_test = scale_data(X_train, y_train, X_test, y_test)
        return train_dataset, test_dataset, scaler_y_train, scaler_y_test
    else:
        return train_dataset, test_dataset

def scale_data(X_train, y_train, X_test, y_test):
    """
    Scale the input data and labels using StandardScaler.

    Parameters:
    -----------
    X_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing input data.
    y_test : np.ndarray
        Testing labels.

    Returns:
    --------
    train_dataset : torch.utils.data.TensorDataset
        Dataset containing scaled training data.
    test_dataset : torch.utils.data.TensorDataset
        Dataset containing scaled testing data.
    scaler_y_train : sklearn.preprocessing.StandardScaler
        Scaler for training labels.
    scaler_y_test : sklearn.preprocessing.StandardScaler
        Scaler for testing labels.

    Example:
    --------
    train_dataset, test_dataset, scaler_y_train, scaler_y_test = scale_data(X_train, y_train, X_test, y_test)
    """
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
        torch.tensor(X_train, dtype=torch.uint8),
        torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.uint8),
        torch.tensor(y_test, dtype=torch.float32))

    return train_dataset, test_dataset, scaler_y_train, scaler_y_test

def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader