"""
Data Preprocessing for 2D CNN Using JUNO Simulation Data

CREATED BY : Lukas Péron
LAST UPDATE : 17/12/2024

This script prepares data for a 2D Convolutional Neural Network (CNN) by loading, scaling, and structuring images (4368 × 2184 resolution) 
and corresponding labels (e.g., energy or position). It supports dataset splitting into training and testing subsets and allows optional scaling of the labels.

Overview:
---------
1. `load_data`:
   - Loads image data and labels from specified files.
   - Converts images to NumPy arrays and labels to continuous arrays.

2. `createDataset`:
   - Splits the dataset into training and testing subsets.
   - Optionally applies scaling to the labels.

3. `scale_data`:
   - Scales the input and label data using sklearn's `StandardScaler`.

Dependencies:
-------------
- numpy
- pathlib
- PIL (Pillow)
- torch (PyTorch)
- sklearn.preprocessing (StandardScaler)

Instructions:
-------------
1. Place the image and label files in the specified `data_dir` path.
2. Adjust `label_type` to either "energy" or "positions" depending on the desired labels.
3. Run the script to process data, split it, and optionally scale it.

"""

from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

data_dir = Path("/sps/l2it/lperon/JUNO/data_cnn/2d/")

def load_data(label_type, num_file_min=0, num_file_max=118):
    """
    Load 2D image data and corresponding labels from disk.

    Parameters:
    -----------
    label_type : str
        The type of label to load. Must be "energy" or "positions".
    num_file_min : int, optional
        The first file number to load (default is 0).
    num_file_max : int, optional
        The last file number to load (default is 118).

    Returns:
    --------
    X : np.ndarray
        Array of image data with shape (N, H, W, C), where N is the total number of entries, 
        H and W are the image height and width, and C is the number of channels.
    y : np.ndarray
        Flattened array of labels corresponding to the images.

    Raises:
    -------
    ValueError:
        If `label_type` is not "energy" or "positions".

    Example:
    --------
    X, y = load_data(label_type="energy", num_file_min=0, num_file_max=10)
    """
    if label_type not in {"energy", "positions"}:
        raise ValueError('label_type must be "energy" or "positions"')

    X_list = []
    y_list = []

    for num_file in range(num_file_min, num_file_max + 1):
        # Load labels
        y_file = np.loadtxt(data_dir / f"true_{label_type}_file_{num_file}.txt")
        y_list.append(y_file)

        # Load images
        entry_files = sorted(
            data_dir.glob(f"cnn_2d_data_file_{num_file}_entry_*.png"),
            key=lambda x: int(x.stem.split("entry_")[1])
        )
        for entry_file in entry_files:
            image = np.array(Image.open(entry_file).convert("RGB"))
            X_list.append(image)

    # Convert lists to NumPy arrays
    X = np.array(X_list, dtype=np.uint8)  # Ensure efficient storage for images
    y = np.concatenate(y_list, axis=0)    # Flatten label arrays into one

    return X, y

def createDataset(X, y, scaler=True):
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
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32))

    return train_dataset, test_dataset, scaler_y_train, scaler_y_test
