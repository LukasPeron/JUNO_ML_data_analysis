"""
Training and Testing Loop for CNN Models on JUNO Data

CREATED BY : Lukas Péron  
LAST UPDATE : 17/12/2024

This script defines a training and evaluation loop for 2D CNN models to process JUNO data.  
The training process computes the loss for each batch, adjusts model weights, and saves the best-performing model.  
The evaluation process calculates metrics to compare predictions against ground truth labels.

Parameters:
-----------
label_type : str
    Specifies the type of output being predicted:
    - "energy": Single value prediction for energy.
    - "positions": 3D vector prediction for spatial coordinates.
    - "both": Combines "energy" and "positions" predictions.

train_loader : torch.utils.data.DataLoader
    DataLoader for the training set.

test_loader : torch.utils.data.DataLoader
    DataLoader for the test set.

scaler_y_train : sklearn.preprocessing.StandardScaler
    Scaler object for normalizing training labels.

scaler_y_test : sklearn.preprocessing.StandardScaler
    Scaler object for normalizing test labels.

cnn : torch.nn.Module
    The CNN model to be trained and evaluated.

criterion : torch.nn.Module
    The loss function (e.g., MSELoss).

optimizer : torch.optim.Optimizer
    The optimizer (e.g., Adam).

n_epochs : int, optional (default=10)
    Number of training epochs.

Returns:
--------
tuple
    Training and testing metrics, including:
    - `train_losses`, `test_losses`: Average losses per epoch.
    - If `label_type == "energy"`:
      - Mean and standard deviation of energy differences for train/test.
      - Energy differences between true and predicted values.
    - If `label_type == "positions"`:
      - Mean and standard deviation of differences for x, y, z coordinates for train/test.
      - Differences for each coordinate between true and predicted values.

"""

import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_loop(label_type, train_loader, test_loader, scaler_y_train, scaler_y_test, cnn, criterion, optimizer, n_epochs=10):
    """
    Train and evaluate a CNN model on JUNO 2D simulation data.

    This function performs the training and testing of a CNN model, saving metrics 
    and the best-performing model during the process.

    Parameters:
    -----------
    label_type : str
        Specifies the output type ("energy", "positions", or "both").
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.
    scaler_y_train : sklearn.preprocessing.StandardScaler
        Scaler for normalizing training labels.
    scaler_y_test : sklearn.preprocessing.StandardScaler
        Scaler for normalizing test labels.
    cnn : torch.nn.Module
        CNN model to train and evaluate.
    criterion : torch.nn.Module
        Loss function for training (e.g., MSELoss).
    optimizer : torch.optim.Optimizer
        Optimizer for weight updates (e.g., Adam).
    n_epochs : int, optional (default=10)
        Number of training epochs.

    Returns:
    --------
    tuple
        Training and testing metrics:
        - If `label_type == "energy"`:
          - Train/test losses.
          - Mean and standard deviations of energy differences.
          - Energy differences between true and predicted values.
        - If `label_type == "positions"`:
          - Train/test losses.
          - Mean and standard deviations of differences for x, y, z coordinates.
          - Differences for x, y, z coordinates between true and predicted values.
    """
    best_vloss = 1_000_000

    # Variables to store training and testing metrics
    train_losses, test_losses = [], []

    avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train = [], [], [], []
    std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train = [], [], [], []

    avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test = [], [], [], []
    std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test = [], [], [], []

    # Training loop
    for epoch in range(n_epochs + 1):
        cnn.train()
        epoch_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_labels.append(labels)
        
        # Inverse transform training labels/predictions and calculate metrics
        all_train_labels = scaler_y_train.inverse_transform(torch.cat(all_train_labels).cpu().numpy())
        all_train_preds = scaler_y_train.inverse_transform(torch.cat(all_train_preds).cpu().numpy())
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        cnn.eval()
        with torch.no_grad():
            test_epoch_loss = 0.0
            all_test_preds, all_test_labels = [], []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                test_epoch_loss += loss.item()
                all_test_preds.append(outputs)
                all_test_labels.append(labels)
        
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Inverse transform test labels/predictions and calculate metrics
        all_test_labels = scaler_y_test.inverse_transform(torch.cat(all_test_labels).cpu().numpy())
        all_test_preds = scaler_y_test.inverse_transform(torch.cat(all_test_preds).cpu().numpy())
        print(f"{epoch + 1}/{n_epochs + 1}, Train Loss: {avg_train_loss:.2e}")

        # Calculate differences every 10 epochs
        if epoch % 10 == 0:
            if label_type in {"energy", "both"}:
                diff_E_train = all_train_labels[:, 0] - all_train_preds[:, 0]
                diff_E_test = all_test_labels[:, 0] - all_test_preds[:, 0]

                avg_diff_E_train.append(np.mean(diff_E_train))
                std_diff_E_train.append(np.std(diff_E_train))
                avg_diff_E_test.append(np.mean(diff_E_test))
                std_diff_E_test.append(np.std(diff_E_test))

            if label_type in {"positions", "both"}:
                diff_x_train = all_train_labels[:, 0] - all_train_preds[:, 0]
                diff_y_train = all_train_labels[:, 1] - all_train_preds[:, 1]
                diff_z_train = all_train_labels[:, 2] - all_train_preds[:, 2]

                diff_x_test = all_test_labels[:, 0] - all_test_preds[:, 0]
                diff_y_test = all_test_labels[:, 1] - all_test_preds[:, 1]
                diff_z_test = all_test_labels[:, 2] - all_test_preds[:, 2]

                avg_diff_x_train.append(np.mean(diff_x_train))
                std_diff_x_train.append(np.std(diff_x_train))
                avg_diff_y_train.append(np.mean(diff_y_train))
                std_diff_y_train.append(np.std(diff_y_train))
                avg_diff_z_train.append(np.mean(diff_z_train))
                std_diff_z_train.append(np.std(diff_z_train))

                avg_diff_x_test.append(np.mean(diff_x_test))
                std_diff_x_test.append(np.std(diff_x_test))
                avg_diff_y_test.append(np.mean(diff_y_test))
                std_diff_y_test.append(np.std(diff_y_test))
                avg_diff_z_test.append(np.mean(diff_z_test))
                std_diff_z_test.append(np.std(diff_z_test))

        # Save the best model
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            torch.save(cnn.state_dict(), "/pbs/home/l/lperon/work_JUNO/cnns/CNN/2D/cnn.pth")

    # Return results based on `label_type`
    if label_type == "energy":
        return train_losses, test_losses, avg_diff_E_train, std_diff_E_train, avg_diff_E_test, std_diff_E_test, diff_E_train, diff_E_test
    elif label_type == "positions":
        return train_losses, test_losses, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, std_diff_x_test, std_diff_y_test, std_diff_z_test, diff_x_train, diff_y_train, diff_z_train, diff_x_test, diff_y_test, diff_z_test
