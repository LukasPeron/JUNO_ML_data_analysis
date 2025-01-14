"""
Training and Testing Loop for CNN Models on JUNO Data

CREATED BY   : Lukas Péron  
LAST UPDATED : 14/01/2025  
PURPOSE      : Define a robust training and evaluation loop for 2D CNN models on JUNO simulation data.

Description:
------------
This script implements the training and testing loops for 2D CNN models, enabling predictions for energy 
and spatial positions based on JUNO simulation data. It computes metrics to evaluate the model's performance 
and saves the best-performing model based on test loss.

Key Features:
-------------
1. **Flexible Prediction Types**:
   - `label_type="energy"`: Predicts energy as a single value.
   - `label_type="positions"`: Predicts spatial coordinates (x, y, z).
   - `label_type="both"`: Simultaneously predicts energy and spatial positions.

2. **Metrics and Results**:
   - Tracks training and testing losses across epochs.
   - Computes mean and standard deviation of prediction errors.
   - Saves the best model based on minimum test loss.

3. **Efficient Resource Management**:
   - Includes garbage collection and GPU memory cleanup for optimal performance.

Parameters:
-----------
label_type : str  
    Type of prediction: `"energy"`, `"positions"`, or `"both"`.  

train_loader : torch.utils.data.DataLoader  
    DataLoader for the training dataset.  

test_loader : torch.utils.data.DataLoader  
    DataLoader for the testing dataset.  

cnn : torch.nn.Module  
    The CNN model to be trained and evaluated.  

criterion : torch.nn.Module  
    Loss function (e.g., `torch.nn.MSELoss`).  

optimizer : torch.optim.Optimizer  
    Optimizer for weight updates (e.g., `torch.optim.Adam`).  

lr : float  
    Learning rate for the optimizer.  

batch_size : int  
    Batch size used for training.  

n_epochs : int, optional (default=10)  
    Number of epochs for training.  

Returns:
--------
tuple  
    Training and testing metrics, including:  
    - `train_losses`, `test_losses`: Average losses per epoch.  
    - If `label_type == "energy"`:  
      - Mean and standard deviation of energy prediction errors.  
      - Energy differences between true and predicted values for train/test sets.  
    - If `label_type == "positions"`:  
      - Mean and standard deviation of prediction errors for x, y, z coordinates.  
      - Prediction differences for x, y, z coordinates for train/test sets.  

Features:
---------
- **Automatic Best Model Saving**:  
  Saves the model with the lowest test loss to disk for future evaluation.  
- **Loss Curve Visualization**:  
  Produces `.svg` plots of training and testing losses for detailed analysis.  
- **Scalable Performance**:  
  Compatible with both CPU and GPU environments.  

Usage:
------
1. Ensure the dataset is preprocessed and loaders (`train_loader`, `test_loader`) are configured.  
2. Initialize the CNN model, loss function, and optimizer.  
3. Set the appropriate `label_type` based on the prediction task.  
4. Call `train_test_loop` and pass the required parameters.  

"""


import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import gc
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def train_test_loop(model_type, label_type, train_loader, test_loader, cnn, criterion, optimizer, lr, batch_size, n_epochs=10):
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
    best_epoch = -1

    # Variables to store training and testing metrics
    train_losses, test_losses = [], []
    # Training loop
    for epoch in range(n_epochs):
        cnn.train()
        epoch_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_labels.append(labels)
        # Inverse transform training labels/predictions and calculate metrics
        # all_train_labels = scaler_y_train.inverse_transform(torch.cat(all_train_labels).cpu().numpy())
        # all_train_preds = scaler_y_train.inverse_transform(torch.cat(all_train_preds).cpu().numpy())
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        all_train_preds = torch.cat(all_train_preds).detach().cpu().numpy()
        all_train_labels = torch.cat(all_train_labels).detach().cpu().numpy()

        # Clear unused variables and call garbage collection
        del inputs, labels, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        # Evaluation phase
        cnn.eval()
        with torch.no_grad():
            test_epoch_loss = 0.0
            all_test_preds, all_test_labels = [], []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs.permute(0, 3, 1, 2))
                loss = criterion(outputs, labels)
                test_epoch_loss += loss.item()
                all_test_preds.append(outputs)
                all_test_labels.append(labels)
        
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        all_test_preds = torch.cat(all_test_preds).detach().cpu().numpy()
        all_test_labels = torch.cat(all_test_labels).detach().cpu().numpy()

        # Inverse transform test labels/predictions and calculate metrics
        # all_test_labels = scaler_y_test.inverse_transform(torch.cat(all_test_labels).cpu().numpy())
        # all_test_preds = scaler_y_test.inverse_transform(torch.cat(all_test_preds).cpu().numpy())
        print(f"{epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.2e}")

        # Calculate differences every 10 epochs
        # Save the best model
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            best_epoch = epoch
            torch.save(cnn.state_dict(), f"/pbs/home/l/lperon/work_JUNO/models/CNN/2D/cnn_2d_{model_type}_{label_type}.pth")

        # Clear unused variables and call garbage collection
        del inputs, labels, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    pwd_saving = f"/pbs/home/l/lperon/work_JUNO/figures/CNN/2d/{label_type}/"

    # Plot loss curves
    fig, ax1 = plt.subplots()
    ax1.loglog(range(n_epochs), train_losses, color='b', label='Training Loss')
    ax1.loglog(range(n_epochs), test_losses, color='r', label='Test Loss')
    ax1.vlines(best_epoch, min(train_losses + test_losses), max(train_losses + test_losses), color='g', linestyle='--', label='Best Model')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    fig.legend()
    fig.tight_layout()
    plt.savefig(pwd_saving + f"unique_{label_type}_loss_long_lr={lr:.0e}_batch_size={batch_size:.0f}_model_{model_type}.svg")

    # Return results based on `label_type`
    return train_losses, test_losses, best_epoch
