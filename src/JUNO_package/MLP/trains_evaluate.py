"""
Training and Evaluation Loop for JUNO Simulation Model

This script defines a function to train and evaluate a neural network model for JUNO simulation data. The `train_test_loop` 
function performs a training loop over several epochs, calculates evaluation metrics on the training and test sets, 
and saves the model state if it achieves the best validation loss.

Created by: Lukas Peron
Last Update: 18/11/2024

Overview:
---------
The script performs the following steps:
1. Train the model on the training set using Mean Squared Error (MSE) loss.
2. Evaluate the model on the test set and compute differences between true and predicted values.
3. Store metrics including average and standard deviations of differences between true and predicted values for energy (E) and spatial coordinates (x, y, z).
4. Save the model's parameters if the validation loss improves.

Error Handling:
---------------
- Assumes that the input data loaders, model, criterion, and optimizer are configured correctly.
- The function will raise a `RuntimeError` if there is a mismatch in data dimensions during training or evaluation.

Dependencies:
-------------
- torch
- numpy

Instructions:
-------------
1. Ensure that `torch` and `numpy` are installed and configured in your environment.
2. Use the `train_test_loop` function to perform training and evaluation by providing the data loaders, model, criterion, and optimizer.
"""

import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_loop(data_type, train_loader, test_loader, scaler_y_train, scaler_y_test, model, criterion, optimizer, n_epochs=1000):
    """
    Train and evaluate the model on JUNO simulation data.

    The function performs training and evaluation over the specified number of epochs. It saves the best model
    based on the minimum validation loss and calculates metrics such as average and standard deviations of the 
    differences between true and predicted values for energy and spatial coordinates.

    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader containing the training dataset.
    test_loader : DataLoader
        DataLoader containing the testing dataset.
    scaler_y_train : StandardScaler
        Scaler for inverse transforming the training labels.
    scaler_y_test : StandardScaler
        Scaler for inverse transforming the testing labels.
    model : torch.nn.Module
        The neural network model to be trained.
    criterion : torch.nn.Module
        Loss function used for optimization (Mean Squared Error).
    optimizer : torch.optim.Optimizer
        Optimizer for model training (e.g., Adam).
    n_epochs : int, optional
        Number of epochs for training (default is 100).

    Returns:
    --------
    tuple
        A tuple containing lists of training and testing losses, average and standard deviations of differences 
        for energy and coordinates, and lists of differences for each coordinate in the training and testing datasets.
    
    Notes:
    ------
    - Saves the model with the best validation loss in `model.pth`.
    - Calculates differences in predicted and true values for metrics at every 10th epoch.
    """
    best_vloss = 1_000_000
    opti_epochs = 0
    # Variables to store training and testing metrics
    train_losses = []
    test_losses = []

    avg_diff_E_train = []
    avg_diff_x_train = []
    avg_diff_y_train = []
    avg_diff_z_train = []

    std_diff_E_train = []
    std_diff_x_train = []
    std_diff_y_train = []
    std_diff_z_train = []

    avg_diff_E_test = []
    avg_diff_x_test = []
    avg_diff_y_test = []
    avg_diff_z_test = []

    std_diff_E_test = []
    std_diff_x_test = []
    std_diff_y_test = []
    std_diff_z_test = []

    # Training loop
    for epoch in range(n_epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_labels.append(labels)
        
        # Inverse transform and calculate metrics for training data
        all_train_labels = torch.cat(all_train_labels).detach().cpu().numpy()
        all_train_preds = torch.cat(all_train_preds).detach().cpu().numpy()
        all_train_labels = scaler_y_train.inverse_transform(all_train_labels)
        all_train_preds = scaler_y_train.inverse_transform(all_train_preds)
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_epoch_loss = 0.0
            all_test_preds = []
            all_test_labels = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_epoch_loss += loss.item()
                all_test_preds.append(outputs)
                all_test_labels.append(labels)
        
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Inverse transform and calculate metrics for test data
        all_test_labels = torch.cat(all_test_labels).detach().cpu().numpy()
        all_test_preds = torch.cat(all_test_preds).detach().cpu().numpy()
        all_test_labels = scaler_y_test.inverse_transform(all_test_labels)
        all_test_preds = scaler_y_test.inverse_transform(all_test_preds)

        print(f"{epoch + 1}/{n_epochs + 1}, Train Loss: {avg_train_loss:.2e}")

        # Calculate differences every 10 epochs
        if epoch % 10 == 0:
            if data_type=="both" or data_type=="energy":
                diff_E = all_test_labels[:, 0] - all_test_preds[:, 0]
                avg_diff_E_test.append(np.mean(diff_E))
                std_diff_E_test.append(np.std(diff_E))

                diff_E_train = all_train_labels[:, 0] - all_train_preds[:, 0]
                avg_diff_E_train.append(np.mean(diff_E_train))
                std_diff_E_train.append(np.std(diff_E_train))
                
            if data_type=="both" or data_type=="spatial":
                diff_x = all_test_labels[:, 1] - all_test_preds[:, 1]
                avg_diff_x_test.append(np.mean(diff_x))
                std_diff_x_test.append(np.std(diff_x))

                diff_y = all_test_labels[:, 2] - all_test_preds[:, 2]
                avg_diff_y_test.append(np.mean(diff_y))
                std_diff_y_test.append(np.std(diff_y))

                diff_z = all_test_labels[:, 3] - all_test_preds[:, 3]
                avg_diff_z_test.append(np.mean(diff_z))
                std_diff_z_test.append(np.std(diff_z))

                diff_x_train = all_train_labels[:, 1] - all_train_preds[:, 1]
                avg_diff_x_train.append(np.mean(diff_x_train))
                std_diff_x_train.append(np.std(diff_x_train))

                diff_y_train = all_train_labels[:, 2] - all_train_preds[:, 2]
                avg_diff_y_train.append(np.mean(diff_y_train))
                std_diff_y_train.append(np.std(diff_y_train))

                diff_z_train = all_train_labels[:, 3] - all_train_preds[:, 3]
                avg_diff_z_train.append(np.mean(diff_z_train))
                std_diff_z_train.append(np.std(diff_z_train))

        # Save the best model
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            opti_epochs = epoch
            if data_type=="both" or data_type=="energy":
                diff_E_test = (all_test_labels[:, 0] - all_test_preds[:, 0])
            if data_type=="both" or data_type=="spatial":
                diff_x_test = (all_test_labels[:, 1] - all_test_preds[:, 1])
                diff_y_test = (all_test_labels[:, 2] - all_test_preds[:, 2])
                diff_z_test = (all_test_labels[:, 3] - all_test_preds[:, 3])

            model_path = f"/pbs/home/l/lperon/work_JUNO/models/MLP/model.pth"  # Path to save the best model
            torch.save(model.state_dict(), model_path)

    if data_type=="both":
        return (train_losses, test_losses,
                avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,
                diff_E_train, diff_x_train, diff_y_train, diff_z_train, 
                diff_E_test, diff_x_test, diff_y_test, diff_z_test)
    
    elif data_type=="energy":
        return (train_losses, test_losses,
            avg_diff_E_train, std_diff_E_train, 
            avg_diff_E_test, std_diff_E_test, 
            diff_E_train, diff_E_test)
    
    elif data_type=="spatial":
        return (train_losses, test_losses,
                avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                std_diff_x_test, std_diff_y_test, std_diff_z_test,
                diff_x_train, diff_y_train, diff_z_train, 
                diff_x_test, diff_y_test, diff_z_test)