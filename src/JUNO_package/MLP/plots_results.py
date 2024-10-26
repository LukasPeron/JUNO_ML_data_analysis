"""
Plotting and Visualization of Model Training and Testing Metrics for JUNO Simulation Data

This script defines a function to plot and save various metrics of a trained neural network model for JUNO simulation data. 
The `plot_all_figures` function visualizes model performance over training epochs, including the differences between true 
and predicted values for energy and spatial coordinates. The function generates line plots for average differences and 
standard deviations over time, as well as histograms of prediction errors.

Overview:
---------
The function performs the following:
1. Line plots of average and standard deviations of differences between true and predicted values for energy (E) and 
   coordinates (x, y, z) over training epochs.
2. A plot of training and test loss across epochs.
3. Histograms for the distributions of errors between true and predicted values for each coordinate.

Error Handling:
---------------
- Assumes input data arrays are correctly formatted and of matching lengths for plotting.
- The output directory should be writable to avoid `IOError` when saving figures.

Dependencies:
-------------
- numpy
- matplotlib

Instructions:
-------------
1. Ensure that `matplotlib` and `numpy` are installed and configured in your environment.
2. Call `plot_all_figures` with appropriate metric lists and model parameters to save visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
import matplotlib.pyplot as plt

pwd_saving = "/pbs/home/l/lperon/work_JUNO/figures/MLP/"

def plot_all_figures(train_losses, test_losses,
                     avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                     std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                     avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                     std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test,
                     diff_E_train, diff_x_train, diff_y_train, diff_z_train,
                     diff_E_test, diff_x_test, diff_y_test, diff_z_test,
                     batch_size=50, dropout_prop=0.2, lr=1e-5, n_epochs=1000):
    """
    Generate and save plots of model training and testing metrics.

    Parameters:
    -----------
    train_losses : list of float
        Training losses recorded over each epoch.
    test_losses : list of float
        Test losses recorded over each epoch.
    avg_diff_E_train, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train : list of float
        Lists of average differences between true and predicted values for energy (E) and coordinates (x, y, z) in training data.
    std_diff_E_train, std_diff_x_train, std_diff_y_train, std_diff_z_train : list of float
        Lists of standard deviations of differences for E, x, y, z in training data.
    avg_diff_E_test, avg_diff_x_test, avg_diff_y_test, avg_diff_z_test : list of float
        Lists of average differences between true and predicted values for E, x, y, z in testing data.
    std_diff_E_test, std_diff_x_test, std_diff_y_test, std_diff_z_test : list of float
        Lists of standard deviations of differences for E, x, y, z in testing data.
    diff_E_train, diff_x_train, diff_y_train, diff_z_train : numpy.ndarray
        Arrays of differences between true and predicted values for E, x, y, z in training data.
    diff_E_test, diff_x_test, diff_y_test, diff_z_test : numpy.ndarray
        Arrays of differences for E, x, y, z in testing data.
    batch_size : int, optional
        Batch size used during model training (default is 50).
    dropout_prop : float, optional
        Dropout probability used in model training (default is 0.2).
    lr : float, optional
        Learning rate used during training (default is 1e-5).
    n_epochs : int, optional
        Total number of training epochs (default is 1000).

    Returns:
    --------
    None
        The function saves all generated plots directly to the specified directory.

    Notes:
    ------
    - All plots are saved as SVG files, with file names indicating the learning rate, batch size, and dropout proportion.
    - Histograms of prediction errors are provided for error analysis.
    """

    x = range(0, n_epochs + 1, 10)

    # Plot average difference in energy predictions over epochs
    fig = plt.figure(1)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs", fontsize=14)
    ax.set_ylabel(r"$\langle E_{true}-E_{model} \rangle [MeV]$", fontsize=14)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(pwd_saving + f"diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    # Plot standard deviation of energy differences over epochs (figure 5)
    fig = plt.figure(5)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs", fontsize=14)
    ax.set_ylabel(r"$\sigma(E_{true}-E_{model}) [MeV]$", fontsize=14)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(pwd_saving + f"std_diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    # Repeat for x, y, z coordinates
    for i, (avg_train, avg_test, std_train, std_test, coord) in enumerate(zip(
            [avg_diff_x_train, avg_diff_y_train, avg_diff_z_train],
            [avg_diff_x_test, avg_diff_y_test, avg_diff_z_test],
            [std_diff_x_train, std_diff_y_train, std_diff_z_train],
            [std_diff_x_test, std_diff_y_test, std_diff_z_test],
            ['x', 'y', 'z'])):

        fig = plt.figure(i + 2)
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(x, avg_train, color="b", marker="o", linestyle="", label="Train")
        ax.plot(x, avg_test, color="r", marker="o", linestyle="", label="Test")
        ax.set_xlabel("Number of epochs", fontsize=14)
        ax.set_ylabel(fr"$\langle {coord}_{{true}}-{coord}_{{model}} \rangle [mm]$", fontsize=14)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(pwd_saving + f"diff_{coord}_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

        fig = plt.figure(i + 6)
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(x, std_train, color="b", marker="o", linestyle="", label="Train")
        ax.plot(x, std_test, color="r", marker="o", linestyle="", label="Test")
        ax.set_xlabel("Number of epochs", fontsize=14)
        ax.set_ylabel(fr"$\sigma({coord}_{{true}}-{coord}_{{model}}) [mm]$", fontsize=14)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(pwd_saving + f"std_diff_{coord}_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    # Plot loss curves
    fig, ax1 = plt.subplots()
    ax1.loglog(range(n_epochs + 1), train_losses, color='b', label='Training Loss')
    ax1.loglog(range(n_epochs + 1), test_losses, color='r', label='Test Loss')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid()
    fig.legend()
    fig.tight_layout()
    plt.savefig(pwd_saving + f"loss_long_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")

    # Plot histograms of prediction errors for training and test sets
    plt.figure(figsize=(10, 8))
    for i, (diff_train, diff_test, coord) in enumerate(zip(
            [diff_E_train, diff_x_train, diff_y_train, diff_z_train],
            [diff_E_test, diff_x_test, diff_y_test, diff_z_test],
            ['E', 'x', 'y', 'z'])):

        plt.subplot(2, 2, i + 1)
        plt.hist(diff_train, bins=50, color='b', alpha=0.7, density=True, label=f'Train\n $\langle {coord}\\rangle = {np.mean(diff_train):.2f}$ {unit}\n $\sigma_{coord} = {np.std(diff_train):.2f}$ {unit}')
        plt.hist(diff_test, bins=50, color='r', alpha=0.7, density=True, label=f'Test\n $\langle {coord}\\rangle = {np.mean(diff_test):.2f}$ {unit}\n $\sigma_{coord} = {np.std(diff_test):.2f} $ {unit}')
        plt.xlabel(fr'${coord}_{{true}} - {coord}_{{model}}$', fontsize=14)
        plt.legend()

    plt.tight_layout()
    plt.savefig(pwd_saving + f"diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}_dropout_prop={dropout_prop:.1f}.svg")
