"""
Plotting Results for 2D CNN Training on JUNO Simulation Data

CREATED BY : Lukas Péron  
LAST UPDATE : 17/12/2024  

This script generates a series of visualizations for evaluating the performance of a 2D Convolutional Neural Network (CNN) 
trained on JUNO simulation data. The visualizations include loss curves, prediction error histograms, and evolution of average 
and standard deviations of prediction errors across epochs.  

The script supports both energy prediction (plot_all_figures_energy) and spatial position prediction (plot_all_figures_spatial).  

Overview:  
---------  
1. `plot_all_figures_energy`:  
   - Generates visualizations for training and testing energy prediction performance over epochs.  
   - Plots include average and standard deviation of energy prediction errors, loss curves, and histograms of prediction errors.  

2. `plot_all_figures_spatial`:  
   - Generates visualizations for training and testing spatial (x, y, z) position prediction performance over epochs.  
   - Plots include average and standard deviation of spatial prediction errors, loss curves, and histograms of prediction errors.  

Dependencies:  
-------------  
- numpy  
- matplotlib  

Instructions:  
-------------  
1. Ensure the paths for saving the figures (`pwd_saving`) are accessible and writable.  
2. Provide input data arrays for training and testing losses, average prediction errors, and standard deviations.  
3. Adjust the `batch_size`, `lr`, and `n_epochs` parameters as needed for labeling the plots appropriately.  
4. Call the appropriate function (`plot_all_figures_energy` or `plot_all_figures_spatial`) based on the type of predictions.  

"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

def plot_all_figures_energy(train_losses, test_losses,
                     avg_diff_E_train, std_diff_E_train, 
                     avg_diff_E_test, std_diff_E_test, 
                     diff_E_train, diff_E_test,
                     batch_size=50, lr=1e-5, n_epochs=5):
    """
    Generate visualizations for energy prediction results over epochs.  

    Parameters:  
    -----------  
    train_losses : list  
        Training loss values for each epoch.  
    test_losses : list  
        Testing loss values for each epoch.  
    avg_diff_E_train : list  
        Average prediction errors (train) over epochs.  
    std_diff_E_train : list  
        Standard deviation of prediction errors (train) over epochs.  
    avg_diff_E_test : list  
        Average prediction errors (test) over epochs.  
    std_diff_E_test : list  
        Standard deviation of prediction errors (test) over epochs.  
    diff_E_train : np.ndarray  
        Array of prediction errors for the training set.  
    diff_E_test : np.ndarray  
        Array of prediction errors for the testing set.  
    batch_size : int, optional  
        Batch size used during training (default is 50).  
    lr : float, optional  
        Learning rate used during training (default is 1e-5).  
    n_epochs : int, optional  
        Number of epochs for training (default is 5).  

    Saves:  
    ------  
    - Loss curve, average difference, standard deviation, and histogram plots as `.svg` files in the specified directory.  

    Example:  
    --------  
    plot_all_figures_energy(train_losses, test_losses, avg_diff_E_train, std_diff_E_train, 
                            avg_diff_E_test, std_diff_E_test, diff_E_train, diff_E_test)  
    """
    pwd_saving = "/pbs/home/l/lperon/work_JUNO/figures/CNN/2d/energy/"
    x = range(0, n_epochs + 1)
    # Plot average difference in energy predictions over epochs
    fig = plt.figure(1)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, avg_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, avg_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\langle E_{true}-E_{model} \rangle [MeV]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(pwd_saving + f"unique_energy_diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

    # Plot standard deviation of energy differences over epochs (figure 5)
    fig = plt.figure(5)
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x, std_diff_E_train, color="b", marker="o", linestyle="", label="Train")
    ax.plot(x, std_diff_E_test, color="r", marker="o", linestyle="", label="Test")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel(r"$\sigma(E_{true}-E_{model}) [MeV]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(pwd_saving + f"unique_energy_std_diff_E_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

    # Plot loss curves
    fig, ax1 = plt.subplots()
    ax1.loglog(range(n_epochs + 1), train_losses, color='b', label='Training Loss')
    ax1.loglog(range(n_epochs + 1), test_losses, color='r', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    fig.legend()
    fig.tight_layout()
    plt.savefig(pwd_saving + f"unique_energy_loss_long_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

    # Plot histogram of prediction errors for energy
    plt.figure(figsize=(6, 5))
    # Energy-specific variables
    diff_train = diff_E_train
    diff_test = diff_E_test
    # Create the histogram
    plt.hist(diff_train, bins=50, color='b', alpha=0.7, density=True, 
            label=f'Train\n $\langle E\\rangle = {np.mean(diff_train):.2f}$ MeV\n $\sigma_E = {np.std(diff_train):.2f}$ MeV')
    plt.hist(diff_test, bins=50, color='r', alpha=0.7, density=True, 
            label=f'Test\n $\langle E\\rangle = {np.mean(diff_test):.2f}$ MeV\n $\sigma_E = {np.std(diff_test):.2f}$ MeV')

    # Add labels and legend
    plt.xlabel(fr'$E_{{true}} - E_{{model}}$ [MeV]')
    plt.ylabel('Density')
    plt.legend(fontsize=12)
    plt.title('Energy Prediction Error', fontsize=16)
    plt.tight_layout()
    plt.savefig(pwd_saving + f"unique_energy_diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

def plot_all_figures_spatial(train_losses, test_losses,
                     avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                     std_diff_x_train, std_diff_y_train, std_diff_z_train, 
                     avg_diff_x_test, avg_diff_y_test, avg_diff_z_test, 
                     std_diff_x_test, std_diff_y_test, std_diff_z_test,
                     diff_x_train, diff_y_train, diff_z_train,
                     diff_x_test, diff_y_test, diff_z_test,
                     batch_size=50, lr=1e-5, n_epochs=5):
    """
    Generate visualizations for spatial (x, y, z) prediction results over epochs.  

    Parameters:  
    -----------  
    train_losses : list  
        Training loss values for each epoch.  
    test_losses : list  
        Testing loss values for each epoch.  
    avg_diff_x_train, avg_diff_y_train, avg_diff_z_train : list  
        Average prediction errors for x, y, and z (train) over epochs.  
    std_diff_x_train, std_diff_y_train, std_diff_z_train : list  
        Standard deviations of prediction errors for x, y, and z (train) over epochs.  
    avg_diff_x_test, avg_diff_y_test, avg_diff_z_test : list  
        Average prediction errors for x, y, and z (test) over epochs.  
    std_diff_x_test, std_diff_y_test, std_diff_z_test : list  
        Standard deviations of prediction errors for x, y, and z (test) over epochs.  
    diff_x_train, diff_y_train, diff_z_train : np.ndarray  
        Arrays of prediction errors for x, y, and z in the training set.  
    diff_x_test, diff_y_test, diff_z_test : np.ndarray  
        Arrays of prediction errors for x, y, and z in the testing set.  
    batch_size : int, optional  
        Batch size used during training (default is 50).  
    lr : float, optional  
        Learning rate used during training (default is 1e-5).  
    n_epochs : int, optional  
        Number of epochs for training (default is 5).  

    Saves:  
    ------  
    - Loss curve, average difference, standard deviation, and histogram plots for x, y, and z coordinates as `.svg` files.  

    Example:  
    --------  
    plot_all_figures_spatial(train_losses, test_losses, avg_diff_x_train, avg_diff_y_train, avg_diff_z_train, 
                             std_diff_x_train, std_diff_y_train, std_diff_z_train, avg_diff_x_test, avg_diff_y_test, 
                             avg_diff_z_test, std_diff_x_test, std_diff_y_test, std_diff_z_test, diff_x_train, 
                             diff_y_train, diff_z_train, diff_x_test, diff_y_test, diff_z_test)  
    """
    pwd_saving = "/pbs/home/l/lperon/work_JUNO/figures/CNN/2d/positions/"
    x = range(0, n_epochs + 1)
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
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel(fr"$\langle {coord}_{{true}}-{coord}_{{model}} \rangle [mm]$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(pwd_saving + f"unique_spatial_diff_{coord}_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

        fig = plt.figure(i + 6)
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(x, std_train, color="b", marker="o", linestyle="", label="Train")
        ax.plot(x, std_test, color="r", marker="o", linestyle="", label="Test")
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel(fr"$\sigma({coord}_{{true}}-{coord}_{{model}}) [mm]$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(pwd_saving + f"unique_spatial_std_diff_{coord}_vs_epochs_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

    # Plot loss curves
    fig, ax1 = plt.subplots()
    ax1.loglog(range(n_epochs + 1), train_losses, color='b', label='Training Loss')
    ax1.loglog(range(n_epochs + 1), test_losses, color='r', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    fig.legend()
    fig.tight_layout()
    plt.savefig(pwd_saving + f"unique_spatial_loss_long_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")

    # Plot histograms of prediction errors for training and test sets
    plt.figure(figsize=(10, 8))
    for i, (diff_train, diff_test, coord, unit) in enumerate(zip(
            [diff_x_train, diff_y_train, diff_z_train],
            [diff_x_test, diff_y_test, diff_z_test],
            ['x', 'y', 'z'],
            ['mm', 'mm', 'mm'])):

        plt.subplot(2, 2, i + 1)
        plt.hist(diff_train, bins=50, color='b', alpha=0.7, density=True, label=f'Train\n $\langle {coord}\\rangle = {np.mean(diff_train):.2f}$ {unit}\n $\sigma_{coord} = {np.std(diff_train):.2f}$ {unit}')
        plt.hist(diff_test, bins=50, color='r', alpha=0.7, density=True, label=f'Test\n $\langle {coord}\\rangle = {np.mean(diff_test):.2f}$ {unit}\n $\sigma_{coord} = {np.std(diff_test):.2f} $ {unit}')
        plt.xlabel(fr'${coord}_{{true}} - {coord}_{{model}}$ [{unit}]')
        plt.legend()

    plt.tight_layout()
    plt.savefig(pwd_saving + f"unique_spatial_diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}.svg")
