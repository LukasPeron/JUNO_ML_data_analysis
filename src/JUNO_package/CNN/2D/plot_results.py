"""
Plotting Results for 2D CNN Training on JUNO Simulation Data

CREATED BY   : Lukas Péron  
LAST UPDATED : 14/01/2025
PURPOSE      : Generate visualizations for evaluating 2D CNN training performance on JUNO simulation data.

Description:
------------
This script provides functions to create visualizations for assessing the training and testing performance
of 2D Convolutional Neural Networks (CNNs) applied to JUNO simulation data. The visualizations help analyze 
model predictions for energy and spatial positions.

Key Functions:
--------------
1. `plot_all_figures_energy`:  
   - Generates visualizations for energy prediction over epochs.  
   - Includes plots for:
     - Loss curves
     - Average and standard deviation of energy prediction errors
     - Histograms of prediction errors

2. `plot_all_figures_spatial`:  
   - Generates visualizations for spatial position predictions (x, y, z) over epochs.  
   - Includes plots for:
     - Loss curves
     - Average and standard deviation of spatial prediction errors
     - Histograms of prediction errors for each coordinate (x, y, z)

Dependencies:
-------------
- numpy  
- matplotlib  

Usage Instructions:
-------------------
1. Ensure the specified directories for saving figures (`pwd_saving`) exist and are writable.
2. Provide input data arrays containing training and testing losses, prediction errors, and statistics.
3. Adjust hyperparameters such as `batch_size`, `lr` (learning rate), and `n_epochs` as needed for labeling the plots.
4. Call the appropriate function:
   - Use `plot_all_figures_energy` for visualizing energy prediction results.
   - Use `plot_all_figures_spatial` for visualizing spatial position prediction results.

Features:
---------
- Customizable figure paths for organized output storage.  
- Supports batch processing of training and testing data for performance analysis.  
- Generates high-quality `.svg` figures for energy and spatial predictions.  

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running without display
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def plot_all_figures_energy(model_type,
                     diff_E_train, diff_E_test,
                     batch_size=50, lr=1e-5):
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

    # Plot histogram of prediction errors for energy
    plt.figure(figsize=(6, 5))
    # Energy-specific variables
    diff_train = diff_E_train
    diff_test = diff_E_test
    # Create the histogram
    plt.hist(diff_train, bins=50, color='b', alpha=0.7, density=True, 
            label=f'Train\n $\langle E \\rangle = {np.mean(diff_train):.3f}$ MeV\n $\sigma_E = {np.std(diff_train):.3f}$ MeV')
    plt.hist(diff_test, bins=50, color='r', alpha=0.7, density=True, 
            label=f'Test\n $\langle E \\rangle = {np.mean(diff_test):.3f}$ MeV\n $\sigma_E = {np.std(diff_test):.3f}$ MeV')

    # Add labels and legend
    plt.xlabel(fr'$E_{{true}} - E_{{model}}$ [MeV]')
    plt.ylabel('Density')
    plt.legend(fontsize=12)
    plt.title('Energy Prediction Error', fontsize=16)
    plt.tight_layout()
    plt.savefig(pwd_saving + f"unique_energy_diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}_model_{model_type}.svg")

def plot_all_figures_spatial(model_type,
                            diff_x_train, diff_y_train, diff_z_train,
                            diff_x_test, diff_y_test, diff_z_test,
                            batch_size=50, lr=1e-5):
    """
    Generate visualizations for spatial (x, y, z) prediction results over epochs.  

    Parameters:  
    -----------  
    (See the original function docstring for details.)  
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pwd_saving = "/pbs/home/l/lperon/work_JUNO/figures/CNN/2d/positions/"

    # Create a figure with 1 row and 3 columns for histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figure size as needed
    for ax, diff_train, diff_test, coord, unit in zip(
            axes,
            [diff_x_train, diff_y_train, diff_z_train],
            [diff_x_test, diff_y_test, diff_z_test],
            ['x', 'y', 'z'],
            ['mm', 'mm', 'mm']):

        ax.hist(diff_train, bins=50, color='b', alpha=0.7, density=True,
                label=f'Train\n $\langle {coord}\\rangle = {np.mean(diff_train):.3f}$ {unit}\n $\sigma_{coord} = {np.std(diff_train):.3f}$ {unit}')
        ax.hist(diff_test, bins=50, color='r', alpha=0.7, density=True,
                label=f'Test\n $\langle {coord}\\rangle = {np.mean(diff_test):.3f}$ {unit}\n $\sigma_{coord} = {np.std(diff_test):.3f}$ {unit}')
        ax.set_xlabel(f'${coord}_{{true}} - {coord}_{{model}}$ [{unit}]')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid()

    fig.tight_layout()
    plt.savefig(pwd_saving + f"unique_spatial_diff_long_lr={lr:.0e}_batch_size={batch_size:.0f}_model_{model_type}.svg")

