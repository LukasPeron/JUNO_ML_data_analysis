"""
! All the pwd used in the following are in the context of my own analysis performed with the CCIN2P3 and cannot be accessed for unauthorized people.

Utility Functions and Information for JUNO Data Analysis

This script provides a set of utility functions for processing and analyzing JUNO data. 
It includes functions for creating ADC counts, removing pedestals from waveforms, plotting waveform data, creating animations, and performing various types of mathematical fits.

Created by: Lukas Péron
Last Update: 01/09/2024

Overview:
---------
The script includes the following functions:
1. `create_adc_count(WaveformQ)`: Converts waveform charge data into ADC counts.
2. `pedestal_remove(adc_count)`: Removes the pedestal (noise) from ADC counts.
3. `plot_colormesh_waveform(data, num_file, event_num)`: Creates a color mesh plot of waveform data.
4. `make_animation(num_file, event_num)`: Creates and saves a 3D event display animation for JUNO data.
5. `update(frame, ax, vertices, lst_pmt_on, data, event_num, num_file)` : The canvas update function used in make_animation().
6. Various fit functions (`linear_fit`, `quadratic_fit`, etc.) for data fitting.
7. Calibration functions to convert ADC counts to energy.

Error Handling:
---------------
- The `create_adc_count`, `piedestal_remove`, and `plot_colormesh_waveform` functions include basic checks to ensure valid input data.
- The `make_animation` and `update` functions check the validity of input data and handle potential issues.
- The plotting and fitting functions assume that the input data is valid; additional error handling may be added as needed.

Dependencies:
-------------
- ROOT
- numpy
- matplotlib

Instructions:
-------------
1. Ensure that ROOT, numpy, and matplotlib are installed and configured in your environment.
2. Use the provided functions to process and analyze JUNO data.
3. The output plots and animations will be saved in the specified directories.
"""

import cppyy
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

# Detector PMT geometry/pbs/home/l/lperon/work_JUNO/JUNO_info/
vertices = np.genfromtxt("../PMTPos_CD_LPMT.csv", usecols=(0, 1, 2, 3))
treshold = 4

def create_adc_count(WaveformQ, no_pedestal=False):
    """
    Converts waveform charge data into ADC counts.

    Parameters:
    -----------
    WaveformQ : list of lists or array-like
        The waveform charge data for each PMT.

    Returns:
    --------
    numpy.ndarray
        An array of ADC counts with pedestals removed.

    Raises:
    -------
    ValueError
        If the input WaveformQ is not in the expected format or is empty.
    """
    if not WaveformQ or not isinstance(WaveformQ, (list, np.ndarray, cppyy.gbl.std.vector[cppyy.gbl.std.vector['unsigned short']])):
        raise ValueError("Input WaveformQ must be a non-empty list or array-like structure.")

    adc_count = np.zeros((len(WaveformQ), 1000))
    waveform_array = np.zeros((len(WaveformQ), 1000))  # Convert WaveformQ to a numpy array with the appropriate shape

    for i, waveform in enumerate(WaveformQ):
        waveform_array[i, :len(waveform)] = waveform

    adc_count[:, :1000] = waveform_array  # Assign waveform_array to adc_count
    if no_pedestal:
        pass
    else:
        adc_count = pedestal_remove(adc_count)
    return adc_count

def pedestal_remove(adc_count):
    """
    Removes the pedestal (noise) from ADC counts by analyzing the waveform data.

    Parameters:
    -----------
    adc_count : numpy.ndarray
        Array of ADC counts for each PMT.

    Returns:
    --------
    numpy.ndarray
        The ADC counts with pedestals removed.

    Raises:
    -------
    ValueError
        If adc_count is empty or not in the expected format.
    """
    if adc_count.size == 0 or not isinstance(adc_count, np.ndarray):
        raise ValueError("Input adc_count must be a non-empty numpy array.")

    for pmt in range(len(adc_count)):
        max_loc = np.where(adc_count[pmt] == np.max(adc_count[pmt]))[0][0]
        max_window_min = max_loc - 20
        max_window_max = max_loc + 51
        noise = np.concatenate((adc_count[pmt][:max_window_min], adc_count[pmt][max_window_max:]))
        mean = np.mean(noise)
        std = np.std(noise)
        adc_count[pmt][adc_count[pmt] <= mean + treshold * std] = 0
        adc_count[pmt][adc_count[pmt] > mean + treshold * std] -= mean
    return adc_count

def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Parameters:
    -----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    z : float
        z-coordinate.

    Returns:
    --------
    tuple
        Tuple containing theta (polar angle in radians) and phi (azimuthal angle in radians).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(z / r)  # Output in radians for matplotlib's projection
    phi = np.arctan2(y, x)  # Output in radians
    return theta, phi

def make_animation(num_file, entry_num, lst_pmt_on, adc_count, true_vertex, display=False):
    """
    Create an animation for event display with 3D and 2D scatter plots.

    Parameters:
    -----------
    num_file : int
        The file number of the event.
    entry_num : int
        The entry number of the event.
    lst_pmt_on : list
        List of PMT IDs that are active.
    adc_count : numpy.ndarray
        Array of ADC counts for each PMT.
    true_vertex : tuple
        Tuple containing x, y, z coordinates, theta, and phi for the true vertex.
    display : bool
        If set on False activate "matplotlib.use('Agg')" to ensure no display is done.
    Returns:
    --------
    None
    """
    if not display:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    height = 9
    fig = plt.figure(figsize=(height*1.618, height))
    fig.set_constrained_layout(True)
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1])  # Adjust for colorbar space

    # Subplot for 3D scatter plot
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')  # Left subplot for 3D scatter plot

    # Subplot for 2D scatter plot using Hammer projection in Matplotlib
    ax_2d = fig.add_subplot(gs[0, 1], projection='hammer')  # Right subplot with Hammer projection

    # Calculate the maximum and minimum non-zero signal for consistent color scale across frames
    max_signal = np.max(adc_count)
    min_signal = np.min(adc_count[np.nonzero(adc_count)])  # Minimum non-zero value in adc_count

    # Normalize the colormap range based on the max and min non-zero signal
    norm = LogNorm(vmin=min_signal, vmax=max_signal)
    cmap = plt.cm.hot

    # Calculate mean radius of all vectors of vertices (3D norm)
    mean_radius = np.max(np.linalg.norm(vertices[:, 1:]/1000, axis=1))

    # Draw a static shaded gray sphere representing the full detector geometry
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = mean_radius * np.cos(u) * np.sin(v)
    y = mean_radius * np.sin(u) * np.sin(v)
    z = mean_radius * np.cos(v)
    rect = plt.Rectangle((-np.pi, -np.pi / 2), 2 * np.pi, np.pi,  # Extent of the background
                        color='gray', alpha=0.075, transform=ax_2d.transAxes, zorder=0)

    # Create the colorbar centered at the bottom
    scat1 = ax_3d.scatter([], [], [], c=[], cmap=cmap, norm=norm)
    cbar_ax = fig.add_subplot(gs[1, :])  # Use the entire bottom row for the colorbar
    cbar1 = fig.colorbar(scat1, cax=cbar_ax, orientation='horizontal', fraction=0.75)
    cbar1.set_label('PMT Charge [ADC]', fontsize=14)
    cbar1.set_ticks(ticks=[min_signal, 100, max_signal], labels=[f'{min_signal:.0f}', f'{100}', f'{max_signal:.0f}'], fontsize=14)
    # Create the animation
    sphere_coord = (x,y,z)
    fig_info = ax_3d, ax_2d, entry_num, num_file, rect, cmap, norm, fig
    ani = FuncAnimation(fig, update, frames=1000, interval=1, fargs=(sphere_coord, true_vertex, fig_info, lst_pmt_on, adc_count))
    # plt.show()
    ani.save(f'event_display_file{num_file}_event{entry_num}_pileup.gif', writer='ffmpeg', fps=10)
    # ani.save(f'event_display_file{num_file}_event{entry_num}_pileup.mp4', writer='ffmpeg', fps=10)

def update(frame, sphere_coord, true_vertex, fig_info, lst_pmt_on, adc_count):
    """
    Update the 3D and 2D scatter plots with the current frame information for event display.

    Parameters:
    -----------
    frame : int
        Current frame number.
    sphere_coord : tuple
        Tuple containing x, y, z coordinates for the sphere.
    true_vertex : tuple
        Tuple containing x, y, z coordinates, theta, and phi for the true vertex.
    fig_info : tuple
        Tuple containing 3D axis, 2D axis, entry number, file number, rectangle patch, colormap, normalization, and figure.
    lst_pmt_on : list
        List of PMT IDs that are on.
    adc_count : numpy.ndarray
        Array of ADC counts for each PMT.
    Returns:
    --------
    None
    """
    x, y, z = sphere_coord
    true_x, true_y, true_z, true_theta, true_phi = true_vertex
    ax_3d, ax_2d, entry_num, num_file, rect, cmap, norm, fig = fig_info
    ax_3d.clear()  # Clear the current frame on the 3D scatter plot
    ax_2d.clear()  # Clear the current frame on the 2D scatter plot

    # Redraw the static sphere in the 3D plot
    ax_3d.plot_surface(x, y, z, color='gray', alpha=0.075, shade=True)
    ax_3d.plot(true_x, true_y, true_z, color="green", marker='o', linestyle="", markersize=10, label=f"Primary vertex\n x={true_x:.2f} m\n y={true_y:.2f} m\n z={true_z:.2f} m\n $\\theta$={np.degrees(true_theta):.2f}°\n $\phi$={np.degrees(true_phi):.2f}°")
    ax_2d.add_patch(rect)
    # Plot the primary vertex in the 2D plot in Hammer projection
    ax_2d.grid(True, zorder=1)
    ax_2d.scatter(true_phi, true_theta, color='green', marker='o', s=40, zorder=0)

    ax_3d.set_xlabel("x [m]", fontsize=12)
    ax_3d.set_ylabel("y [m]", fontsize=12)
    ax_3d.set_zlabel("z [m]", fontsize=12)
    ax_3d.set_xlim(-20, 20)
    ax_3d.set_ylim(-20, 20)
    ax_3d.set_zlim(-20, 20)
    colors = []  # List to hold the color adc_count for scatter points
    positions_3d = []
    theta_vals = []
    phi_vals = []

    for j, id_on in enumerate(lst_pmt_on):
        pos = vertices[id_on, :]/1000
        q = adc_count[j][frame]

        if q > 0:  # Only plot points with non-zero values
            color = cmap(norm(q))
            colors.append(color)
            positions_3d.append(pos)
            theta, phi = cartesian_to_spherical(pos[1], pos[2], pos[3])
            theta_vals.append(theta)
            phi_vals.append(phi)

    if positions_3d:  # If there are non-zero positions to plot
        positions_3d = np.array(positions_3d)
        ax_3d.scatter(positions_3d[:, 1], positions_3d[:, 2], positions_3d[:, 3], c=colors, cmap=cmap, norm=norm, s=5)

        # Plot the 2D scatter plot of (theta, phi) in Hammer projection
        ax_2d.scatter(phi_vals, theta_vals, c=colors, cmap=cmap, norm=norm, s=17, zorder=0)
        
    # Centered title for the entire figure
    fig.suptitle(f"Event display for event {entry_num} of file {num_file}\n signal treshold = {treshold}$\sigma$\n t = {frame + 1} ns", fontsize=14, ha="center", va="top")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc=(0.44,0.15), fontsize=14)

def plot_colormesh_waveform(data, num_file, event_num, display=False):
    """
    Creates a color mesh plot of the waveform data.

    Parameters:
    -----------
    data : numpy.ndarray
        The waveform data to be plotted.
    num_file : int
        The file number associated with the data.
    event_num : int
        The event number within the file.
    display : bool
        If set on False activate "matplotlib.use('Agg')" to ensure no display is done.
    Returns:
    --------
    int
        Returns 0 upon successful completion.

    Raises:
    -------
    ValueError
        If the data input is empty or not a numpy array.
    """
    if not display:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if data.size == 0 or not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a non-empty numpy array.")

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(data, cmap='pink', rasterized=True)
    cbar = plt.colorbar(pcm)
    cbar.set_label("ADC counts", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel("Time elapsed since trigger [ns]", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f'Large PMTs Q waveform for sim {num_file} entry {event_num}', fontsize=14)
    plt.savefig(f"PMT_WF_Q_pileup_file{num_file}_event{event_num}.svg")
    plt.savefig(f"PMT_WF_Q_pileup_file{num_file}_event{event_num}.png")
    return 0

# Mathematical fitting functions
def linear_fit(x, a):
    """
    Linear fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Slope of the linear function.

    Returns:
    --------
    numpy.ndarray
        Linear fit result.
    """
    return a * x

def quadratic_fit(x, a, x0, c):
    """
    Quadratic fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the quadratic term.
    x0 : float
        Offset of the quadratic term.
    c : float
        Constant term.

    Returns:
    --------
    numpy.ndarray
        Quadratic fit result.
    """
    return a * (x - x0) ** 2 + c

def affine_fit(x, a, b):
    """
    Affine fit function (linear function with offset).

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Slope of the line.
    b : float
        Intercept of the line.

    Returns:
    --------
    numpy.ndarray
        Affine fit result.
    """
    return a * x + b

def sqrt_fit(x, a, b):
    """
    Square root fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the square root function.
    b : float
        Constant term.

    Returns:
    --------
    numpy.ndarray
        Square root fit result.
    """
    return a * np.sqrt(x) + b

def power_law_fit(x, a, b):
    """
    Power law fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the power law.
    b : float
        Exponent of the power law.

    Returns:
    --------
    numpy.ndarray
        Power law fit result.
    """
    return a * x ** b

# Calibration functions
#! The constants used for convertion come from previous calibration. If needed, run your own calibration.
def calib_ADC_to_Energy_peak(adc):
    """
    Calibration function to convert ADC counts to energy (MeV) based on the peak method.

    Parameters:
    -----------
    adc : float or numpy.ndarray
        ADC counts.

    Returns:
    --------
    float or numpy.ndarray
        Calibrated energy in MeV.
    """
    return (adc / 559) ** (4 / 3) / 1852

def calib_ADC_to_Energy_integrated(adc):
    """
    Calibration function to convert ADC counts to energy (MeV) based on the integrated method.

    Parameters:
    -----------
    adc : float or numpy.ndarray
        ADC counts.

    Returns:
    --------
    float or numpy.ndarray
        Calibrated energy in MeV.
    """
    return adc / (1852 * 808)