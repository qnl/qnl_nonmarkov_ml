import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import os
from utils import save_options

def dark_mode_compatible(dark_mode_color=r'#86888A'):
    """
    Set the plotting style to dark mode, which works well on dark backgrounds but is also
    readable on white backgrounds with the default dark_mode_color
    :param dark_mode_color: color of the plot axes, labels and titles
    :return: None
    """
    matplotlib.rc('axes', edgecolor=dark_mode_color)
    matplotlib.rc('text', color=dark_mode_color)
    matplotlib.rc('xtick', color=dark_mode_color)
    matplotlib.rc('ytick', color=dark_mode_color)
    matplotlib.rc('axes', labelcolor=dark_mode_color)
    matplotlib.rc('axes', facecolor='none')
    matplotlib.rc('figure', edgecolor='none')  # .edgecolor', (1, 1, 1, 0))
    matplotlib.rc('figure', facecolor='none')  # (1, 1, 1, 0))

def get_histogram(weak_meas_times, X, Y, Z, n_bins=101, bin_min=-1, bin_max=+1):
    """
    Return the histogram vs. time for three trajectory arrays X, Y and Z. These histograms are not normalized.
    :param weak_meas_times: 1d array of length n_timesteps. Sets the time axis for the histograms.
    :param X: 2d array of shape (n_reps, n_timesteps), i.e. each row in X represents a trajectory.
    :param Y: 2d array of shape (n_reps, n_timesteps)
    :param Z: 2d array of shape (n_reps, n_timesteps)
    :param n_bins: Number of bins in the direction orthogonal to time
    :param bin_min: Value of the minimum bin
    :param bin_max: Value of the maximum bin
    :return: bins, and three histograms for X, Y and Z
    """
    sequence_length = len(weak_meas_times)

    histX = np.zeros((n_bins, sequence_length))
    histY = np.zeros((n_bins, sequence_length))
    histZ = np.zeros((n_bins, sequence_length))

    for b in range(sequence_length):
        histX[:, b], bins = np.histogram(X[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))
        histY[:, b], bins = np.histogram(Y[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))
        histZ[:, b], bins = np.histogram(Z[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))

    return bins, histX, histY, histZ


def plot_qubit_histogram(weak_meas_times, X, Y, Z, tomography_times, expX, expY, expZ, n_bins=101):
    """
    Constructs a histogram and plots it for all 3 qubit coordinates X, Y and Z as function of time.
    Also shows how the histogram compares with the tomography results.
    :param weak_meas_times: 1d array of length n_timesteps. Sets the time axis for the histograms.
    :param X: 2d array of shape (n_reps, n_timesteps)
    :param Y: 2d array of shape (n_reps, n_timesteps)
    :param Z: 2d array of shape (n_reps, n_timesteps)
    :param tomography_times: 1d array of length n_t. Sets the time axis for the tomography.
    :param expX: 1d array of length n_t. Averaged tomography results in the X direction.
    :param expY: 1d array of length n_t. Averaged tomography results in the Y direction.
    :param expZ: 1d array of length n_t. Averaged tomography results in the Z direction.
    :param n_bins: Number of bins for the X, Y and Z coordinates.
    :return:
    """
    bins, histX, histY, histZ = get_histogram(weak_meas_times, X, Y, Z, n_bins=n_bins)

    cmap = plt.cm.hot
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histZ, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histZ.max()))
    plt.plot(tomography_times * 1e6, expZ, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Z$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histY, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histY.max()))
    plt.plot(tomography_times * 1e6, expY, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Y$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histX, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histX.max()))
    plt.plot(tomography_times * 1e6, expX, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$X$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()
    return fig


def plot_qutrit_histogram(weak_meas_times, Pg_trajectories, Pe_trajectories, Pf_trajectories,
                          tomography_times, Pg, Pe, Pf, n_bins=101):
    """
    Qutrit version of `plot_qubit_histogram`
    :param weak_meas_times:
    :param Pg_trajectories: 2d array of shape (n_reps, n_timesteps)
    :param Pe_trajectories: 2d array of shape (n_reps, n_timesteps)
    :param Pf_trajectories: 2d array of shape (n_reps, n_timesteps)
    :param tomography_times: 1d array of length n_t. Sets the time axis for the tomography.
    :param Pg: 1d array of length n_t. Averaged tomography results for the ground state population vs time.
    :param Pe: 1d array of length n_t. Averaged tomography results for the excited state population vs time.
    :param Pf: 1d array of length n_t. Averaged tomography results for the 2nd excited state population vs time.
    :param n_bins: Number of bins for the qutrit populations
    :return: figure handle, for saving
    """
    bins, histPg, histPe, histPf = get_histogram(weak_meas_times, Pg_trajectories, Pe_trajectories,
                                                 Pf_trajectories, n_bins=n_bins, bin_min=0, bin_max=1)

    cmap = plt.cm.hot
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPg, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPg.max()))
    plt.plot(tomography_times * 1e6, Pg, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_g$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPe, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPe.max()))
    plt.plot(tomography_times * 1e6, Pe, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_e$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPf, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPf.max()))
    plt.plot(tomography_times * 1e6, Pf, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_f$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()
    return fig


def plot_individual_trajs(weak_meas_times, X, Y, Z, traj_indices=np.arange(4), n_bins=101):
    """
    Plot a histogram with individual trajectories on top.
    :param weak_meas_times: time axis for the weak measurement
    :param X: 2d array of shape (n_reps, n_timesteps)
    :param Y: 2d array of shape (n_reps, n_timesteps)
    :param Z: 2d array of shape (n_reps, n_timesteps)
    :param traj_indices: array-like, which trajectories to plot
    :param n_bins: int, number of bins for the qubit x-axis.
    :return: figure handle, for saving.
    """
    bins, histX, histY, histZ = get_histogram(weak_meas_times, X, Y, Z, n_bins=n_bins)

    traj_cols = ['r', 'darkorange', 'yellow', 'forestgreen']
    cmap = plt.cm.Greys_r
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histZ, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histZ.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, Z[k, :], color=traj_cols[l])

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Z$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histY, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histY.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, Y[k, :], color=traj_cols[l])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Y$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histX, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histX.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, X[k, :], color=traj_cols[l])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$X$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()

    return fig

def make_a_pie(time_series_lengths, title="", savepath=None):
    """
    Create a pie chart, where the slices will be ordered and plotted counter-clockwise.
    :param time_series_lengths: 1d array of integers
    :param title: string, will be displayed in the figure
    :param savepath: string, defaults to None. If none, no figure will be saved.
    :return: None
    """
    labels = np.unique(time_series_lengths)
    sizes = [len(np.where(time_series_lengths == label)[0]) for label in labels]
    colors = plt.cm.viridis(np.arange(len(labels)) / (len(labels) - 1))

    fig = plt.figure()
    plt.title(title)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=colors)
    plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if savepath is not None:
        fig.savefig(os.path.join(savepath, title.replace(" ", "_") + ".png"), **save_options)
