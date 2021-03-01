import sys, os, h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import matplotlib

from qnl_trajectories.analysis import data_analysis
from qnl_trajectories.analysis.load_hdf5 import load_hdf5
from qnl_trajectories.analysis.utils import greek
from qnl_trajectories.analysis.nn_plotting import *
from qnl_trajectories import x_color, y_color, z_color

from qnl_nonmarkov_ml.vanilla_lstm.utils import *
from qnl_nonmarkov_ml.vanilla_lstm.vanilla_lstm import *
from rich import print
from rich.console import Console
console = Console()

dark_mode_compatible(dark_mode_color=r'#86888A')

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
last_timestep = 99

datapath = r'/run/media/qnl/Seagate Expansion Drive/non_markovian/local_data/2021_02_27/cr_trajectories_test_029/data_transfer/2021_02_27/cr_trajectories_test_029'
filepath = r"analysis/cr/cr_trajectories_test_029/210301_092924_cr_trajectories_test_029_prep_C+X_T+Y" # Path of the trained trajectories

arrow_length_multiplier = 1 # Artificially lengthens the arrows. Default 1.0 means length is true to actual length
ROTATION_ANGLE = 0 # Rotation angle of the data
seq_lengths = np.arange(6, 594, 6) # Sequence lengths to process for the quiver maps, in units of weak measurement dt

sweep_time = True # Bin the trajectories in time to fit parameters as function of time.
time_window = 0.1e-6 # Use this time window when sweep_time = True
t_min_ = 0.1e-6
t_max_ = 2e-6

t_mins = np.arange(t_min_, t_max_, time_window) # Left side of the time window
t_maxs = np.arange(t_min_ + time_window, t_max_ + time_window, time_window) # Right side of the time window

plot_sro = False  # plot strong readout results

console.print(f"Loading data...", style="bold green")

h5 = True
load = False

if load:
    if h5:
        load_hdf5_ = load_hdf5.LoadHDF5()

        def keys_(ax):
            return ['prep_C+X_T+Y', f'meas_C+{ax}_T+{ax}', 'prep_C+X_T+Y', f'meas_+{ax}_T+{ax}']

        dX = load_hdf5_.load_data(datapath, keys=keys_('X'), qubit='Q6', last_timestep=last_timestep)
        console.print("Loaded X", style="bold red")

        dY = load_hdf5_.load_data(datapath, keys=keys_('Y'), qubit='Q6', last_timestep=last_timestep)
        console.print("Loaded Y", style="bold red")

        dZ = load_hdf5_.load_data(datapath, keys=keys_('Z'), qubit='Q6', last_timestep=last_timestep)
        console.print("Loaded Z", style="bold red")

    else:

        meas_X = r"meas_C+Z_T+X"
        meas_Y = r"meas_C+Z_T+Y"
        meas_Z = r"meas_C+Z_T+Z"

        dX = data_analysis.load_data(os.path.join(datapath, meas_X), qubit='Q6', method='final')
        dY = data_analysis.load_data(os.path.join(datapath, meas_Y), qubit='Q6', method='final')
        dZ = data_analysis.load_data(os.path.join(datapath, meas_Z), qubit='Q6', method='final')

if plot_sro:
    Tm, expX, expY, expZ = data_analysis.plot_strong_ro_results(dX, dY, dZ,
                                                                   timesteps=np.arange(0, last_timestep+1),
                                                                   fit_curves=[],
                                                                   artificial_detuning=False,
                                                                   savepath=None)

    expX = np.array(expX)
    expY = np.array(expY)
    expZ = np.array(expZ)

# Load the longest trained trajectories
with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
    try:
        xyz_pred = f.get(f'predictions_{seq_lengths[-1]}')[:]
    except:
        print(list(f.keys()))
    time = f.get('t')[:]

dt = np.diff(time)[0]
print(dt)

Xf = xyz_pred[..., 0]
Yf = xyz_pred[..., 1]
Zf = xyz_pred[..., 2]

# # Comparison of trajectories with strong readout
# fig = plt.figure()
# plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Xf, axis=0), color=x_color, lw=2)
# plt.plot(Tm * 1e6, expX, 'o', color=x_color, markersize=4)
# plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Yf, axis=0), color=y_color, lw=2)
# plt.plot(Tm * 1e6, expY, 'o', color=y_color, markersize=4)
# plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Zf, axis=0), color=z_color, lw=2)
# plt.plot(Tm * 1e6, expZ, 'o', color=z_color, markersize=4)
# plt.title("Comparison average trajectories and strong readout")
# plt.xlabel(f"Time ({greek('mu')}s)")
# plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
# fig.savefig(os.path.join(filepath, "001_traj_strong_ro_comparison.png"), dpi=200, bbox_inches='tight')
#
# # Average purity vs. time
# average_purity = np.mean(np.sqrt(Xf**2 + Yf**2 + Zf**2), axis=0)
# fig = plt.figure()
# plt.plot(time[:np.shape(Xf)[1]]*1e6, average_purity)
# plt.ylabel(r"$\mathrm{Tr}(\rho^2)$")
# plt.xlabel(f"Time ({greek('mu')}s)")
# plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
# fig.savefig(os.path.join(filepath, "001_traj_avg_purity.png"), dpi=200, bbox_inches='tight')

dX = Xf[:, 1:] - Xf[:, :-1]
dY = Yf[:, 1:] - Yf[:, :-1]
dZ = Zf[:, 1:] - Zf[:, :-1]

Xf_centered = (Xf[:, 1:] + Xf[:, :-1]) / 2.
Yf_centered = (Yf[:, 1:] + Yf[:, :-1]) / 2.
Zf_centered = (Zf[:, 1:] + Zf[:, :-1]) / 2.

x_bins = np.arange(-1.0, 1.02, 0.02)
y_bins = np.arange(-1.0, 1.02, 0.02)
z_bins = np.arange(-1.0, 1.02, 0.02)

Hyz, edges1, edges2 = np.histogram2d(Yf_centered.flatten(), Zf_centered.flatten(), bins=(y_bins, z_bins))
Hxy, edges1, edges2 = np.histogram2d(Xf_centered.flatten(), Yf_centered.flatten(), bins=(x_bins, y_bins))
Hxz, edges1, edges2 = np.histogram2d(Xf_centered.flatten(), Zf_centered.flatten(), bins=(x_bins, z_bins))

# Histograms
fig = plt.figure(figsize=(6.,6.))
ax = plt.gca()
im = plt.pcolormesh(edges1, edges2, Hyz.T, cmap=plt.cm.hot, vmin=0, vmax=200)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title("Occurences")
x_circle = np.linspace(-np.pi, np.pi)
plt.plot(np.cos(x_circle), np.sin(x_circle), color='white')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Y")
plt.ylabel("Z")
ax.set_aspect('equal')
fig.savefig(os.path.join(filepath, "001_traj_histogram_yz.png"), dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(6.,6.))
plt.pcolormesh(edges1, edges2, Hxz.T, cmap=plt.cm.hot, vmin=0, vmax=600)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title("Occurences")
plt.plot(np.cos(x_circle), np.sin(x_circle), color='white')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("X")
plt.ylabel("Z")
plt.gca().set_aspect('equal')
fig.savefig(os.path.join(filepath, "001_traj_histogram_xz.png"), dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(6.,6.))
plt.pcolormesh(edges1, edges2, Hxy.T, cmap=plt.cm.hot, vmin=0, vmax=600)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title("Occurences")
plt.plot(np.cos(x_circle), np.sin(x_circle), color='white')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')
fig.savefig(os.path.join(filepath, "001_traj_histogram_xy.png"), dpi=200, bbox_inches='tight')

plt.close('all')

# Grid spacing
d_bin = 0.1
x_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
y_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
z_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)

console.print(f"Making yz quiver plots", style="bold green")

# Bin trajectories in the YZ plane, now taking into account trained trajectories of all lengths
y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, eig1, eig2 = calculate_drho([filepath], x_bins, y_bins,
                                                                                     z_bins, seq_lengths, horizontal_axis="Y",
                                                                                     vertical_axis="Z",
                                                                                     other_coordinate=0.0,
                                                                                     t_min=1e-6, t_max=20e-6)
# From the average values we can fit the Hamiltonian parameters in the YZ plane
fr_all_times, ferr_all_times = plot_and_fit_hamiltonian(y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, dt,
                                                        theta=ROTATION_ANGLE, savepath=filepath, axis_identifier='yz',
                                                        arrow_length_multiplier=arrow_length_multiplier)

# From the eigenvectors eig1 and eig2 we can find the measurement back-action in the YZ plane
plot_stochastic(y_bin_centers, z_bin_centers, eig1, eig2, filepath, theta=ROTATION_ANGLE, axis_identifier='yz')

# Repeat the same procedure for the XY plane
console.print(f"Making xy quiver plots", style="bold green")
x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, eig1, eig2 = calculate_drho([filepath], x_bins, y_bins,
                                                                                     z_bins, seq_lengths, horizontal_axis="X",
                                                                                     vertical_axis="Y",
                                                                                     other_coordinate=0.0,
                                                                                     t_min=None, t_max=None)
plot_and_fit_hamiltonian(x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, dt, savepath=filepath,
                         axis_identifier='xy', fit=False, arrow_length_multiplier=arrow_length_multiplier)
plot_stochastic(x_bin_centers, y_bin_centers, eig1, eig2, filepath, axis_identifier='xy')

# Repeat the same procedure for the xz plane
console.print(f"Making xz quiver plots", style="bold green")
x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, eig1, eig2 = calculate_drho([filepath], x_bins, y_bins,
                                                                                     z_bins, seq_lengths, horizontal_axis="X",
                                                                                     vertical_axis="Z",
                                                                                     other_coordinate=0.0,
                                                                                     t_min=None, t_max=None)
plot_and_fit_hamiltonian(x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, dt, savepath=filepath,
                         axis_identifier='xz', fit=False, arrow_length_multiplier=arrow_length_multiplier)
plot_stochastic(x_bin_centers, z_bin_centers, eig1, eig2, filepath, axis_identifier='xz',
                arrow_length_multiplier=arrow_length_multiplier)

plt.close('all')

omegas = []
gammas = []
domegas = []
dgammas = []

if sweep_time:
    console.print("Sweeping over time axis", style="bold red")
    # fig = plt.figure(figsize=(14, 14))
    k = 0

    if not(os.path.exists(os.path.join(filepath, 'traj_swarm'))):
        os.mkdir(os.path.join(filepath, 'traj_swarm'))

    # Grid spacing for the time sweep
    d_bin = 0.1
    x_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    y_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    z_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)

    for t_min, t_max in tqdm(zip(t_mins, t_maxs)):
        x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, eig1, eig2 = calculate_drho([filepath], x_bins, y_bins,
                                                                                             z_bins, seq_lengths, horizontal_axis="X",
                                                                                             vertical_axis="Y",
                                                                                             other_coordinate=0.0,
                                                                                             t_min=t_min, t_max=t_max)

        out = plot_and_fit_hamiltonian(x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, dt,
                                            theta=ROTATION_ANGLE,
                                            savepath=os.path.join(filepath, 'traj_swarm'),
                                            axis_identifier='xy',
                                            plot=True,
                                            ax_fig=None,
                                            fit=False,
                                            arrow_length_multiplier=arrow_length_multiplier)

        # fr, ferr = out
        # omegas.append(fr[1] / (2 * np.pi))
        # gammas.append(fr[0] / (2 * np.pi))
        # domegas.append(ferr[1] / (2 * np.pi))
        # dgammas.append(ferr[0] / (2 * np.pi))

    # fit_results = np.array(fit_results)
    # yerr = fit_results * np.sqrt((ferr[0]/fr[0])**2 + (ferr[1]/fr[1])**2)

    # fig = plt.figure()
    # plt.errorbar(0.5 * (t_mins + t_maxs) * 1e6, np.array(omegas) / (1e6), yerr=np.array(domegas)/1e6, color='gray',
    #              fmt='o', label=f"{greek('Omega')}/2{greek('pi')} (instantaneous)")
    # plt.hlines(fr_all_times[1] / (2 * np.pi * 1e6), 0, np.max(t_maxs)*1e6, linestyles='--', color='gray',
    #            label=f"{greek('Omega')}/2{greek('pi')} (all trajectories)")
    #
    # plt.errorbar(0.5 * (t_mins + t_maxs) * 1e6, np.array(gammas) / (1e6), yerr=np.array(dgammas)/1e6, color='navy',
    #              fmt='o', label=f"{greek('Gamma')}/2{greek('pi')} (instantaneous)")
    # plt.hlines(fr_all_times[0] / (2 * np.pi * 1e6), 0, np.max(t_maxs)*1e6, linestyles='--', color='navy',
    #            label=f"{greek('Gamma')}/2{greek('pi')} (all trajectories)")
    #
    # plt.xlim(0, np.max(t_maxs) * 1e6)
    # plt.xlabel(f"Time ({greek('mu')}s)")
    # plt.ylabel(f"{greek('Omega')}/2{greek('pi')}, {greek('Gamma')}/2{greek('pi')} (MHz)")
    # plt.legend(loc=0, frameon=False)
    # fig.savefig(os.path.join(filepath, "001_traj_hamiltonian_fit_vs_time.png"), dpi=200, bbox_inches='tight')