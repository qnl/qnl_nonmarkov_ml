import sys, os, h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import matplotlib

sys.path.append(r"/home/qnl/Git-repositories")
from qnl_trajectories import data_analysis
from qnl_trajectories.utils import greek
from qnl_trajectories import x_color, y_color, z_color
from machine_learning_test.utils import *
from machine_learning_test.all_timesteps import *
from plotting import *
from rich import print
from rich.console import Console
console = Console()

dark_mode_compatible(dark_mode_color=r'#86888A')

last_timestep = 39

# datapath = filepath
# filepath = m.savepath

# datapath = r"/home/qnl/Git-repositories/machine_learning_test/data/cts_rabi_amp_2/prep_Y"
# filepath = r"/home/qnl/Git-repositories/machine_learning_test/analysis/rabi_amp_sweep/200518_152413_cts_rabi_amp_2_prep_Y_all_times"

dX = data_analysis.load_data(os.path.join(datapath, 'meas_X'), last_timestep=last_timestep, qubit='Q6')
dY = data_analysis.load_data(os.path.join(datapath, 'meas_Y'), last_timestep=last_timestep, qubit='Q6')
dZ = data_analysis.load_data(os.path.join(datapath, 'meas_Z'), last_timestep=last_timestep, qubit='Q6')

Tm, expX, expY, expZ = data_analysis.plot_average_trajectories(dX, dY, dZ,
                                                               timesteps=np.arange(0, last_timestep+1),
                                                               fit_curves=[],
                                                               artificial_detuning=False,
                                                               savepath=None)

expX = np.array(expX)
expY = np.array(expY)
expZ = np.array(expZ)

with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
    xyz_pred = f.get('predictions_190')[:]
    time = f.get('t')[:]

Xf = xyz_pred[..., 0]
Yf = xyz_pred[..., 1]
Zf = xyz_pred[..., 2]

# Comparison of trajectories with strong readout
fig = plt.figure()
plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Xf, axis=0), color=x_color, lw=2)
plt.plot(Tm * 1e6, expX, 'o', color=x_color, markersize=4)
plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Yf, axis=0), color=y_color, lw=2)
plt.plot(Tm * 1e6, expY, 'o', color=y_color, markersize=4)
plt.plot(time[:np.shape(Xf)[1]]*1e6, np.mean(Zf, axis=0), color=z_color, lw=2)
plt.plot(Tm * 1e6, expZ, 'o', color=z_color, markersize=4)
plt.title("Comparison average trajectories and strong readout")
plt.xlabel(f"Time ({greek('mu')}s)")
plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
fig.savefig(os.path.join(filepath, "001_traj_strong_ro_comparison.png"), dpi=200, bbox_inches='tight')

# Average purity vs. time
average_purity = np.mean(np.sqrt(Xf**2 + Yf**2 + Zf**2), axis=0)
fig = plt.figure()
plt.plot(time[:np.shape(Xf)[1]]*1e6, average_purity)
plt.ylabel(r"$\mathrm{Tr}(\rho^2)$")
plt.xlabel(f"Time ({greek('mu')}s)")
plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
fig.savefig(os.path.join(filepath, "001_traj_avg_purity.png"), dpi=200, bbox_inches='tight')

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
d_bin = 0.05
x_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
y_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
z_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)

for timesteps in trange(190, 195, 5):
    with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
        xyz_pred = f.get(f'predictions_{timesteps}')[:]

    console.print(f"Making yz quiver plots", style="bold green")
    y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, eig1, eig2 = plot_drho(xyz_pred, x_bins, y_bins,
                                                                                         z_bins, horizontal_axis="Y",
                                                                                         vertical_axis="Z",
                                                                                         other_coordinate=0.0)
    plot_and_fit_hamiltonian(y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, filepath, axis_identifier='yz')
    plot_quiver(y_bin_centers, z_bin_centers, eig1, eig2, filepath, axis_identifier='yz')

    console.print(f"Making xy quiver plots", style="bold green")
    x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, eig1, eig2 = plot_drho(xyz_pred, x_bins, y_bins,
                                                                                         z_bins, horizontal_axis="X",
                                                                                         vertical_axis="Y",
                                                                                         other_coordinate=0.0)
    plot_and_fit_hamiltonian(x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, filepath,
                             axis_identifier='xy', fit=False)
    plot_quiver(x_bin_centers, y_bin_centers, eig1, eig2, filepath, axis_identifier='xy')

    console.print(f"Making xz quiver plots", style="bold green")
    x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, eig1, eig2 = plot_drho(xyz_pred, x_bins, y_bins,
                                                                                         z_bins, horizontal_axis="X",
                                                                                         vertical_axis="Z",
                                                                                         other_coordinate=0.0)
    plot_and_fit_hamiltonian(x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, filepath,
                             axis_identifier='xz', fit=False)
    plot_quiver(x_bin_centers, z_bin_centers, eig1, eig2, filepath, axis_identifier='xz')

    plt.close('all')