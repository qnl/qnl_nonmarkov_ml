import sys, os, h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import matplotlib
from scipy.linalg import sqrtm

sys.path.append(r"/home/qnl/Git-repositories")
from qnl_trajectories.analysis.utils import greek
from qnl_trajectories.analysis.nn_plotting import *
from qnl_trajectories import x_color, y_color, z_color, Id, sigmaX, sigmaY, sigmaZ

from qnl_nonmarkov_ml.gate_diagnosis_lstm.utils import *
from qnl_nonmarkov_ml.gate_diagnosis_lstm.qutrit_lstm_network import *
from rich import print
from rich.console import Console
console = Console()

# dark_mode_compatible(dark_mode_color=r'#86888A')

# settings = load_settings(r"/home/qnl/Git-repositories/qnl_nonmarkov_ml/gate_diagnosis_lstm/settings.yaml")
yaml_file = r"/home/qnl/noah/projects/2020-NonMarkovTrajectories/code/qnl_nonmarkov_ml/gate_diagnosis_lstm/settings.yaml"
settings = load_settings(yaml_file)

datapath = settings['voltage_records']['filepath'] # Path of the data
dataname = settings['voltage_records']['filename'] # Filename of h5 file used to feed into prep.py
filepath = os.path.join(datapath, 'analysis', settings['analysis']['subdir']) # Path of the trained trajectories

trajectory_dt = settings['analysis']['trajectory_dt']
multiple_prep_states = settings['voltage_records']['multiple_prep_states']
prep_states = settings['voltage_records']['prep_states']
num_prep_states = len(prep_states)
data_points_for_prep_state = settings['voltage_records']['data_points_for_prep_state']

# arrow_length_multiplier = 1.25 # Artificially lengthens the arrows. Default 1.0 means length is true to actual length
ROTATION_ANGLE = 0 # Rotation angle of the data
fit_guess = settings['analysis']['hamiltonian_map']['fit_guess'] # Fit guess for Hamiltonian map fit, gamma, omega
derivative_order = settings['analysis']['derivative_order']
omega_fixed = settings['analysis']['hamiltonian_map']['omega_fixed']
t_min = settings['analysis']['t_min']
t_max = settings['analysis']['t_max']

sweep_time = settings['analysis']['sweep_time']['sweep_time'] # Bin the trajectories in time to fit parameters as function of time.
time_window = 0.2e-6 # Use this time window when sweep_time = True

plot_average = settings.get('analysis/sweep_time/plot_average')
running_sweep = settings.get('analysis/sweep_time/running_sweep')
if running_sweep:
    t_mins = np.linspace(0.4e-6, 3.8e-6,
                         1 + np.int(np.round((6.8e-6 - 0.4e-6) / 5e-9)))  # Left side of the time window
else:
    t_mins = np.linspace(0.4e-6, 3.8e-6, 1 + np.int(np.round((6.8e-6 - 0.4e-6) / time_window))) # Left side of the time window
t_maxs = t_mins + time_window # Right side of the time window

x_for_yz_fit = settings['analysis']['x_for_yz_fit'] # Keep None if you don't want to select on the x coordinate
y_for_xz_fit = settings['analysis']['y_for_xz_fit']
z_for_xy_fit = settings['analysis']['z_for_xy_fit']

for k, p in enumerate(prep_states):
    console.print(f"Loading data...", style="bold green")

    # Load the data from the h5 file
    if k == 0:
        d = load_repackaged_data(os.path.join(datapath, dataname), multi_prep_state=multiple_prep_states)

    dX = d[f'prep_{p}']['meas_X']
    dY = d[f'prep_{p}']['meas_Y']
    dZ = d[f'prep_{p}']['meas_Z']

    Px = np.array([np.sum(dX[key]['final_ro_results'] == 1) / len(dX[key]['final_ro_results']) for key in dX.keys()])
    Py = np.array([np.sum(dY[key]['final_ro_results'] == 1) / len(dY[key]['final_ro_results']) for key in dY.keys()])
    Pz = np.array([np.sum(dZ[key]['final_ro_results'] == 1) / len(dZ[key]['final_ro_results']) for key in dZ.keys()])

    expX = 1 - 2 * Px
    expY = 1 - 2 * Py
    expZ = 1 - 2 * Pz

    # dt = dZ['t_0']['dt_binned']
    dt = dZ['t_1']['dt_binned']
    timesteps = np.sort([int(key[2:]) for key in list(dZ.keys()) if key[:2] == 't_'])
    # Sequence lengths to process for the quiver maps, in units of trajectory dt
    Tm = np.array([np.round(dZ[f't_{ts}']['time_axis_filtered'][-1], decimals=9) for ts in timesteps])
    seq_lengths = np.array([np.shape(dZ[f't_{ts}']['I_binned_filtered'])[1] for ts in timesteps])
    tfinal = Tm[-1]

    # Load the longest trained trajectories
    with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
        try:
            xyz_pred = f.get(f'prep_{p}/predictions_{seq_lengths[-1]}')[:]
        except:
            print(seq_lengths)
            print(list(f.keys()))
        time = f.get('t')[:]

    dt = np.diff(time)[0]
    print(dt)

    Xf = xyz_pred[:, data_points_for_prep_state:, 0]
    Yf = xyz_pred[:, data_points_for_prep_state:, 1]
    Zf = xyz_pred[:, data_points_for_prep_state:, 2]
    time = time[data_points_for_prep_state:]

    # Loop over the strong readout results to get the fidelity
    fidelities = list()
    trace_dist = list()
    for k, t in enumerate(Tm):
        nearest_traj_idx = find_nearest(time, t)
        rho_tilde = 0.5 * (Id + np.mean(Xf, axis=0)[nearest_traj_idx] * sigmaX +
                           np.mean(Yf, axis=0)[nearest_traj_idx] * sigmaY +
                           np.mean(Zf, axis=0)[nearest_traj_idx] * sigmaZ)

        # Find the real density matrix from tomography results
        rho = 0.5 * (Id + expX[k] * sigmaX + expY[k] * sigmaY + expZ[k] * sigmaZ)
        fidelities.append(np.trace(sqrtm(sqrtm(rho) @ rho_tilde @ sqrtm(rho))) ** 2)
        trace_dist.append(0.5 * np.trace(sqrtm((rho - rho_tilde).conj().T @ (rho - rho_tilde))))

    # Calcalate the fidelity averaged over all timesteps
    avg_fid = np.mean(fidelities)
    avg_trace_dist = np.mean(trace_dist)
    print("Maximum and average fidelity = ", np.max(fidelities), avg_fid)
    print("Maximum and average trace dist = ", np.max(trace_dist), avg_trace_dist)

    # Comparison of trajectories with strong readout
    fig = plt.figure()
    plt.plot(time*1e6, np.mean(Xf, axis=0), color=x_color, lw=2, label="X")
    plt.plot(Tm * 1e6, expX, 'o', color=x_color, markersize=4)
    plt.plot(time*1e6, np.mean(Yf, axis=0), color=y_color, lw=2, label="Y")
    plt.plot(Tm * 1e6, expY, 'o', color=y_color, markersize=4)
    plt.plot(time*1e6, np.mean(Zf, axis=0), color=z_color, lw=2, label="Z")
    plt.plot(Tm * 1e6, expZ, 'o', color=z_color, markersize=4,
             label=f"$F$ = {np.abs(avg_fid):.3f}, $T$ = {np.abs(avg_trace_dist):.2e}")
    plt.title(r"Average trajectories ($-$) and tomography ($\bullet$)")
    plt.xlabel(f"Time ({greek('mu')}s)")
    plt.ylabel(f"Qubit coordinates X, Y, Z")
    plt.legend(loc=0, frameon=False)
    plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
    fig.savefig(os.path.join(filepath, f"001_traj_strong_ro_comparison_prep_{p}.png"), **settings['figure_options'])

    # Average purity vs. time
    average_purity = np.mean(np.sqrt(Xf**2 + Yf**2 + Zf**2), axis=0)
    fig = plt.figure()
    plt.plot(time[:np.shape(Xf)[1]]*1e6, average_purity)
    plt.ylabel(r"$\mathrm{Tr}(\rho^2)$")
    plt.xlabel(f"Time ({greek('mu')}s)")
    plt.xlim(0, np.max(time[:np.shape(Xf)[1]]*1e6))
    fig.savefig(os.path.join(filepath, f"001_traj_avg_purity_prep_{p}.png"), **settings['figure_options'])

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
    fig.savefig(os.path.join(filepath, f"001_traj_histogram_yz_prep_{p}.png"), **settings['figure_options'])

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
    fig.savefig(os.path.join(filepath, f"001_traj_histogram_xz_prep_{p}.png"), **settings['figure_options'])

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
    fig.savefig(os.path.join(filepath, f"001_traj_histogram_xy_prep_{p}.png"), **settings['figure_options'])

    plt.close('all')

    # Grid spacing
    d_bin = settings['analysis']['bin_size']
    x_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    y_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    z_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)

    console.print(f"Making yz quiver plots", style="bold green")

    # Bin trajectories in the YZ plane, now taking into account trained trajectories of all lengths
    yz_output = calculate_drho([filepath], x_bins, y_bins, z_bins, seq_lengths, horizontal_axis="Y",
                               main_group=f'prep_{p}',
                               # prefix=f"prep_{p}/predictions",
                               vertical_axis="Z", other_coordinate=x_for_yz_fit, t_min=t_min, t_max=t_max,
                               derivative_order=derivative_order)
    y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, eig1, eig2 = yz_output

    # From the average values we can fit the Hamiltonian parameters in the YZ plane
    fr_all_times, ferr_all_times = plot_and_fit_hamiltonian(y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, dt,
                                                            theta=ROTATION_ANGLE, savepath=filepath, axis_identifier='yz',
                                                            arrow_length_multiplier=settings['analysis']['hamiltonian_map']['arrow_length_multiplier'],
                                                            fix_omega=omega_fixed, fit_guess=fit_guess)

    # From the eigenvectors eig1 and eig2 we can find the measurement back-action in the YZ plane
    plot_stochastic(y_bin_centers, z_bin_centers, eig1, eig2, filepath, theta=ROTATION_ANGLE, axis_identifier='yz',
                    arrow_length_multiplier=settings['analysis']['backaction_map']['arrow_length_multiplier'],
                    color_min=settings['analysis']['backaction_map']['color_min'],
                    color_max=settings['analysis']['backaction_map']['color_max'])

    # Repeat the same procedure for the XY plane
    console.print(f"Making xy quiver plots", style="bold green")
    xy_output = calculate_drho([filepath], x_bins, y_bins, z_bins, seq_lengths, horizontal_axis="X",
                               main_group=f'prep_{p}',
                               # prefix=f"prep_{p}/predictions",
                               vertical_axis="Y", other_coordinate=z_for_xy_fit, t_min=t_min, t_max=t_max,
                               derivative_order=derivative_order)
    x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, eig1, eig2 = xy_output

    plot_and_fit_hamiltonian(x_bin_centers, y_bin_centers, mean_binned_dX, mean_binned_dY, dt, savepath=filepath,
                             axis_identifier='xy', fit=False,
                             arrow_length_multiplier=settings['analysis']['hamiltonian_map']['arrow_length_multiplier'])
    plot_stochastic(x_bin_centers, y_bin_centers, eig1, eig2, filepath, axis_identifier='xy',
                    arrow_length_multiplier=settings['analysis']['backaction_map']['arrow_length_multiplier'],
                    color_min=settings['analysis']['backaction_map']['color_min'],
                    color_max=settings['analysis']['backaction_map']['color_max'])

    # Repeat the same procedure for the xz plane
    console.print(f"Making xz quiver plots", style="bold green")
    xz_output = calculate_drho([filepath], x_bins, y_bins, z_bins, seq_lengths, horizontal_axis="X",
                               main_group=f'prep_{p}',
                               # prefix=f"prep_{p}/predictions",
                               vertical_axis="Z", other_coordinate=y_for_xz_fit, t_min=t_min, t_max=t_max,
                               derivative_order=derivative_order)
    x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, eig1, eig2 = xz_output

    plot_and_fit_hamiltonian(x_bin_centers, z_bin_centers, mean_binned_dX, mean_binned_dZ, dt, savepath=filepath,
                             axis_identifier='xz', fit=False,
                             arrow_length_multiplier=settings['analysis']['hamiltonian_map']['arrow_length_multiplier'])
    plot_stochastic(x_bin_centers, z_bin_centers, eig1, eig2, filepath, axis_identifier='xz',
                    arrow_length_multiplier=settings['analysis']['backaction_map']['arrow_length_multiplier'],
                    color_min=settings['analysis']['backaction_map']['color_min'],
                    color_max=settings['analysis']['backaction_map']['color_max'])

    plt.close('all')
#
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
    d_bin = settings['analysis']['bin_size']
    x_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    y_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)
    z_bins = np.arange(-1.0, 1.0 + d_bin, d_bin)

    for t_min, t_max in tqdm(zip(t_mins, t_maxs)):
        try:
            yz_output = calculate_drho(
                [filepath],
                x_bins,
                y_bins,
                z_bins,
                seq_lengths,
                horizontal_axis="Y",
                vertical_axis="X",
                other_coordinate=z_for_xy_fit,
                t_min=t_min,
                t_max=t_max,
                derivative_order=derivative_order,
                main_group=f'prep_{p}',
                )
            y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, eig1, eig2 = yz_output

            fr, ferr = plot_and_fit_hamiltonian(y_bin_centers, z_bin_centers, mean_binned_dY, mean_binned_dZ, dt,
                                                theta=ROTATION_ANGLE, savepath=os.path.join(filepath, 'traj_swarm'),
                                                fit_guess=fit_guess, axis_identifier='xy', plot=True, ax_fig=None,
                                                fit=True,
                                                # fit=False,
                                                fix_omega=omega_fixed,
                                                arrow_length_multiplier=settings['analysis']['hamiltonian_map']['arrow_length_multiplier'])

            omegas.append(fr[1] / (2 * np.pi))
            gammas.append(fr[0] / (2 * np.pi))
            domegas.append(ferr[1] / (2 * np.pi))
            dgammas.append(ferr[0] / (2 * np.pi))
        except:
            omegas.append(np.inf)
            gammas.append(np.inf)
            domegas.append(np.inf)
            dgammas.append(np.inf)

    # fit_results = np.array(fit_results)
    # yerr = fit_results * np.sqrt((ferr[0]/fr[0])**2 + (ferr[1]/fr[1])**2)
    fig = plt.figure()
    plt.errorbar(0.5 * (t_mins + t_maxs) * 1e6, np.array(omegas) / (1e6), yerr=np.array(domegas)/1e6, color='gray',
                 fmt='o', label=f"{greek('Omega')}/2{greek('pi')} (instantaneous)")
    plt.errorbar(0.5 * (t_mins + t_maxs) * 1e6, np.array(gammas) / (1e6), yerr=np.array(dgammas)/1e6,
                 color=plt.cm.Blues(0.6), fmt='o', label=f"{greek('Gamma')}/2{greek('pi')} (instantaneous)")
    if plot_average:
        plt.hlines(fr_all_times[0] / (2 * np.pi * 1e6), 0, np.max(t_maxs)*1e6, linestyles='--', color=plt.cm.Blues(0.6),
                   label=f"{greek('Gamma')}/2{greek('pi')} (all trajectories)")
        plt.hlines(fr_all_times[1] / (2 * np.pi * 1e6), 0, np.max(t_maxs) * 1e6, linestyles='--', color='gray',
                   label=f"{greek('Omega')}/2{greek('pi')} (all trajectories)")

    plt.xlim(0, np.max(t_maxs) * 1e6)
    plt.xlabel(f"Time ({greek('mu')}s)")
    plt.ylabel(f"{greek('Omega')}/2{greek('pi')}, {greek('Gamma')}/2{greek('pi')} (MHz)")
    plt.legend(loc=0, frameon=False)
    fig.savefig(os.path.join(filepath, "001_traj_hamiltonian_fit_vs_time.png"), **settings['figure_options'])