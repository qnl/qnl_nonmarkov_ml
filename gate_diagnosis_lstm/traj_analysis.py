from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os, matplotlib, h5py
from scipy import optimize
from tabulate import tabulate
from glob import glob
from utils import greek

def rotate_points(x, y, theta):
    """
    Rotates pairs of points (x, y) by an angle theta
    :param x: array
    :param y: array
    :param theta: rotation angle in radians
    :return: [rotated x points, rotated y points]
    """
    return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

def colorbar(mappable):
    """
    Acts the same as matplotlib's colorbar but works with subplots and rescales the colorbar correctly.
    Ex: pcm=plt.pcolormesh(x, y, z); colorbar(pcm)
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_state_labels(ax, x_label, y_label):
    shift = 0.02
    ax.text(0.0, 1.0 + shift, r"|+$%s\rangle$" % y_label, ha='center', va='bottom')
    ax.text(0.0, -1.0 - shift, r"|-$%s\rangle$" % y_label, ha='center', va='top')
    ax.text(1.0 + shift, 0.0, r"|+$%s\rangle$" % x_label, ha='left', va='center')
    ax.text(-1.0 - shift, 0.0, r"|-$%s\rangle$" % x_label, ha='right', va='center')

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    x and y are arrays of the same size. Plots the covariance ellipse on a matplotlib axes object.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def calculate_drho(filepaths, x_bins, y_bins, z_bins, seq_lengths, horizontal_axis="Y", vertical_axis="Z",
                   other_coordinate=0.0, derivative_order=1, t_min=None, t_max=None, convert_to_spherical=False,
                   main_group='', verbose=True, xy_rotation=0.0):
    """
    filepaths: list of filepaths that contain trajectories.h5 files
    x_bins: 1d array of bin edges
    y_bins: 1d array of bin edges for Y
    z_bins: 1d array of bin edges for Z
    horizontal_axis: "X", "Y" or "Z". To process data for a quiver map of X, Z set horizontal_axis to "X" and vertical_axis to "Z"
    vertical_axis: "X", "Y" or "Z", cannot be equal to horizontal_axis
    other_coordinate: float. Condition for the coordinate that is not horizontal_axis or vertical_axis. Default=0.0
    t_min: Sets the lower bound for the time window. Trajectory values below t_min are ignored
    t_max: Upper bound for the time_window. Trajectory values that occur after t_max are ignored.
    convert_to_spherical: Converts the X, Y, Z values in the trajectory to spherical coordinates. Default=False
    """
    for n, seq_length in enumerate(seq_lengths):
        for jj, filepath in enumerate(filepaths):
            with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
                group_name = f'predictions_{seq_length}'
                available_keys = list(f.keys()) if main_group == '' else list(f[main_group].keys())
                if group_name in available_keys:
                    sub_xyz = f.get(group_name)[:] if main_group == "" else f.get(os.path.join(main_group, group_name))[:]
                    if jj == 0:
                        xyz_array = sub_xyz
                    else:
                        xyz_array = np.vstack((xyz_array, sub_xyz))

                    time_key = 't'
                    times = f.get(time_key)[:][:seq_length]
                else:
                    # For backwards compatibility
                    closest_group_name = available_keys[np.argmax([f"_{seq_length}" in ak for ak in available_keys])]
                    if jj == 0:
                        if n == 0: # Only print the error for the first sequence_length
                            if verbose:
                                print(f"Error in loading key: {group_name}, found {closest_group_name} instead.")
                        xyz_array = f.get(closest_group_name)[:]
                    else:
                        xyz_array = np.vstack((xyz_array, f.get(closest_group_name)[:]))
                    times = f.get('t')[:][:seq_length]

        Xf = xyz_array[..., 0]
        Yf = xyz_array[..., 1]
        Zf = xyz_array[..., 2]

        # Apply a rotation in the Xf, Yf plane
        for k in range(Xf.shape[0]):
            X, Y = rotate_points(Xf[k, :], Yf[k, :], xy_rotation)
            Xf[k, :] = X
            Yf[k, :] = Y

        if convert_to_spherical:
            # convert trajectories to polar coordinates
            R = np.sqrt(Xf ** 2 + Yf ** 2 + Zf ** 2)
            T = np.arctan2(np.sqrt(Xf ** 2 + Yf ** 2), Zf)
            P = np.arctan2(Yf, Xf)

            Xf = R
            Yf = T
            Zf = P

        if derivative_order == 2:
            # Take centered derivatives. The second dimension is the time axis.
            dX = (Xf[:, 2:] - Xf[:, :-2]) / 2.
            dY = (Yf[:, 2:] - Yf[:, :-2]) / 2.
            dZ = (Zf[:, 2:] - Zf[:, :-2]) / 2.

            # Make sure Xf..Zf have the same dimension as dX...dZ
            Xf = Xf[:, 1:-1]
            Yf = Yf[:, 1:-1]
            Zf = Zf[:, 1:-1]
        elif derivative_order == 1:
            # Take centered derivatives. The second dimension is the time axis.
            dX = Xf[:, 1:] - Xf[:, :-1]
            dY = Yf[:, 1:] - Yf[:, :-1]
            dZ = Zf[:, 1:] - Zf[:, :-1]

            # Make sure Xf..Zf have the same dimension as dX...dZ
            Xf = Xf[:, :-1]
            Yf = Yf[:, :-1]
            Zf = Zf[:, :-1]

        assert np.all(np.shape(dX) == np.shape(Xf))

        # These are the bin numbers for the X-coordinate
        x_idcs = np.digitize(Xf, x_bins)
        y_idcs = np.digitize(Yf, y_bins)
        z_idcs = np.digitize(Zf, z_bins)

        if (t_min is None) or (t_max is None):
            time_mask_2d = np.ones(len(times[1:-1]), dtype=bool) if derivative_order==2 else np.ones(len(times[:-1]), dtype=bool)
        else:
            if derivative_order == 2:
                time_mask_1d = (times[1:-1] >= t_min) * (times[1:-1] <= t_max)
            elif derivative_order == 1:
                time_mask_1d = (times[:-1] >= t_min) * (times[:-1] <= t_max)
            time_mask_2d = np.repeat(np.reshape(time_mask_1d, (1, len(time_mask_1d))),
                                     np.shape(Xf)[0], axis=0)

        # print(np.shape(times))
        # print(np.shape(time_mask_2d))
        # print(np.shape(Xf))

        if horizontal_axis == "X":
            horizontal_bins = x_bins
            horizontal_idcs = x_idcs
            dhor = dX
        elif horizontal_axis == "Y":
            horizontal_bins = y_bins
            horizontal_idcs = y_idcs
            dhor = dY
        elif horizontal_axis == "Z":
            horizontal_bins = z_bins
            horizontal_idcs = z_idcs
            dhor = dZ

        if vertical_axis == "X":
            vertical_bins = x_bins
            vertical_idcs = x_idcs
            dvert = dX
        elif vertical_axis == "Y":
            vertical_bins = y_bins
            vertical_idcs = y_idcs
            dvert = dY
        elif vertical_axis == "Z":
            vertical_bins = z_bins
            vertical_idcs = z_idcs
            dvert = dZ

        if "X" not in [horizontal_axis, vertical_axis]:
            other_idcs = x_idcs
            other_bins = x_bins
        elif "Y" not in [horizontal_axis, vertical_axis]:
            other_idcs = y_idcs
            other_bins = y_bins
        else:
            other_bins = z_bins
            other_idcs = z_idcs

        if other_coordinate is not None:
            other_idx = np.digitize(other_coordinate, other_bins)

        if n == 0:
            mean_binned_hor = np.zeros((len(vertical_bins), len(horizontal_bins)))
            std_binned_hor = np.zeros((len(vertical_bins), len(horizontal_bins)))
            mean_binned_vert = np.zeros((len(vertical_bins), len(horizontal_bins)))
            std_binned_vert = np.zeros((len(vertical_bins), len(horizontal_bins)))

            eig1 = np.zeros((len(vertical_bins), len(horizontal_bins), 2))
            eig2 = np.zeros((len(vertical_bins), len(horizontal_bins), 2))

            num_samps_per_bin = np.zeros((len(vertical_bins), len(horizontal_bins)))

        k = 0
        for iter_1 in range(len(horizontal_bins)):
            for iter_2 in range(len(vertical_bins)):
                if other_coordinate is not None:
                    mask = (horizontal_idcs == iter_1) * (vertical_idcs == iter_2) * (
                            other_idcs == other_idx) * time_mask_2d
                else:
                    # Don't mask on the third coordinate if it's None
                    mask = (horizontal_idcs == iter_1) * (vertical_idcs == iter_2) * time_mask_2d
                # Don't mask on the last timestep, we're looking at derivative arrays
                # mask = mask[:, :-1]
                n_traj_this_bin = np.sum(mask)

                # At least 5 trajectories must make up a bin
                if n_traj_this_bin > 5:
                    # Keep track of the total number of trajectories in a single bin
                    num_samps_per_bin[iter_2, iter_1] += n_traj_this_bin
                    # Captures the coherent rotation part of the master eq. Multiply by n_traj to get weighted avg.
                    mean_binned_hor[iter_2, iter_1] += n_traj_this_bin * np.mean(dhor[mask])
                    mean_binned_vert[iter_2, iter_1] += n_traj_this_bin * np.mean(dvert[mask])

                    # Processing on the stochastic part of the master eq.
                    std_binned_hor[iter_2, iter_1] += n_traj_this_bin * np.std(dhor[mask])
                    std_binned_vert[iter_2, iter_1] += n_traj_this_bin * np.std(dvert[mask])

                    # Calculate the 2x2 covariance matrix
                    cov = np.cov(dhor[mask], dvert[mask])
                    w, v = np.linalg.eig(cov)
                    ev_order = np.argsort(w)
                    lambda1, lambda2 = w[ev_order]
                    ev1, ev2 = v[:, ev_order[0]], v[:, ev_order[1]]
                    k += 1

                    # Eigenvectors of the covariance matrix weighted by their eigenvalues.
                    # 1 standard deviations in the semi-major/minor axis
                    eig1[iter_2, iter_1] += n_traj_this_bin * np.sqrt(lambda1) * ev1
                    eig2[iter_2, iter_1] += n_traj_this_bin * np.sqrt(lambda2) * ev2

    not_zero = (num_samps_per_bin > 5)
    mean_binned_hor[not_zero] /= num_samps_per_bin[not_zero]
    mean_binned_vert[not_zero] /= num_samps_per_bin[not_zero]
    eig1[not_zero, 0] /= num_samps_per_bin[not_zero]
    eig1[not_zero, 1] /= num_samps_per_bin[not_zero]
    eig2[not_zero, 0] /= num_samps_per_bin[not_zero]
    eig2[not_zero, 1] /= num_samps_per_bin[not_zero]

    horizontal_bin_centers = (horizontal_bins[1:] + horizontal_bins[:-1]) / 2.
    vertical_bin_centers = (vertical_bins[1:] + vertical_bins[:-1]) / 2.
    return horizontal_bin_centers, vertical_bin_centers, mean_binned_hor, mean_binned_vert, eig1, eig2

def plot_and_fit_hamiltonian(horizontal_bin_centers, vertical_bin_centers, mean_binned_hor, mean_binned_vert, dt,
                             theta=0.0, savepath=None, axis_identifier="__", plot=True, ax_fig=None, fit=True,
                             print_fit_result=True, plot_residuals=False, fix_omega=False, fit_guess=[0.2e6, 1e6, 0.01],
                             arrow_length_multiplier=1.0):
    """
    horizontal_bin_centers: 1d array returned by calculate_drho
    vertical_bin_centers: 1d array returned by calculate drho
    mean_binned_hor: horizontal arrow magnitudes on the meshgrid spanned by horizontal_bin_centers, vertical_bin_centers
    mean_binned_vert: vertical arrow magnitude on the same meshgrid as mean_binned_hor
    dt: timestep between successive samples along a trajectory (alazar sampling time)
    savepath: filepath to save the figure png to. If None, no figure will be saved.
    axis_identifier: string of length 2, such as "XY"
    plot: bool, if True plots the 2d plot
    ax_fig: (ax, fig) matplotlib tuple. If None, a new figure will be created.
    fit: bool, if True, fit the quiver map to a Hamiltonian and Lindbladian dissipation
    arrow_length_multiplier: float, values larger than 1.0 increase the length of arrows on the quiver map.
    fix_omega: float or False. If larger than 0.0, the fit fixes the rabi rate parameter and only fits to the decay rate.
    """
    # There are three fit functions we can use.
    if len(fit_guess) == 1:
        assert fix_omega, "Please specify a numerical value for fix_omega"
        # fit_guess must be a list and contain [gamma]
        def dr_dt(x, y, z, gamma):
            Lambda = np.array([[-gamma, 0, 0], [0, -gamma, -fix_omega], [0, fix_omega, 0]])
            r = np.reshape(np.array([x, y, z]), (3, 1))
            return np.dot(Lambda, r) * dt

        def fitfunc(x, *p):
            gamma = p[0]
            ylist, zlist = x
            output = np.zeros(2 * len(ylist))
            k = 0

            for y, z in zip(ylist, zlist):
                ys, zs = dr_dt(0.0, y, z, gamma)[1:]
                output[2 * k] = ys
                output[2 * k + 1] = zs
                k += 1
            return output.flatten()
        bounds = ([-np.inf], [np.inf])
    elif len(fit_guess) == 2:
        # fit_guess must be a list and contain [gamma, omega]
        def dr_dt(x, y, z, gamma, omega):
            Lambda = np.array([[-gamma, 0, 0], [0, -gamma, -omega], [0, omega, 0]])
            r = np.reshape(np.array([x, y, z]), (3, 1))
            return np.dot(Lambda, r) * dt

        def fitfunc(x, *p):
            gamma, omega = p
            ylist, zlist = x
            output = np.zeros(2 * len(ylist))
            k = 0

            for y, z in zip(ylist, zlist):
                ys, zs = dr_dt(0.0, y, z, gamma, omega)[1:]
                output[2 * k] = ys
                output[2 * k + 1] = zs
                k += 1
            return output.flatten()
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
    elif len(fit_guess) == 3:
        # fit_guess must be a list and contain [gamma, omega, epsilon]
        def dr_dt(x, y, z, gamma, omega, epsilon):
            Lambda = np.array([[-gamma, 0, 0],
                               [0, -gamma*(1-np.sin(epsilon)**2), -omega+gamma*np.cos(epsilon)*np.sin(epsilon)],
                               [0, omega+gamma*np.cos(epsilon)*np.sin(epsilon), 0-gamma*(1-np.cos(epsilon)**2)]])
            r = np.reshape(np.array([x, y, z]), (3, 1))
            return np.dot(Lambda, r) * dt

        def fitfunc(x, *p):
            gamma, omega, epsilon = p
            ylist, zlist = x
            output = np.zeros(2 * len(ylist))
            k = 0

            for y, z in zip(ylist, zlist):
                ys, zs = dr_dt(0.0, y, z, gamma, omega, epsilon)[1:]
                output[2 * k] = ys
                output[2 * k + 1] = zs
                k += 1
            return output.flatten()
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    # This gives you the option to artificially apply a rotation to the vectors and coordinates
    _output = rotate_quiver_map(horizontal_bin_centers, vertical_bin_centers,
                                mean_binned_hor[1:, 1:], mean_binned_vert[1:, 1:], theta)
    Ymg_rotated, Zmg_rotated, Yvec_rotated, Zvec_rotated = _output

    # Hamiltonian part of the master equation
    ydata = np.zeros((np.shape(Ymg_rotated)[0], np.shape(Ymg_rotated)[1], 2))
    ydata[..., 0] = Yvec_rotated
    ydata[..., 1] = Zvec_rotated
    # Don't fit to coordinates that have vector components that are both zero and arrows outside the unit circle
    mask = ((ydata[..., 0] != 0) + (ydata[..., 1] != 0)) * (Ymg_rotated ** 2 + Zmg_rotated ** 2 <= 1)

    if fit:
        fr, fcov = optimize.curve_fit(fitfunc, (Ymg_rotated[mask], Zmg_rotated[mask]), ydata[mask, :].flatten(),
                                      p0=fit_guess, bounds=bounds)
        ferr = np.sqrt(np.diag(fcov))

        # For displaying the right fit results.
        if len(fit_guess) == 1:
            gamma = fr[0]
            dgamma = ferr[0]

            parnames = [f"{greek('Gamma')}/2{greek('pi')} (MHz)"]
            params = [gamma/(2*np.pi*1e6)]
            param_errs = [dgamma/(2*np.pi*1e6)]
        elif len(fit_guess) == 2:
            gamma, omega = fr
            dgamma, domega = ferr

            parnames = [f"{greek('Gamma')}/2{greek('pi')} (MHz)",
                        f"{greek('Omega')}/2{greek('pi')} (MHz)"]
            params = [gamma/(2*np.pi*1e6), omega/(2*np.pi*1e6)]
            param_errs = [dgamma/(2*np.pi*1e6), domega/(2*np.pi*1e6)]
        elif len(fit_guess) == 3:
            gamma, omega, epsilon = fr
            dgamma, domega, depsilon = ferr
            epsilon = epsilon % (-np.pi)

            parnames = [f"{greek('Gamma')}/2{greek('pi')} (MHz)",
                        f"{greek('Omega')}/2{greek('pi')} (MHz)",
                        f"{greek('epsilon')} (deg)"]
            params = [gamma/(2*np.pi*1e6), omega/(2*np.pi*1e6), epsilon * 180 / np.pi]
            param_errs = [dgamma/(2*np.pi*1e6), domega/(2*np.pi*1e6), depsilon * 180 / np.pi]

        if print_fit_result:
            print(tabulate(zip(parnames, params, param_errs),
                           headers=["Parameter", "Value", "Std"],
                           tablefmt="fancy_grid", floatfmt=".2f", numalign="center", stralign='left'))

        y_fit = fitfunc((Ymg_rotated.flatten(), Zmg_rotated.flatten()), *fr)
        fitted_dys = np.reshape(y_fit[::2], np.shape(Ymg_rotated))
        fitted_dzs = np.reshape(y_fit[1::2], np.shape(Ymg_rotated))

    x_circle = np.linspace(-np.pi, np.pi, 1000)

    if plot:
        if ax_fig is None:
            fig = plt.figure(figsize=(6, 6))
            ax = plt.gca()
        else:
            ax, fig = ax_fig

        if fit:
            ax.quiver(Ymg_rotated[mask], Zmg_rotated[mask], fitted_dys[mask], fitted_dzs[mask], color=plt.cm.Reds(0.6),
                      angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003, label='Model')
                      # label=r'Model: $\Gamma/2\pi = $%.2f $\pm$ %.2f MHz $\Omega/2\pi = $%.2f $\pm$ %.2f MHz \n$\epsilon = $%.2f rad'
                      #       %(gamma/(2*np.pi*1e6), dgamma/(2*np.pi*1e6), omega/(2*np.pi*1e6), domega/(2*np.pi*1e6), epsilon))

        ax.quiver(Ymg_rotated[mask], Zmg_rotated[mask], Yvec_rotated[mask], Zvec_rotated[mask],
                  angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003,
                  color='black', label='Trajectories')
        plot_state_labels(ax, axis_identifier[0].upper(), axis_identifier[1].upper())
        ax.set_ylim(-1.18, 1.18)
        ax.set_xlim(-1.18, 1.18)
        ax.plot(np.cos(x_circle), np.sin(x_circle), color='gray', lw=2)
        ax.set_aspect('equal')
        ax.set_xlabel(axis_identifier[0].upper())
        ax.set_ylabel(axis_identifier[1].upper())
        ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
        ax.set_title(r"Hamiltonian dynamics")
        ax.set_axis_off()
        legend = ax.legend(loc=1, frameon=False)
        legend.get_frame().set_facecolor('white')

        if savepath is not None:
            idx = len(glob(os.path.join(savepath, '*.png')))
            fig.savefig(os.path.join(savepath, f"{idx:04d}_traj_hamiltonian_dynamics_{axis_identifier}.png"), dpi=300,
                        bbox_inches='tight', transparent=True, pad_inches=0.05)

        if fit:
            residuals = np.sqrt((fitted_dys-Yvec_rotated)**2 + (fitted_dzs-Zvec_rotated)**2)
            masked_residuals = np.ma.masked_where(np.logical_not(mask), residuals)

        if plot_residuals and fit:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            pcm = ax.pcolormesh(Ymg_rotated, Zmg_rotated, masked_residuals, cmap=plt.cm.viridis)
            ax.plot(np.cos(x_circle), np.sin(x_circle), color='gray', lw=2)
            plt.title("Residuals")
            ax.set_xlabel(axis_identifier[0].upper())
            ax.set_ylabel(axis_identifier[1].upper())
            plot_state_labels(ax, axis_identifier[0].upper(), axis_identifier[1].upper())
            ax.set_ylim(-1.18, 1.18)
            ax.set_xlim(-1.18, 1.18)
            ax.set_aspect('equal')
            plt.yticks(np.arange(-1, 1, 0.5))
            plt.xticks(np.arange(-1, 1, 0.5))
            colorbar(pcm)

    if fit:
        if fix_omega:
            return np.array([fr[0], fix_omega]), np.array([ferr[0], 0.0])
        else:
            return fr, ferr

def plot_stochastic(horizontal_bin_centers, vertical_bin_centers, eig1, eig2, savepath, theta=0.0, axis_identifier="",
                    ax_fig=None, arrow_length_multiplier=1.0, do_colorplot=False, color_min=0.0, color_max=0.1,
                    arrow_color='gray'):
    """
    horizontal_bin_centers: 1d array, output from calculate_drho
    vertical_bin_centers: 1d array, output from calcaulte_drho
    eig1 3d array of shape (len(vertical_bin_centers), len(horizontal_bin_centers, 2) containing the 1st of 2 eigenvectors of the covariance matrix
    eig2: 3d array of same shape as eig1, containing the 2nd eigenvectors of the covariance matrix
    savepath: filepath, if  None, no figure is saved.
    axis_identifier: string of length 2, e.g. "XY" which serves as label for the axes.
    arrow_length_multiplier: float, a value larger than 1.0 increases the length of the arrows in the quiver plot.
    color_min:  float, clips the bottom of the colorbar of the backaction magnitude at this value.
    color_max: float, clips the colorbar of the backaction magnitude at this value
    """
    # Stochastic part of the master equation
    _output = rotate_quiver_map(horizontal_bin_centers, vertical_bin_centers, eig2[1:, 1:, 0], eig2[1:, 1:, 1], theta=theta)
    hor_mg_rotated, vert_mg_rotated, hor_eig2_rotated, vert_eig2_rotated = _output

    _output = rotate_quiver_map(horizontal_bin_centers, vertical_bin_centers, eig1[1:, 1:, 0], eig1[1:, 1:, 1], theta=theta)
    hor_mg_rotated, vert_mg_rotated, hor_eig1_rotated, vert_eig1_rotated = _output
    mask = np.sqrt(hor_mg_rotated ** 2 + vert_mg_rotated ** 2) <= 1

    x_circle = np.linspace(-np.pi, np.pi, 1000)

    if ax_fig is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
    else:
        ax, fig = ax_fig

    ax.quiver(hor_mg_rotated[mask], vert_mg_rotated[mask], hor_eig1_rotated[mask], vert_eig1_rotated[mask],
              angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003, color=arrow_color)
    ax.quiver(hor_mg_rotated[mask], vert_mg_rotated[mask], -hor_eig1_rotated[mask], -vert_eig1_rotated[mask],
              angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003, color=arrow_color)
    ax.quiver(hor_mg_rotated[mask], vert_mg_rotated[mask], hor_eig2_rotated[mask], vert_eig2_rotated[mask],
              angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003, color=arrow_color)
    ax.quiver(hor_mg_rotated[mask], vert_mg_rotated[mask], -hor_eig2_rotated[mask], -vert_eig2_rotated[mask],
              angles='xy', scale_units='xy', scale=1/arrow_length_multiplier, width=0.003, color=arrow_color)
    plot_state_labels(ax, axis_identifier[0].upper(), axis_identifier[1].upper())
    ax.set_ylim(-1.18, 1.18)
    ax.set_xlim(-1.18, 1.18)
    ax.plot(np.cos(x_circle), np.sin(x_circle), color='gray', lw=2)
    ax.set_aspect('equal')
    ax.set_xlabel(axis_identifier[0].upper())
    ax.set_ylabel(axis_identifier[1].upper())
    ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
    ax.set_title(r"Measurement back action")
    ax.set_axis_off()

    # Crude method to determine the measurement pole.
    Ybins, Zbins = hor_mg_rotated, vert_mg_rotated #np.meshgrid(horizontal_bin_centers, vertical_bin_centers)
    window = (-0.15, 0.15)
    msk = (Ybins >= window[0]) * (Ybins <= window[1]) * (Zbins >= window[0]) * (Zbins <= window[1])
    meas_vec_hor = np.mean(hor_eig2_rotated[msk])
    meas_vec_vert = np.mean(vert_eig2_rotated[msk])

    # print(f"Axis tilt w.r.t. vertical: {greek('theta')} = {(np.arctan2(meas_vec_hor, meas_vec_vert) * 180 / np.pi)%-180 : .1f} deg")

    b = meas_vec_vert / meas_vec_hor
    ax.plot(horizontal_bin_centers, b * horizontal_bin_centers, color=plt.cm.Reds(0.6), lw=3, alpha=0.5)
    if savepath is not None:
        idx = len(glob(os.path.join(savepath, '*.png')))
        fig.savefig(os.path.join(savepath, f"{idx:04d}_traj_stochastic_dynamics_{axis_identifier}.png"), dpi=200,
                    bbox_inches='tight', transparent=True, pad_inches=0.05)

    outside_sphere = (Ybins**2 + Zbins**2) >= 1.0
    back_action_magnitude = np.sqrt(hor_eig2_rotated**2 + vert_eig2_rotated**2)
    back_action_magnitude[outside_sphere] = np.nan
    masked_magnitude = np.ma.masked_where(np.isnan(back_action_magnitude), back_action_magnitude)
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    if do_colorplot:
        # Back-action color plot
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        pcm = plt.pcolormesh(Ybins, Zbins, masked_magnitude, vmin=color_min, vmax=color_max, cmap=current_cmap)
        colorbar(pcm)
        plot_state_labels(ax, axis_identifier[0].upper(), axis_identifier[1].upper())
        ax.set_ylim(-1.18, 1.18)
        ax.set_xlim(-1.18, 1.18)
        ax.plot(np.cos(x_circle), np.sin(x_circle), color='gray', lw=2)
        ax.plot(horizontal_bin_centers, b * horizontal_bin_centers, color=plt.cm.Reds(0.6), lw=3, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlabel(axis_identifier[0].upper())
        ax.set_ylabel(axis_identifier[1].upper())
        plt.yticks(np.arange(-1, 1.5, 0.5))
        plt.xticks(np.arange(-1, 1.5, 0.5))
        plt.title(r"Measurement back action")

        if savepath is not None:
            idx = len(glob(os.path.join(savepath, '*.png')))
            fig.savefig(os.path.join(savepath, f"{idx:04d}_traj_stochastic_magnitude_{axis_identifier}.png"), dpi=300,
                        bbox_inches='tight', transparent=True, pad_inches=0.05)

    return (np.arctan2(meas_vec_hor, meas_vec_vert) * 180 / np.pi) %-180

def fit_unbinned_trajectories(filepaths, seq_lengths, dt, tmins=None, tmaxs=None, derivative_order=1.0,
                              fit_guess=[0.5e6, 0.2e6, 0.2e6, 0.2e6], prep_state=None):

    def fit_func(x, *p):
        omega_x, omega_y, delta, gamma = p
        Lambda = 2 * np.pi * np.array([[-gamma, -delta, omega_y],
                                       [-delta, -gamma, -omega_x],
                                       [-omega_y, omega_x, 0.0]])

        # Lambda: 3x3, x: 3xn
        dxyz = np.dot(Lambda, x) * dt

        return dxyz.flatten()

    tmins = [0.0e-6] if tmins is None else tmins
    tmaxs = [10e-6] if tmaxs is None else tmaxs

    assert len(tmins) == len(tmaxs)

    avg_fit_parameters = np.zeros((len(tmins), 4))
    avg_fit_errors = np.zeros((len(tmins), 4))

    for k, tmin, tmax in zip(range(len(tmins)), tmins, tmaxs):
        fit_parameters = np.zeros((1, 4))
        fit_errors = np.zeros((1, 4))
        for n, seq_length in enumerate(seq_lengths): # Include all sequence lengths
            for jj, filepath in enumerate(filepaths): # Include multiple filepaths for different prep states
                with h5py.File(os.path.join(filepath, 'trajectories.h5'), 'r') as f:
                    group_name = f'predictions_{seq_length}'
                    available_keys = list(f[prep_state].keys()) if prep_state is not None else list(f.keys())
                    if group_name in available_keys:
                        if jj == 0:
                            xyz_array = f[prep_state].get(group_name)[:] if prep_state is not None else f.get(group_name)[:]
                        else:
                            if prep_state is not None:
                                xyz_array = np.vstack((xyz_array, f[prep_state].get(group_name)[:]))
                            else:
                                xyz_array = np.vstack((xyz_array, f.get(group_name)[:]))
                        times = f.get(f't')[:][:seq_length]

            Xf = xyz_array[..., 0]
            Yf = xyz_array[..., 1]
            Zf = xyz_array[..., 2]

            if derivative_order == 2:
                # Take centered derivatives. The second dimension is the time axis.
                dX = (Xf[:, 2:] - Xf[:, :-2]) / 2.
                dY = (Yf[:, 2:] - Yf[:, :-2]) / 2.
                dZ = (Zf[:, 2:] - Zf[:, :-2]) / 2.

                # Make sure Xf..Zf have the same dimension as dX...dZ
                Xf = Xf[:, 1:-1]
                Yf = Yf[:, 1:-1]
                Zf = Zf[:, 1:-1]
            elif derivative_order == 1:
                # Take forward derivatives. The second dimension is the time axis.
                dX = Xf[:, 1:] - Xf[:, :-1]
                dY = Yf[:, 1:] - Yf[:, :-1]
                dZ = Zf[:, 1:] - Zf[:, :-1]

                # Make sure Xf..Zf have the same dimension as dX...dZ
                Xf = Xf[:, :-1]
                Yf = Yf[:, :-1]
                Zf = Zf[:, :-1]

            time_mask_1d = (times[1:-1] >= tmin) * (times[1:-1] <= tmax) if derivative_order==2 else (times[:-1] >= tmin) * (times[:-1] <= tmax)
            time_mask_2d = np.repeat(np.reshape(time_mask_1d, (1, len(time_mask_1d))), np.shape(Xf)[0], axis=0)

            assert np.all(np.shape(dX) == np.shape(Xf))

            if np.sum(time_mask_2d) > 0:
                num_samples = len(Yf[time_mask_2d].flatten())
                xdata = np.vstack((np.reshape(Xf[time_mask_2d].flatten(), (1, num_samples)),
                                   np.reshape(Yf[time_mask_2d].flatten(), (1, num_samples))))
                xdata = np.vstack((xdata,
                                   np.reshape(Zf[time_mask_2d].flatten(), (1, num_samples))))

                ydata = np.vstack((np.reshape(dX[time_mask_2d].flatten(), (1, num_samples)),
                                   np.reshape(dY[time_mask_2d].flatten(), (1, num_samples))))
                ydata = np.vstack((ydata,
                                   np.reshape(dZ[time_mask_2d].flatten(), (1, num_samples))))

                fitpars, fcov = optimize.curve_fit(fit_func, xdata, ydata.flatten(), p0=fit_guess)
                fiterrs = np.sqrt(np.diag(fcov))

                if n == 0:
                    fit_parameters[0, :] = fitpars
                    fit_errors[0, :] = fiterrs
                else:
                    fit_parameters = np.vstack((fit_parameters, np.reshape(fitpars, (1, 4))))
                    fit_errors = np.vstack((fit_errors, np.reshape(fiterrs, (1, 4))))

        avg_fit_errors[k, :] = np.mean(fit_errors, axis=0)
        avg_fit_parameters[k, :] = np.mean(fit_parameters, axis=0)

    return avg_fit_parameters, avg_fit_errors

def rotate_quiver_map(horizontal_bin_centers, vertical_bin_centers, horizontal_arrows, vertical_arrows, theta):
    """
    Rotates a quiver map and can be used to rotate the measurement axis in the plot_stochastic function.
    :param horizontal_bin_centers: 1d array, bins in the horizontal direction
    :param vertical_bin_centers: 1d array, bins in the vertical direction
    :param horizontal_arrows: 2d array of shape (len(vertical_bin_centers), len(horizontal_bin_centers))
    :param vertical_arrows: vertical component of the arrows, same shape as horizontal_arrows
    :param theta: rotation angle in radians
    :return: rotated meshgrid horizontal, vertical coord, rotated vector horizontal, vertical arrow components.
    """
    Ymg, Zmg = np.meshgrid(horizontal_bin_centers, vertical_bin_centers)
    Ymg_rotated = np.zeros(np.shape(Ymg))
    Zmg_rotated = np.zeros(np.shape(Zmg))
    Yvec_rotated = np.zeros(np.shape(Ymg))
    Zvec_rotated = np.zeros(np.shape(Zmg))

    for k in range(np.shape(Ymg)[0]):
        for l in range(np.shape(Ymg)[1]):
            Ymg_rotated[k, l] = Ymg[k, l] * np.cos(theta) - Zmg[k, l] * np.sin(theta)
            Zmg_rotated[k, l] = Ymg[k, l] * np.sin(theta) + Zmg[k, l] * np.cos(theta)
            Yvec_rotated[k, l] = horizontal_arrows[k, l] * np.cos(theta) - vertical_arrows[k, l] * np.sin(theta)
            Zvec_rotated[k, l] = horizontal_arrows[k, l] * np.sin(theta) + vertical_arrows[k, l] * np.cos(theta)

    return Ymg_rotated, Zmg_rotated, Yvec_rotated, Zvec_rotated