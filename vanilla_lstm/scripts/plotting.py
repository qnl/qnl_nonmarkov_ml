from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os, matplotlib, scipy, sys
sys.path.append(r"/home/qnl/Git-repositories")
from qnl_trajectories.utils import greek

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
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

def plot_drho(xyz_array, x_bins, y_bins, z_bins, horizontal_axis="Y", vertical_axis="Z", other_coordinate=0.0):
    Xf = xyz_array[..., 0]
    Yf = xyz_array[..., 1]
    Zf = xyz_array[..., 2]

    # Take forward derivatives. The second dimension is the time axis.
    dX = Xf[:, 1:] - Xf[:, :-1]
    dY = Yf[:, 1:] - Yf[:, :-1]
    dZ = Zf[:, 1:] - Zf[:, :-1]

    # These are the bin numbers for the X-coordinate
    x_idcs = np.digitize(Xf, x_bins)
    y_idcs = np.digitize(Yf, y_bins)
    z_idcs = np.digitize(Zf, z_bins)

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

    other_idx = np.digitize(other_coordinate, other_bins)

    mean_binned_hor = np.zeros((len(vertical_bins), len(horizontal_bins)))
    std_binned_hor = np.zeros((len(vertical_bins), len(horizontal_bins)))
    mean_binned_vert = np.zeros((len(vertical_bins), len(horizontal_bins)))
    std_binned_vert = np.zeros((len(vertical_bins), len(horizontal_bins)))

    eig1 = np.zeros((len(vertical_bins), len(horizontal_bins), 2))
    eig2 = np.zeros((len(vertical_bins), len(horizontal_bins), 2))
    ev1s = np.zeros((len(vertical_bins), len(horizontal_bins)))
    ev2s = np.zeros((len(vertical_bins), len(horizontal_bins)))
    k = 0

    for iter_1 in range(len(horizontal_bins)):
        for iter_2 in range(len(vertical_bins)):
            mask = (horizontal_idcs == iter_1) * (vertical_idcs == iter_2) * (other_idcs == other_idx)
            # Don't mask on the last timestep, we're looking at derivative arrays
            mask = mask[:, :-1]

            # At least 10 trajectories must make up a bin
            if np.sum(mask) > 10:
                # Captures the coherent rotation part of the master eq.
                mean_binned_hor[iter_2, iter_1] += np.mean(dhor[mask])
                mean_binned_vert[iter_2, iter_1] += np.mean(dvert[mask])

                # Processing on the stochastic part of the master eq.
                std_binned_hor[iter_2, iter_1] += np.std(dhor[mask])
                std_binned_vert[iter_2, iter_1] += np.std(dvert[mask])

                # Calculate the 2x2 covariance matrix
                cov = np.cov(dhor[mask], dvert[mask])
                w, v = np.linalg.eig(cov)
                ev_order = np.argsort(w)
                lambda1, lambda2 = w[ev_order]
                ev1, ev2 = v[:, ev_order[0]], v[:, ev_order[1]]
                ev1s[iter_2, iter_1] = lambda1
                ev2s[iter_2, iter_1] = lambda2

                scale = 3.0
                ell_radius_x = scale * np.sqrt(lambda1)
                ell_radius_y = scale * np.sqrt(lambda2)

                if k in [0, 25, 50]:
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.scatter(dhor[mask], dvert[mask], color='k')
                    ax.arrow(np.mean(dhor[mask]), np.mean(dvert[mask]), ell_radius_x * ev1[0],
                             ell_radius_x * ev1[1], color='red')
                    ax.arrow(np.mean(dhor[mask]), np.mean(dvert[mask]), ell_radius_y * ev2[0],
                             ell_radius_y * ev2[1], color='red')
                    ax.set_aspect('equal')
                    ax.set_ylabel(f"d{vertical_axis}")
                    ax.set_xlabel(f"d{horizontal_axis}")
                    ax.set_title(f"({horizontal_axis}, {vertical_axis}) = ({horizontal_bins[iter_1]:.2f}, {vertical_bins[iter_2]:.2f})")

                    confidence_ellipse(dhor[mask], dvert[mask], ax, n_std=3.0,
                                       facecolor='red', alpha=0.2, edgecolor='none')

                k += 1

                # Eigenvectors of the covariance matrix weighted by their eigenvalues.
                eig1[iter_2, iter_1] += 3.0 * np.sqrt(lambda1) * ev1
                eig2[iter_2, iter_1] += 3.0 * np.sqrt(lambda2) * ev2

    horizontal_bin_centers = (horizontal_bins[1:] + horizontal_bins[:-1]) / 2.
    vertical_bin_centers = (vertical_bins[1:] + vertical_bins[:-1]) / 2.
    return horizontal_bin_centers, vertical_bin_centers, mean_binned_hor, mean_binned_vert, eig1, eig2


def plot_and_fit_hamiltonian(horizontal_bin_centers, vertical_bin_centers, mean_binned_hor, mean_binned_vert, savepath,
                             axis_identifier="__", fit=True):

    def dr_dt(x, y, z, gamma, omega):
        Lambda = np.array([[-gamma, 0, 0], [0, -gamma, -omega], [0, omega, 0]])
        r = np.reshape(np.array([x, y, z]), (3, 1))
        return np.dot(Lambda, r)

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

    # Hamiltonian part of the master equation
    Ymg, Zmg = np.meshgrid(horizontal_bin_centers, vertical_bin_centers)
    ydata = np.zeros((np.shape(Ymg)[0], np.shape(Ymg)[1], 2))
    ydata[..., 0] = mean_binned_hor[1:, 1:]
    ydata[..., 1] = mean_binned_vert[1:, 1:]
    mask = ((ydata[..., 0] != 0) + (ydata[..., 1] != 0)) * (Ymg ** 2 + Zmg ** 2 < 1)

    if fit:
        fr, fcov = scipy.optimize.curve_fit(fitfunc, (Ymg[mask], Zmg[mask]), ydata[mask, :].flatten(), p0=[0.2, 1])
        ferr = np.sqrt(np.diag(fcov))

        omega_over_gamma = fr[1] / fr[0]
        print(f"{greek('Omega')}/{greek('Gamma')} = {omega_over_gamma:.2f}")

        y_fit = fitfunc((Ymg.flatten(), Zmg.flatten()), *fr)
        fitted_dys = np.reshape(y_fit[::2], np.shape(Ymg))
        fitted_dzs = np.reshape(y_fit[1::2], np.shape(Ymg))

    x_circle = np.linspace(-np.pi, np.pi, 1000)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    if fit:
        ax.quiver(Ymg[mask], Zmg[mask], fitted_dys[mask], fitted_dzs[mask], color='red',
                  label=r'fit: $\Omega/\Gamma = $%.2f' % omega_over_gamma)
    ax.quiver(horizontal_bin_centers, vertical_bin_centers, mean_binned_hor[1:, 1:], mean_binned_vert[1:, 1:],
              color='black', label='NN pred.')
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    ax.plot(np.cos(x_circle), np.sin(x_circle), color=plt.cm.Reds(0.6), lw=2)
    ax.set_aspect('equal')
    ax.set_xlabel(axis_identifier[0].upper())
    ax.set_ylabel(axis_identifier[1].upper())
    plt.yticks(np.arange(-1, 1, 0.5))
    plt.xticks(np.arange(-1, 1, 0.5))
    plt.title(r"$\langle \mathrm{d}\rho\rangle$")
    plt.legend(loc=0, frameon=False)
    fig.savefig(os.path.join(savepath, f"001_traj_hamiltonian_dynamics_{axis_identifier}.png"), dpi=200,
                bbox_inches='tight')

def plot_quiver(horizontal_bin_centers, vertical_bin_centers, eig1, eig2, savepath, axis_identifier=""):
    # Stochastic part of the master equation
    # A smaller arrow_scale makes the arrows bigger
    arrow_scale = 1 * 4
    x_circle = np.linspace(-np.pi, np.pi, 1000)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.quiver(horizontal_bin_centers, vertical_bin_centers, eig1[1:, 1:, 0], eig1[1:, 1:, 1], scale=arrow_scale)
    ax.quiver(horizontal_bin_centers, vertical_bin_centers, -eig1[1:, 1:, 0], -eig1[1:, 1:, 1], scale=arrow_scale)
    ax.quiver(horizontal_bin_centers, vertical_bin_centers, eig2[1:, 1:, 0], eig2[1:, 1:, 1], scale=arrow_scale)
    ax.quiver(horizontal_bin_centers, vertical_bin_centers, -eig2[1:, 1:, 0], -eig2[1:, 1:, 1], scale=arrow_scale)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    ax.plot(np.cos(x_circle), np.sin(x_circle), color=plt.cm.Blues(0.6), lw=2)
    ax.set_aspect('equal')
    ax.set_xlabel(axis_identifier[0].upper())
    ax.set_ylabel(axis_identifier[1].upper())
    plt.yticks(np.arange(-1, 1.5, 0.5))
    plt.xticks(np.arange(-1, 1.5, 0.5))
    plt.title(r"Measurement back action")

    # Crude method to determine the measurement pole.
    Ybins, Zbins = np.meshgrid(horizontal_bin_centers, vertical_bin_centers)
    window = (-0.2, 0.2)
    msk = (Ybins >= window[0]) * (Ybins <= window[1]) * (Zbins >= window[0]) * (Zbins <= window[1])
    meas_vec_hor = np.mean(eig2[1:, 1:, 0][msk])
    meas_vec_vert = np.mean(eig2[1:, 1:, 1][msk])

    print(meas_vec_hor)
    print(meas_vec_vert)
    print("Measurement axis angle:", np.arctan2(meas_vec_vert, meas_vec_hor) * 180 / np.pi)

    b = meas_vec_vert / meas_vec_hor
    plt.plot(horizontal_bin_centers, b * horizontal_bin_centers, color=plt.cm.Reds(0.6), lw=3, alpha=0.5)
    fig.savefig(os.path.join(savepath, f"001_traj_stochastic_dynamics_{axis_identifier}.png"), dpi=200, bbox_inches='tight')

    outside_sphere = (Ybins**2 + Zbins**2) >= 1.0
    back_action_magnitude = np.sqrt(eig2[1:, 1:, 0]**2 + eig2[1:, 1:, 1]**2)
    back_action_magnitude[outside_sphere] = np.nan
    masked_magnitude = np.ma.masked_where(np.isnan(back_action_magnitude), back_action_magnitude)
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    # Back-action color plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plt.pcolormesh(horizontal_bin_centers, vertical_bin_centers, masked_magnitude, vmin=0, vmax=0.22)
    plt.colorbar()
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    ax.plot(np.cos(x_circle), np.sin(x_circle), color=plt.cm.Blues(0.6), lw=2)
    plt.plot(horizontal_bin_centers, b * horizontal_bin_centers, color=plt.cm.Reds(0.6), lw=3, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel(axis_identifier[0].upper())
    ax.set_ylabel(axis_identifier[1].upper())
    plt.yticks(np.arange(-1, 1.5, 0.5))
    plt.xticks(np.arange(-1, 1.5, 0.5))
    plt.title(r"Measurement back action")

    fig.savefig(os.path.join(savepath, f"001_traj_stochastic_magnitude_{axis_identifier}.png"), dpi=200, bbox_inches='tight')