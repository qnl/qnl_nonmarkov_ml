import numpy as np
from scipy.optimize import curve_fit
from tensorflow.keras import backend as K
import tensorflow as tf
from utils import x_color, y_color, z_color
from matplotlib import pyplot as plt

cmap = plt.get_cmap('Accent')
zero_color, one_color, two_color = [cmap.colors[z] for z in range(3)]

def get_trajectories_within_window(predictions, target_value, RO_results, n_levels, pass_window=0.025, verbose=True):
    """
    Finds trajectories in `predictions` within a certain window around a `target_value`, then returns the averaged
    RO_results. This is used for verification of the predictions from the RNN.
    :param predictions: array of predictions from the RNN.
    :param target_value: Find predictions around this target_value. If predictions are x, y, z values, so must target_value.
    :param RO_results: Array of strong readout results containing 0 and 1 (possibly 2 for qutrit data)
    :param n_levels: 2 for qubit data, 3 for qutrit data
    :param pass_window: Defines the tolerance around the target value. A larger pass_window will increase the number of
                        trajectories to average RO results more accurately, but limit the total number of target_values.
    :param verbose: True for printing statements, False for silent operation.
    :return: Indices for which predictions hit the target value, Mean prediction for these indicies, and RO result array
    """
    if n_levels == 2:
        # Select traces where the final index is within Z ± the pass_window
        passed_idcs = np.where(np.abs(predictions - target_value) < pass_window)[0]
        N_verification_trajs = np.shape(predictions)[0]

        if verbose:
            print(f"Post-selecting trajectories with target = {target_value:.3f} ± {pass_window:.3f}")
            print(
                f"{len(passed_idcs)} trajectories left after post-selection ({len(passed_idcs) / N_verification_trajs * 100:.1f}% pass rate)")

        verification_strong_RO = RO_results[passed_idcs]
        avg_verification_value = 1 - 2 * np.mean(verification_strong_RO)
        passed_RO_results = RO_results[passed_idcs]
    elif n_levels == 3:
        for ro_result in range(3):
            # Select traces where the final index is within target Pg/Pe/Pf ± the pass_window
            passed_idcs = np.where(np.abs(predictions[:, ro_result] - target_value) < pass_window)[0]
            N_verification_trajs = np.shape(predictions)[0]
            verification_strong_RO = RO_results[passed_idcs]
            if ro_result == 0:
                avg_P0 = np.sum(verification_strong_RO[:, 0]) / np.shape(verification_strong_RO)[0]
                passed_idcs0 = passed_idcs
            elif ro_result == 1:
                avg_P1 = np.sum(verification_strong_RO[:, 1]) / np.shape(verification_strong_RO)[0]
                passed_idcs1 = passed_idcs
            elif ro_result == 2:
                avg_P2 = np.sum(verification_strong_RO[:, 2]) / np.shape(verification_strong_RO)[0]
                passed_idcs2 = passed_idcs
        avg_verification_value = [avg_P0, avg_P1, avg_P2]
        passed_idcs = [passed_idcs0, passed_idcs1, passed_idcs2]
        passed_RO_results = [RO_results[passed_idcs0], RO_results[passed_idcs1], RO_results[passed_idcs2]]

    return passed_idcs, avg_verification_value, passed_RO_results

def get_error(strong_ro_results, readout_value=1):
    """
    Returns the error bar on the mean of on an array of strong readout results (e.g. [0, 1, 1, 0, 1, 1, ...])
    :param strong_ro_results: array of binary readout results
    :param readout_value: optional: set to 2 for qutrit data if you're interested in the error bar on Pf
    :return: Error bar on the mean probability, assuming Bernouilli distribution.
    """
    N = len(strong_ro_results)
    p = np.sum(strong_ro_results == readout_value) / N
    return np.sqrt(p * (1-p) / N)

def get_xyz(probabilities):
    """
    Convert a 3d array of probabilities to x, y and z values. Only works for qubit data.
    :param probabilities: 3d array of shape (batch size, time steps, 6)
    :return: 3d array of shape (batch size, time steps, 3)
    """
    return 2 * probabilities[:, :, ::2] - 1

def pairwise_softmax(y_pred, n_levels):
    """
    When training on labels from different tomography axes, each tomography axis has 2 results that sum to 1.0
    This function takes y_pred of the form [L0x, L1x, L0y, L1y, L0z, L1z], where Lix,y,z are the predicted logits, and
    converts the logits to probabilities in a pairwise fashion such that probabilities = [P0x, P1x, P0y, P1y, P0z, P1z]
    and P0i + P1i = 1 for i = x, y, z.
    Note: if n_levels = 3, we assume to only have Z measurements. In that case we can apply a regular softmax.
    :param y_pred: Predicted logits from the RNN. 3D array with shape (batch size, time steps, 6)
    :param n_levels: 2 for a qubit, 3 for a qutrit.
    :return: Array of probabilities
    """
    # In the case of qubits, we should do a pairwise softmax.
    if n_levels == 2:
        probabilities = np.zeros(np.shape(y_pred))
        batch_size, seq_length, _ = np.shape(y_pred)
        for k in [0, 1, 2]:  # px, py, pz for qubits
            numerator = np.exp(y_pred[:, :, 2 * k:2 * k + 2])
            denominator = np.expand_dims(np.sum(np.exp(y_pred[:, :, 2 * k:2 * k + 2]), axis=2), axis=2)
            probabilities[:, :, 2 * k:2 * k + 2] = numerator / denominator
    elif n_levels == 3:
        # For a qutrit we can use the standard Keras activation function, because for a single measurement axes,
        # the qutrit probabilities should add up to 1.0.
        probabilities = tf.keras.activations.softmax(K.constant(y_pred)).numpy()

    return probabilities

def _simple_line(x, *p):
    """
    Linear relation y = ax + b, used for fitting
    :param x: array
    :param p: parameter list [a, b]
    :return: y = ax + b
    """
    slope, offset = p
    return slope * x + offset

def weighted_line_fit(xdata, ydata, yerr, guess_slope, guess_offset, no_weights=False):
    """
    Simple linear regression with the option to specify weights in `yerr`
    :param xdata: array
    :param ydata: array, must be the same size as xdata
    :param yerr: array of the same size as xdata and ydata
    :param guess_slope: guess for the slope (a in y = ax + b)
    :param guess_offset: guess for the offset (b in y = ax + b)
    :param no_weights: bool, set to True to perform a simple linear regression without weights.
    :return: Optimal parameters, Standard deviation of the parameters
    """
    if no_weights:
        popt, pcov = curve_fit(_simple_line, xdata, ydata, p0=[guess_slope, guess_offset])
    else:
        try:
            popt, pcov = curve_fit(_simple_line, xdata, ydata, p0=[guess_slope, guess_offset],
                                   sigma=yerr, absolute_sigma=True, check_finite=True,
                                   bounds=(-np.inf, np.inf), method=None, jac=None)
        except RuntimeError:
            popt, pcov = curve_fit(_simple_line, xdata, ydata, p0=[guess_slope, guess_offset])

    perr = np.sqrt(np.diag(pcov))

    return popt, perr

def plot_qubit_verification(predicted_labels, verification_labels):
    """
    Plots the predicted values of the RNN against the averaged readout results. This groups all readout results
    irrespective of time.
    :param predicted_labels: array of size (n_reps, n_timesteps, 6), predicted probabilities by the RNN
    :param verification_labels: array of size (n_reps, n_timesteps, 6), ground truth probabilities.
    :return: Figure handle for saving.
    """
    xyz_pred = get_xyz(predicted_labels)
    measurement_axis = -1 * np.ones((np.shape(verification_labels)[0],
                                     np.shape(verification_labels)[1]))

    for k in range(np.shape(verification_labels)[0]):
        for ts in range(np.shape(verification_labels)[1]):
            if verification_labels[k, ts, 0] != -1:
                measurement_axis[k, ts] = 0
            elif verification_labels[k, ts, 2] != -1:
                measurement_axis[k, ts] = 1
            elif verification_labels[k, ts, 4] != -1:
                measurement_axis[k, ts] = 2

    x_measurements = np.where(measurement_axis == 0)
    y_measurements = np.where(measurement_axis == 1)
    z_measurements = np.where(measurement_axis == 2)

    # Readout results associated with trajectories that have predictions of epsilon around the target are averaged.
    # Note that the actual window is 2 * epsilon.
    epsilon = 0.02

    x_axis_idx, y_axis_idx, z_axis_idx = 1, 3, 5
    x_pred, y_pred, z_pred = list(), list(), list()
    x_pred_trajs, y_pred_trajs, z_pred_trajs = list(), list(), list()
    x_errs, y_errs, z_errs = list(), list(), list()

    x_targets = np.arange(-1 + epsilon, 1 + epsilon, 2 * epsilon)
    y_targets = np.arange(-1 + epsilon, 1 + epsilon, 2 * epsilon)
    z_targets = np.arange(-1 + epsilon, 1 + epsilon, 2 * epsilon)

    x_RO = verification_labels[x_measurements[0], x_measurements[1], x_axis_idx]
    y_RO = verification_labels[y_measurements[0], y_measurements[1], y_axis_idx]
    z_RO = verification_labels[z_measurements[0], z_measurements[1], z_axis_idx]

    for tx, ty, tz in zip(x_targets, y_targets, z_targets):
        passed_idcs, avg_ver, ro_res = get_trajectories_within_window(xyz_pred[x_measurements[0], x_measurements[1], 0],
                                                                      tx, x_RO, n_levels=2, pass_window=epsilon,
                                                                      verbose=False)

        # Note that the error is calculated on the probabilities, and since X = 1 - 2Px, we multiply the output of
        # get_error by 2.
        x_pred.append(avg_ver)
        x_errs.append(2.0 * get_error(ro_res))
        x_pred_trajs.append(len(passed_idcs))

        passed_idcs, avg_ver, ro_res = get_trajectories_within_window(xyz_pred[y_measurements[0], y_measurements[1], 1],
                                                                      ty, y_RO, n_levels=2, pass_window=epsilon,
                                                                      verbose=False)
        y_pred.append(avg_ver)
        y_errs.append(2.0 * get_error(ro_res))
        y_pred_trajs.append(len(passed_idcs))

        passed_idcs, avg_ver, ro_res = get_trajectories_within_window(xyz_pred[z_measurements[0], z_measurements[1], 2],
                                                                      tz, z_RO, n_levels=2, pass_window=epsilon,
                                                                      verbose=False)
        z_pred.append(avg_ver)
        z_errs.append(2.0 * get_error(ro_res))
        z_pred_trajs.append(len(passed_idcs))

    x_targets = np.array(x_targets)
    x_pred = np.array(x_pred)
    x_errs = np.array(x_errs)

    y_targets = np.array(y_targets)
    y_pred = np.array(y_pred)
    y_errs = np.array(y_errs)

    z_targets = np.array(z_targets)
    z_pred = np.array(z_pred)
    z_errs = np.array(z_errs)

    mask = np.array(x_pred_trajs) >= 20
    fr, ferr = weighted_line_fit(x_targets[mask], x_pred[mask], x_errs[mask], 1.0, 0.0)

    fig = plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x_targets[mask], x_pred[mask], 'o', color=x_color)
    plt.errorbar(x_targets[mask], x_pred[mask], yerr=x_errs[mask], color=x_color, fmt='.')
    plt.plot(x_targets[mask], _simple_line(x_targets[mask], *fr), '-k',
             label=r"$\varepsilon_x$ = %.2f $\pm$ %.2f" % (np.abs(fr[0]-1), ferr[0]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot([-1, 1], [-1, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $\langle X \rangle$ given by NN")
    plt.ylabel(r"$\langle X \rangle$ verified by strong readout")
    plt.yticks([-1, -0.5, 0, 0.5, 1.0])
    plt.legend(loc=0, frameon=False)
    plt.gca().set_aspect('equal')

    mask = np.array(y_pred_trajs) >= 20
    fr, ferr = weighted_line_fit(y_targets[mask], y_pred[mask], y_errs[mask], 1.0, 0.0)

    plt.subplot(1, 3, 2)
    plt.plot(y_targets[mask], y_pred[mask], 'o', color=y_color)
    plt.errorbar(y_targets[mask], y_pred[mask], yerr=y_errs[mask], color=y_color, fmt='.')
    plt.plot(y_targets[mask], _simple_line(y_targets[mask], *fr), '-k',
             label=r"$\varepsilon_y$ = %.2f $\pm$ %.2f"%(np.abs(fr[0]-1), ferr[0]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot([-1, 1], [-1, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $\langle Y \rangle$ given by NN")
    plt.ylabel(r"$\langle Y \rangle$ verified by strong readout")
    plt.yticks([-1, -0.5, 0, 0.5, 1.0])
    plt.legend(loc=0, frameon=False)
    # plt.title(f"Verification for t = {Tm[timesteps[0]]*1e6} {chr(956)}s")
    plt.gca().set_aspect('equal')

    mask = np.array(z_pred_trajs) >= 20
    fr, ferr = weighted_line_fit(z_targets[mask], z_pred[mask], z_errs[mask], 1.0, 0.0)

    plt.subplot(1, 3, 3)
    plt.plot(z_targets[mask], z_pred[mask], 'o', color=z_color)
    plt.errorbar(z_targets[mask], z_pred[mask], yerr=z_errs[mask], color=z_color, fmt='.')
    plt.plot(z_targets[mask], _simple_line(z_targets[mask], *fr), '-k',
             label=r"$\varepsilon_z$ = %.2f $\pm$ %.2f"%(np.abs(fr[0]-1), ferr[0]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot([-1, 1], [-1, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $\langle Z \rangle$ given by NN")
    plt.ylabel(r"$\langle Z \rangle$ verified by strong readout")
    plt.yticks([-1, -0.5, 0, 0.5, 1.0])
    plt.legend(loc=0, frameon=False)
    plt.gca().set_aspect('equal')

    plt.tight_layout()

    return fig

def plot_qutrit_verification(predicted_labels, verification_labels):
    """
    Plots the predicted populations by the RNN vs. averaged readout results (our estimate of the ground truth) to
    assess the prediction accuracy of the RNN.
    :param predicted_labels: array of size (n_reps, n_timesteps, 3), predicted probabilities by the RNN.
    :param verification_labels: array of size (n_reps, n_timesteps, 3), ground truth probabilities for Pg, Pe and Pf.
    :return: figure handle
    """
    # This is the window size. We will average readout results of trajectories that fall within gef_target +/- epsilon
    # just before tomography was performed.
    epsilon = 0.02

    Pg_pred, Pe_pred, Pf_pred = list(), list(), list()
    Pg_pred_trajs, Pe_pred_trajs, Pf_pred_trajs = list(), list(), list()
    Pg_errs, Pe_errs, Pf_errs = list(), list(), list()

    gef_targets = np.arange(0 + epsilon, 1 + epsilon, epsilon)

    # Select the strong readout points
    strong_ro_selection = np.where(verification_labels != -1)
    print(predicted_labels[strong_ro_selection[0], strong_ro_selection[1], :].shape)

    for target in gef_targets:
        passed_idcs, avg_probs, ro_res = get_trajectories_within_window(predicted_labels[strong_ro_selection[0][::3], strong_ro_selection[1][::3], :],
                                                                        target,
                                                                        verification_labels[strong_ro_selection[0][::3], strong_ro_selection[1][::3], :],
                                                                        n_levels=3, pass_window=epsilon, verbose=False)
        # Convert back to probability
        Pg_pred.append(avg_probs[0])
        Pe_pred.append(avg_probs[1])
        Pf_pred.append(avg_probs[2])

        if len(ro_res[0]) > 0:
            Pg_errs.append(get_error(ro_res[0][:, 0], readout_value=0))
        else:
            # This can happen if there's no trajectories that fell within the target window.
            Pg_errs.append(0.0)
        if len(ro_res[1]) > 0:
            Pe_errs.append(get_error(ro_res[1][:, 1], readout_value=1))
        else:
            Pe_errs.append(0.0)
        if len(ro_res[2]) > 0:
            Pf_errs.append(get_error(ro_res[2][:, 2], readout_value=2))
        else:
            Pf_errs.append(0.0)

        Pg_pred_trajs.append(len(passed_idcs[0]))
        Pe_pred_trajs.append(len(passed_idcs[1]))
        Pf_pred_trajs.append(len(passed_idcs[2]))

    gef_targets = np.array(gef_targets)
    Pg_pred = np.array(Pg_pred)
    Pg_errs = np.array(Pg_errs)

    Pe_pred = np.array(Pe_pred)
    Pe_errs = np.array(Pe_errs)

    Pf_pred = np.array(Pf_pred)
    Pf_errs = np.array(Pf_errs)

    mask = np.logical_not(np.isnan(Pg_pred)) * (np.array(Pg_pred_trajs) >= 20)
    fr, ferr = weighted_line_fit(gef_targets[mask], Pg_pred[mask], Pg_errs[mask], 1.0, 0.0)

    fig = plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1)
    plt.plot(gef_targets[mask], Pg_pred[mask], 'o', color=zero_color)
    plt.errorbar(gef_targets[mask], Pg_pred[mask], yerr=Pg_errs[mask], color=zero_color, fmt='.')
    plt.plot(gef_targets[mask], _simple_line(gef_targets[mask], *fr), '-k',
             label=r"$\varepsilon_g$ = %.2f $\pm$ %.2f" % (np.abs(fr[0]-1), ferr[0]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $P_g$ given by NN")
    plt.ylabel(r"$P_g$ verified by strong readout")
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.legend(loc=0, frameon=False)
    plt.gca().set_aspect('equal')

    mask = np.logical_not(np.isnan(Pe_pred)) * (np.array(Pe_pred_trajs) >= 20)
    fr, ferr = weighted_line_fit(gef_targets[mask], Pe_pred[mask], Pe_errs[mask], 1.0, 0.0)

    plt.subplot(1, 3, 2)
    plt.plot(gef_targets[mask], Pe_pred[mask], 'o', color=one_color)
    plt.errorbar(gef_targets[mask], Pe_pred[mask], yerr=Pe_errs[mask], color=one_color, fmt='.')
    plt.plot(gef_targets[mask], _simple_line(gef_targets[mask], *fr), '-k',
             label=r"$\varepsilon_e$ = %.2f $\pm$ %.2f" % (np.abs(fr[0] - 1), ferr[0]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $P_e$ given by NN")
    plt.ylabel(r"$P_e$ verified by strong readout")
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.legend(loc=0, frameon=False)
    plt.gca().set_aspect('equal')

    mask = np.logical_not(np.isnan(Pf_pred)) * (np.array(Pf_pred_trajs) >= 20)
    fr, ferr = weighted_line_fit(gef_targets[mask], Pf_pred[mask], Pf_errs[mask], 1.0, 0.0)

    plt.subplot(1, 3, 3)
    plt.plot(gef_targets[mask], Pf_pred[mask], 'o', color=two_color)
    plt.errorbar(gef_targets[mask], Pf_pred[mask], yerr=Pf_errs[mask], color=two_color, fmt='.')
    plt.plot(gef_targets[mask], _simple_line(gef_targets[mask], *fr), '-k',
             label=r"$\varepsilon_f$ = %.2f $\pm$ %.2f" % (np.abs(fr[0] - 1), ferr[0]))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '-', color='gray', lw=2, alpha=0.5)
    plt.xlabel(r"Target $P_f$ given by NN")
    plt.ylabel(r"$P_f$ verified by strong readout")
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.legend(loc=0, frameon=False)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    return fig