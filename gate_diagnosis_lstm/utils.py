import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import h5py
import yaml
import os
import copy

x_color = plt.cm.Reds(0.6)
y_color = plt.cm.Blues(0.6)
z_color = plt.cm.Greys(0.6)

save_options = {"dpi" : 200,
                "bbox_inches" : "tight",
                "pad_inches" : 0.05,
                "transparent" : True}

qubit_prep_dict = {"+X" : {"prep_x" : [1.0, 0.0],
                           "prep_y" : [0.5, 0.5],
                           "prep_z" : [0.5, 0.5]},
                   "-X" : {"prep_x" : [0.0, 1.0],
                           "prep_y" : [0.5, 0.5],
                           "prep_z" : [0.5, 0.5]},
                   "+Y" : {"prep_x" : [0.5, 0.5],
                           "prep_y" : [1.0, 0.0],
                           "prep_z" : [0.5, 0.5]},
                   "-Y" : {"prep_x" : [0.5, 0.5],
                           "prep_y" : [0.0, 1.0],
                           "prep_z" : [0.5, 0.5]},
                   "+Z" : {"prep_x" : [0.5, 0.5],
                           "prep_y" : [0.5, 0.5],
                           "prep_z" : [1.0, 0.0]},
                   "-Z": {"prep_x": [0.5, 0.5],
                          "prep_y": [0.5, 0.5],
                          "prep_z": [0.0, 1.0]},
                   "g": {"prep_x": [0.5, 0.5],
                         "prep_y": [0.5, 0.5],
                         "prep_z": [1.0, 0.0]},
                   "e": {"prep_x": [0.5, 0.5],
                         "prep_y": [0.5, 0.5],
                         "prep_z": [0.0, 1.0]}}

qutrit_prep_dict = {"+X": {"prep_z": [0.5, 0.5, 0.0]},
                    "-X": {"prep_z": [0.5, 0.5, 0.0]},
                    "+Y": {"prep_z": [0.5, 0.5, 0.0]},
                    "-Y": {"prep_z": [0.5, 0.5, 0.0]},
                    "+Z": {"prep_z": [1.0, 0.0, 0.0]},
                    "-Z": {"prep_z": [0.0, 1.0, 0.0]},
                    "f": {"prep_z": [0.0, 0.0, 1.0]}}


def load_settings(yaml_path, relative=False):

    if relative:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path_ = os.path.join(dir_path, yaml_path)
    else:
        yaml_path_ = yaml_path

    with open(yaml_path_) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def save_settings(yaml_path, settings_dict):
    with open(yaml_path, 'w') as file:
        yaml.dump(settings_dict, file, sort_keys=True)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def load_repackaged_data(filename, multi_prep_state=False, verbose_xyzs=False):
    """
    Load the h5 file produced by filtering.repackage_raw_data
    :param filename: string, HDF5 Filename including the file extension
    :return: dictionary, contents of the h5 file for further processing.
    """
    with h5py.File(filename, "r") as f:
        # Here are all the keys available
        print("Available keys in this datafile:", list(f.keys()))

        return_dict = {}
        if multi_prep_state:
            # Support for multiple prep states in the data file. Return_dict will have a different structure
            prep_states = list(f.keys())

            for prep_state in prep_states:
                return_dict[prep_state] = {}
                meas_axes = list(f[prep_state].keys())
                xyzs = [x[-1:] for x in meas_axes]

                # get data for different measurement axes labels
                # if verbose_xyzs is False, will look for 'meas_X', 'meas_Y', ...
                # if it's true, it will get everything after the underscore in the measurement file
                # originally designed for CR measurement data with meas keys of the form 'meas_C+X_T+X', ...
                if verbose_xyzs:
                    xyzs_in = [x.split('_', 1)[-1] for x in meas_axes]
                else:
                    xyzs_in = copy.deepcopy(xyzs)

                for xyz, xyz_in, meas_axis in zip(xyzs, xyzs_in, meas_axes):
                    # print(list(f[prep_state][f'meas_{xyz_in}'].keys()))
                    timesteps = np.sort([int(key[2:]) for key in list(f[prep_state][f'meas_{xyz_in}'].keys()) if key[:2] == 't_'])
                    return_dict[prep_state][f'meas_{xyz}'] = {}

                    for n in timesteps:
                        return_dict[prep_state][f'meas_{xyz}'][f't_{n}'] = {}
                        for key in f[prep_state][f'meas_{xyz_in}'][f't_{n}'].keys():
                            if key in ['cutoff_freq', 'dt_binned', 'dt_unbinned', 'dt_filtered']:
                                return_dict[prep_state][f'meas_{xyz}'][f't_{n}'][key] = f[prep_state][f'meas_{xyz_in}'][f't_{n}'][key][()]
                            else:
                                return_dict[prep_state][f'meas_{xyz}'][f't_{n}'][key] = f[prep_state][f'meas_{xyz_in}'][f't_{n}'][key][:]
        else:
            # Old data structure with one single prep state in the h5 file.
            meas_axes = list(f.keys())
            xyzs = [x[-1:] for x in meas_axes]

            # get data for different measurement axes labels
            # if verbose_xyzs is False, will look for 'meas_X', 'meas_Y', ...
            # if it's true, it will get everything after the underscore in the measurement file
            # originally designed for CR measurement data with meas keys of the form 'meas_C+X_T+X', ...
            if verbose_xyzs:
                xyzs_in = [x.split('_', 1)[-1] for x in meas_axes]
            else:
                xyzs_in = copy.deepcopy(xyzs)

            for xyz, xyz_in, meas_axis in zip(xyzs, xyzs_in, meas_axes):
                print(xyz_in)
                print(list(f.keys()))
                print(list(f[f'meas_{xyz_in}'].keys()))
                timesteps = np.sort([int(key[2:]) for key in list(f[f'meas_{xyz_in}'].keys()) if key[:2] == 't_'])
                return_dict[f'meas_{xyz}'] = {}

                for n in timesteps:
                    return_dict[f'meas_{xyz}'][f't_{n}'] = {}
                    for key in f[f'meas_{xyz_in}'][f't_{n}'].keys():
                        if key in ['cutoff_freq', 'dt_binned', 'dt_unbinned', 'dt_filtered']:
                            return_dict[f'meas_{xyz}'][f't_{n}'][key] = f[f'meas_{xyz_in}'][f't_{n}'][key][()]
                        else:
                            return_dict[f'meas_{xyz}'][f't_{n}'][key] = f[f'meas_{xyz_in}'][f't_{n}'][key][:]

    return return_dict

def dark_mode_compatible(dark_mode_color=r'#86888A'):
    matplotlib.rc('axes', edgecolor=dark_mode_color)
    matplotlib.rc('text', color=dark_mode_color)
    matplotlib.rc('xtick', color=dark_mode_color)
    matplotlib.rc('ytick', color=dark_mode_color)
    matplotlib.rc('axes', labelcolor=dark_mode_color)
    matplotlib.rc('axes', facecolor='none')
    matplotlib.rc('figure', edgecolor='none')  # .edgecolor', (1, 1, 1, 0))
    matplotlib.rc('figure', facecolor='none')  # (1, 1, 1, 0))

def split_data(I, Q, labels, training_fraction, rep_list):
    new_timestep_idcs = [0] + np.cumsum(rep_list).tolist()
    validation_idcs = list()
    training_idcs = list()
    total_batch_size, sequence_length = np.shape(I)

    for r_min, r_max in zip(new_timestep_idcs[:-1], new_timestep_idcs[1:]):
        N_samples = int(training_fraction * (r_max - r_min))  # Oversampling is OK to avoid unique drawings
        # Make sure idcs are unique
        idcs_for_training = np.unique(np.random.choice(range(0, r_max - r_min), size=N_samples))
        idcs_for_val = np.delete(np.arange(0, r_max - r_min), idcs_for_training)

        if r_min == 0:
            validation_idcs = idcs_for_val + r_min
            training_idcs = idcs_for_training + r_min
        else:
            validation_idcs = np.hstack((validation_idcs, idcs_for_val + r_min))
            training_idcs = np.hstack((training_idcs, idcs_for_training + r_min))

    print(f"Training batch size: {len(training_idcs)} ({len(training_idcs) / total_batch_size * 100:.1f}%)")
    print(f"Validation batch size: {len(validation_idcs)} ({len(validation_idcs) / total_batch_size * 100:.1f}%)")

    train_x = np.ndarray(shape=(len(training_idcs), sequence_length, 2))
    valid_x = np.ndarray(shape=(len(validation_idcs), sequence_length, 2))

    train_x[:, :, 0] = I[training_idcs, :]
    train_x[:, :, 1] = Q[training_idcs, :]

    train_y = labels[training_idcs, :].astype(np.int)

    valid_x[:, :, 0] = I[validation_idcs, :]
    valid_x[:, :, 1] = Q[validation_idcs, :]

    valid_y = labels[validation_idcs, :].astype(np.int)

    return train_x, train_y, valid_x, valid_y

def split_data_same_each_time(I, Q, labels, training_fraction, num_features, prep_state_encoding=None, start_idx=0):
    total_batch_size, sequence_length = np.shape(I)
    val_each_n = int(1 / (1 - training_fraction))
    all_idcs = np.arange(total_batch_size)
    validation_idcs = all_idcs[start_idx::val_each_n]
    training_idcs = np.delete(all_idcs, validation_idcs)
    num_prep_states = len(prep_state_encoding) if prep_state_encoding is not None else 0

    print(f"Training batch size: {len(training_idcs)} ({len(training_idcs) / total_batch_size * 100:.1f}%)")
    print(f"Validation batch size: {len(validation_idcs)} ({len(validation_idcs) / total_batch_size * 100:.1f}%)")

    train_x = np.ndarray(shape=(len(training_idcs), sequence_length, num_features + num_prep_states), dtype=np.float32)
    valid_x = np.ndarray(shape=(len(validation_idcs), sequence_length, num_features + num_prep_states), dtype=np.float32)

    if num_features == 2:
        # Heterodyne support (phase preserving amplification)
        train_x[:, :, 0] = I[training_idcs, :].astype(np.float32)
        train_x[:, :, 1] = Q[training_idcs, :].astype(np.float32)
        valid_x[:, :, 0] = I[validation_idcs, :].astype(np.float32)
        valid_x[:, :, 1] = Q[validation_idcs, :].astype(np.float32)
    else:
        # Homodyne support (phase sensitive amplification)
        train_x[:, :, 0] = I[training_idcs, :].astype(np.float32)
        valid_x[:, :, 0] = I[validation_idcs, :].astype(np.float32)

    train_y = labels[training_idcs, :].astype(np.int)
    valid_y = labels[validation_idcs, :].astype(np.int)

    if prep_state_encoding is not None:
        if num_features == 2:
            # Heterodyne support (phase preserving amplification)
            valid_x[:, :, 2:] = prep_state_encoding.astype(np.float32)
            train_x[:, :, 2:] = prep_state_encoding.astype(np.float32)
        else:
            # Homodyne support (phase sensitive amplification)
            train_x[:, :, 1:] = I[training_idcs, :].astype(np.float32)
            valid_x[:, :, 1:] = I[validation_idcs, :].astype(np.float32)

    return train_x, train_y, valid_x, valid_y

def prep_state_encoding(n_levels, prep_state, **kwargs):
    # Rename equivalent prep states
    if prep_state == "g":
        prep_state = "+Z"
    if prep_state == "e":
        prep_state = "-Z"

    if n_levels == 2:
        if prep_state in qubit_prep_dict.keys():
            # Pull the prep state label from `qubit_prep_dict`, which is defined at the top.
            prep_x = qubit_prep_dict[prep_state]["prep_x"]
            prep_y = qubit_prep_dict[prep_state]["prep_y"]
            prep_z = qubit_prep_dict[prep_state]["prep_z"]
        elif prep_state is None:
            # Determine the prep state based on the strong readout results at the first timestep
            expX = kwargs['expX']
            expY = kwargs['expY']
            expZ = kwargs['expZ']
            R0 = np.sqrt(expX[0] ** 2 + expY[0] ** 2 + expZ[0] ** 2)
            X0 = expX[0] if R0 <= 1.0 else expX[0] / R0
            Y0 = expY[0] if R0 <= 1.0 else expY[0] / R0
            Z0 = expZ[0] if R0 <= 1.0 else expZ[0] / R0
            px = 0.5 * (1 + X0)
            py = 0.5 * (1 + Y0)
            pz = 0.5 * (1 + Z0)
            prep_x = [px, 1 - px]
            prep_y = [py, 1 - py]
            prep_z = [pz, 1 - pz]
            print(f"Assumed prep state from strong RO results is: ({X0:.3f}, {Y0:.3f}, {Z0:.3f})")
            print(f"Purity of measured prep state was {R0:.4f}.")
            if R0 >= 1.0:
                print(f"Prep state has been automatically scaled such that purity = 1.")
        else:
            raise ValueError(f"Prep state {prep_state} is not supported for qubits")
    elif n_levels == 3:
        if prep_state in qutrit_prep_dict.keys():
            # Pull the prep state label from `qutrit_prep_dict`, which is defined at the top.
            prep_z = qutrit_prep_dict[prep_state]["prep_z"]
        elif prep_state is None:
            # Determine the prep state based on the strong readout results at the first timestep
            Pg = kwargs['Pg']
            Pe = kwargs['Pe']
            Pf = kwargs['Pf']
            prep_z = [Pg[0], Pe[0], Pf[0]]
        else:
            raise ValueError(f"Prep state {prep_state} is not supported for qutrits")
        prep_x = None
        prep_y = None
    return prep_x, prep_y, prep_z

def get_data(data_dict, n_levels, axis, timesteps, prep_state_encoding=None, scaling=1.0, label_mask_value=-1, take_max=np.inf):
    raw_I = list()
    raw_Q = list()
    reps_per_timestep = list()
    lmv = label_mask_value

    if n_levels == 2:
        if prep_state_encoding is None:
            # This one hot encoding assumes a maximum of three measurement axes, namely 'X', 'Y' or 'Z'
            # More measurement axes can be implemented by adding columns to one_hot.
            # Current encoding strategy is one_hot = [P0x, P1x, P0y, P1y, P0z, P1z] where e.g. P0x is the probability of
            # measuring "0" along axis X (i.e. a +X)
            if axis == 'X':
                one_hot = np.array([[1, 0, lmv, lmv, lmv, lmv],
                                    [0, 1, lmv, lmv, lmv, lmv]])
            elif axis == 'Y':
                one_hot = np.array([[lmv, lmv, 1, 0, lmv, lmv],
                                    [lmv, lmv, 0, 1, lmv, lmv]])
            elif axis == 'Z':
                one_hot = np.array([[lmv, lmv, lmv, lmv, 1, 0],
                                    [lmv, lmv, lmv, lmv, 0, 1]])
        else:
            # If there are multiple prep states, we encode the measurement result in the final 6 elements of one_hot
            # The first n elements are reserved for prep state encoding.
            num_prep_states = len(prep_state_encoding)
            one_hot = np.zeros((2, num_prep_states + 6))
            one_hot[:, :num_prep_states] = np.tile(prep_state_encoding, (2, 1))

            if axis == 'X':
                one_hot[:, num_prep_states:] = np.array([[1, 0, lmv, lmv, lmv, lmv],
                                                         [0, 1, lmv, lmv, lmv, lmv]])
            elif axis == 'Y':
                one_hot[:, num_prep_states:] = np.array([[lmv, lmv, 1, 0, lmv, lmv],
                                                         [lmv, lmv, 0, 1, lmv, lmv]])
            elif axis == 'Z':
                one_hot[:, num_prep_states:] = np.array([[lmv, lmv, lmv, lmv, 1, 0],
                                                         [lmv, lmv, lmv, lmv, 0, 1]])

    elif n_levels == 3:
        # For now, let's assume we have a single measurement axis and we encode (P0, P1, P2)
        # For multiple measurement axes, we can always change the one-hot encoding below, for example for qubit
        # measurement axis with 3 level support, change this to
        # one_hot = [P0x, P1x, P2x, P0y, P1y, P2y, P0z, P1z, P2z] so the length of one_hot is 3 * n_meas_axes
        one_hot = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    for k, t in enumerate(timesteps):
        strong_ro_results = data_dict[f't_{t}']['final_ro_results']
        if n_levels == 2:
            qubit_idcs = np.where(strong_ro_results != 2)[0]
            selected_strong_ro_results = strong_ro_results[qubit_idcs]
            selected_i = data_dict[f't_{t}']['I_binned_filtered'][qubit_idcs, :]
            selected_q = data_dict[f't_{t}']['Q_binned_filtered'][qubit_idcs, :]
        elif n_levels == 3:
            selected_strong_ro_results = strong_ro_results
            selected_i = data_dict[f't_{t}']['I_binned_filtered']
            selected_q = data_dict[f't_{t}']['Q_binned_filtered']

        N_reps = len(selected_strong_ro_results)
        take = int(np.min([N_reps, take_max]))
        reps_per_timestep.append(take)

        # Compile the strong RO results in a long vector
        if k == 0:
            labels = one_hot[selected_strong_ro_results[:take]]
        else:
            labels = np.vstack((labels, one_hot[selected_strong_ro_results[:take]]))

        # Compile the I, Q values in one big 2D array as well. We will apply masking later
        for n in range(take):
            I_n = selected_i[n, :] * scaling
            Q_n = selected_q[n, :] * scaling

            # We append this to a list because the padding function takes a list of lists
            raw_I.append(I_n.tolist())
            raw_Q.append(Q_n.tolist())

    return raw_I, raw_Q, labels, reps_per_timestep

def append_to_h5key(f, key, data):
    f[key].resize((f[key].shape[0] + data.shape[0]), axis=0)
    f[key][-data.shape[0]:] = data