import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import h5py
import yaml
import os
from qnl_trajectories.analysis.load_hdf5 import load_hdf5
from rich.console import Console


x_color = plt.cm.Reds(0.6)
y_color = plt.cm.Blues(0.6)
z_color = plt.cm.Greys(0.6)

save_options = {"dpi" : 200,
                "bbox_inches" : "tight",
                "pad_inches" : 0.05,
                "transparent" : True}


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

def load_repackaged_data(filename):
    """
    Load the h5 file produced by filtering.repackage_raw_data
    :param filename: string, HDF5 Filename including the file extension
    :return: dictionary, contents of the h5 file for further processing.
    """
    with h5py.File(filename, "r") as f:
        # Here are all the keys available
        print("Available keys in this datafile:", list(f.keys()))
        meas_axes = list(f.keys())
        xyzs = [x[-1:] for x in meas_axes]
        return_dict = {}

        for xyz, meas_axis in zip(xyzs, meas_axes):
            timesteps = np.sort([int(key[2:]) for key in list(f[f'meas_{xyz}'].keys()) if key[:2] == 't_'])
            return_dict[f'meas_{xyz}'] = {}

            for n in timesteps:
                return_dict[f'meas_{xyz}'][f't_{n}'] = {}
                for key in f[f'meas_{xyz}'][f't_{n}'].keys():
                    if key in ['cutoff_freq', 'dt_binned', 'dt_unbinned', 'dt_filtered']:
                        return_dict[f'meas_{xyz}'][f't_{n}'][key] = f[f'meas_{xyz}'][f't_{n}'][key][()]
                    else:
                        return_dict[f'meas_{xyz}'][f't_{n}'][key] = f[f'meas_{xyz}'][f't_{n}'][key][:]

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

def split_data_same_each_time(I, Q, labels, training_fraction, start_idx=0):
    total_batch_size, sequence_length = np.shape(I)
    val_each_n = int(1 / (1 - training_fraction))
    all_idcs = np.arange(total_batch_size)
    validation_idcs = all_idcs[start_idx::val_each_n]
    training_idcs = np.delete(all_idcs, validation_idcs)

    print(f"Training batch size: {len(training_idcs)} ({len(training_idcs) / total_batch_size * 100:.1f}%)")
    print(f"Validation batch size: {len(validation_idcs)} ({len(validation_idcs) / total_batch_size * 100:.1f}%)")

    train_x = np.ndarray(shape=(len(training_idcs), sequence_length, 2), dtype=np.float32)
    valid_x = np.ndarray(shape=(len(validation_idcs), sequence_length, 2), dtype=np.float32)

    train_x[:, :, 0] = I[training_idcs, :].astype(np.float32)
    train_x[:, :, 1] = Q[training_idcs, :].astype(np.float32)

    train_y = labels[training_idcs, :].astype(np.int)

    valid_x[:, :, 0] = I[validation_idcs, :].astype(np.float32)
    valid_x[:, :, 1] = Q[validation_idcs, :].astype(np.float32)

    valid_y = labels[validation_idcs, :].astype(np.int)

    return train_x, train_y, valid_x, valid_y

def get_data(data_dict, axis, timesteps, scaling=1.0, label_mask_value=-1, take_max=np.inf):
    raw_I = list()
    raw_Q = list()
    reps_per_timestep = list()
    lmv = label_mask_value

    if axis == 'X':
        one_hot = np.array([[1, 0, lmv, lmv, lmv, lmv],
                            [0, 1, lmv, lmv, lmv, lmv]])
    elif axis == 'Y':
        one_hot = np.array([[lmv, lmv, 1, 0, lmv, lmv],
                            [lmv, lmv, 0, 1, lmv, lmv]])
    elif axis == 'Z':
        one_hot = np.array([[lmv, lmv, lmv, lmv, 1, 0],
                            [lmv, lmv, lmv, lmv, 0, 1]])

    for k, t in enumerate(timesteps):
        strong_ro_results = data_dict[f't_{t}']['final_ro_results']
        qubit_idcs = np.where(strong_ro_results != 2)[0]
        selected_strong_ro_results = strong_ro_results[qubit_idcs]
        selected_i = data_dict[f't_{t}']['I_binned_filtered'][qubit_idcs, :]
        selected_q = data_dict[f't_{t}']['Q_binned_filtered'][qubit_idcs, :]
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

