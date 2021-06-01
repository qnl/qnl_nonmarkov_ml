import sys, os, h5py
import numpy as np
import tensorflow as tf
from rich.console import Console
from tqdm import tqdm
console = Console()

from qnl_nonmarkov_ml.gate_diagnosis_lstm.utils import load_settings, load_repackaged_data, get_data, split_data_same_each_time, dark_mode_compatible
from qnl_nonmarkov_ml.gate_diagnosis_lstm.utils import prep_state_encoding as pse
from qnl_nonmarkov_ml.gate_diagnosis_lstm.qutrit_lstm_network import pad_labels

dark_mode_compatible(dark_mode_color=r'#86888A')

yaml_file = r"/home/qnl/noah/projects/2020-NonMarkovTrajectories/code/qnl_nonmarkov_ml/gate_diagnosis_lstm/settings.yaml"
settings = load_settings(yaml_file)

# NOTE: Note that most of the settings below must be equal to the settings in prep.py
# Path that contains the training/validation dataset.
filepath = settings['voltage_records']['filepath']
filename = settings['voltage_records']['filename']

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
mask_value = settings['training']['mask_value'] # This is the mask value for the data, not the missing labels
num_features = settings['voltage_records']['num_features'] # I and Q
multiple_prep_states = settings['voltage_records']['multiple_prep_states']
n_levels = settings['voltage_records']['n_levels']
data_points_for_prep_state = settings['voltage_records']['data_points_for_prep_state']

# Compressing the saved h5 file can save up to an order of magnitude of disk space, but unzipping may take a long time.
if settings['prep']['compress_output_file']:
    h5_save_settings = {'compression' : "gzip", 'chunks' : True}
else:
    h5_save_settings = {}

console.print("Loading data...", style="bold red")

# Load the data from the h5 file
d = load_repackaged_data(os.path.join(filepath, filename), multi_prep_state=multiple_prep_states)
if multiple_prep_states:
    prep_states = settings['voltage_records']['prep_states'] #[prep_key for prep_key in d.keys() if "prep" in prep_key]
    prep_keys = [f"prep_{key}" for key in prep_states]
    num_prep_states = len(prep_states)
    num_meas_axes = len([meas_key for meas_key in d[prep_keys[0]].keys() if "meas" in meas_key])
else:
    num_meas_axes = len([meas_key for meas_key in d.keys() if "meas" in meas_key])

assert ((n_levels == 2) or (n_levels == 3)), "Only n_levels = 2 or 3 is supported at this point"

for p, prep_key in tqdm(enumerate(prep_keys)):
    if n_levels == 2:
        assert num_meas_axes == 3, f"Accurate training on qubit data requires three measurement axes. I only found {num_meas_axes}"
        dX = d[prep_key]['meas_X']
        dY = d[prep_key]['meas_Y']
        dZ = d[prep_key]['meas_Z']

        Px = np.array([np.sum(dX[key]['final_ro_results'] == 1) / len(dX[key]['final_ro_results']) for key in dX.keys()])
        Py = np.array([np.sum(dY[key]['final_ro_results'] == 1) / len(dY[key]['final_ro_results']) for key in dY.keys()])
        Pz = np.array([np.sum(dZ[key]['final_ro_results'] == 1) / len(dZ[key]['final_ro_results']) for key in dZ.keys()])

        expX = 1 - 2 * Px
        expY = 1 - 2 * Py
        expZ = 1 - 2 * Pz
    elif n_levels == 3:
        if num_meas_axes > 1:
            console.print(f"Warning: the qutrit RNN only supports 1 measurement axis (meas_Z). I found {num_meas_axes}",
                          style="bold red")

        dZ = d['meas_Z']
        Pg = np.array([np.sum(dZ[key]['final_ro_results'] == 0) / len(dZ[key]['final_ro_results']) for key in dZ.keys()])
        Pe = np.array([np.sum(dZ[key]['final_ro_results'] == 1) / len(dZ[key]['final_ro_results']) for key in dZ.keys()])
        Pf = np.array([np.sum(dZ[key]['final_ro_results'] == 2) / len(dZ[key]['final_ro_results']) for key in dZ.keys()])

    # dt = dZ['t_0']['dt_filtered']
    dt = dZ['t_1']['dt_filtered']
    timesteps = np.sort([int(key[2:]) for key in list(dZ.keys()) if key[:2] == 't_'])
    # Create the list of strong readout times
    Tm = np.array([np.round(dZ[f't_{ts}']['time_axis_filtered'][-1], decimals=9) for ts in timesteps])
    # Create a list of weak measurement times
    sequence_lengths = np.array([np.shape(dZ[f't_{ts}']['I_binned_filtered'])[1] for ts in timesteps])
    Twm = np.arange(0, sequence_lengths[-1]) * dt
    tfinal = Tm[-1]

    # Encode the prep state as a vector [1, 0, ...] with length num_prep_states. Backwards compatible
    prep_state_encoding = np.array([prep_key == prep_keys[k] for k in range(num_prep_states)]) if multiple_prep_states else None

    console.print("Dividing data in training and validation sets...", style="bold red")

    # Extract the I and Q voltage records and apply a scaling
    scaling = settings['prep']['iq_scaling']

    # Append I and Q voltage records from different measurement axes
    if n_levels == 2:
        rawZ_I, rawZ_Q, labelsZ, reps_per_timestepZ = get_data(dZ, n_levels, 'Z', timesteps,
                                                               prep_state_encoding=prep_state_encoding, scaling=scaling)
        rawX_I, rawX_Q, labelsX, reps_per_timestepX = get_data(dX, n_levels, 'X', timesteps,
                                                               prep_state_encoding=prep_state_encoding, scaling=scaling)
        rawY_I, rawY_Q, labelsY, reps_per_timestepY = get_data(dY, n_levels, 'Y', timesteps,
                                                               prep_state_encoding=prep_state_encoding, scaling=scaling)
        raw_I = rawX_I + rawY_I + rawZ_I
        raw_Q = rawX_Q + rawY_Q + rawZ_Q
        reps_per_timestep = reps_per_timestepX + reps_per_timestepY + reps_per_timestepZ
        labels = np.vstack((labelsX, labelsY, labelsZ))
        meas_ax = ['X'] * np.sum(reps_per_timestepX) + ['Y'] * np.sum(reps_per_timestepY) + \
                  ['Z'] * np.sum(reps_per_timestepZ)
    elif n_levels == 3:
        raw_I, raw_Q, labels, reps_per_timestep = get_data(dZ, n_levels, 'Z', timesteps,
                                                           prep_state_encoding=prep_state_encoding, scaling=scaling)
        meas_ax = ['Z'] * np.sum(reps_per_timestep)

    # By default, this will pad using 0s; it is configurable via the "value" parameter.
    # Note that you could "pre" padding (at the beginning) or "post" padding (at the end).
    # We recommend using "post" padding when working with RNN layers (in order to be able to use the
    # CuDNN implementation of the layers).
    padded_I = tf.keras.preprocessing.sequence.pad_sequences(raw_I, padding='post',
                                                             dtype='float32', value=mask_value)
    padded_Q = tf.keras.preprocessing.sequence.pad_sequences(raw_Q, padding='post',
                                                             dtype='float32', value=mask_value)

    batch_size, sequence_length = np.shape(padded_I)
    # print(np.shape(labels))
    # print(len(timesteps))
    # Pad the labels such that they can be processed by the NN later
    # Note: this assumes the sequence lengths are the same for meas_+X, meas_+Y and meas_+Z datafiles.
    all_sequence_lengths = sequence_lengths.tolist() * num_meas_axes if n_levels == 2 else sequence_lengths
    padded_labels = pad_labels(labels, all_sequence_lengths, reps_per_timestep, mask_value)
    # print(np.shape(padded_labels))

    # Split validation and data deterministically so we can compare results from run to run
    train_x, train_y, valid_x, valid_y = split_data_same_each_time(padded_I.astype(np.float32), padded_Q.astype(np.float32),
                                                                   padded_labels,
                                                                   settings['prep']['training_validation_ratio'],
                                                                   num_features=num_features,
                                                                   prep_state_encoding=prep_state_encoding,
                                                                   start_idx=0)

    train_msk = train_x != mask_value
    valid_msk = valid_x != mask_value
    all_data_msk = padded_I != mask_value

    all_time_series_lengths = np.sum(all_data_msk, axis=1)
    valid_time_series_lengths = np.sum(valid_msk[:, :, 0], axis=1)
    train_time_series_lengths = np.sum(train_msk[:, :, 0], axis=1)

    if multiple_prep_states:
        # Mask the prep state encoding, to match the masking of the I and Q
        for n, samps in enumerate(train_time_series_lengths):
            train_x[n, samps:, num_features:] = mask_value
        for n, samps in enumerate(valid_time_series_lengths):
            valid_x[n, samps:, num_features:] = mask_value

    # Save a pre-processed file as an h5 file. Note these files can be quite large, typ. 15 GB.
    console.print(f"Saving processed data to {filepath}...", style="bold red")
    output_filename = os.path.join(filepath, settings['prep']['output_filename'])
    # If the file exists then remove it.
    if p == 0 and os.path.exists(output_filename):
        os.remove(output_filename)

    print(np.shape(train_x), np.shape(train_y), np.shape(valid_x), np.shape(valid_y))
    assert np.shape(train_x)[1] == np.shape(valid_x)[1], "Shapes of train_x, valid_x are not right!"

    # Create the h5 file and keep appending to it, if there is more than 1 prep state.
    with h5py.File(output_filename, 'a') as f:
        if p == 0:
            f.create_dataset("train_x", data=train_x, maxshape=(None, sequence_length, num_features + num_prep_states),
                             **h5_save_settings)
            f.create_dataset("valid_x", data=valid_x, maxshape=(None, sequence_length, num_features + num_prep_states),
                             **h5_save_settings)
            f.create_dataset("train_y", data=train_y, maxshape=(None, sequence_length, 6 + num_prep_states),
                             **h5_save_settings)
            f.create_dataset("valid_y", data=valid_y, maxshape=(None, sequence_length, 6 + num_prep_states),
                             **h5_save_settings)

            f.create_dataset("Tm", data=Tm)
            f.create_dataset("weak_meas_times", data=Twm)

            if n_levels == 2:
                f.create_dataset("expX", data=expX.reshape((1, -1)), maxshape=(None, len(expX)), **h5_save_settings)
                f.create_dataset("expY", data=expY.reshape((1, -1)), maxshape=(None, len(expY)), **h5_save_settings)
                f.create_dataset("expZ", data=expZ.reshape((1, -1)), maxshape=(None, len(expZ)), **h5_save_settings)
            elif n_levels == 3:
                f.create_dataset("Pg", data=Pg.reshape((1, -1)), maxshape=(None, len(Pg)), **h5_save_settings)
                f.create_dataset("Pe", data=Pe.reshape((1, -1)), maxshape=(None, len(Pe)), **h5_save_settings)
                f.create_dataset("Pf", data=Pf.reshape((1, -1)), maxshape=(None, len(Pf)), **h5_save_settings)

            f.create_dataset("all_time_series_lengths", data=all_time_series_lengths,
                             maxshape=(None,), **h5_save_settings)
            f.create_dataset("valid_time_series_lengths", data=valid_time_series_lengths,
                             maxshape=(None,), **h5_save_settings)
            f.create_dataset("train_time_series_lengths", data=train_time_series_lengths,
                             maxshape=(None,), **h5_save_settings)

            f.create_dataset("dt", data=np.array([dt]))
            f.create_dataset("tfinal", data=np.array([tfinal]))
        else:
            # Append different prep states to the h5 file
            f["train_x"].resize((f["train_x"].shape[0] + train_x.shape[0]), axis=0)
            f["train_x"][-train_x.shape[0]:] = train_x

            f["train_y"].resize((f["train_y"].shape[0] + train_y.shape[0]), axis=0)
            f["train_y"][-train_y.shape[0]:] = train_y

            f["valid_x"].resize((f["valid_x"].shape[0] + valid_x.shape[0]), axis=0)
            f["valid_x"][-valid_x.shape[0]:] = valid_x

            f["valid_y"].resize((f["valid_y"].shape[0] + valid_y.shape[0]), axis=0)
            f["valid_y"][-valid_y.shape[0]:] = valid_y

            if n_levels == 2:
                f["expX"].resize((f["expX"].shape[0] + 1), axis=0)
                f["expX"][-1] = expX.reshape((1, -1))

                f["expY"].resize((f["expY"].shape[0] + 1), axis=0)
                f["expY"][-1] = expY.reshape((1, -1))

                f["expZ"].resize((f["expZ"].shape[0] + 1), axis=0)
                f["expZ"][-1] = expZ.reshape((1, -1))
            elif n_levels == 3:
                f["Pg"].resize((f["Pg"].shape[0] + 1), axis=0)
                f["Pg"][-1] = Pg.reshape((1, -1))

                f["Pe"].resize((f["Pe"].shape[0] + 1), axis=0)
                f["Pe"][-1] = Pe.reshape((1, -1))

                f["Pf"].resize((f["Pf"].shape[0] + 1), axis=0)
                f["Pf"][-1] = Pf.reshape((1, -1))

            f["all_time_series_lengths"].resize((f["all_time_series_lengths"].shape[0] + all_time_series_lengths.shape[0]), axis=0)
            f["all_time_series_lengths"][-all_time_series_lengths.shape[0]:] = all_time_series_lengths

            f["valid_time_series_lengths"].resize((f["valid_time_series_lengths"].shape[0] + valid_time_series_lengths.shape[0]), axis=0)
            f["valid_time_series_lengths"][-valid_time_series_lengths.shape[0]:] = valid_time_series_lengths

            f["train_time_series_lengths"].resize((f["train_time_series_lengths"].shape[0] + train_time_series_lengths.shape[0]), axis=0)
            f["train_time_series_lengths"][-train_time_series_lengths.shape[0]:] = train_time_series_lengths