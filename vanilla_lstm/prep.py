import sys
import os
import h5py
import numpy as np
import tensorflow as tf
from qnl_trajectories.analysis import data_analysis
from qnl_trajectories.analysis.load_hdf5 import load_hdf5
# from .utils import *
from qnl_nonmarkov_ml.vanilla_lstm.utils import *
# from .vanilla_lstm import pad_labels
from qnl_nonmarkov_ml.vanilla_lstm.vanilla_lstm import pad_labels
from qnl_nonmarkov_ml.vanilla_lstm.formatting_fixes.data_filters import DataFilters as df
from rich.console import Console
console = Console()

dark_mode_compatible(dark_mode_color=r'#86888A')

# NOTE: Note that most of the settings below must be equal to the settings in prep.py
# Path that contains the training/validation dataset.
filepath = r'/run/media/qnl/Seagate Expansion Drive/non_markovian/local_data/2021_02_17/cr_trajectories_test_028/data_transfer/2021_02_17/cr_trajectories_test_028'

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
last_timestep = 99
mask_value = -1.0  # This is the mask value for the data, not the missing labels
num_features = 2  # I and Q
strong_ro_dt = 20e-9  # Time interval for strong readout in the dataset in seconds

console.print("Loading data...", style="bold red")

# Load the data from the pickle files.
h5 = True
if h5:
    load_hdf5_ = load_hdf5.LoadHDF5()

    def keys_(ax):
        return ['prep_C+X_T+Y', f'meas_C+{ax}_T+{ax}', 'prep_C+X_T+Y', f'meas_+{ax}_T+{ax}']

    dX = load_hdf5_.load_data(filepath, keys=keys_('X'), qubit='Q6', last_timestep=last_timestep)
    dY = load_hdf5_.load_data(filepath, keys=keys_('Y'), qubit='Q6', last_timestep=last_timestep)
    dZ = load_hdf5_.load_data(filepath, keys=keys_('Z'), qubit='Q6', last_timestep=last_timestep)

else:

    meas_X = r"meas_C+Z_T+X"
    meas_Y = r"meas_C+Z_T+Y"
    meas_Z = r"meas_C+Z_T+Z"

    dX = data_analysis.load_data(os.path.join(filepath, meas_X), qubit='Q6', method='final')
    dY = data_analysis.load_data(os.path.join(filepath, meas_Y), qubit='Q6', method='final')
    dZ = data_analysis.load_data(os.path.join(filepath, meas_Z), qubit='Q6', method='final')

dX = df.correct_timestep(dX)
dY = df.correct_timestep(dY)
dZ = df.correct_timestep(dZ)

# Get the expectation value from the data containers for each measurement axis.
Tm, expX, expY, expZ = data_analysis.plot_average_trajectories(dX, dY, dZ,
                                                               timesteps=np.arange(0, last_timestep+1),
                                                               fit_curves=[],
                                                               artificial_detuning=False,
                                                               savepath=None)
expX = np.array(expX)
expY = np.array(expY)
expZ = np.array(expZ)

dt = dZ['t_0']['integration_time']
timesteps = range(1, last_timestep+1)
tfinal = dZ[f't_{timesteps[-1]}']['dt']

console.print("Loaded data...", style="bold red")

# Extract the I and Q voltage records and apply a scaling
rawX_I, rawX_Q, labelsX, reps_per_timestepX = get_data(dX, 'X', timesteps, scaling=0.005)
rawY_I, rawY_Q, labelsY, reps_per_timestepY = get_data(dY, 'Y', timesteps, scaling=0.005)
rawZ_I, rawZ_Q, labelsZ, reps_per_timestepZ = get_data(dZ, 'Z', timesteps, scaling=0.005)

# Append I and Q voltage records from different measurement axes
raw_I = rawX_I + rawY_I + rawZ_I
raw_Q = rawX_Q + rawY_Q + rawZ_Q
reps_per_timestep = reps_per_timestepX + reps_per_timestepY + reps_per_timestepZ
labels = np.vstack((np.vstack((labelsX, labelsY)), labelsZ))
meas_ax = ['X'] * np.sum(reps_per_timestepX) + ['Y'] * np.sum(reps_per_timestepY) + \
          ['Z'] * np.sum(reps_per_timestepZ)

# By default, this will pad using 0s; it is configurable via the "value" parameter.
# Note that you could "pre" padding (at the beginning) or "post" padding (at the end).
# We recommend using "post" padding when working with RNN layers (in order to be able to use the
# CuDNN implementation of the layers).
padded_I = tf.keras.preprocessing.sequence.pad_sequences(raw_I, padding='post',
                                                         dtype='float32', value=mask_value)
padded_Q = tf.keras.preprocessing.sequence.pad_sequences(raw_Q, padding='post',
                                                         dtype='float32', value=mask_value)

batch_size, sequence_length = np.shape(padded_I)

# Pad the labels such that they can be processed by the NN later
# n = settings['voltage_records']['data_points_for_prep_state']
# padded_labels = pad_labels(labels, (n + int(strong_ro_dt/dt) * np.array(timesteps)).tolist() * 3,
#                            reps_per_timestep, mask_value)
_n = 0  # placeholder value to get code to run; need to figure out source of error and fix
padded_labels = pad_labels(labels, _n + np.array((int(strong_ro_dt/dt) * np.array(timesteps)).tolist() * 3), reps_per_timestep, mask_value)

# print(np.shape(padded_labels))
# _sel = 5*np.arange(1, 5) - 1
# print(_sel)
# # print([padded_labels[0, _sel, i] for i in range(6)])
# print(labels[_sel])
# print(padded_labels[0, _sel, :])
# sys.exit()


# Split validation and data deterministically so we can compare results from run to run
train_x, train_y, valid_x, valid_y = split_data_same_each_time(padded_I.astype(np.float32), padded_Q.astype(np.float32),
                                                               padded_labels, 0.90, start_idx=0)

train_msk = train_x != mask_value
valid_msk = valid_x != mask_value
all_data_msk = padded_I != mask_value

all_time_series_lengths = np.sum(all_data_msk, axis=1)
valid_time_series_lengths = np.sum(valid_msk[:, :, 0], axis=1)
train_time_series_lengths = np.sum(train_msk[:, :, 0], axis=1)

# Save a pre-processed file as an h5 file. Note these files can be quite large, typ. 15 GB.
console.print(f"Saving processed data to {filepath}...", style="bold red")
with h5py.File(os.path.join(filepath, "training_validation_split.h5"), 'w') as f:
    f.create_dataset("train_x", data=train_x)
    f.create_dataset("train_y", data=train_y)
    f.create_dataset("valid_x", data=valid_x)
    f.create_dataset("valid_y", data=valid_y)

    f.create_dataset("Tm", data=Tm[np.array(timesteps)])
    f.create_dataset("expX", data=expX[np.array(timesteps)])
    f.create_dataset("expY", data=expY[np.array(timesteps)])
    f.create_dataset("expZ", data=expZ[np.array(timesteps)])

    f.create_dataset("all_time_series_lengths", data=all_time_series_lengths)
    f.create_dataset("valid_time_series_lengths", data=valid_time_series_lengths)
    f.create_dataset("train_time_series_lengths", data=train_time_series_lengths)

    f.create_dataset("dt", data=np.array([dt]))
    f.create_dataset("tfinal", data=np.array([tfinal]))