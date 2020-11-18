import sys, os, matplotlib
import numpy as np
import tensorflow as tf
from shutil import copyfile
from rich import print
from rich.console import Console
console = Console()
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append(r"/home/qnl/Git-repositories")
from qnl_trajectories.analysis import data_analysis
import machine_learning_test as ml
from machine_learning_test.utils import *
from machine_learning_test.all_timesteps import *

dark_mode_compatible(dark_mode_color=r'#86888A')

print(tf.config.experimental.list_physical_devices('CPU'))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

tf.debugging.set_log_device_placement(False)

#filepath = r"/home/qnl/Git-repositories/machine_learning_test/data/cts_rabi_amp_1/prep_Y"
last_timestep = 39
mask_value = -1.0 # This is the mask value for the data, not the missing labels
num_features = 2 # I and Q
total_epochs = 15
mini_batch_size = 1024
lstm_neurons = 32
prep_state = "+Y"

rabi_amp = os.path.split(os.path.split(filepath)[0])[1]
experiment_name = f"{rabi_amp}_prep_Y_all_times"
model_savepath = r"/home/qnl/Git-repositories/machine_learning_test/analysis/rabi_amp_sweep"

console.print("Loading data...", style="bold red")

dX = data_analysis.load_data(os.path.join(filepath, 'meas_X'), last_timestep=last_timestep, qubit='Q6')
dY = data_analysis.load_data(os.path.join(filepath, 'meas_Y'), last_timestep=last_timestep, qubit='Q6')
dZ = data_analysis.load_data(os.path.join(filepath, 'meas_Z'), last_timestep=last_timestep, qubit='Q6')

Tm, expX, expY, expZ = data_analysis.plot_average_trajectories(dX, dY, dZ,
                                                               timesteps=np.arange(0, last_timestep+1),
                                                               fit_curves=[],
                                                               artificial_detuning=False,
                                                               savepath=None)
expX = np.array(expX)
expY = np.array(expY)
expZ = np.array(expZ)

console.print("Loaded data...", style="bold red")

timesteps = range(1, last_timestep+1)
rawX_I, rawX_Q, labelsX, reps_per_timestepX = get_data(dX, 'X', timesteps, scaling=0.005)
rawY_I, rawY_Q, labelsY, reps_per_timestepY = get_data(dY, 'Y', timesteps, scaling=0.005)
rawZ_I, rawZ_Q, labelsZ, reps_per_timestepZ = get_data(dZ, 'Z', timesteps, scaling=0.005)

raw_I = rawX_I + rawY_I + rawZ_I
raw_Q = rawX_Q + rawY_Q + rawZ_Q
reps_per_timestep = reps_per_timestepX + reps_per_timestepY + reps_per_timestepZ
labels = np.vstack((np.vstack((labelsX, labelsY)), labelsZ))
meas_ax = ['X'] * np.sum(reps_per_timestepX) + ['Y'] * np.sum(reps_per_timestepY) + \
          ['Z'] * np.sum(reps_per_timestepZ)

del rawX_I, rawX_Q, rawY_I, rawY_Q, rawZ_I, rawZ_Q, labelsX, labelsY, labelsZ

# By default, this will pad using 0s; it is configurable via the "value" parameter.
# Note that you could "pre" padding (at the beginning) or "post" padding (at the end).
# We recommend using "post" padding when working with RNN layers (in order to be able to use the
# CuDNN implementation of the layers).
padded_I = tf.keras.preprocessing.sequence.pad_sequences(raw_I, padding='post',
                                                         dtype='float32', value=mask_value)
padded_Q = tf.keras.preprocessing.sequence.pad_sequences(raw_Q, padding='post',
                                                         dtype='float32', value=mask_value)

batch_size, sequence_length = np.shape(padded_I)

padded_labels = pad_labels(labels, (5 * np.array(timesteps)).tolist() * 3, reps_per_timestep, mask_value)

# Split the data into training and validation data
# train_x, train_y, valid_x, valid_y = split_data(padded_I.astype(np.float32),
#                                                 padded_Q.astype(np.float32),
#                                                 padded_labels,
#                                                 2.25,
#                                                 reps_per_timestep)

train_x, train_y, valid_x, valid_y = split_data_same_each_time(padded_I.astype(np.float32), padded_Q.astype(np.float32),
                                                               padded_labels, 0.90, start_idx=0)

train_msk = train_x != mask_value
valid_msk = valid_x != mask_value
all_data_msk = padded_I != mask_value

all_time_series_lengths = np.sum(all_data_msk, axis=1)
valid_time_series_lengths = np.sum(valid_msk[:, :, 0], axis=1)
train_time_series_lengths = np.sum(train_msk[:, :, 0], axis=1)

# Initialize the model
m = MultiTimeStep(train_x, train_y, valid_x, valid_y, prep_state,
                  lstm_neurons=lstm_neurons, mini_batch_size=mini_batch_size, expX=expX[np.array(timesteps)],
                  expY=expY[np.array(timesteps)], expZ=expZ[np.array(timesteps)],
                  savepath=model_savepath,
                  experiment_name=experiment_name)

_, _, _, _ = data_analysis.plot_average_trajectories(dX, dY, dZ,
                                                     timesteps=np.arange(0, last_timestep+1),
                                                     fit_curves=[],
                                                     artificial_detuning=False,
                                                     savepath=m.savepath)

this_script = "single_prep_multi_timestep.py"
copyfile(os.path.join(r"/home/qnl/Git-repositories/machine_learning_test/scripts", this_script),
         os.path.join(m.savepath, this_script))

# Check if the training data is equally distributed: whether it has equal number of sequence lengths and meas. axes
make_a_pie(all_time_series_lengths, "All data - sequence lengths",
           savepath=m.savepath)
make_a_pie(train_time_series_lengths, "Training data - sequence lengths",
           savepath=m.savepath)
make_a_pie(valid_time_series_lengths, "Validation data - sequence lengths",
           savepath=m.savepath)
make_a_pie(np.array(meas_ax), "All data - measurement axes",
           savepath=m.savepath)

epochs = np.arange(total_epochs)
learning_rate = [m.learning_rate_schedule(epoch) for epoch in epochs]
dropout_rate = [m.dropout_schedule(epoch) for epoch in epochs]

fig = plt.figure()
plt.plot(epochs, learning_rate)
plt.yscale('log')
plt.xlabel("Number of epochs")
plt.ylabel("Scheduled learning rate")

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_learning_rate.png"), bbox_inches='tight', dpi=200)

fig = plt.figure()
plt.plot(epochs, dropout_rate)
plt.yscale('log')
plt.xlabel("Number of epochs")
plt.ylabel("Scheduled dropout rate")

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_dropout_rate.png"), bbox_inches='tight', dpi=200)

m.build_model()
m.compile_model()
m.get_expected_accuracy()
m.init_learning_rate = 1e-3

plt.close('all')

dt = dZ['t_0']['integration_time']
tfinal = dZ[f't_{timesteps[-1]}']['dt']

del dX
del dY
del dZ

# Training
history = m.fit_model(total_epochs)

m.plot_history(history)
m.model.save_weights(os.path.join(m.savepath, "weights.h5"))

console.print("Feeding vaildation data back into network...", style="bold red")

# Pass the validation data through the network once more to produce some plots
y_pred = m.model.predict(valid_x)
y_pred_probabilities = pairwise_softmax(y_pred)
xyz_pred = get_xyz(pairwise_softmax(y_pred))
last_time_idcs = np.where(valid_time_series_lengths == valid_time_series_lengths[-1])[0]
time_axis = np.arange(dt, tfinal + dt, dt)[:np.shape(xyz_pred)[1]]

bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                          xyz_pred[last_time_idcs, :, 0],
                                          xyz_pred[last_time_idcs, :, 1],
                                          xyz_pred[last_time_idcs, :, 2])

fig = plot_histogram(time_axis, xyz_pred[last_time_idcs, :, 0],
                     xyz_pred[last_time_idcs, :, 1], xyz_pred[last_time_idcs, :, 2],
                     Tm, expX, expY, expZ)
if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_histogram.png"), dpi=200)

m.save_trajectories(time_axis, xyz_pred, valid_time_series_lengths)

fig = ml.all_timesteps.plot_verification(y_pred_probabilities, valid_y)
if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "verification_on_validation_data.png"), dpi=200)

plt.close('all')

console.print(f"Training finished. Results saved in {m.savepath}", style="bold red")
