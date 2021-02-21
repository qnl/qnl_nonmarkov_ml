import os, h5py
import numpy as np
import tensorflow as tf
from shutil import copyfile
from rich import print
from rich.console import Console
console = Console()
import matplotlib.pyplot as plt
from qnl_nonmarkov_ml.vanilla_lstm.utils import *
from qnl_nonmarkov_ml.vanilla_lstm.vanilla_lstm import *

dark_mode_compatible(dark_mode_color=r'#86888A')

print(tf.config.experimental.list_physical_devices('CPU'))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

tf.debugging.set_log_device_placement(False)

# NOTE: Note that most of the settings below must be equal to the settings in prep.py
# Path that contains the training/validation dataset.
filepath = r"/home/qnl/noah/projects/2020-NonMarkovTrajectories/local-data/2020_07_31/cr_trajectories_test_024/prep_C+X_T+Y"
prep_state = "+Y"  # Prep state, VERY IMPORTANT

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
last_timestep = 249
mask_value = -1  # This is the mask value for the data, not the missing labels
total_epochs = 50  # Number of epochs for the training
mini_batch_size = 1024  # Batch size
lstm_neurons = 32  # Depth of the LSTM layer
strong_ro_dt = 20e-9  # Time interval for strong readout in the dataset in seconds

rabi_amp = os.path.split(os.path.split(filepath)[0])[1]
experiment_name = f"{rabi_amp}_prep_{prep_state}"
# This is where the trained trajectories will be saved to
model_savepath = r"analysis/cr"

# Load the data prepaired in prep.py
with h5py.File(os.path.join(filepath, 'training_validation_split.h5'), "r") as f:
    train_x = f.get('train_x')[:]
    train_y = f.get('train_y')[:]
    valid_x = f.get('valid_x')[:]
    valid_y = f.get('valid_y')[:]

    dt = f.get('dt')[:]
    tfinal = f.get('tfinal')[:]

    Tm = f.get('Tm')[:]
    expX = f.get('expX')[:]
    expY = f.get('expY')[:]
    expZ = f.get('expZ')[:]

    all_time_series_lengths = f.get('all_time_series_lengths')[:]
    valid_time_series_lengths = f.get('valid_time_series_lengths')[:]
    train_time_series_lengths = f.get('train_time_series_lengths')[:]

# Initialize the model
console.print("Creating model...", style="bold red")
m = MultiTimeStep(train_x, train_y, valid_x, valid_y, prep_state,
                  lstm_neurons=lstm_neurons, mini_batch_size=mini_batch_size, expX=expX, expY=expY, expZ=expZ,
                  savepath=model_savepath, experiment_name=experiment_name)
m.mask_value = mask_value

# Check if the training data is equally distributed: whether it has equal number of sequence lengths and meas. axes
make_a_pie(all_time_series_lengths, "All data - sequence lengths", savepath=m.savepath)
make_a_pie(train_time_series_lengths, "Training data - sequence lengths", savepath=m.savepath)
make_a_pie(valid_time_series_lengths, "Validation data - sequence lengths", savepath=m.savepath)

# Copy the script to the analysis folder to keep track of settings
this_script = "train.py"
copyfile(this_script, os.path.join(m.savepath, this_script))

# Plot the learning rate settings etc.
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

console.print("Building model...", style="bold red")
m.build_model()
console.print("Compiling model...", style="bold red")
m.compile_model()
m.get_expected_accuracy()
m.init_learning_rate = 1e-3

plt.close('all')


# Start the training
console.print("Training started...", style="bold red")
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

# Make a histogram of the validation trajectories
bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                          xyz_pred[last_time_idcs, :, 0],
                                          xyz_pred[last_time_idcs, :, 1],
                                          xyz_pred[last_time_idcs, :, 2])

fig = plot_histogram(time_axis,
                     xyz_pred[last_time_idcs, :, 0],
                     xyz_pred[last_time_idcs, :, 1],
                     xyz_pred[last_time_idcs, :, 2],
                     Tm, expX, expY, expZ)
if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_histogram.png"), dpi=200)

m.save_trajectories(time_axis, xyz_pred, valid_time_series_lengths)

fig = plot_verification(y_pred_probabilities, valid_y)
if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "verification_on_validation_data.png"), dpi=200)

plt.close('all')

console.print(f"Training finished. Results saved in {m.savepath}", style="bold red")
