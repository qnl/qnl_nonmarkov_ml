import os, h5py
import numpy as np
import tensorflow as tf
from shutil import copyfile
from rich import print
from rich.console import Console
console = Console()
import matplotlib.pyplot as plt
from utils import dark_mode_compatible, load_settings, save_settings
from qutrit_lstm_network import MultiTimeStep, make_a_pie, pairwise_softmax, get_xyz, get_histogram
from qutrit_lstm_network import plot_verification, plot_qutrit_verification, plot_qubit_histogram, plot_qutrit_histogram

dark_mode_compatible(dark_mode_color=r'#86888A')

yaml_file = r"/home/qnl/Git-repositories/qnl_nonmarkov_ml/gate_diagnosis_lstm/settings.yaml"
settings = load_settings(yaml_file)

print(tf.config.experimental.list_physical_devices('CPU'))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

tf.debugging.set_log_device_placement(False)

# NOTE: Note that most of the settings below must be equal to the settings in prep.py
# Path that contains the training/validation dataset.
filepath = settings['voltage_records']['filepath']
prep_state = settings['voltage_records']['prep_state'] # Prep state, VERY IMPORTANT
experiment_name = settings['training']['experiment_id']
n_levels = settings['voltage_records']['n_levels']

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
mask_value = settings['training']['mask_value'] # This is the mask value for the data, not the missing labels
total_epochs = settings['training']['epochs'] # Number of epochs for the training
mini_batch_size = settings['training']['mini_batch_size'] # Batch size
lstm_neurons = settings['training']['lstm_neurons'] # Depth of the LSTM layer

# This is where the trained trajectories will be saved to
model_savepath = os.path.join(filepath, "analysis")
if not(os.path.exists(model_savepath)):
    os.makedirs(model_savepath)

# Load the data prepaired in prep.py
with h5py.File(os.path.join(filepath, settings['prep']['output_filename']), "r") as f:
    train_x = f.get('train_x')[:]
    train_y = f.get('train_y')[:]
    valid_x = f.get('valid_x')[:]
    valid_y = f.get('valid_y')[:]

    dt = f.get('dt')[:]
    time_axis = f.get('weak_meas_times')[:]
    tfinal = f.get('tfinal')[:]

    Tm = f.get('Tm')[:]

    if n_levels == 2:
        expX = f.get('expX')[:]
        expY = f.get('expY')[:]
        expZ = f.get('expZ')[:]
        avgd_strong_ro_results = {'expX': expX,
                                  'expY': expY,
                                  'expZ': expZ}
    else:
        Pg = f.get('Pg')[:]
        Pe = f.get('Pe')[:]
        Pf = f.get('Pf')[:]
        avgd_strong_ro_results = {'Pg': Pg,
                                  'Pe': Pe,
                                  'Pf': Pf}

    all_time_series_lengths = f.get('all_time_series_lengths')[:]
    valid_time_series_lengths = f.get('valid_time_series_lengths')[:]
    train_time_series_lengths = f.get('train_time_series_lengths')[:]

console.print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

# Initialize the model
console.print("Creating model...", style="bold red")
m = MultiTimeStep(train_x, train_y, valid_x, valid_y, prep_state, n_levels,
                  lstm_neurons=lstm_neurons, mini_batch_size=mini_batch_size,
                  savepath=model_savepath, experiment_name=experiment_name, **avgd_strong_ro_results)

m.init_learning_rate = settings['training']['learning_rate_init']
m.reduce_learning_rate_after = settings['training']['learning_rate_reduce_after']
m.learning_rate_epoch_constant = settings['training']['learning_rate_epoch_constant']

settings['analysis']['subdir'] = m.savepath
settings['analysis']['trajectory_dt'] = float(dt)
save_settings(yaml_file, settings)

# Check if the training data is equally distributed: whether it has equal number of sequence lengths and meas. axes
make_a_pie(all_time_series_lengths, "All data - sequence lengths", savepath=m.savepath)
make_a_pie(train_time_series_lengths, "Training data - sequence lengths", savepath=m.savepath)
make_a_pie(valid_time_series_lengths, "Validation data - sequence lengths", savepath=m.savepath)

# Copy the script to the analysis folder to keep track of settings
this_script = "train.py"
copyfile(this_script, os.path.join(m.savepath, this_script))
copyfile(yaml_file, os.path.join(m.savepath, os.path.split(yaml_file)[-1]))

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
    fig.savefig(os.path.join(m.savepath, "000_learning_rate.png"), **settings['figure_options'])

fig = plt.figure()
plt.plot(epochs, dropout_rate)
plt.yscale('log')
plt.xlabel("Number of epochs")
plt.ylabel("Scheduled dropout rate")

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_dropout_rate.png"), **settings['figure_options'])

console.print("Building model...", style="bold red")
m.build_model()
console.print("Compiling model...", style="bold red")
m.compile_model()
if n_levels == 2:
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
y_pred_probabilities = pairwise_softmax(y_pred, n_levels)

# Select the longest trajectories
last_time_idcs = np.where(valid_time_series_lengths == valid_time_series_lengths[-1])[0]
# print(len(time_axis), np.shape(xyz_pred)[1])

# Make a histogram of the validation trajectories
if n_levels == 2:
    xyz_pred = get_xyz(pairwise_softmax(y_pred, n_levels))
    bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                              xyz_pred[last_time_idcs, :, 0],
                                              xyz_pred[last_time_idcs, :, 1],
                                              xyz_pred[last_time_idcs, :, 2],
                                              bin_min=-1, bin_max=1)

    fig = plot_qubit_histogram(time_axis, xyz_pred[last_time_idcs, :, 0],
                               xyz_pred[last_time_idcs, :, 1], xyz_pred[last_time_idcs, :, 2],
                               Tm, expX, expY, expZ)
elif n_levels == 3:
    xyz_pred = y_pred_probabilities
    print(xyz_pred.shape)
    bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                              xyz_pred[last_time_idcs, :, 0],
                                              xyz_pred[last_time_idcs, :, 1],
                                              xyz_pred[last_time_idcs, :, 2],
                                              bin_min=0, bin_max=1)

    fig = plot_qutrit_histogram(time_axis, xyz_pred[last_time_idcs, :, 0],
                                xyz_pred[last_time_idcs, :, 1], xyz_pred[last_time_idcs, :, 2],
                                Tm, Pg, Pe, Pf)

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "000_histogram.png"), **settings['figure_options'])

m.save_trajectories(time_axis, xyz_pred, valid_time_series_lengths, history)

if n_levels == 2:
    fig = plot_verification(y_pred_probabilities, valid_y)
elif n_levels == 3:
    fig = plot_qutrit_verification(y_pred_probabilities, valid_y)

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, "verification_on_validation_data.png"), **settings['figure_options'])

plt.close('all')

console.print(f"Training finished. Results saved in {m.savepath}", style="bold red")