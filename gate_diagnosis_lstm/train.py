import os, h5py
import numpy as np
import tensorflow as tf
from shutil import copyfile
from rich import print
from rich.console import Console
from scipy.special import softmax
console = Console()
import matplotlib.pyplot as plt
from qnl_nonmarkov_ml.gate_diagnosis_lstm.utils import dark_mode_compatible, load_settings, save_settings
from qnl_nonmarkov_ml.gate_diagnosis_lstm.qutrit_lstm_network import MultiTimeStep, make_a_pie, pairwise_softmax, get_xyz, get_histogram
from qnl_nonmarkov_ml.gate_diagnosis_lstm.qutrit_lstm_network import plot_qubit_verification, plot_qutrit_verification, plot_qubit_histogram, plot_qutrit_histogram
import gc

# dark_mode_compatible(dark_mode_color=r'#86888A')

# yaml_file = r"/home/qnl/noah/projects/2020-NonMarkovTrajectories/code/qnl_nonmarkov_ml/gate_diagnosis_lstm/lstm_test/settings.yaml"
yaml_file = r"/home/qnl/noah/projects/2020-NonMarkovTrajectories/code/qnl_nonmarkov_ml/gate_diagnosis_lstm/settings.yaml"
settings = load_settings(yaml_file)

print(tf.config.experimental.list_physical_devices('CPU'))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

tf.debugging.set_log_device_placement(False)

# NOTE: Note that most of the settings below must be equal to the settings in prep.py
# Path that contains the training/validation dataset.
filepath = settings['voltage_records']['filepath']
prep_state_from_ro = settings['training']['prep_state_from_ro'] # Prep state, VERY IMPORTANT
experiment_name = settings['training']['experiment_id']
n_levels = settings['voltage_records']['n_levels']
prep_states = settings['voltage_records']['prep_states']
num_features = settings['voltage_records']['num_features']
num_prep_states = len(prep_states)
data_points_for_prep_state = settings['voltage_records']['data_points_for_prep_state']
annealing_steps = int(settings['training']['annealing_steps'])
bidirectional = settings['training']['bidirectional']

# last_timestep determines the length of trajectories used for training in units of strong_ro_dt.
# Must be <= the last strong readout point
mask_value = settings['training']['mask_value'] # This is the mask value for the data, not the missing labels
epochs_per_anneal = settings['training']['epochs_per_anneal'] # Number of epochs for the training
total_epochs = epochs_per_anneal * annealing_steps
mini_batch_size = settings['training']['mini_batch_size'] # Batch size
lstm_neurons = settings['training']['lstm_neurons'] # Depth of the LSTM layer

# This is where the trained trajectories will be saved to
model_savepath = os.path.join(filepath, "analysis")
if not(os.path.exists(model_savepath)):
    os.makedirs(model_savepath)

# Load the data prepaired in prep.py
console.print("Unzipping data from prep.py...", style="bold red")
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

if len(prep_states) == 1:
    # Don't give the NN the prep state encoding, since it's not used in the cost function.
    train_x = train_x[..., :-1] # Order in the last index is [I, Q, prep state encoding]
    train_y = train_y[..., 1:] # Order in the last index is [prep state encoding, P0x, P1x, etc.]

    valid_x = valid_x[..., :-1]
    valid_y = valid_y[..., 1:]

# Construct Tensorflow dataset
# class generator:
#     def __init__(self):
#         self.file = os.path.join(filepath, settings['prep']['output_filename'])
#         self.training_data = True
#
#     def __call__(self):
#         with h5py.File(self.file, 'r') as f:
#             if self.training_data:
#                 for train_x, train_y in zip(f["train_x"], f["train_y"]):
#                     yield train_x, train_y
#                     # yield np.expand_dims(train_x, axis=0), np.expand_dims(train_y, axis=0)
#             else:
#                 for valid_x, valid_y in zip(f["valid_x"], f["valid_y"]):
#                     yield np.expand_dims(valid_x, axis=0), np.expand_dims(valid_y, axis=0)
#
# ds = tf.data.Dataset.from_generator(generator(), output_types=(tf.float32, tf.float32),
#                                     output_shapes=(tf.TensorShape([valid_x.shape[1], valid_x.shape[2]]),
#                                                    tf.TensorShape([valid_y.shape[1], valid_y.shape[2]])))
# batched_dataset = ds.batch(mini_batch_size)

console.print(valid_x.shape, valid_y.shape)

# Initialize the model
console.print("Creating model...", style="bold red")
m = MultiTimeStep(valid_x, valid_y, prep_states, n_levels,
                  data_points_for_prep_state=data_points_for_prep_state,
                  prep_state_from_ro=prep_state_from_ro,
                  lstm_neurons=lstm_neurons, mini_batch_size=mini_batch_size,
                  annealing_steps=annealing_steps, epochs_per_annealing=epochs_per_anneal,
                  savepath=model_savepath, experiment_name=experiment_name, bidirectional=bidirectional,
                  **avgd_strong_ro_results)

del valid_x, valid_y

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

del all_time_series_lengths, train_time_series_lengths
gc.collect()

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
console.print("Model compiled", style='bold green')
if n_levels == 2:
    m.get_expected_accuracy()
m.init_learning_rate = 1e-3

plt.close('all')

# Start the training
console.print("Training started...", style="bold red")
# history = m.fit_model_with_generator(batched_dataset, total_epochs)
history = m.fit_model(train_x, train_y)
gc.collect()

m.plot_history(history)
m.model.save_weights(os.path.join(m.savepath, "weights.h5"))

console.print("Feeding validation data back into network...", style="bold red")

# Pass the validation data through the network once more to produce some plots
y_pred = m.model.predict(m.validation_features)
if num_prep_states > 1:
    y_pred_probabilities = pairwise_softmax(y_pred[..., num_prep_states:], n_levels)
    prep_state_characterization = softmax(y_pred[..., :num_prep_states], axis=2)
else:
    y_pred_probabilities = pairwise_softmax(y_pred, n_levels)

# Select the longest trajectories
last_time_idcs = np.where(valid_time_series_lengths == valid_time_series_lengths[-1])[0]

if num_prep_states > 1:
    # Convert probabilities from the NN to x, y and z qubit-coordinates.
    # xyz_pred = get_xyz(pairwise_softmax(y_pred_probabilities, n_levels))
    xyz_pred = get_xyz(y_pred_probabilities)
    # Organize the predicted trajectories by prep state
    # The prep state is encoded in the features.
    for p in range(num_prep_states):
        # Select trajectories of all length from one particular prep state
        selected_prep_idcs_all_t = np.where(m.validation_features[:, 0, num_features + p] == 1)[0]
        # Select trajectories of the maximum length for one particular prep state
        selected_prep_idcs = np.intersect1d(selected_prep_idcs_all_t, last_time_idcs)

        # Plot: how well does the RNN classify the initial state.
        fig = plt.figure()
        for jj in range(num_prep_states):
            plt.plot(time_axis * 1e6, np.mean(prep_state_characterization[selected_prep_idcs], axis=0)[:, jj],
                     label=f"RNN predicts {prep_states[jj]}")
        plt.title(f"Preparation fidelity for prep {prep_states[p]}")
        plt.ylabel("Classification probability")
        plt.xlabel(f"Weak measurement time ({chr(956)}s)")
        plt.xlim(np.min(time_axis * 1e6), np.max(time_axis * 1e6))
        plt.legend(loc=0, frameon=False)
        fig.savefig(os.path.join(m.savepath, f"000_prep_state_class_{prep_states[p]}.png"), **settings['figure_options'])

        if n_levels == 2:
            bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                                      xyz_pred[selected_prep_idcs, :, 0],
                                                      xyz_pred[selected_prep_idcs, :, 1],
                                                      xyz_pred[selected_prep_idcs, :, 2],
                                                      bin_min=-1, bin_max=1)

            # Note: expX is now a 2d array with shape (num_prep_states, num_time_steps)
            # So, each row in expX, expY and expZ corresponds to a different prep state.
            fig = plot_qubit_histogram(time_axis, xyz_pred[selected_prep_idcs, :, 0],
                                       xyz_pred[selected_prep_idcs, :, 1], xyz_pred[selected_prep_idcs, :, 2],
                                       Tm, expX[p, :], expY[p, :], expZ[p, :])
        elif n_levels == 3:
            xyz_pred = y_pred_probabilities
            print(xyz_pred.shape)
            bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                                      xyz_pred[selected_prep_idcs, :, 0],
                                                      xyz_pred[selected_prep_idcs, :, 1],
                                                      xyz_pred[selected_prep_idcs, :, 2],
                                                      bin_min=0, bin_max=1)

            fig = plot_qutrit_histogram(time_axis, xyz_pred[selected_prep_idcs, :, 0],
                                        xyz_pred[selected_prep_idcs, :, 1], xyz_pred[selected_prep_idcs, :, 2],
                                        Tm, Pg[p, :], Pe[p, :], Pf[p, :])

        if m.savepath is not None:
            fig.savefig(os.path.join(m.savepath, f"000_histogram_prep_{prep_states[p]}.png"), **settings['figure_options'])

        # Plot the qubit verification for each prep state separately
        if n_levels == 2:
            fig = plot_qubit_verification(y_pred_probabilities[selected_prep_idcs_all_t, ...],
                                          m.validation_labels[selected_prep_idcs_all_t, :, num_prep_states:])
        elif n_levels == 3:
            fig = plot_qutrit_verification(y_pred_probabilities[selected_prep_idcs_all_t, ...],
                                           m.validation_labels[selected_prep_idcs_all_t, :, num_prep_states:])

        if m.savepath is not None:
            fig.savefig(os.path.join(m.savepath, f"000_verification_prep_{prep_states[p]}.png"), **settings['figure_options'])

        m.save_trajectories(time_axis, xyz_pred[selected_prep_idcs_all_t, ...],
                            valid_time_series_lengths[selected_prep_idcs_all_t],
                            history, prep_label=prep_states[p])
else:
    # In the case of just a single prep state (backwards compatible). Make a histogram of the validation trajectories
    if n_levels == 2:
        xyz_pred = get_xyz(y_pred_probabilities)
        bins, histX, histY, histZ = get_histogram(time_axis * 1e6,
                                                  xyz_pred[last_time_idcs, :, 0],
                                                  xyz_pred[last_time_idcs, :, 1],
                                                  xyz_pred[last_time_idcs, :, 2],
                                                  bin_min=-1, bin_max=1)

        fig = plot_qubit_histogram(time_axis, xyz_pred[last_time_idcs, :, 0],
                                   xyz_pred[last_time_idcs, :, 1], xyz_pred[last_time_idcs, :, 2],
                                   Tm, expX[0, :], expY[0, :], expZ[0, :])
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

    if n_levels == 2:
        fig = plot_qubit_verification(y_pred_probabilities, m.validation_labels)
    elif n_levels == 3:
        fig = plot_qutrit_verification(y_pred_probabilities, m.validation_labels)

    if m.savepath is not None:
        fig.savefig(os.path.join(m.savepath, "000_verification_on_validation_data.png"), **settings['figure_options'])

    m.save_trajectories(time_axis, xyz_pred, valid_time_series_lengths, history, prep_label=prep_states[0])

# Plot the qubit verification averaged over all prep states.
if n_levels == 2:
    if num_prep_states > 1:
        fig = plot_qubit_verification(y_pred_probabilities, m.validation_labels[..., num_prep_states:])
    else:
        fig = plot_qubit_verification(y_pred_probabilities, m.validation_labels)
elif n_levels == 3:
    if num_prep_states > 1:
        fig = plot_qutrit_verification(y_pred_probabilities, m.validation_labels[..., num_prep_states:])
    else:
        fig = plot_qutrit_verification(y_pred_probabilities, m.validation_labels)

if m.savepath is not None:
    fig.savefig(os.path.join(m.savepath, f"000_verification_all_prep_states.png"), **settings['figure_options'])

plt.close('all')

console.print(f"Training finished. Results saved in {m.savepath}", style="bold red")