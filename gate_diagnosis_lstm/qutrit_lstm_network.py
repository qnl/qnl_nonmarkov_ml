import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from callbacks import TrainingPlot, LossTracker
from matplotlib import pyplot as plt
from matplotlib import colors
import os, time, h5py
from scipy.optimize import curve_fit
from utils import x_color, y_color, z_color, save_options, qubit_prep_dict, qutrit_prep_dict

cmap = plt.get_cmap('Accent')
zero_color, one_color, two_color = [cmap.colors[z] for z in range(3)]

def get_trajectories_within_window(predictions, target_value, RO_results, n_levels, pass_window=0.025, verbose=True):
    if n_levels == 2:
        # Select traces where the final index is within Z +/- the pass_window
        passed_idcs = np.where(np.abs(predictions - target_value) < pass_window)[0]
        N_verification_trajs = np.shape(predictions)[0]

        if verbose:
            print(f"Post-selecting trajectories with target = {target_value:.3f} +/- {pass_window:.3f}")
            print(
                f"{len(passed_idcs)} trajectories left after post-selection ({len(passed_idcs) / N_verification_trajs * 100:.1f}% pass rate)")

        verification_strong_RO = RO_results[passed_idcs]
        avg_verification_value = 1 - 2 * np.mean(verification_strong_RO)
        passed_RO_results = RO_results[passed_idcs]
    elif n_levels == 3:
        for ro_result in range(3):
            # Select traces where the final index is within target Pg/Pe/Pf +/- the pass_window
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
    N = len(strong_ro_results)
    p = np.sum(strong_ro_results == readout_value) / N
    return np.sqrt(p * (1-p) / N)


def pairwise_softmax(y_pred, n_levels):
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

def pad_labels(labels, sequence_lengths, reps_per_timestep, mask_value):
    # n_labels depends on the one_hot encoding type. For example, for qubit data with three measurement axes,
    # the length of one_hot = 2 * 3 = 6. However, for qutrit data with a single measurement axis, n_labels = 3 * 1 = 3
    batch_size, n_labels = np.shape(labels)
    sequence_length = np.max(sequence_lengths)
    # Start with a filled array with mask values.
    padded_labels = mask_value * np.ones((batch_size, sequence_length, n_labels))
    cum_reps = 0
    # Then replace the mask value with the appropriate label at the right time step.
    for ts, r in zip(sequence_lengths, reps_per_timestep):
        padded_labels[cum_reps:cum_reps + r, ts - 1, :] = labels[cum_reps:cum_reps + r, :]
        cum_reps += r

    return padded_labels

def get_xyz(probabilities):
    return 2 * probabilities[:, :, ::2] - 1

class MultiTimeStep():
    def __init__(self, validation_features, validation_labels, prep_states, n_levels,
                 data_points_for_prep_state, prep_state_from_ro=False, lstm_neurons=32, mini_batch_size=500,
                 epochs_per_annealing=10, annealing_steps=1, savepath=None, experiment_name='', **kwargs):

        tf.keras.backend.set_floatx('float32')  # Set the standard float format to float32
        self.lstm_neurons = lstm_neurons
        self.mini_batch_size = mini_batch_size
        _, self.sequence_length, self.num_features = np.shape(validation_features)
        self.init_dropout = 0.2
        self.reduce_dropout_rate_after = 10
        self.dropout_epoch_constant = 15
        self.init_learning_rate = 0.001
        self.reduce_learning_rate_after = 6
        self.learning_rate_epoch_constant = 10
        self.epochs_per_annealing = epochs_per_annealing
        self.annealing_steps = annealing_steps
        self.total_epochs = self.annealing_steps * self.epochs_per_annealing
        self.l2_regularization = 0.0
        self.validation_features = validation_features
        self.validation_labels = validation_labels
        self.n_levels = n_levels
        self.num_measurement_axes = 3 if n_levels == 2 else 1
        # List of prep states in order as they are encoded in the first N columns of the labels
        self.prep_states = prep_states
        # Apply the initial state constraint for the following timestep
        self.data_points_for_prep_state = data_points_for_prep_state

        if n_levels == 2:
            self.expX = kwargs['expX']
            self.expY = kwargs['expY']
            self.expZ = kwargs['expZ']
            self.avgd_strong_ro_results = {'expX': self.expX,
                                           'expY': self.expY,
                                           'expZ': self.expZ}
            self.num_prep_states = np.shape(self.expX)[0]

            # For calculation of the cost function. init_x, init_y and init_z are arrays of shape (num_prep_states, 2)
            if prep_state_from_ro:
                self.init_x = np.array([[0.5 * (1 + self.expX[p, 0]),
                                         0.5 * (1 - self.expX[p, 0])] for p in range(self.num_prep_states)])
                self.init_y = np.array([[0.5 * (1 + self.expY[p, 0]),
                                         0.5 * (1 - self.expY[p, 0])] for p in range(self.num_prep_states)])
                self.init_z = np.array([[0.5 * (1 + self.expZ[p, 0]),
                                         0.5 * (1 - self.expZ[p, 0])] for p in range(self.num_prep_states)])
                print("Prep states inferred from strong readout results:")
                for p, ps in enumerate(prep_states):
                    print(f"Prep state {ps} - (Px, Py, Pz) = ({self.init_x[p, 1]:.3f}, {self.init_x[p, 1]:.3f}, {self.init_x[p, 1]:.3f})")
            else:
                self.init_x = np.array([qubit_prep_dict[key]['prep_x'] for key in prep_states])
                self.init_y = np.array([qubit_prep_dict[key]['prep_y'] for key in prep_states])
                self.init_z = np.array([qubit_prep_dict[key]['prep_z'] for key in prep_states])

        elif n_levels == 3:
            self.Pg = kwargs['Pg']
            self.Pe = kwargs['Pe']
            self.Pf = kwargs['Pf']
            self.avgd_strong_ro_results = {'Pg': self.Pg,
                                           'Pe': self.Pe,
                                           'Pf': self.Pf}
            self.num_prep_states = np.shape(self.Pg)[0]

        self.mask_value = -1.0
        # self.prep_state_encoding(n_levels=n_levels, prep_state=prep_state)

        if savepath is not None:
            subfolder = time.strftime(f'%y%m%d_%H%M%S_{experiment_name}')
            if not (os.path.exists(os.path.join(savepath, subfolder))):
                os.makedirs(os.path.join(savepath, subfolder))

            self.savepath = os.path.join(savepath, subfolder)
        else:
            self.savepath = None

    def prep_state_encoding(self, n_levels, prep_state):
        # Rename equivalent prep states
        if prep_state == "g":
            prep_state = "+Z"
        if prep_state == "e":
            prep_state = "-Z"

        if n_levels == 2:
            if prep_state in qubit_prep_dict.keys():
                self.prep_x = qubit_prep_dict[prep_state]["prep_x"]
                self.prep_y = qubit_prep_dict[prep_state]["prep_y"]
                self.prep_z = qubit_prep_dict[prep_state]["prep_z"]
            elif prep_state is None:
                # Determine the prep state based on the strong readout results at the first timestep
                R0 = np.sqrt(self.expX[0] ** 2 + self.expY[0] ** 2 + self.expZ[0] ** 2)
                X0 = self.expX[0] if R0 <= 1.0 else self.expX[0] / R0
                Y0 = self.expY[0] if R0 <= 1.0 else self.expY[0] / R0
                Z0 = self.expZ[0] if R0 <= 1.0 else self.expZ[0] / R0
                px = 0.5 * (1 + X0)
                py = 0.5 * (1 + Y0)
                pz = 0.5 * (1 + Z0)
                self.prep_x = [px, 1 - px]
                self.prep_y = [py, 1 - py]
                self.prep_z = [pz, 1 - pz]
                print(f"Assumed prep state from strong RO results is: ({X0:.3f}, {Y0:.3f}, {Z0:.3f})")
                print(f"Purity of measured prep state was {R0:.4f}.")
                if R0 >= 1.0:
                    print(f"Prep state has been automatically scaled such that purity = 1.")
            else:
                raise ValueError(f"Prep state {prep_state} is not supported for qubits")
        elif n_levels == 3:
            if prep_state in qutrit_prep_dict.keys():
                self.prep_z = qutrit_prep_dict[prep_state]["prep_z"]
            elif prep_state is None:
                self.prep_z = [self.Pg[0], self.Pe[0], self.Pf[0]]
            else:
                raise ValueError(f"Prep state {prep_state} is not supported for qutrits")
            self.prep_x = None
            self.prep_y = None
        return

    def build_model(self):
        self.model = tf.keras.Sequential()

        # Mask it
        self.model.add(layers.Masking(mask_value=self.mask_value,
                                      input_shape=(self.sequence_length, self.num_features)))

        # Add an LSTM layer
        self.model.add(layers.LSTM(self.lstm_neurons,
                                   batch_input_shape=(self.sequence_length, self.num_features),
                                   dropout=0.0, # Dropout of the hidden state
                                   stateful=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize input weights
                                   recurrent_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize recurrent weights
                                   bias_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize bias weights
                                   return_sequences=True))

        # Add a dropout layer
        # self.model.add(layers.TimeDistributed(layers.Dropout(self.init_dropout)))

        # Cast to the output
        self.model.add(layers.TimeDistributed(layers.Dense(self.num_prep_states + self.n_levels * self.num_measurement_axes)))

        self.model.summary()

    def compile_model(self, optimizer='adam'):
        if self.n_levels == 2:
            if self.num_prep_states > 1:
                self.model.compile(loss=self.qubit_multi_prep_loss_function, optimizer=optimizer,
                                   metrics=[self.masked_multi_prep_accuracy])
            else:
                self.model.compile(loss=self.qubit_loss_function, optimizer=optimizer, metrics=[self.masked_accuracy])
        if self.n_levels == 3:
            self.model.compile(loss=self.qutrit_loss_function, optimizer=optimizer, metrics=[self.masked_accuracy])

    def fit_model(self, training_features, training_labels, verbose_level=1):
        LRScheduler = tf.keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.savepath, histogram_freq=1)
        # loss_tracker_callback = LossTracker(self.validation_features,
        #                                     self.validation_labels,
        #                                     self.n_levels,
        #                                     mask_value=self.mask_value,
        #                                     savepath=self.savepath,
        #                                     prep_x=self.prep_x, prep_y=self.prep_y, prep_z=self.prep_z)
        history = self.model.fit(training_features, training_labels, epochs=self.total_epochs,
                                 batch_size=self.mini_batch_size,
                                 validation_data=(self.validation_features, self.validation_labels),
                                 verbose=verbose_level, shuffle=True,
                                 callbacks=[TrainingPlot(),
                                            LRScheduler])
                                            # loss_tracker_callback,
                                            # ValidationPlot(self.validation_features,
                                            #                self.validation_labels, self.n_levels, self.mini_batch_size,
                                            #                self.savepath, **self.avgd_strong_ro_results),
                                            # DropOutScheduler(self.dropout_schedule)])
        return history

    def fit_model_with_generator(self, dataset, epochs, verbose_level=1):
        LRScheduler = tf.keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.savepath, histogram_freq=1)
        # loss_tracker_callback = LossTracker(self.validation_features,
        #                                     self.validation_labels,
        #                                     self.n_levels,
        #                                     mask_value=self.mask_value,
        #                                     savepath=self.savepath,
        #                                     prep_x=self.prep_x, prep_y=self.prep_y, prep_z=self.prep_z)

        history = self.model.fit(x=dataset, epochs=epochs,
                                 steps_per_epoch=np.int(self.validation_features.shape[0] / self.mini_batch_size),
                                 batch_size=self.mini_batch_size,
                                 validation_data=(self.validation_features, self.validation_labels),
                                 verbose=verbose_level, shuffle=True,
                                 callbacks=[TrainingPlot(),
                                            LRScheduler])
                                            # loss_tracker_callback,
                                            # ValidationPlot(self.validation_features,
                                            #                self.validation_labels, self.n_levels, self.mini_batch_size,
                                            #                self.savepath, **self.avgd_strong_ro_results),
                                            # DropOutScheduler(self.dropout_schedule)])
        return history

    def learning_rate_schedule(self, epoch):
        epoch = tf.math.floormod(epoch, self.epochs_per_annealing)
        if epoch < self.reduce_learning_rate_after:
            return self.init_learning_rate
        else:
            # Drops an order of magnitude every self.learning_rate_epoch_constant epochs
            return self.init_learning_rate * tf.math.exp((self.reduce_learning_rate_after - epoch) / self.learning_rate_epoch_constant)

    def dropout_schedule(self, epoch):
        if epoch < self.reduce_dropout_rate_after:
            return self.init_dropout
        else:
            # Starts at 0.2 and Drops an order of magnitude every 50 epochs
            return self.init_dropout * tf.math.exp((self.reduce_dropout_rate_after - epoch) / self.dropout_epoch_constant)

    def get_expected_accuracy(self, verbose=True):
        assert self.n_levels == 2, "Expected accuracy is only defined for n_levels = 2 at the moment."
        expected_accuracy = np.mean([0.5 * np.max([1 + self.expX, 1 - self.expX]),
                                     0.5 * np.max([1 + self.expY, 1 - self.expY]),
                                     0.5 * np.max([1 + self.expZ, 1 - self.expZ])])
        if verbose:
            print("Expected accuracy should converge to", expected_accuracy)
        return expected_accuracy

    def qubit_multi_prep_loss_function(self, y_true, y_pred):
        # Extract initial state information
        y_true_prep_encoding = y_true[..., :self.num_prep_states]
        y_true_ro_results = y_true[..., self.num_prep_states:]
        y_pred_prep_encoding = y_pred[..., :self.num_prep_states]
        y_pred_ro_results = y_pred[..., self.num_prep_states:]

        # Processing on the readout labels
        batch_size = K.cast(K.shape(y_true_ro_results)[0], K.floatx())
        # Finds out where a readout is available
        mask = K.cast(K.not_equal(y_true_ro_results, self.mask_value), K.floatx())
        # First do a softmax (when from_logits = True) and then calculate the cross-entropy: CE_i = -log(prob_i)
        # where prob_i is the predicted probability for y_true_i = 1.0
        # Note: this assumes that each voltage record has exactly 1 label associated with it.
        pred_logits = K.reshape(tf.boolean_mask(y_pred_ro_results, mask), (batch_size, 2))
        true_probs = K.reshape(tf.boolean_mask(y_true_ro_results, mask), (batch_size, 2))
        CE = K.categorical_crossentropy(true_probs, pred_logits, from_logits=True)
        L_readout = K.sum(CE) / batch_size

        # Penalize deviation from the known initial state at the first time step
        # Do a softmax to get the predicted probabilities
        mask = K.cast(K.not_equal(y_true_prep_encoding, self.mask_value), K.floatx())
        pred_encoding = K.reshape(tf.boolean_mask(y_pred_prep_encoding, mask), (batch_size, self.num_prep_states))
        true_encoding = K.reshape(tf.boolean_mask(y_true_prep_encoding, mask), (batch_size, self.num_prep_states))
        CE = K.categorical_crossentropy(true_encoding, pred_encoding, from_logits=True)
        L_prep_encoding = K.sum(CE) / batch_size

        init_x = tf.linalg.matmul(true_encoding, tf.constant(self.init_x, dtype=K.floatx()))
        init_y = tf.linalg.matmul(true_encoding, tf.constant(self.init_y, dtype=K.floatx()))
        init_z = tf.linalg.matmul(true_encoding, tf.constant(self.init_z, dtype=K.floatx()))

        # init_x_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, :2])
        # init_y_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 2:4])
        # init_z_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 4:])

        init_x_pred = K.softmax(y_pred_ro_results[:, 0, :2])
        init_y_pred = K.softmax(y_pred_ro_results[:, 0, 2:4])
        init_z_pred = K.softmax(y_pred_ro_results[:, 0, 4:])

        L_init_state = K.sqrt(K.square(init_x - init_x_pred)[0] + \
                              K.square(init_y - init_y_pred)[0] + \
                              K.square(init_z - init_z_pred)[0])

        # Constrain the purity of the qubit state < 1
        X_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 0:2], axis=-1)[:, :, 1]
        Y_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 2:4], axis=-1)[:, :, 1]
        Z_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 4:6], axis=-1)[:, :, 1]
        L_outside_sphere = K.relu(K.sqrt(K.square(X_all_t) + K.square(Y_all_t) + K.square(Z_all_t)), threshold=1.0)

        # Force the state of average readout results to be equal to the strong readout results.
        lagrange_1 = tf.constant(1.0, dtype=K.floatx()) # Readout cross-entropy
        lagrange_2 = tf.constant(0.0, dtype=K.floatx()) # Initial state
        lagrange_3 = tf.constant(0.0, dtype=K.floatx()) # Purity constraint
        lagrange_4 = tf.constant(0.1, dtype=K.floatx()) # Prep state encoding

        return lagrange_1 * L_readout + lagrange_2 * L_init_state[0] + lagrange_3 * K.mean(L_outside_sphere) + lagrange_4 * L_prep_encoding

    def masked_multi_prep_accuracy(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        # Finds out where a readout is available, mask has shape (batch_size, max_seq_length, 6) for qubits
        mask = K.not_equal(y_true[..., self.num_prep_states:], self.mask_value)
        # Selects logits with a readout, pred_logits has shape (batch_size, 2) for qubits
        pred_logits = K.reshape(tf.boolean_mask(y_pred[..., self.num_prep_states:], mask), (batch_size, self.n_levels))
        # Do a softmax to get the predicted probabilities, pred_probs has shape (batch_size, 2) for qubits
        pred_probs = K.softmax(pred_logits)
        # True readout results are [0, 1] or [1, 0] for qubits or [0, 0, 1], [1, 0, 0] or [0, 1, 0] for qutrits
        # Note: this assumes that each voltage record has exactly 1 label associated with it.
        true_probs = K.reshape(tf.boolean_mask(y_true[..., self.num_prep_states:], mask), (batch_size, self.n_levels))
        # Categorical accuracy returns a 1 when |pred_probs - true_probs| < 0.5 and else a 0.
        well_predicted = tf.keras.metrics.categorical_accuracy(true_probs, pred_probs)
        return tf.reduce_mean(well_predicted)

    def qubit_loss_function(self, y_true, y_pred):
        batch_size = K.cast(K.shape(y_true)[0], K.floatx())
        # Finds out where a readout is available
        mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
        # First do a softmax (when from_logits = True) and then calculate the cross-entropy: CE_i = -log(prob_i)
        # where prob_i is the predicted probability for y_true_i = 1.0
        # Note: this assumes that each voltage record has exactly 1 label associated with it.
        pred_logits = K.reshape(tf.boolean_mask(y_pred, mask), (batch_size, 2))
        true_probs = K.reshape(tf.boolean_mask(y_true, mask), (batch_size, 2))
        CE = K.categorical_crossentropy(true_probs, pred_logits, from_logits=True)
        L_readout = K.sum(CE) / batch_size

        # Penalize deviation from the known initial state at the first time step
        # Do a softmax to get the predicted probabilities
        init_x = tf.repeat(tf.constant([self.prep_x], dtype=K.floatx()), repeats=K.cast(batch_size, "int32"), axis=0)
        init_x_pred = K.softmax(y_pred[:, 0, 0:2])
        # todo: pull the 0 from the number of samples for the first timestep

        init_y = tf.repeat(tf.constant([self.prep_y], dtype=K.floatx()), repeats=K.cast(batch_size, "int32"), axis=0)
        init_y_pred = K.softmax(y_pred[:, 0, 2:4])

        init_z = tf.repeat(tf.constant([self.prep_z], dtype=K.floatx()), repeats=K.cast(batch_size, "int32"), axis=0)
        init_z_pred = K.softmax(y_pred[:, 0, 4:6])

        L_init_state = K.sqrt(K.square(init_x - init_x_pred)[0] + \
                              K.square(init_y - init_y_pred)[0] + \
                              K.square(init_z - init_z_pred)[0])

        # NEW
        X_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 0:2], axis=-1)[:, :, 1]
        Y_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 2:4], axis=-1)[:, :, 1]
        Z_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 4:6], axis=-1)[:, :, 1]
        L_outside_sphere = K.relu(K.sqrt(K.square(X_all_t) + K.square(Y_all_t) + K.square(Z_all_t)), threshold=1.0)

        # Force the state of average readout results to be equal to the strong readout results.
        lagrange_1 = tf.constant(1.0, dtype=K.floatx())
        lagrange_2 = tf.constant(0.5, dtype=K.floatx())
        lagrange_3 = tf.constant(0.1, dtype=K.floatx())

        return lagrange_1 * L_readout + lagrange_2 * L_init_state[0] + lagrange_3 * K.mean(L_outside_sphere)

    def qutrit_loss_function(self, y_true, y_pred):
        batch_size = K.cast(K.shape(y_true)[0], K.floatx())
        # Finds out where a readout is available
        mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())
        # First do a softmax (when from_logits = True) and then calculate the cross-entropy: CE_i = -log(prob_i)
        # where prob_i is the predicted probability for y_true_i = 1.0
        # Note: this assumes that each voltage record has exactly 1 label associated with it.
        pred_logits = K.reshape(tf.boolean_mask(y_pred, mask), (batch_size, 3))
        true_probs = K.reshape(tf.boolean_mask(y_true, mask), (batch_size, 3))
        CE = K.categorical_crossentropy(true_probs, pred_logits, from_logits=True)
        L_readout = K.sum(CE) / batch_size

        # Penalize deviation from the known initial state at the first time step
        # Do a softmax to get the predicted probabilities
        init_gef = tf.repeat(tf.constant([self.prep_z], dtype=K.floatx()), repeats=K.cast(batch_size, "int32"), axis=0)
        init_gef_pred = K.softmax(y_pred[:, 0, :])
        L_init_state = K.sum(K.abs(init_gef - init_gef_pred)) / batch_size

        # Force the state of average readout results to be equal to the strong readout results.
        lagrange_1 = tf.constant(1.0, dtype=K.floatx())
        lagrange_2 = tf.constant(0.5, dtype=K.floatx())

        return lagrange_1 * L_readout + lagrange_2 * L_init_state

    def masked_accuracy(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        # Finds out where a readout is available
        mask = K.not_equal(y_true, self.mask_value)
        # Selects logits with a readout
        pred_logits = K.reshape(tf.boolean_mask(y_pred, mask), (batch_size, self.n_levels))
        # Do a softmax to get the predicted probabilities
        pred_probs = K.softmax(pred_logits)
        # True readout results are [0, 1] or [1, 0] for qubits or [0, 0, 1], [1, 0, 0] or [0, 1, 0] for qutrits
        # Note: this assumes that each voltage record has exactly 1 label associated with it.
        true_probs = K.reshape(tf.boolean_mask(y_true, mask), (batch_size, self.n_levels))
        # Categorical accuracy returns a 1 when |pred_probs - true_probs| < 0.5 and else a 0.
        well_predicted = tf.keras.metrics.categorical_accuracy(true_probs, pred_probs)
        return tf.reduce_mean(well_predicted)

    def get_predictions(self, features):
        for k in range(int(np.shape(features)[0] / self.mini_batch_size)):
            y_pred = self.model(features[k * self.mini_batch_size:((k + 1) * self.mini_batch_size)]).numpy()
            if k == 0:
                xyz_pred = get_xyz(pairwise_softmax(y_pred, self.n_levels))
            else:
                xyz_pred = np.vstack((xyz_pred, get_xyz(pairwise_softmax(y_pred, self.n_levels))))

        return xyz_pred

    def plot_history(self, history):
        # plot history
        fig = plt.figure()
        plt.plot(history.history['loss'], label=f"training loss (final: {history.history['loss'][-1]:.4f})")
        plt.plot(history.history['val_loss'],
                 label=f"validation loss (final epoch: {history.history['val_loss'][-1]:.4f})")
        plt.xlabel("Epochs")
        plt.ylabel("Categorical crossentropy loss (a.u.)")
        plt.xlim(0, len(history.history['loss']))
        plt.legend(loc=0, frameon=False)

        if self.savepath is not None:
            fig.savefig(os.path.join(self.savepath, "training_history_loss_absolute.png"), **save_options)

        # plot history
        fig = plt.figure()
        min_value = np.min(history.history['loss'])
        plt.plot(history.history['loss'] - min_value, label='training loss')
        min_value = np.min(history.history['val_loss'])
        plt.plot(history.history['val_loss'] - min_value, label='validation loss')
        plt.xlabel("Epochs")
        plt.yscale('log')
        plt.ylabel("Categorical crossentropy loss (a.u.)")
        plt.xlim(0, len(history.history['loss']))
        plt.legend(loc=0, frameon=False)

        if self.savepath is not None:
            fig.savefig(os.path.join(self.savepath, "training_history_loss_relative.png"), **save_options)

        fig = plt.figure()
        if self.num_prep_states > 1:
            plt.plot(history.history['masked_multi_prep_accuracy'],
                     label=f"training accuracy (final epoch: {history.history['masked_multi_prep_accuracy'][-1]:.4f})")
            plt.plot(history.history['val_masked_multi_prep_accuracy'],
                     label=f"validation accuracy (final epoch: {history.history['val_masked_multi_prep_accuracy'][-1]:.4f})")
        else:
            plt.plot(history.history['masked_accuracy'],
                     label=f"training accuracy (final epoch: {history.history['masked_accuracy'][-1]:.4f})")
            plt.plot(history.history['val_masked_accuracy'],
                     label=f"validation accuracy (final epoch: {history.history['val_masked_accuracy'][-1]:.4f})")
        plt.ylabel("Accuracy (categorical)")
        plt.xlabel("Epochs")
        plt.xlim(0, len(history.history['loss']))
        plt.legend(loc=0, frameon=False)
        # plt.show()

        if self.savepath is not None:
            fig.savefig(os.path.join(self.savepath, "training_history_accuracy.png"), **save_options)

    def save_trajectories(self, time, predictions, indices, history, prep_label=None):
        file_exists = os.path.exists(os.path.join(self.savepath, "trajectories.h5"))

        with h5py.File(os.path.join(self.savepath, "trajectories.h5"), 'a') as f:
            # Compatible with multiple prep states
            if not file_exists:
                f.create_dataset("t", data=time)

                epochs = np.arange(1, 1 + len(history.history['loss']))
                f.create_dataset(f"training/epochs", data=epochs)
                # f.create_dataset(f"training/loss_components", data=self.model.losses)
                if self.num_prep_states > 1:
                    history_keys = ["loss", "val_loss", "masked_multi_prep_accuracy", "val_masked_multi_prep_accuracy"]
                else:
                    history_keys = ["loss", "val_loss", "masked_accuracy", "val_masked_accuracy"]
                for key in history_keys:
                    f.create_dataset(f"training/{key}", data=history.history[key])

                learning_rate = np.array([self.learning_rate_schedule(e) for e in epochs])
                f.create_dataset(f"training/learning_rate", data=learning_rate)

            unique_indices = np.unique(indices)

            # Divide the trajectories according to their weak measurement length
            for k in range(len(unique_indices)):
                select = np.where(indices == unique_indices[k])[0]
                if prep_label is not None:
                    f.create_dataset(f"prep_{prep_label}/predictions_{unique_indices[k]}",
                                     data=predictions[select, :unique_indices[k], :])
                else:
                    f.create_dataset(f"predictions_{unique_indices[k]}",
                                     data=predictions[select, :unique_indices[k], :])

class DropOutScheduler(tf.keras.callbacks.Callback):
    def __init__(self, dropout_schedule):
        self.dropout_schedule = dropout_schedule

    def on_epoch_end(self, epoch, logs={}):
        try:
            self.model.layers[1].dropout = self.dropout_schedule(epoch)
        except:
            print("Dropout scheduling failed.")

# class ValidationPlot(tf.keras.callbacks.Callback):
#     def __init__(self, validation_features, validation_labels, n_levels, mini_batch_size, savepath, **kwargs):
#         self.validation_features = validation_features
#         self.validation_labels = validation_labels
#         self.mini_batch_size = mini_batch_size
#         self.savepath = savepath
#         self.n_levels = n_levels
#         if self.n_levels == 2:
#             self.expX = kwargs['expX']
#             self.expY = kwargs['expY']
#             self.expZ = kwargs['expZ']
#         elif self.n_levels == 3:
#             self.Pg = kwargs['Pg']
#             self.Pe = kwargs['Pe']
#             self.Pf = kwargs['Pf']
#
#     def on_epoch_end(self, epoch, logs={}):
#         pass
        # if not (epoch % 5):
        #     max_size = int(8e4)
        #     y_pred = self.model.predict(self.validation_features[:max_size, ...])
        #     y_pred_probabilities = pairwise_softmax(y_pred)
        #     fig = plot_verification(y_pred_probabilities, self.validation_labels[:max_size, ...])
        #
        #     if self.savepath is not None:
        #         fig.savefig(os.path.join(self.savepath, "xyz_validation_epoch_%03d.png" % epoch), dpi=200)
        #     plt.close(fig)


def plot_qubit_verification(predicted_labels, verification_labels):
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
        x_pred.append(avg_ver)
        x_errs.append(get_error(ro_res))
        x_pred_trajs.append(len(passed_idcs))

        passed_idcs, avg_ver, ro_res = get_trajectories_within_window(xyz_pred[y_measurements[0], y_measurements[1], 1],
                                                                      ty, y_RO, n_levels=2, pass_window=epsilon,
                                                                      verbose=False)
        y_pred.append(avg_ver)
        y_errs.append(get_error(ro_res))
        y_pred_trajs.append(len(passed_idcs))

        passed_idcs, avg_ver, ro_res = get_trajectories_within_window(xyz_pred[z_measurements[0], z_measurements[1], 2],
                                                                      tz, z_RO, n_levels=2, pass_window=epsilon,
                                                                      verbose=False)
        z_pred.append(avg_ver)
        z_errs.append(get_error(ro_res))
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

def _simple_line(x, *p):
    slope, offset = p
    return slope * x + offset

def weighted_line_fit(xdata, ydata, yerr, guess_slope, guess_offset):
    try:
        popt, pcov = curve_fit(_simple_line, xdata, ydata, p0=[guess_slope, guess_offset],
                               sigma=yerr, absolute_sigma=True, check_finite=True,
                               bounds=(-np.inf, np.inf), method=None, jac=None)
    except RuntimeError:
        popt, pcov = curve_fit(_simple_line, xdata, ydata, p0=[guess_slope, guess_offset])

    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def get_histogram(weak_meas_times, X, Y, Z, n_bins=101, bin_min=-1, bin_max=+1):
    sequence_length = len(weak_meas_times)

    histX = np.zeros((n_bins, sequence_length))
    histY = np.zeros((n_bins, sequence_length))
    histZ = np.zeros((n_bins, sequence_length))

    for b in range(sequence_length):
        histX[:, b], bins = np.histogram(X[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))
        histY[:, b], bins = np.histogram(Y[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))
        histZ[:, b], bins = np.histogram(Z[:, b], bins=np.linspace(bin_min, bin_max, n_bins + 1))

    return bins, histX, histY, histZ


def plot_qubit_histogram(weak_meas_times, X, Y, Z, tomography_times, expX, expY, expZ, n_bins=101):
    bins, histX, histY, histZ = get_histogram(weak_meas_times, X, Y, Z, n_bins=n_bins)

    cmap = plt.cm.hot
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histZ, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histZ.max()))
    plt.plot(tomography_times * 1e6, expZ, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Z$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histY, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histY.max()))
    plt.plot(tomography_times * 1e6, expY, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Y$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histX, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histX.max()))
    plt.plot(tomography_times * 1e6, expX, '.-', color='gray')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$X$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()
    return fig


def plot_qutrit_histogram(weak_meas_times, Pg_trajectories, Pe_trajectories, Pf_trajectories,
                          tomography_times, Pg, Pe, Pf, n_bins=101):
    bins, histPg, histPe, histPf = get_histogram(weak_meas_times, Pg_trajectories, Pe_trajectories,
                                                 Pf_trajectories, n_bins=n_bins, bin_min=0, bin_max=1)

    cmap = plt.cm.hot
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPg, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPg.max()))
    plt.plot(tomography_times * 1e6, Pg, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_g$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPe, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPe.max()))
    plt.plot(tomography_times * 1e6, Pe, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_e$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histPf, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histPf.max()))
    plt.plot(tomography_times * 1e6, Pf, '-', color='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$P_f$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()
    return fig


def plot_individual_trajs(weak_meas_times, X, Y, Z, traj_indices = np.arange(4), n_bins=101):
    bins, histX, histY, histZ = get_histogram(weak_meas_times, X, Y, Z, n_bins=n_bins)

    traj_cols = ['r', 'darkorange', 'yellow', 'forestgreen']
    cmap = plt.cm.Greys_r
    cmap.set_bad(color='k')

    fig = plt.figure(figsize=(6., 8))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histZ, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histZ.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, Z[k, :], color=traj_cols[l])

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Z$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 2)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histY, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histY.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, Y[k, :], color=traj_cols[l])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$Y$")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    plt.subplot(3, 1, 3)
    plt.pcolormesh(weak_meas_times * 1e6, bins, histX, cmap=cmap, norm=colors.LogNorm(vmin=5, vmax=histX.max()))
    for l, k in enumerate(traj_indices):
        plt.plot(weak_meas_times * 1e6, X[k, :], color=traj_cols[l])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(f"Probability (bin size {np.diff(bins)[0]:.2f})")
    plt.ylabel(r"$X$")
    plt.xlabel(f"Weak measurement time ({chr(956)}s)")
    plt.xlim(np.min(weak_meas_times * 1e6), np.max(weak_meas_times * 1e6))

    fig.tight_layout()

    return fig

def make_a_pie(time_series_lengths, title="", savepath=None):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = np.unique(time_series_lengths)
    sizes = [len(np.where(time_series_lengths == label)[0]) for label in labels]
    colors = plt.cm.viridis(np.arange(len(labels)) / (len(labels) - 1))

    fig = plt.figure()
    plt.title(title)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=colors)
    plt.gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if savepath is not None:
        fig.savefig(os.path.join(savepath, title.replace(" ", "_") + ".png"), **save_options)