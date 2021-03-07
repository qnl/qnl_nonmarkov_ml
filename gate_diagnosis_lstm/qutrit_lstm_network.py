import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from callbacks import TrainingPlot, LossTracker
from matplotlib import pyplot as plt
import os, time, h5py
from utils import save_options, qubit_prep_dict, qutrit_prep_dict
from verification import get_xyz, pairwise_softmax

cmap = plt.get_cmap('Accent')
zero_color, one_color, two_color = [cmap.colors[z] for z in range(3)]

class MultiTimeStep():
    def __init__(self, validation_features, validation_labels, prep_states, n_levels,
                 data_points_for_prep_state, prep_state_from_ro=False, lstm_neurons=32, mini_batch_size=500,
                 bidirectional=False, epochs_per_annealing=10, annealing_steps=1, savepath=None,
                 experiment_name='', **kwargs):

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
        self.bidirectional = bidirectional

        if n_levels == 2:
            self.expX = kwargs['expX']
            self.expY = kwargs['expY']
            self.expZ = kwargs['expZ']
            self.avgd_strong_ro_results = {'expX': self.expX,
                                           'expY': self.expY,
                                           'expZ': self.expZ}
            self.num_prep_states = np.shape(self.expX)[0]

            # For calculation of the cost function. init_x, init_y and init_z are arrays of shape (num_prep_states, 2)
            # Note: init_x, init_y and init_z are the probabilities, not the qubit coordinates x0, y0 and z0
            if prep_state_from_ro:
                self.init_x = np.array([[0.5 * (1 + self.expX[p, 0]),
                                         0.5 * (1 - self.expX[p, 0])] for p in range(self.num_prep_states)])
                self.init_y = np.array([[0.5 * (1 + self.expY[p, 0]),
                                         0.5 * (1 - self.expY[p, 0])] for p in range(self.num_prep_states)])
                self.init_z = np.array([[0.5 * (1 + self.expZ[p, 0]),
                                         0.5 * (1 - self.expZ[p, 0])] for p in range(self.num_prep_states)])
                print("Prep states inferred from strong readout results:")
                for p, ps in enumerate(prep_states):
                    print(f"Prep state {ps} - (Px, Py, Pz) = ({self.init_x[p, 1]:.3f}, {self.init_y[p, 1]:.3f}, {self.init_z[p, 1]:.3f})")
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

            self.init_x = None
            self.init_y = None
            if prep_state_from_ro:
                self.init_z = np.array([[self.Pg[k, 0], self.Pe[k, 0], self.Pf[k, 0]] for k in range(self.num_prep_states)])
                print("init_z: ", self.init_z)

        self.mask_value = -1.0
        # if self.num_prep_states == 1:
        #     self.prep_state_encoding(n_levels=n_levels, prep_state=self.prep_states[0])

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
        lstm_layer = layers.LSTM(self.lstm_neurons,
                                 batch_input_shape=(self.sequence_length, self.num_features),
                                 dropout=0.0, # Dropout of the hidden state
                                 stateful=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize input weights
                                 recurrent_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize recurrent weights
                                 bias_regularizer=tf.keras.regularizers.l2(self.l2_regularization), # regularize bias weights
                                 return_sequences=True)

        if self.bidirectional:
            self.model.add(layers.Bidirectional(lstm_layer, merge_mode='concat'))
        else:
            self.model.add(lstm_layer)

        # Add a dropout layer
        # self.model.add(layers.TimeDistributed(layers.Dropout(self.init_dropout)))

        # Cast to the output
        if self.num_prep_states > 1:
            self.model.add(layers.TimeDistributed(layers.Dense(self.num_prep_states + self.n_levels * self.num_measurement_axes)))
        else:
            # If there's just a single prep state, we don't need to use the prep state encoding.
            self.model.add(layers.TimeDistributed(layers.Dense(self.n_levels * self.num_measurement_axes)))

        self.model.summary()

    def compile_model(self, optimizer='adam'):
        if self.n_levels == 2:
            if self.num_prep_states > 1:
                self.model.compile(loss=self.qubit_multi_prep_loss_function, optimizer=optimizer,
                                   metrics=[self.masked_multi_prep_accuracy])
            else:
                self.model.compile(loss=self.qubit_loss_function, optimizer=optimizer, metrics=[self.masked_accuracy])
        if self.n_levels == 3:
            if self.num_prep_states > 1:
                self.model.compile(loss=self.qutrit_multi_prep_loss_function, optimizer=optimizer,
                                   metrics=[self.masked_multi_prep_accuracy])
            else:
                self.model.compile(loss=self.qutrit_loss_function, optimizer=optimizer, metrics=[self.masked_accuracy])

    def fit_model(self, training_features, training_labels, verbose_level=1):

        LRScheduler = tf.keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.savepath, histogram_freq=1)

        loss_tracker_callback = LossTracker(self.validation_features,
                                            self.validation_labels,
                                            self.n_levels,
                                            num_prep_states=self.num_prep_states,
                                            mask_value=self.mask_value,
                                            savepath=self.savepath,
                                            init_x=self.init_x, init_y=self.init_y, init_z=self.init_z)

        history = self.model.fit(training_features, training_labels, epochs=self.total_epochs,
                                 batch_size=self.mini_batch_size,
                                 validation_data=(self.validation_features, self.validation_labels),
                                 verbose=verbose_level, shuffle=True,
                                 callbacks=[TrainingPlot(),
                                            LRScheduler,
                                            loss_tracker_callback])
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

    def qutrit_multi_prep_loss_function(self, y_true, y_pred):
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
        pred_logits = K.reshape(tf.boolean_mask(y_pred_ro_results, mask), (batch_size, 3))
        true_probs = K.reshape(tf.boolean_mask(y_true_ro_results, mask), (batch_size, 3))
        CE = K.categorical_crossentropy(true_probs, pred_logits, from_logits=True)
        L_readout = K.sum(CE) / batch_size

        # Penalize deviation from the known initial state at the first time step
        # Do a softmax to get the predicted probabilities
        mask = K.cast(K.not_equal(y_true_prep_encoding, self.mask_value), K.floatx())
        pred_encoding = K.reshape(tf.boolean_mask(y_pred_prep_encoding, mask), (batch_size, self.num_prep_states))
        true_encoding = K.reshape(tf.boolean_mask(y_true_prep_encoding, mask), (batch_size, self.num_prep_states))
        CE = K.categorical_crossentropy(true_encoding, pred_encoding, from_logits=True)
        L_prep_encoding = K.sum(CE) / batch_size

        # Penalize deviation from the known initial state at the first time step
        # Do a softmax to get the predicted probabilities
        # This will enforce the x, y and z values of the prep state on the first sample.
        init_z = tf.linalg.matmul(true_encoding, tf.constant(self.init_z, dtype=K.floatx()))
        init_z_pred = K.softmax(y_pred_ro_results[:, 0, :])
        L_init_state = K.sum(K.abs(init_z - init_z_pred)) / batch_size

        # Force the state of average readout results to be equal to the strong readout results.
        lagrange_1 = tf.constant(1.0, dtype=K.floatx()) # Readout cross-entropy
        lagrange_2 = tf.constant(1.0, dtype=K.floatx()) # Initial state
        lagrange_4 = tf.constant(1.0, dtype=K.floatx()) # Prep state encoding

        return lagrange_1 * L_readout + lagrange_2 * L_init_state + lagrange_4 * L_prep_encoding

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

        # I think this is useless, because this is enforced in the loss function above
        # init_x_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, :2])
        # init_y_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 2:4])
        # init_z_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 4:])

        # This will enforce the x, y and z values of the prep state on the first sample.
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
        lagrange_2 = tf.constant(1.0, dtype=K.floatx()) # Initial state
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
        init_x = tf.repeat(tf.constant(self.init_x, dtype=K.floatx()),
                           repeats=K.cast(batch_size, "int32"), axis=0)
        init_x_pred = K.softmax(y_pred[:, 0, 0:2])
        # todo: pull the 0 from the number of samples for the first timestep

        init_y = tf.repeat(tf.constant(self.init_y, dtype=K.floatx()),
                           repeats=K.cast(batch_size, "int32"), axis=0)
        init_y_pred = K.softmax(y_pred[:, 0, 2:4])

        init_z = tf.repeat(tf.constant(self.init_z, dtype=K.floatx()),
                           repeats=K.cast(batch_size, "int32"), axis=0)
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
        lagrange_2 = tf.constant(1.0, dtype=K.floatx())
        lagrange_3 = tf.constant(1.0, dtype=K.floatx())

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
            if "t" not in list(f.keys()):
                print(f.keys())
            # if not file_exists:
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

                learning_rate = np.array([self.learning_rate_schedule(e-1) for e in epochs])
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
    """
    Adjust the dropout of `models.layer[1]` after each epoch according to a specified dropout schedule.
    """
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
