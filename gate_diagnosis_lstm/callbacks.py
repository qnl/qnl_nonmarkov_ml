import tensorflow as tf
import visdom, os, h5py
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, backend as K

class TrainingPlot(tf.keras.callbacks.Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

        self.plotter = visdom.Visdom()

        if self.plotter.win_exists('Loss'):
            self.plotter.close('Loss')
        if self.plotter.win_exists('Accuracy'):
            self.plotter.close('Accuracy')

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        #         if len(self.losses) > 1:
        if self.plotter.win_exists('Loss'):
            kwargs = {'update': 'append'}
        else:
            kwargs = {}

        plot = self.plotter.line(np.array([logs.get('loss')]),
                                 X=np.array([epoch]),
                                 win='Loss',
                                 opts=dict(title="Loss", xlabel='Epoch', ylabel='Loss'),
                                 name='Training_Loss',
                                 **kwargs)

        plot = self.plotter.line(np.array([logs.get('val_loss')]),
                                 X=np.array([epoch]),
                                 win='Loss',
                                 opts=dict(title="Loss", xlabel='Epoch', ylabel='Loss'),
                                 name='Validation_Loss',
                                 **kwargs)

        if self.plotter.win_exists('Accuracy'):
            kwargs = {'update': 'append'}
        else:
            kwargs = {}

        plot = self.plotter.line(np.array([logs.get('masked_accuracy')]),
                                 X=np.array([epoch]),
                                 win='Accuracy',
                                 opts=dict(title="Accuracy", xlabel='Epoch', ylabel='Accuracy'),
                                 name='Training_Accuracy',
                                 **kwargs)

        plot = self.plotter.line(np.array([logs.get('val_masked_accuracy')]),
                                 X=np.array([epoch]),
                                 win='Accuracy',
                                 opts=dict(title="Accuracy", xlabel='Epoch', ylabel='Accuracy'),
                                 name='Validation_Accuracy',
                                 **kwargs)


class DropOutScheduler(tf.keras.callbacks.Callback):
    def __init__(self, dropout_schedule):
        self.dropout_schedule = dropout_schedule

    def on_epoch_end(self, epoch, logs={}):
        self.model.layers[1].rate = self.dropout_schedule(epoch)

class LossTracker(tf.keras.callbacks.Callback):
    def __init__(self, test_features, test_labels, n_levels, mask_value, savepath, prep_x, prep_y, prep_z):
        self.mask_value = mask_value
        self.test_x = test_features
        self.test_y = test_labels
        self.n_levels = n_levels
        self.savepath = savepath
        if self.n_levels == 2:
            self.prep_x = prep_x
            self.prep_y = prep_y
            self.prep_z = prep_z
        elif self.n_levels == 3:
            self.prep_z = prep_z
        self.avg_validation_losses = np.empty((0, 3), dtype=np.float)

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def qubit_loss_components(self, y_true, y_pred):
        assert np.shape(y_true) == np.shape(y_pred)
        batch_size = np.shape(y_true)[0]
        mask = (y_true != self.mask_value)
        # print(np.shape(y_true), np.shape(mask))
        pred_logits = np.reshape(y_pred[mask], (batch_size, 2))
        true_probs = np.reshape(y_true[mask], (batch_size, 2))
        # print(K.constant(pred_logits)[:10, :], K.constant(true_probs)[:10, :])
        CE = K.categorical_crossentropy(K.constant(true_probs), K.constant(pred_logits), from_logits=True)
        # print(np.shape(CE))
        # print(CE[:10])
        L_readout = np.sum(CE) / batch_size

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

        # Purity
        X_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 0:2], axis=-1)[:, :, 1]
        Y_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 2:4], axis=-1)[:, :, 1]
        Z_all_t = 1.0 - 2.0 * K.softmax(y_pred[:, :, 4:6], axis=-1)[:, :, 1]
        L_outside_sphere = K.mean(K.relu(K.sqrt(K.square(X_all_t) + K.square(Y_all_t) + K.square(Z_all_t)),
                                         threshold=1.0))

        return L_readout, L_init_state[0], L_outside_sphere

    def qutrit_loss_components(self, y_true, y_pred):
        assert np.shape(y_true) == np.shape(y_pred)
        batch_size = np.shape(y_true)[0]
        mask = (y_true != self.mask_value)
        # print(np.shape(y_true), np.shape(mask))
        pred_logits = np.reshape(y_pred[mask], (batch_size, 3))
        true_probs = np.reshape(y_true[mask], (batch_size, 3))
        # print(K.constant(pred_logits)[:10, :], K.constant(true_probs)[:10, :])
        CE = K.categorical_crossentropy(K.constant(true_probs), K.constant(pred_logits), from_logits=True)
        # print(np.shape(CE))
        # print(CE[:10])
        L_readout = np.sum(CE) / batch_size

        init_z = tf.repeat(tf.constant([self.prep_z], dtype=K.floatx()), repeats=K.cast(batch_size, "int32"), axis=0)
        init_z_pred = K.softmax(y_pred[:, 0, :])

        L_init_state = K.mean(K.sqrt(K.square(init_z - init_z_pred)))

        return L_readout, L_init_state, 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.test_x)
        if self.n_levels == 2:
            current_loss = self.qubit_loss_components(self.test_y, y_pred)
        elif self.n_levels == 3:
            current_loss = self.qutrit_loss_components(self.test_y, y_pred)
        print(np.shape(current_loss[0]), np.shape(current_loss[1]), np.shape(current_loss[2]))
        self.avg_validation_losses = np.vstack((self.avg_validation_losses, current_loss))
        print('\n')
        print(self.avg_validation_losses[-1])

    def on_train_end(self, logs=None):
        with h5py.File(os.path.join(self.savepath, "trajectories.h5"), 'a') as f:
            f.create_dataset('training/loss_components', data=self.avg_validation_losses)