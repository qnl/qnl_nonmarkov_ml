import tensorflow as tf
import visdom, os
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