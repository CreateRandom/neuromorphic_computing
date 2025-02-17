"""End-to-end example for SNN Toolbox.

This script sets up a small CNN using Keras and tensorflow, trains it for one
epoch on MNIST, stores model and dataset in a temporary folder on disk, creates
a configuration file for SNN toolbox, and finally calls the main function of
SNN toolbox to convert the trained ANN to an SNN and run it using pyNN/nest
simulator.
"""

import os
import pickle
import time

import keras
from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomUniform

from custom_regularizers import ModifiedL2Cost, ModifiedL2Callback, SparseCodingCallback, SparseCodingRegularizer
from dropout_learning_schedule import DropoutScheduler
from utils.data import load_mnist, save_data_for_toolbox
from utils.saving import save_model_config
# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(time.time())))
os.makedirs(path_wd)

# GET DATASET #
###############

(x_train, y_train), (x_test, y_test) = load_mnist()
# store a part of the train data and the test data in the path for the toolbox to be able to use it
save_data_for_toolbox(x_train, x_test, y_test, path_wd)

# CREATE ANN #
##############

# Create the ANN in Keras and train

def build_model(hidden_units=1200, dropout_rate=0.5, activity_regularizer=None):

    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)

    uniform_init = RandomUniform(minval=-0.1, maxval=0.1)

    # flatten to 784
    layer = Flatten()(input_layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(units=hidden_units, kernel_initializer=uniform_init, activity_regularizer=activity_regularizer, activation='relu')(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(units=hidden_units, kernel_initializer=uniform_init, activity_regularizer=activity_regularizer, activation='relu')(layer)
    layer = Dropout(dropout_rate)(layer)
    layer = Dense(units=10, kernel_initializer=uniform_init, activity_regularizer=activity_regularizer,
                  activation='softmax')(layer)
    # create model
    model = Model(input_layer, layer)

    model.summary()
    # basic training with SGD
    sgd = SGD(learning_rate=0.01, momentum=0.1)
    model.compile(sgd, 'categorical_crossentropy', ['accuracy'])

    return model


def train_model(model, epochs, callbacks=None):
    # Train model with backprop.
    history = model.fit(x_train, y_train, batch_size=512, epochs=epochs, verbose=2,
              validation_data=(x_test, y_test), callbacks=callbacks)

    # remove activity_regularizers before serializing
    for layer in model.layers:
        if hasattr(layer, 'activity_regularizer'):
            if isinstance(layer.activity_regularizer, ModifiedL2Cost):
                layer.activity_regularizer = None
            elif isinstance(layer.activity_regularizer, SparseCodingRegularizer):
                layer.activity_regularizer = None

    return history

def save_history(history, model_name):
    model_path = os.path.join(path_wd, model_name + '.history')
    with open(model_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def save_model(model, model_name):
    model_path = os.path.join(path_wd, model_name + '.h5')
    keras.models.save_model(model, model_path)
    return model_path

def full_pipeline(config=None):
    save_model_config(config, path_wd)
    total_epochs = config['epochs']
    activity_regularizer = None
    callbacks = []
    if 'activity_regularizer' in config:
        if config['activity_regularizer'] is 'l2':
            # c_act always needs to start at 0 to allow for pre-training
            start_epochs = config['regularizer_params']['start_epochs']
            activity_regularizer = ModifiedL2Cost(c_min= config['regularizer_params']['c_min'], c_act=0)
            callback = ModifiedL2Callback(c_act_target=config['regularizer_params']['c_act'],
                                          start_epochs=start_epochs)
            callbacks.append(callback)
            # add the pre-training epochs to the total
            total_epochs += start_epochs
        elif config['activity_regularizer'] is 'sparse':
            start_epochs = config['regularizer_params']['start_epochs']
            activity_regularizer = SparseCodingRegularizer(s_cost=config['regularizer_params']['s_cost'],
                                                           s_target=0)
            callback = SparseCodingCallback(s_cost_target=config['regularizer_params']['s_target'],
                                            start_epochs=start_epochs)
            callbacks.append(callback)
            total_epochs += start_epochs
        else:
            raise ValueError('Unknown activity regularizer {}'.format(config['activity_regularizer']))

    dropout_rate = config.get('dropout', 0.5)
    hidden_units = config.get('hidden_units', 1200)
    model = build_model(hidden_units = hidden_units,
                        dropout_rate = dropout_rate, activity_regularizer=activity_regularizer)

    if 'dropout_scheduler' in config:
        scheduler_config = config['dropout_scheduler']
        scheduler = DropoutScheduler(**scheduler_config)
        # override the total number of epochs
        total_epochs = scheduler_config['extra_epochs'] + scheduler.start_epochs
        callbacks.append(scheduler)
    history = train_model(model, epochs=total_epochs, callbacks=callbacks)
    model_name = config['model_name']
    model_path = save_model(model, model_name)
    save_history(history, model_name)

    return model_path

# Dropout
# 6 models
# 0, all missing
for dropout in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
    config = {'model_name': 'mnist_dropout_' + str(dropout), 'dropout': dropout, 'epochs': 50}
    full_pipeline(config)

# Sparse coding
# 36 models
# 28, 8 missing
# 0.0001 --
for s_cost in [0.1, 0.01, 0.001, 0.0001]:
    for s_target in [0.2, 0.05, 0.01]:
        # Unsure in paper, it says [5, 50, 50]
        for sec_epochs in [5, 20, 50]:
            config = {'model_name': 'mnist_sparse_s_cost_' + str(s_cost) + '_s_target_' + str(s_target) + '_sec_epochs_' + str(sec_epochs),
                      'activity_regularizer': 'sparse', 'regularizer_params': {'s_cost': s_cost, 's_target': s_target, 'start_epochs': 20},
                      'epochs': sec_epochs}
            full_pipeline(config)

# L2 Cost
# 48 models
for c_act in [0.1, 0.01, 0.001, 0.0001]:
    # split on this, 0.5 and 1 to flutter
    for c_min in [0.5, 1.0, 1.5, 2.0]:
        # after 2 epochs of pre-training
        for epochs in [5, 20, 50]:
            config = {'model_name': 'mnist_l2_c_min_' + str(c_min) + '_c_act_' + str(c_act) + '_epochs_' + str(epochs), 'activity_regularizer': 'l2',
                      'regularizer_params': {'c_min': c_min, 'c_act': c_act, 'start_epochs': 2}, 'epochs': epochs}
            full_pipeline(config)

# Dropout scheduler
# 18 models
for p_final in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for e2 in [20, 50, 80]:
       config = {'model_name': 'mnist_dropout_scheduler_p_final_' + str(p_final) + '_e2_' + str(e2),
                                   'dropout_scheduler': {'p_final': p_final, 'extra_epochs': e2}, 'epochs' : 0}
       full_pipeline(config)


# cheerilee: 64 --> 33 models, so put all the sparse models (36) on there
# fancypants: 64 --> 33 models, so put dropout_scheduler and dropout on there 24 + 9 from sparse
# fluttershy: 40 --> 21, so put half of l2, 24 --> blossomforth
# rarity: 40 --> 21, the other half of l2, 24 --> thunderlane

# cheerilee is out
# fancypants is out
# fluttershy is out
# rarity is out
# pipsqueak is out
# scootaloo is out

# twist is available
# featherweight is available
# blossomforth / thunderlane


# twist: 27 models, sparse except for 0.001 --> did not finish
# featherweight: 9 sparse, 18 dropout_scheduler --> 27 --> finished
# bf: 24 l2 --> finished
# thunderlane: 24 l2 + 6 dropout --> 30 --> did not finish