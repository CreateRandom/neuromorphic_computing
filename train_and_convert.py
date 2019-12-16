"""End-to-end example for SNN Toolbox.

This script sets up a small CNN using Keras and tensorflow, trains it for one
epoch on MNIST, stores model and dataset in a temporary folder on disk, creates
a configuration file for SNN toolbox, and finally calls the main function of
SNN toolbox to convert the trained ANN to an SNN and run it using pyNN/nest
simulator.
"""

import os
import time

import keras
from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomUniform

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from custom_regularizers import ModifiedL2Cost, ModifiedL2Callback
from dropout_learning_schedule import DropoutScheduler
from utils.data import load_mnist, save_data_for_toolbox

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

input_shape = x_train.shape[1:]
input_layer = Input(input_shape)

uniform_init = RandomUniform(minval=-0.1, maxval=0.1)
dropout_rate = 0.0
# flatten to 784


layer = Flatten()(input_layer)
layer = Dropout(dropout_rate)(layer)
#layer = Dense(units=1200, kernel_initializer=uniform_init, activation='relu')(layer)
layer = Dense(units=1200, kernel_initializer=uniform_init, activity_regularizer=ModifiedL2Cost(), activation='relu')(layer)
layer = Dropout(dropout_rate)(layer)
layer = Dense(units=1200, kernel_initializer=uniform_init, activation='relu')(layer)
#layer = Dense(units=1200, kernel_initializer=uniform_init, activity_regularizer=ModifiedL2Cost(), activation='relu')(layer)
layer = Dropout(dropout_rate)(layer)
#layer = Dense(units=10, kernel_initializer=uniform_init, activity_regularizer=ModifiedL2Cost(),
 #             activation='softmax')(layer)
layer = Dense(units=10, kernel_initializer=uniform_init,
              activation='softmax')(layer)
# create model
model = Model(input_layer, layer)

model.summary()
# basic training with SGD
sgd = SGD(learning_rate=0.01, momentum=0.1)
model.compile(sgd, 'categorical_crossentropy', ['accuracy'])

scheduler = DropoutScheduler(final_rate=0.9,extra_epochs=10, start_epochs=0)
# Train model with backprop.
model.fit(x_train, y_train, batch_size=64, epochs=22, verbose=2,
          validation_data=(x_test, y_test), callbacks=[ModifiedL2Callback()])

# remove activity_regularizers before serializing
for layer in model.layers:
    if hasattr(layer, 'activity_regularizer'):
        if isinstance(layer.activity_regularizer, ModifiedL2Cost):
            layer.activity_regularizer = None

# Store model so SNN Toolbox can find it.
model_name = 'mnist_vanilla'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': True,              # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',            # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 5,               # How many test samples to run.
    'batch_size': 1,                # Batch size for simulation.
    'keras_backend': 'tensorflow'
}

config['output'] = {
    # log all possible variables to the disk
    'log_vars': {'all'},
    # 'plot_vars': {                  # Various plots (slows down simulation).
    #     'spiketrains',              # Leave section empty to turn off plots.
    #     'spikerates',
    #     'activations',
    #     'correlation',
    #     'v_mem',
    #     'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)

