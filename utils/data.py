import os

import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def save_data_for_toolbox(x_train, x_test, y_test, path_wd):
    # Save dataset so SNN toolbox can find it.
    np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
    np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
    # SNN toolbox will not do any training, but we save a subset of the training
    # set so the toolbox can use it when normalizing the network parameters.
    np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

def load_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize input so we can train ANN with it.
    # Will be converted back to integers for SNN layer.
    x_train = x_train / 255
    x_test = x_test / 255

    # Add a channel dimension.
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)

    # One-hot encode target vectors.
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)