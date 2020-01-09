# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:45:29 2019

@author: Lars Koopman
"""

from keras import backend as K
from keras.callbacks import Callback
import keras.losses as Loss
from keras.regularizers import Regularizer
import numpy as np
from numpy import linalg as LA

class SparseCodingCallback(Callback):
    
    def __init__(self, s_cost_target, start_epochs=20):
        super(SparseCodingCallback, self).__init__()
        self.s_cost_target = s_cost_target
        self.start_epochs = start_epochs

    def on_epoch_end(self, epoch, logs=None):
        # counting starts at 0 here, so e.g. if you want to do 2, do 0 and 1, callback thereafter
        if epoch == self.start_epochs -1:
            for layer in self.model.layers:
                if hasattr(layer, 'activity_regularizer'):
                    if isinstance(layer.activity_regularizer, SparseCodingRegularizer):
                        s_cost = layer.activity_regularizer.s_cost
                        print('Sparse cost weight was {}'.format(s_cost))
                        K.set_value(s_cost,self.s_cost_target)
                        print('Sparse cost set to {}'.format(s_cost))

class SparseCodingRegularizer(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, s_cost=2.0, s_target=0):
        self.s_cost = K.variable(K.cast_to_floatx(s_cost))
        self.s_target = K.variable(K.cast_to_floatx(s_target))

    def __call__(self, y):
        l_sparse = self.s_cost * K.l2_normalize(y - self.s_target * K.ones(y.shape[1]))
        return l_sparse

    def get_config(self):
        return {'s_cost': self.s_cost.numpy(),
                's_target': self.s_target.numpy()}