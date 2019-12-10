# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:45:29 2019

@author: Lars Koopman
"""

from keras import backend as K
import keras.losses as Loss
import numpy as np
from numpy import linalg as LA

class Sparse_Coding_Loss(Loss.LossFunctionWrapper):
    
    def __init__(self, s_target, extra_epochs, start_epochs=20, s_cost):
        super(Sparse_Coding_Loss, self).__init__()
        self.s_target = s_target
        self.s_cost = s_cost
        self.extra_epochs = extra_epochs
        self.start_epochs = start_epochs
        
    def customLoss(true,pred):
        #Need some way to get the epoch number
        l_sparse = s_cost * LA.norm(y - s_target * np.ones(len(y)))
        #Need some way to get the "y" vector which was discribed as neuron activations
        #Need some way to add this penalty to the currect loss function