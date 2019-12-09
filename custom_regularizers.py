from keras import backend as K
from keras.regularizers import Regularizer


class ModifiedL2Cost(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, c_min=0.5, c_act=0.1):
        self.c_min = K.cast_to_floatx(c_min)
        self.c_act = K.cast_to_floatx(c_act)

    def __call__(self, x):
        norm = self.c_act * K.sum(K.cast(K.identity(K.greater(x, self.c_min)), 'float32') * K.square(x))
        return norm

    def get_config(self):
        return {'c_min': float(self.c_min),
                'c_act': float(self.c_act)}