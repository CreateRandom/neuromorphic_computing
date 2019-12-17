from keras import backend as K
from keras.regularizers import Regularizer
from keras.callbacks import Callback


class ModifiedL2Callback(Callback):

    def __init__(self, start_epochs=2, c_act_target=0.01):
        super(ModifiedL2Callback, self).__init__()
        self.c_act_target = c_act_target
        self.start_epochs = start_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.start_epochs:
            for layer in self.model.layers:
                if hasattr(layer, 'activity_regularizer'):
                    if isinstance(layer.activity_regularizer, ModifiedL2Cost):
                        c_act = layer.activity_regularizer.c_act
                        print('Activation cost weight was {}'.format(c_act))
                        K.set_value(c_act,self.c_act_target)
                        print('Activation cost set to {}'.format(c_act))

class ModifiedL2Cost(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, c_min=2.0, c_act=0):
        self.c_min = K.variable(K.cast_to_floatx(c_min))
        self.c_act = K.variable(K.cast_to_floatx(c_act))

    def __call__(self, x):
        norm = self.c_act * K.sum(K.cast(K.identity(K.greater(x, self.c_min)), 'float32') * K.square(x))
        return norm

    def get_config(self):
        return {'c_min': self.c_min.numpy(),
                'c_act': self.c_act.numpy()}