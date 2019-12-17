from keras.callbacks import Callback
from keras.layers import Dropout

class DropoutScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and adjusts the dropout rate of each dropout layer.
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, final_rate, extra_epochs, start_epochs=50, verbose=0):
        super(DropoutScheduler, self).__init__()
        self.final_rate = final_rate
        self.verbose = verbose
        self.extra_epochs = extra_epochs
        self.start_epochs = start_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epochs: step = 0
        else: step = epoch - self.start_epochs
        rate = self.final_rate/self.extra_epochs * step
        print('Setting rate to {}'.format(rate))
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                print('Rate was {}'.format(layer.rate))
                layer.rate = rate