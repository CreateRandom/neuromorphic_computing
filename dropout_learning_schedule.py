import keras as K

class DropoutScheduler(K.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and adjusts the dropout rate of each dropout layer.
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, final_rate, extra_epochs, verbose=0):
        super(DropoutScheduler, self).__init__()
        self.final_rate = final_rate
        self.verbose = verbose
        self.extra_epochs = extra_epochs    

    def on_epoch_end(self, epoch, logs=None):
        step = 0
        if epoch < 50: step = 0
        else: step = extra_epochs - 50
        rate = final_rate/extra_epochs * step
        for layer in self.model.layers[]:
            if if isinstance(layer, Dropout):
                layer.rate = rate



