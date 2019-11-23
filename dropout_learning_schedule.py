import keras as K

class DropoutScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and adjusts the dropout rate of each dropout layer.
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, final_rate, max_epoch, verbose=0):
        super(DropoutScheduler, self).__init__()
        self.final_rate = final_rate
        self.verbose = verbose
        self.max_epoch = max_epoch

    def on_epoch_end(self, epoch, logs=None):
        rate = epoch - max_epoch/2 
        if rate < 0: rate = 0
        rate = rate * final_rate/(max_epoch/2)
        for layer in self.model.layers[]:
            if if isinstance(layer, Dropout):
                layer.rate = rate



