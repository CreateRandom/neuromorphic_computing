import os

from snntoolbox.utils.utils import import_configparser

loihi_config_dict = {
    # nWB has a maximum of 8
    # weightExponent must be 0 anyway
    'connection_kwargs': {'numWeightBits': 8, 'weightExponent': 0},
    # vThMant: no idea how to set this, max is 2 ** 17 - 1
    # biasExp must be 6 anyway
    'compartment_kwargs': {'vThMant': 2 ** 8, 'biasExp': 6},
    # no clue how to set this
    'desired_threshold_to_input_ratio': 1

}

def convert_model(path_wd, model_name, simulator='INI'):
    # Store model so SNN Toolbox can find it.

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
        'simulator': simulator,            # Chooses execution backend of SNN toolbox.
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

    if simulator is 'loihi':
        config['loihi'] = loihi_config_dict

    # Store config file.
    config_filepath = os.path.join(path_wd, 'config')
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    return config_filepath
