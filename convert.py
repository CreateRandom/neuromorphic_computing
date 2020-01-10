import os
import time
import shutil
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
import datetime

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

def generate_snn_config(path_wd, model_name, simulator='INI'):
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
    config_filepath = os.path.join(path_wd, model_name) + '.config'
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    return config_filepath

def run_all_on_path(in_path):
    assert os.path.exists(os.path.join(in_path, 'x_norm.npz')), 'Could not find x_norm.npz on path'
    candidates = []
    for filename in os.listdir(in_path):
        if filename.endswith(".h5"):
            candidates.append(filename)

    print('Found a total of {} candidates.'.format(len(candidates)))
    durations = []
    for i, candidate in enumerate(candidates):
        # strip the extension
        candidate = candidate.split('.h5')[0]
        print('Running candidate {}'.format(candidate))
        config = generate_snn_config(in_path,candidate, simulator='loihi')
        start_time = time.time()
        main(config)
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        durations.append(duration)
        print('Duration for candidate {}: {}'.format(candidate, duration))
        mean_duration = sum(durations) / len(durations)
        n_left = len(candidates[i+1:])
        pred_duration = mean_duration * n_left
        print('Predicted duration for remaining candidates: {}'.format(datetime.timedelta(seconds=pred_duration)))

        # rename the log dir
        shutil.move(os.path.join(in_path,'log'), os.path.join(in_path,candidate))

if __name__ == '__main__':
    nrc_path = '/homes/klux/models'
    run_all_on_path(nrc_path)