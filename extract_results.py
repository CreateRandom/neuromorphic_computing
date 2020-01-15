import os
import pickle

import numpy


def main():
    base_result_path = '/vol/tensusers/klux/neuromorphic_computing/models/all_results/'
    config_pkl_path = '/vol/tensusers/klux/neuromorphic_computing/models/all_results/pkls'

    model_paths = [os.path.join(base_result_path,x) for x in os.listdir(base_result_path)]


    for model_path in model_paths:
        config = get_config_for_model(model_path, config_pkl_path)
        print(config)
        results = extract_results_for_model(model_path)

    pass


def get_config_for_model(model_path, pkl_path):
    # the last element of the model_path
    config_name = model_path.split('/')[-1]
    config_name = config_name + '.pkl'
    pkl_path = os.path.join(pkl_path, config_name)

    with open(pkl_path, 'rb') as f:
        config = pickle.load(f)

    return config


def extract_results_for_model(model_path):
    log_var_location = os.path.join(model_path, 'gui', 'test', 'log_vars')

    log_var_files = [os.path.join(log_var_location,x) for x in os.listdir(log_var_location)]

    for log_var_batch in log_var_files:
        print(log_var_batch)
        batch_results = numpy.load(log_var_batch)
        # TODO do something here to get the relevant results out
        print(batch_results['synaptic_operations_b_t'].shape)

    pass

if __name__ == '__main__':

    main()
