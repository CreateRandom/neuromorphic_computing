import os
import pickle
import numpy
import matplotlib.pyplot as plt
import pandas as pd

def main():
    base_result_path = '/vol/tensusers/klux/neuromorphic_computing/models/all_results/'
    config_pkl_path = '/vol/tensusers/klux/neuromorphic_computing/models/all_results/pkls'

    model_paths = [os.path.join(base_result_path,x) for x in os.listdir(base_result_path)]

    df = pd.DataFrame()
    for model_path in model_paths:
        config = get_config_for_model(model_path, config_pkl_path)
        acc, comp = extract_results_for_model(model_path)
        config['accuracy']=acc
        config['computation']=comp
        df.append(config)
    df.to_csv(base_result_path+'results.csv')
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
    log_var_files = os.listdir(log_var_location)
    log_var_files = sorted(log_var_files, key=lambda x: int(x[:-4]))
    batch_operations = []


    ##for every batch: store number of operations by time step
    for log_var_batch in log_var_files:
        log_var_batch = os.path.join(log_var_location, log_var_batch)
        batch_results = numpy.load(log_var_batch)
        operations_by_t = numpy.swapaxes(batch_results['synaptic_operations_b_t'], 0, 1)
        #average operations over samples
        for time in range(2000):
            batch_operations.append(sum(operations_by_t[time])/256)

    #Reshape and swap so that we can average over batches

    computation = numpy.reshape(batch_operations,(39,2000)
    computation = numpy.swapaxes(computation,0,1)
    comp_by_t = []

    accuracy = numpy.reshape(batch_results['acc_at_t'],(39,2000))
    accuracy = numpy.swapaxes(accuracy,0,1)
    acc_by_t = []


    #average over batches
    for step in range(2000):
        acc_by_t.append(sum(accuracy[step])/39)
        comp_by_t.append(sum(computation[step]/39))

    return acc_by_t,comp_by_t


if __name__ == '__main__':

    main()
