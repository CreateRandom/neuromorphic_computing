import os
import pickle
import numpy
import pandas as pd
from multiprocessing import Pool

def main():
    base_result_path = '/vol/tensusers/klux/neuromorphic_computing/models/all_results/'
    model_paths = [os.path.join(base_result_path,x) for x in os.listdir(base_result_path) if x is not 'pkls']

    p = Pool(32)
    results = p.map(process_model,model_paths)
    p.close()
    df = pd.DataFrame(results)
    result_path = os.path.join(base_result_path,'results.csv')
    df.to_csv(result_path)
    print('Exported result to {}'.format(result_path))

def process_model(model_path):
    config_pkl_path = '/vol/tensusers/klux/neuromorphic_computing/models/pkls'
    config = get_config_for_model(model_path, config_pkl_path)
    # flatten the inner dicts
    config = flatten_dict(config)
    acc, comp = extract_results_for_model(model_path)
    config['accuracy'] = acc
    config['computation'] = comp

    return config
def get_config_for_model(model_path, pkl_path):
    # the last element of the model_path
    config_name = model_path.split('/')[-1]
    config_name = config_name + '.pkl'
    pkl_path = os.path.join(pkl_path, config_name)

    with open(pkl_path, 'rb') as f:
        config = pickle.load(f)

    return config

def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items())

def extract_results_for_model(model_path):
    print('Processing model {}'.format(model_path))
    log_var_location = os.path.join(model_path, 'gui', 'test', 'log_vars')
    log_var_files = os.listdir(log_var_location)
    log_var_files = sorted(log_var_files, key=lambda x: int(x[:-4]))
    # average n_op for every time stp
    batch_operations = []

    final_batch = None
    ##for every batch: store number of operations by time step
    for i, log_var_batch in enumerate(log_var_files):
        log_var_batch = os.path.join(log_var_location, log_var_batch)
        batch_results = numpy.load(log_var_batch)
        operations_by_t = numpy.swapaxes(batch_results['synaptic_operations_b_t'], 0, 1)
        #average operations over samples
        for time in range(2000):
            batch_operations.append(sum(operations_by_t[time])/256)
        if i == len(log_var_files) -1 :
            final_batch = batch_results

    #Reshape and swap so that we can average over batches

    computation = numpy.reshape(batch_operations,(39,2000))
    computation = numpy.swapaxes(computation,0,1)
    comp_by_t = []

    accuracy = numpy.reshape(final_batch['acc_at_t'],(39,2000))
    accuracy = numpy.swapaxes(accuracy,0,1)
    acc_by_t = []


    #average over batches
    for step in range(2000):
        acc_by_t.append(sum(accuracy[step])/39)
        comp_by_t.append(sum(computation[step]/39))

    return acc_by_t,comp_by_t


if __name__ == '__main__':

    main()
