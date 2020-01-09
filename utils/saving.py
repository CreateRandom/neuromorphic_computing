import os
import pickle

def save_model_config(config, path_wd):
    file_name = config['model_name'] + ".pkl"
    path = os.path.join(path_wd, file_name)
    pickle_object(config, path)

def pickle_object(object, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(object, f)