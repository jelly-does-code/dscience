from pandas import DataFrame, set_option
import json

from time import localtime, strftime

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from objective_functions import obj_rf, obj_xgb, obj_cat, obj_histboost, obj_ridge

# Function to convert sparse matrix to df
def convert_sparse_to_df(data_map, sparse_matrix_key):
    encoded_col_names = data_map['encoder'].get_feature_names_out()
    return DataFrame.sparse.from_spmatrix(data_map[sparse_matrix_key], columns=encoded_col_names, index=data_map['index_col'])

def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

# Initialization
def initialize():
    set_option('display.max_columns', None)
    
    # Clean up previous logs
    with open('log.txt', 'w') as f1:
        pass

# Logging function
def log(msg, fname='log.txt'):
    if isinstance(msg, str):
        with open(fname, 'a') as file:
            file.write(msg + '\n')
            file.write('' + '\n')
    elif isinstance(msg, DataFrame):
        with open(fname, 'a') as file:
            msg.to_csv(fname, index=False)

# Function to make final predictions
def make_predictions(model, X_data, proba_func):
    if proba_func == 'decision_function':
        return model.decision_function(X_data)
    elif proba_func == 'pred_proba':
        return model.predict_proba(X_data)[:, 1]

def update_maps_from_config(data_map_file, model_map_file, runtime_map_file):
    with open(data_map_file, 'r') as f:
        data_map_config = json.load(f)
    
    with open(model_map_file, 'r') as f:
        model_map_config = json.load(f)
    
    with open(runtime_map_file, 'r') as f:
        runtime_map_config = json.load(f)    

    runtime_map_config['runtime'] = get_current_time()

    # Reconstruct the model_map objects
    model_map = {}
    for name, config in model_map_config.items():
        model_class = globals()[config['model']]  # Retrieve the model class from json using globals()
        obj_func = globals()[config['obj_func']]  # Retrieve the function from json using globals() 
        model_map[name] = {
            'model': model_class,
            'handles_cat': config['handles_cat'],
            'params': config['params'],
            'obj_func': obj_func,
            'retune': config['retune'],
            'refit': config['refit'],
            'pred_proba': config['pred_proba'],
            'proba_func': config['proba_func'],
            'perf': config['perf'],
            'kfold_perf': config['kfold_perf']
        }

    return data_map_config, model_map, runtime_map_config
