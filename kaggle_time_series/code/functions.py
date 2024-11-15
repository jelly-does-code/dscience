from pandas import DataFrame, set_option
import json

from time import localtime, strftime

import lightgbm as LGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

#-------------------------------------------------------------------------------------
# Objective function for RandomForestClassifier
def obj_rf(trial, data_map, runtime_map):
    param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 13),  
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42}
    
    model = RandomForestRegressor(**param)
    cv_scores = cross_val_score(
                                model, 
                                convert_sparse_to_df(data_map, 'X_train_encoded'), 
                                data_map['y_train'], 
                                cv=5, 
                                scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for XBGBoostClassifier
def obj_xgb(trial, data_map, runtime_map):
    param = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 7, 16),
        'max_bin': trial.suggest_int('max_bin', 256, 2000), 
        'tree_method': 'auto', 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5),  
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),  
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'enable_categorical': trial.suggest_categorical('enable_categorical', [True])} 
    model = XGBRegressor(**param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for XBGBoostClassifier
def obj_lgb(trial, data_map, runtime_map):
    param = {
        'objective': 'regression',
        'max_depth': trial.suggest_int('max_depth', 7, 16),
        'max_bin': trial.suggest_int('max_bin', 256, 2000), 
        'tree_method': 'auto', 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5),  
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),  
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'enable_categorical': trial.suggest_categorical('enable_categorical', [True])} 
    model = LGBRegressor(**param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for CatBoostClassifier
def obj_cat(trial, data_map, runtime_map):
    param = {
        'early_stopping_rounds': 50,
        'iterations': trial.suggest_int('iterations', 400, 600),  
        'depth': trial.suggest_int('depth', 3, 7),  
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 9.0),
        'random_seed': 42,
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'verbose': trial.suggest_categorical('verbose', [False]),
    }
    model = CatBoostRegressor(cat_features=data_map['cat_cols_engineered'], **param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

#-------------------------------------------------------------------------------------
def convert_sparse_to_df(data_map, sparse_matrix_key):
    
    if 'test' in sparse_matrix_key:
        index_vals =  data_map['X_test_index_values'] 
    elif 'pred' in sparse_matrix_key:
        index_vals = data_map['X_pred_index_values'] 
    else:
        index_vals = data_map['X_train_index_values']    
    
    
    return DataFrame.sparse.from_spmatrix(
        data_map[sparse_matrix_key], 
        columns=data_map['encoded_columns'], 
        index=index_vals  # This should be an array of values
    )

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

def predict(name, data_map, model_map, predict_data):
    model = model_map[name]['model']
    handles_cat = model_map[name]['handles_cat']
    handles_sparse = model_map[name]['handles_sparse']
    
    # Choose the right dataframe for the right model (categorical, encoded, sparse, ...)
    if handles_cat:
        X_to_predict = data_map[predict_data]
    else:
        if handles_sparse:
            X_to_predict = data_map[predict_data + '_encoded']
        else:
            X_to_predict = convert_sparse_to_df(data_map, predict_data + '_encoded')

    try:
        # Try using predict_proba for classification models
        model_map[name]['predictions'] = model.predict_proba(X_to_predict)[:, 1]  # Assuming binary classification
    except AttributeError:
        try:
            # If predict_proba fails, try using decision_function for models like SVM
            model_map[name]['predictions'] = model.decision_function(X_to_predict)
        except AttributeError:
            # If both fail, fallback to predict for regression or other models
            model_map[name]['predictions'] = model.predict(X_to_predict)
    
    return model_map

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
            'handles_sparse': config['handles_sparse'],
            'params': config['params'],
            'obj_func': obj_func,
            'retune': config['retune'],
            'refit': config['refit'],
            'predictions': config['predictions'],
            'perf': config['perf'],
            'kfold_perf': config['kfold_perf']
        }

    return data_map_config, model_map, runtime_map_config
