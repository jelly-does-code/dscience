from ast import literal_eval
from os.path import isfile, join
from os import listdir, remove
from time import gmtime, localtime, time, strftime
from shutil import copy2

import numpy as np
from pandas import DataFrame, NA, Series, concat, read_csv, set_option

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import optuna
from joblib import dump, load

from objective_functions import obj_rf, obj_xgb, obj_cat, obj_ridge
from load_data import load_data
from eda import eda, plot_feature_importances, plot_permutation_importances
from feature_engineering import compare_models_with_without_engineered_features, feature_engineering
from aux_functions import log

runtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
set_option('display.max_columns', None)

def get_params(X_train, y_train, X_test, y_test, model_map, data_map, runtime_map):
    cat_cols = data_map['cat_cols_raw'] 
    perf_metric_direction = runtime_map['perf_metric_direction']
    
    param_df = read_csv('../performance/best.csv', index_col='name')
    for name in model_map:
        # If a retune is requested or if there are no existing params, tune/retune:
        if model_map[name]['retune'] == 1 or len(param_df.loc[name, 'params']) < 3:
            print(f'Hyperparametertuning for {name}..')
            start_time = time()
            study = optuna.create_study(direction=perf_metric_direction)
            
            if name == 'CatBoostClassifier':
                study.optimize(lambda trial: model_map[name]['obj_func'](trial, X_train, y_train, X_test, y_test, cat_cols), n_trials=runtime_map['n_trials'])
            elif name ==  'XGBClassifier':
                study.optimize(lambda trial: model_map[name]['obj_func'](trial, X_train, y_train, X_test, y_test), n_trials=runtime_map['n_trials'])

            model_params = study.best_params
            print(f"Hyperparameter tuning done for {name}. Time elapsed: {strftime('%H:%M:%S', gmtime(time() - start_time))} for {runtime_map['n_trials']}.")
        else:
            model_params = literal_eval(param_df.loc[name, 'params'])
        model_map[name]['params'] = model_params

def get_models(X_train, y_train, model_map, cat_cols):
    print('\nTraining: retrieving fit models or refitting models..')
    for name in model_map:
        if model_map[name]['refit'] == 1 or not isfile(f'../models/{name}_best.joblib'):
            print(f'fitting for {name}..')
            
            if name == 'CatBoostClassifier':
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params'], verbose=0, cat_features=cat_cols).fit(X_train, y_train)
            elif name == 'XGBClassifier':
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params'], enable_categorical=True).fit(X_train, y_train)
            else:
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(X_train, y_train)

        else:
            model_map[name]['model'] = load(f'../models/{name}_best.joblib')

def write_current(model_map):
    curr_perf = DataFrame(columns=['name','perf','kfold_perf','params', 'timestamp'])

    for name in model_map:
        #for model, name, perf, kfold_perf, param in zip(models, names, perfs, kfold_perfs, params):
        new_row = DataFrame({
            'name': name,
            'perf': model_map[name]['perf'],
            'kfold_perf': model_map[name]['kfold_perf'],
            'params': [model_map[name]['params']],
            'timestamp': runtime})
        if curr_perf.empty:
            curr_perf = new_row
        else:
            curr_perf = concat([curr_perf, new_row], ignore_index=True)
        dump(model_map[name]['model'], f'../models/{name}_current.joblib')
    curr_perf.to_csv('../performance/current.csv', index=False)

def write_better(model_map, perf_metric_direction):
    # If there is a best model/perf file, compare it with current run
    curr_perf = read_csv('../performance/current.csv', index_col='name')
    
    if isfile('../performance/best.csv'):
        best_perf = read_csv('../performance/best.csv', index_col='name')
        for name in model_map:
            if isfile(f'../models/{name}_best.joblib'):
                # Fetch current and best performance from the dataframe
                best_perf_value, curr_perf_value = best_perf.loc[name, 'perf'], curr_perf.loc[name, 'perf']
                # Compare based on the direction
                if (perf_metric_direction == 'maximize' and curr_perf_value > best_perf_value) or (perf_metric_direction == 'minimize' and curr_perf_value < best_perf_value):
                    best_perf.loc[name] = curr_perf.loc[name]
                    copy2(f'../models/{name}_current.joblib', f'../models/{name}_best.joblib')
                    print(f'Yeeehaaa! We\'ve got ourselves a new best candidate for model {name}!')
            else:
                copy2(f'../models/{name}_current.joblib', f'../models/{name}_best.joblib')
        best_perf.to_csv('../performance/best.csv')
    else:
        copy2('../performance/current.csv', '../performance/best.csv')

def tune_train(X_train, X_test, y_train, y_test, data_map, model_map, runtime_map):

    get_params(X_train, y_train, X_test, y_test, model_map, data_map, runtime_map)
    get_models(X_train, y_train, model_map, data_map['cat_cols_engineered'])

    # Predict probabilities for classification models
    print('Training: predicting on test set..')
    for name in model_map:
        if model_map[name]['refit'] == 1 or model_map[name]['retune'] == 1:
            if name == 'RidgeClassifier':
                model_map[name]['pred_proba'] = model_map[name]['model'].decision_function(X_test)
            else:
                model_map[name]['pred_proba'] = model_map[name]['model'].predict_proba(X_test)[:, 1]

            if runtime_map['calculate_kfold']:
                print(f'Training: checking kfold performance for {name}..')
                model_map[name]['cross_val_score'] = cross_val_score(model_map[name]['model'], X_train, y_train, cv=runtime_map['kfold'], scoring='roc_auc')
                model_map[name]['kfold_perf'] = "%.2f%% (%.2f%%)" % (model_map[name]['cross_val_score'].mean()*100, model_map[name]['cross_val_score'].std()*100)

            print(f'Training: checking performance with chosen performance metric for {name} .. (roc-auc)')
            model_map[name]['perf'] = roc_auc_score(y_test, model_map[name]['pred_proba'])
            
        
    write_current(model_map)
    write_better(model_map, runtime_map['perf_metric_direction'])
  
def main():
    # Clean up previous logs
    with open('log.txt', 'w') as f1:
        pass
    
    data_map = {'target_col': 'loan_status',
                'index_col': 'id',
                'drop_cols': [],
                'cat_cols_raw': [],
                'num_cols_raw': [],
                'cat_cols_engineered': [],
                'num_cols_engineered': []
                }
    
    runtime_map = {'eda_when': 'both',
                   'task_type': 'classification',
                   'perf_metric_direction': 'maximize', # 'maximize' or 'minimize'. Defined by Optuna function study option
                   'n_trials': 20,
                   'calculate_kfold': False,
                   'plots': [0,        0,        1,          0,           0,    0,  0,    1],
                            #num_plot, cat_plot, mixed_plot, single_plot, skew, mi, corr, imp
                   'feature_engineering_done': 0
                   }   
    
    model_map = {'RandomForestClassifier':                   {'model': RandomForestClassifier,
                                                             'handles_cat': False,
                                                             'params': '',
                                                             'obj_func': obj_rf,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'perf': '',
                                                             'kfold_perf': ''},
                'XGBClassifier':                            {'model': XGBClassifier,
                                                             'handles_cat': True,
                                                             'params': '',
                                                             'obj_func': obj_xgb,
                                                             'retune': 1,
                                                             'refit': 1,
                                                             'pred_proba': '',
                                                             'perf': '',
                                                             'kfold_perf': ''},
                'CatBoostClassifier':                       {'model': CatBoostClassifier,
                                                             'handles_cat': True,
                                                             'params': '',
                                                             'obj_func': obj_cat,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'perf': '',
                                                             'kfold_perf': ''},                                                             
                'RidgeClassifier':                          {'model': RidgeClassifier,
                                                             'handles_cat': False,
                                                             'params': '',
                                                             'obj_func': obj_ridge,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'perf': '',
                                                             'kfold_perf': ''}                                                          
                }
    
    
    # Load raw data and perform EDA on it
    X_train, X_test, y_train, y_test, X_pred, data_map, runtime_map = load_data(data_map, runtime_map)      # Load data, split the datasets and fillna's without data leakage
    eda(X_train, y_train, data_map, runtime_map)                                                            # EDA on raw data

    X_train_no_engineered = X_train
    
    # Feature engineering
    X_train, X_test, y_train, y_test, X_pred, runtime_map['feature_engineering'] = feature_engineering(X_train, X_test, y_train, y_test, X_pred, data_map, runtime_map)  # Feature engineering
    eda(X_train, y_train, data_map, runtime_map)                                                # This EDA runs on engineered data (controlled in runtime_map parameter)
    compare_models_with_without_engineered_features(model_map, X_train, y_train, X_train_no_engineered, runtime_map['kfold'], scoring='roc_auc')
    tune_train(X_train, X_test, y_train, y_test, data_map, model_map, runtime_map)                 # Train and evaluate models
    
    # By now a best model exists for all models. Could have been a previous best, could be a copy of the current model
    for name in model_map:
        model_map[name]['best_model'] = load(f'../models/{name}_best.joblib') 
    
    if runtime_map['plots'][-1]:
        plot_feature_importances(model_map, X_pred.columns)
        plot_permutation_importances(model_map, X_pred.columns, X_train, y_train)

    # Predict and submit
    for name in model_map:
        if model_map[name]['refit'] == 1:
            if name == 'RidgeClassifier':
                predictions_curr = model_map[name]['model'].decision_function(X_pred)
                #predictions_best = model_map[name]['best_model'].decision_function(X_pred)
            else:
                predictions_curr = model_map[name]['model'].predict_proba(X_pred)[:, 1]
                #predictions_best  = model_map[name]['best_model'].predict_proba(X_pred)[:, 1]
            
            submission_curr = DataFrame({data_map['index_col']: X_pred.index, data_map['target_col']: predictions_curr})
            submission_curr.to_csv(f'../submissions/submission_{name}_curr.csv', index=False)

            #submission_best = DataFrame({index_col: X_pred.index, target_col: predictions_best})
            #submission_best.to_csv(f'../submissions/submission_{name}_best.csv', index=False)

    log("\nSubmission file(s) created successfully.")

if __name__ == "__main__":
    main()
