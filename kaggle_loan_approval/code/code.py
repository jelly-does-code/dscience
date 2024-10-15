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
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import optuna
from joblib import dump, load

from objective_functions import obj_rf, obj_xgb, obj_cat, obj_histboost, obj_ridge
from load_data import load_data
from eda import eda, plot_feature_importances, plot_permutation_importances
from feature_engineering import compare_models_with_without_engineered_features, feature_engineering, onehotencode
from aux_functions import log

runtime = strftime("%Y-%m-%d %H:%M:%S", localtime())
set_option('display.max_columns', None)

def get_params(name, data_map, model_map, runtime_map):
    param_df = read_csv('../performance/best.csv', index_col='name')
    
    # Retune if 1. A retune is requested, 2. It has never been tuned before
    if model_map[name]['retune'] == 1 or (name not in param_df.index):
        print(f'Hyperparametertuning for {name}..')
        start_time = time()
        study = optuna.create_study(direction=runtime_map['perf_metric_direction'] )
        
        if name == 'CatBoostClassifier': # CatBoost need extra option cat_cols
            study.optimize(lambda trial: model_map[name]['obj_func'](trial, data_map['X_train'], data_map['y_train'], data_map['X_test'], data_map['y_test'], data_map['cat_cols_engineered']), n_trials=runtime_map['n_trials'])
        
        elif model_map[name]['handles_cat']:
            study.optimize(lambda trial: model_map[name]['obj_func'](trial, data_map['X_train'], data_map['y_train'], data_map['X_test'], data_map['y_test']), n_trials=runtime_map['n_trials'])
        
        else:
            study.optimize(lambda trial: model_map[name]['obj_func'](trial, data_map['X_train_encoded'], data_map['y_train'], data_map['X_test_encoded'], data_map['y_test']), n_trials=runtime_map['n_trials'])

        model_params = study.best_params
        print(f"Hyperparameter tuning done for {name}. Time elapsed: {strftime('%H:%M:%S', gmtime(time() - start_time))} for {runtime_map['n_trials']}.")
    else:
        model_params = literal_eval(param_df.loc[name, 'params'])
    
    model_map[name]['params'] = model_params
    return model_map

def fit_models(name, data_map, model_map):
    print('\nTraining: retrieving fit models or refitting models..')

    if model_map[name]['refit'] == 1 or not isfile(f'../models/{name}_best.joblib'):
        print(f'fitting for {name}..')
        if name == 'CatBoostClassifier':
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params'], verbose=0, cat_features=data_map['cat_cols_engineered']).fit(data_map['X_train'], data_map['y_train'])
        
        elif name == 'XGBClassifier':
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params'], enable_categorical=True).fit(data_map['X_train'], data_map['y_train'])
        
        elif name == 'HistBoostingClassifier':
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params'], categorical_features='from_dtype').fit(data_map['X_train'], data_map['y_train'])
               
        elif model_map[name]['handles_cat']:
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train'], data_map['y_train'])
        
        else:
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train_encoded'], data_map['y_train'])
    else:
        model_map[name]['model'] = load(f'../models/{name}_best.joblib')
    return model_map

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

def tune_train(data_map, model_map, runtime_map):
    for name in model_map:
        model_map = get_params(name, data_map, model_map, runtime_map)
        model_map = fit_models(name, data_map, model_map)
      

    # Predict probabilities for classification models
    print('Training: predicting on test set..')
    for name in model_map:
        if model_map[name]['refit'] == 1 or model_map[name]['retune'] == 1:
            # Predict using the correct function (model dependent)
            if model_map[name]['proba_func'] == 'decision_function':
                model_map[name]['pred_proba'] = model_map[name]['model'].decision_function(data_map['X_test'])
            elif model_map[name]['proba_func'] == 'predict_proba':
                model_map[name]['pred_proba'] = model_map[name]['model'].predict_proba(data_map['X_test'])[:, 1]


            if runtime_map['calculate_kfold']:
                print(f'Training: checking kfold performance for {name}..')
                model_map[name]['cross_val_score'] = cross_val_score(model_map[name]['model'], data_map['X_train'], data_map['y_train'], cv=runtime_map['kfold'], scoring='roc_auc')
                model_map[name]['kfold_perf'] = "%.2f%% (%.2f%%)" % (model_map[name]['cross_val_score'].mean()*100, model_map[name]['cross_val_score'].std()*100)

            print(f'Training: checking performance with chosen performance metric for {name} .. (roc-auc)')
            model_map[name]['perf'] = roc_auc_score(data_map['y_test'], model_map[name]['pred_proba'])
            
        
    write_current(model_map)
    write_better(model_map, runtime_map['perf_metric_direction'])
    return model_map
  
def main():
    # Clean up previous logs
    with open('log.txt', 'w') as f1:
        pass
    
    data_map = {'X_train': DataFrame,
                'X_train_no_engineered': DataFrame,
                'X_test': DataFrame,
                'X_pred': DataFrame,
                'y_train': DataFrame,
                'y_test': DataFrame,
                'target_col': 'loan_status',
                'index_col': 'id',
                'drop_cols': [],
                'cat_cols_raw': [],
                'num_cols_raw': [],
                'cat_cols_engineered': [],
                'num_cols_engineered': [],
                'cat_cols_encoded': [],
                'num_cols_encoded': []
                }
    
    runtime_map = {'eda_when': 'none',
                   'task_type': 'classification',
                   'scoring': 'roc_auc',
                   'perf_metric_direction': 'maximize', # 'maximize' or 'minimize'. Defined by Optuna function study option
                   'n_trials': 1,
                   'calculate_kfold': False,
                   'plots': [0,        0,        0,          0,           0,    0,  0,    1],
                            #num_plot, cat_plot, mixed_plot, single_plot, skew, mi, corr, imp
                   }   
    
    model_map = {
                'XGBClassifier':                            {'model': XGBClassifier,
                                                             'handles_cat': True,
                                                             'params': {},
                                                             'obj_func': obj_xgb,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'proba_func': 'predict_proba',
                                                             'perf': '',
                                                             'kfold_perf': ''},
                'CatBoostClassifier':                       {'model': CatBoostClassifier,
                                                             'handles_cat': True,
                                                             'params': {},
                                                             'obj_func': obj_cat,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'proba_func': 'predict_proba',
                                                             'perf': '',
                                                             'kfold_perf': ''},    
                'HistBoostingClassifier':                   {'model': HistGradientBoostingClassifier,
                                                             'handles_cat': True,
                                                             'params': {},
                                                             'obj_func': obj_histboost,
                                                             'retune': 1,
                                                             'refit': 1,
                                                             'pred_proba': '',
                                                             'proba_func': 'predict_proba',
                                                             'perf': '',
                                                             'kfold_perf': ''},                    
                'RandomForestClassifier':                   {'model': RandomForestClassifier,
                                                             'handles_cat': False,
                                                             'params': {},
                                                             'obj_func': obj_rf,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'proba_func': 'decision_function',
                                                             'perf': '',
                                                             'kfold_perf': ''},                                                                 
                'RidgeClassifier':                          {'model': RidgeClassifier,
                                                             'handles_cat': False,
                                                             'params': {},
                                                             'obj_func': obj_ridge,
                                                             'retune': 0,
                                                             'refit': 0,
                                                             'pred_proba': '',
                                                             'proba_func': 'predict_proba',
                                                             'perf': '',
                                                             'kfold_perf': ''}                                                          
                }
    
    
    # Load raw data and perform EDA on it
    data_map, runtime_map = load_data(data_map, runtime_map)                                                            # Load data, split the datasets and fillna's without data leakage
    
    eda(data_map, runtime_map)                                                                                          # EDA on raw data
    data_map = feature_engineering(data_map)                                                                            # Feature engineering. Write to data_map
    eda(data_map, runtime_map)                                                                                          # EDA on engineered data

    data_map = compare_models_with_without_engineered_features(data_map, model_map, runtime_map)
    data_map = onehotencode(data_map)                                                                                   # Add entries in data map for encoded data
    model_map = tune_train(data_map, model_map, runtime_map)                                                            # Train and evaluate models

    
    # By now a best model exists for all models. Could have been a previous best, could be a copy of the current model
    for name in model_map:
        model_map[name]['best_model'] = load(f'../models/{name}_best.joblib') 
    
    if runtime_map['plots'][-1]:
        plot_feature_importances(data_map, model_map)
        plot_permutation_importances(data_map, model_map, runtime_map)

    # Predict and submit
    for name in model_map:
        if model_map[name]['refit'] == 1:
            # If we're prediction for models that can handle categorical data
            if model_map[name]['handles_cat']:
                if model_map[name]['proba_func'] == 'decision_function':
                    predictions_curr = model_map[name]['model'].decision_function(data_map['X_pred'])
                    predictions_best = model_map[name]['best_model'].decision_function(data_map['X_pred'])
                else:
                    predictions_curr = model_map[name]['model'].predict_proba(data_map['X_pred'])[:, 1]
                    predictions_best  = model_map[name]['best_model'].predict_proba(data_map['X_pred'])[:, 1]
            
                submission_curr = DataFrame({data_map['index_col']: data_map['X_pred'].index, data_map['target_col']: predictions_curr})
                submission_curr.to_csv(f'../submissions/submission_{name}_curr.csv', index=False)
                submission_best = DataFrame({data_map['index_col']: data_map['X_pred'].index, data_map['target_col']: predictions_best})
                submission_best.to_csv(f'../submissions/submission_{name}_best.csv', index=False)
            
            # For models that don't handle categorical, we use the encoded data
            else:
                if model_map[name]['proba_func'] == 'decision_function':
                    predictions_curr = model_map[name]['model'].decision_function(data_map['X_pred_encoded'])
                    predictions_best = model_map[name]['best_model'].decision_function(data_map['X_pred_encoded'])
                else:
                    predictions_curr = model_map[name]['model'].predict_proba(data_map['X_pred_encoded'])[:, 1]
                    predictions_best  = model_map[name]['best_model'].predict_proba(data_map['X_pred_encoded'])[:, 1]
                
                submission_curr = DataFrame({data_map['index_col']: data_map['X_pred_encoded'].index, data_map['target_col']: predictions_curr})
                submission_curr.to_csv(f'../submissions/submission_{name}_curr.csv', index=False)
                submission_best = DataFrame({data_map['index_col']: data_map['X_pred_encoded'].index, data_map['target_col']: predictions_best})
                submission_best.to_csv(f'../submissions/submission_{name}_best.csv', index=False)
            
            

            

    log("\nSubmission file(s) created successfully.")

if __name__ == "__main__":
    main()
