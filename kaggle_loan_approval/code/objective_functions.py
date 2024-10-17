from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import cross_val_score

# Objective function for RandomForestClassifier
def obj_rf(trial, data_map, runtime_map):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 13),  
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42}
    
    model = RandomForestClassifier(**param)
    cv_scores = cross_val_score(model, convert_sparse_to_df(data_map, 'X_train_encoded'), data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for XBGBoostClassifier
def obj_xgb(trial, data_map, runtime_map):
    param = {
        'objective': 'binary:logistic',
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
        'enable_categorical': True}
    
    model = XGBClassifier(**param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for CatBoostClassifier
def obj_cat(trial, data_map, runtime_map):
    param = {
        'early_stopping_rounds': 50,                                # Future note: implement logic to determine this based on number of dataset records
        'iterations': trial.suggest_int('iterations', 400, 600),  
        'depth': trial.suggest_int('depth', 3, 7),  
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 9.0),
        'random_seed': 42,
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'verbose': 0,
        'cat_features': data_map['cat_cols_engineered']
    }
    model = CatBoostClassifier(**param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective Function for HistBoostRegressor
def obj_histboost(trial, data_map, runtime_map):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 200),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-10, 1.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 100, 255),
        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
        'categorical_features': 'from_dtype'
    }
    model = HistGradientBoostingClassifier(**params)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for RidgeClassifier
def obj_ridge(trial, data_map, runtime_map):
    alpha = trial.suggest_float("alpha", 1e-5, 1e2, log=True)
    model = RidgeClassifier(alpha=alpha)
    cv_scores = cross_val_score(model, convert_sparse_to_df(data_map, 'X_train_encoded'), data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()
