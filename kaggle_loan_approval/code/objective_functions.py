
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

# Objective function for RandomForestClassifier
def obj_rf(trial, X_train, y_train, X_test, y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 13),  
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42}
    
    model = RandomForestClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Objective function for XBGBoostClassifier
def obj_xgb(trial, X_train, y_train, X_test, y_test):
    param = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 7, 16),
        'max_bin': trial.suggest_int('max_bin', 256, 5000), 
        'tree_method': 'auto', 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),  
        'n_estimators': trial.suggest_int('n_estimators', 50, 700),  
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True}
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Objective function for CatBoostClassifier
def obj_cat(trial, X_train, y_train, X_test, y_test, cat_cols):
    param = {
        'early_stopping_rounds': 100,
        'iterations': trial.suggest_int('iterations', 400, 600),  
        'depth': trial.suggest_int('depth', 3, 9),  
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 9.0),
        'random_seed': 42,
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'verbose': 0,
        'cat_features': cat_cols
    }
    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Objective Function for HistBoostRegressor
def obj_histboost(trial, X_train, y_train, X_test, y_test):
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
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Objective function for RidgeClassifier
def obj_ridge(trial, X_train, y_train, X_test, y_test):
    alpha = trial.suggest_float("alpha", 1e-5, 1e2, log=True)
    model = RidgeClassifier(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.decision_function(X_test)
    return roc_auc_score(y_test, preds)
