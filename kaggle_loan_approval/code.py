import json
import os
import shutil
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from joblib import dump, load

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import optuna

runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# Logging function
def log(msg, fname='log.txt'):
    if isinstance(msg, str):
        with open(fname, 'a') as file:
            file.write(msg + '\n')
            file.write('' + '\n')
    elif isinstance(msg, pd.DataFrame):
        with open(fname, 'a') as file:
            msg.to_csv(fname, index=False)
    
def get_params(model_name, retrain, objective_func, perf_metric_direction, n_trials=30):
    if retrain:
        start_time = time.time()
        study = optuna.create_study(direction=perf_metric_direction)
        study.optimize(objective_func, n_trials=n_trials)
        params = study.best_params
        log(f"Hyperparameter tuning done for {model_name}. Time elapsed: {start_time - time.time()} for {n_trials}.")
    else:
        param_df = pd.read_csv('performance/best.csv')
        params = param_df.loc[param_df['name'] == model_name, 'params'].values[0]
    return params

def write_current(curr_models, model_names, model_perfs, model_params,  kfold_perfs):
    curr_perf = pd.DataFrame(columns=['name','perf','kfold_perf','params', 'timestamp'])
    for model, model_name, model_perf, model_kfold_perf, model_param in zip(curr_models, model_names, model_perfs, kfold_perfs, model_params):
        new_row = pd.DataFrame({
            'name': model_name,
            'perf': model_perf,
            'kfold_perf': model_kfold_perf,
            'params': [model_param],
            'timestamp': runtime})
    
        if curr_perf.empty:
            curr_perf = new_row
        else:
            curr_perf = pd.concat([curr_perf, new_row], ignore_index=True)
        dump(model, f'models/{model_name}_current.joblib')
    curr_perf.to_csv('performance/current.csv', index=False)

def write_better(curr_models, model_names, perf_metric_direction):
    curr_perf = pd.read_csv('performance/current.csv')
    if os.path.isfile('performance/best.csv'):
        best_perf = pd.read_csv('performance/best.csv')
        for model_name, model in zip(model_names, curr_models):
            # If there is a best model/perf file, see if it can be improved with current run
            if os.path.isfile(f'models/{model_name}_best.joblib'):
                # Fetch current and best performance from the dataframe
                best_perf_row = best_perf[best_perf['name'] == model_name]
                curr_perf_row = curr_perf[curr_perf['name'] == model_name]

                # Check if rows exist
                if not best_perf_row.empty and not curr_perf_row.empty:
                    best_perf_value = best_perf_row['perf'].values[0]
                    curr_perf_value = curr_perf_row['perf'].values[0]
                    curr_params = curr_perf_row['params'].values[0]

                    # Compare based on the direction
                    if (perf_metric_direction == 'maximize' and curr_perf_value > best_perf_value) or \
                    (perf_metric_direction == 'minimize' and curr_perf_value < best_perf_value):
                        best_perf.loc[best_perf['name'] == model_name, 'perf'] = curr_perf_value
                        best_perf.loc[best_perf['name'] == model_name, 'params'] = curr_params
                        best_perf.loc[best_perf['name'] == model_name, 'timestamp'] = runtime
                        shutil.copy2(f'models/{model_name}_current.joblib', f'models/{model_name}_best.joblib')
                        print(f'Yeeehaaa! We\'ve got ourselves a new best candidate for model {model_name}!')
            else:
                shutil.copy2(f'models/{model_name}_current.joblib', f'models/{model_name}_best.joblib')
        best_perf.to_csv('performance/best.csv', index=False)
    else:
        # Copy the current performance file to best performance if it does not exist
        shutil.copy2('performance/current.csv', 'performance/best.csv')
        print("Best performance file did not exist. Current performance file copied.")

# EDA and feature importance functions
def eda_required(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            return False
    return True

def eda_and_insight(X_train_full, y_train_full, models, model_names, feature_names, skew, mi, corr, imp):
    if skew: # Calculate skewness for all columns and visualize the column distributions
        print('EDA: Investigating skewness..')
        for col in X_train_full.columns:
            skewness = round(X_train_full[col].skew(), 2)
            sns.histplot(X_train_full[col], kde=True)
            plt.title(f'Distribution of {col} (skew = {skewness})')
            plt.savefig(f'graphs/col_dist/skew={skewness}_col={col}.png')
            plt.close()
    
    if mi:   # Calculate mutual information scores on prepped data. This ensures dummified cat cols are included too
        mi_scores = mutual_info_regression(X_train_full, y_train_full)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train_full.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        top_10_mi_scores = mi_scores.nlargest(10)
        plt.figure(dpi=100, figsize=(21, 7))
        plt.barh(top_10_mi_scores.index, top_10_mi_scores.values)
        plt.xlabel("Mutual Information Score")
        plt.title("Mutual Information Scores")
        plt.savefig(f'graphs/mi/mutual_information.png')
        plt.close()
    
    if corr: # Investigate_correlations on prepped data. (.corr() can't handle NaNs)
        print('EDA: Investigating correlations ..')
        corr_matrix = X_train_full.corr()
        corr_matrix = corr_matrix.where(np.triu(np.abs(corr_matrix) > 0.5, k=1))
        plt.figure(figsize=(30, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=1)
        plt.title('Significant correlation matrix heatmap')
        plt.savefig(f'graphs/corr_matrix/correlation_matrix.png')
        plt.close()
        threshold, high_corr_pairs = 0.8, []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname1 = corr_matrix.columns[i]
                    colname2 = corr_matrix.columns[j]
                    high_corr_pairs.append((colname1, colname2, corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            log("Highly correlated feature pairs (correlation > 0.8):")
            for pair in high_corr_pairs:
                log(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
        else:
            log("No highly correlated feature pairs found with correlation > 0.8")

    if imp:  # Investigate feature importances of trained models
        """Plot the feature importance from multiple models, showing only the top N features."""
        for model, model_name in zip(models, model_names):
            if model_name == 'RidgeClassifier':
                pass
                '''
                # Access the coefficients
                coefficients = model.coef_

                # Create a DataFrame for better visualization
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

                # Sort by absolute value of coefficients
                feature_importance['Importance'] = feature_importance['Coefficient'].abs()
                feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

                # Get the top N feature names and importances
                top_features = feature_importance['Feature'].head(top_n).values
                top_importances = feature_importance['Importance'].head(top_n).values
                '''
            else:
                importances = model.feature_importances_
                # Get the indices of the top N features or top 30% features
                num_features_to_plot = int(min(10, 0.3 * len(importances)))
                top_indices = np.argsort(importances)[-num_features_to_plot:]

                top_features = [feature_names[i] for i in top_indices]
                top_importances = importances[top_indices]

                log(f"The top features for {model_name} are: {top_features}")
                log(f"The top feature importances for {model_name} are: {top_importances}")

                plt.figure(figsize=(21, 7))
                plt.barh(top_features, top_importances)
                plt.title(f'Feature Importance - {model_name}')
                plt.xlabel('Importance')
                plt.savefig(f'graphs/feature_imp/feature_importance_{model_name}.png')
                plt.close()

def plot_top_columns(X_train, models, top_n=10):
    for model in models:
        """Visualize distributions of the top N most important features for all models."""
        feature_importances = model.feature_importances_
        
        # Get the indices of the top N features
        top_indices = feature_importances.argsort()[-top_n:][::-1]
        top_features = X_train.columns[top_indices]
        
        # Plot distributions for top features
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features):
            plt.subplot(5, 2, i + 1)
            sns.histplot(X_train[feature], kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.savefig(f'{model.__class__.__name__}_top_features_distribution.png')
        plt.close()

# Training and fitting functions
def load_data(train_file, pred_file, target_col, drop_cols):
    train_data = pd.read_csv(train_file)
    pred_data = pd.read_csv(pred_file)

    """Preprocess the data by handling missing values, encoding categorical features and dropping fully duplicated rows."""
    def fill_missing_values(df):
        for col in df.columns:
            if df[col].isna().sum() == 0:
                pass
            elif df[col].dtype in ['object', 'bool']:  # Categorical
                df.fillna({col: df[col].mode()}, inplace=True)
            else:   
                # Numerical
                skewness = round(df[col].skew(), 2)  # Decide whether to use mean or mode
                if abs(skewness) < 0.5:  # Approximately normal distribution
                    mean_value = df[col].mean()
                    df.fillna({col: mean_value}, inplace=True)
                    log(f'{col}: Filled missing values with mean {mean_value}')
                else:  
                    # Skewed distribution
                    median_value = df[col].median()
                    df.fillna({col: median_value}, inplace=True)
                    log(f'{col}: Filled missing values with mode {median_value}')
        return df

    train_data.drop_duplicates(inplace=True)
    train_data.drop(drop_cols, axis=1, inplace=True)

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_pred = pred_data

    fill_missing_values(X_train)
    fill_missing_values(X_pred)

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_pred = pd.get_dummies(X_pred, drop_first=True)
    
    # Align the test set with the training set, filling missing columns with 0
    X_train, X_pred = X_train.align(X_pred, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_pred

def split(X, y):
    # Determine the size of the dataset
    num_samples = len(X)
    log(f"Number of samples in the dataset: {num_samples}")

    # Decide the train/test split ratio
    if num_samples < 100:
        train_size = 0.5  # 50/50 split for small datasets
        log("Using 50/50 split for small dataset.")
    elif num_samples < 1000:
        train_size = 0.7  # 70/30 split for medium datasets
        log("Using 70/30 split for medium dataset.")
    else:
        train_size = 0.8  # 80/20 split for large datasets
        log("Using 80/20 split for large dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test

def tune_train_current_models(X_train, y_train, X_test, y_test, model_names, task_type, perf_metric_direction, n_trials, retrain = False):
    # Objective function for RandomForestClassifier
    def objective_rf(trial):
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
    def objective_xgb(trial):
        param = {
            'objective': 'binary:logistic',
            'max_depth': trial.suggest_int('max_depth', 4, 13),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),  
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),  
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),  
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)} 
        
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)
    
    # Objective function for CatBoostClassifier
    def objective_cat(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 100, 500),  
            'depth': trial.suggest_int('depth', 4, 13),  
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3),  
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'random_seed': 42,
            'verbose': 0,
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1)} 
        
        model = CatBoostClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    # Objective function for RidgeClassifier
    def objective_ridge(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 1e2, log=True)
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.decision_function(X_test)
        return roc_auc_score(y_test, preds)

    print('\nTraining: hyperparameter optimization for RandomForestClassifier..')
    rf_params = get_params("RandomForestClassifier", retrain, objective_rf, perf_metric_direction, n_trials)
    print('Training: hyperparameter optimization for XGBClassifier..')
    xgb_params = get_params("XGBClassifier", retrain, objective_xgb, perf_metric_direction, n_trials)
    print('Training: hyperparameter optimization for CatBoostClassifier..')
    cat_params = get_params("CatBoostClassifier", retrain, objective_cat, perf_metric_direction, n_trials)
    print('Training: hyperparameter optimization for RidgeClassifier..')
    ridge_params = get_params("RidgeClassifier", retrain, objective_ridge, perf_metric_direction, n_trials)

    # Choose and train models
    model_rf = RandomForestClassifier(**rf_params)
    model_xgb = XGBClassifier(**xgb_params)
    model_cat = CatBoostClassifier(**cat_params, silent=True, allow_writing_files=False)
    model_ridge = RidgeClassifier(**ridge_params)

    print('Training: fitting models..')
    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)

    # Collect models
    curr_models = [model_rf, model_xgb, model_cat, model_ridge]
    curr_models_params = [rf_params, xgb_params, cat_params, ridge_params] 
    
    # Predict probabilities for classification models
    print('Training: predicting on test set..')
    y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]
    y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    y_pred_proba_cat = model_cat.predict_proba(X_test)[:, 1]
    y_pred_proba_ridge = model_ridge.decision_function(X_test)  # RidgeClassifier uses decision_function

    print('Training: checking kfold performance..')
    if task_type == 'classification':
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    
    cross_val_scores = [cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc') for model in curr_models]
    
    print('Training: checking performance with chosen performance metric .. (roc-auc)')
    model_perfs = [roc_auc_score(y_test, y_pred_proba_rf),
                            roc_auc_score(y_test, y_pred_proba_xgb),
                            roc_auc_score(y_test, y_pred_proba_cat),
                            roc_auc_score(y_test, y_pred_proba_ridge)]
    kfold_perfs = ["%.2f%% (%.2f%%)" % (cross_val_scores[0].mean()*100, cross_val_scores[0].std()*100),
                                        "%.2f%% (%.2f%%)" % (cross_val_scores[1].mean()*100, cross_val_scores[1].std()*100),
                                        "%.2f%% (%.2f%%)" % (cross_val_scores[2].mean()*100, cross_val_scores[2].std()*100),
                                        "%.2f%% (%.2f%%)" % (cross_val_scores[3].mean()*100, cross_val_scores[3].std()*100)]
    
    write_current(curr_models, model_names, model_perfs, curr_models_params, kfold_perfs)
    write_better(curr_models, model_names, perf_metric_direction)


def main():
    # Clean up previous logs
    with open('log.txt', 'w') as f1:
        pass
    
    # File and folder locations
    train_file = 'input_data/train.csv'
    pred_file = 'input_data/test.csv'
    
    feature_imp = 'graphs/feature_imp'
    col_distr = 'graphs/col_dist'
    mi = 'graphs/mi'
    corr_matrix = 'graphs/corr_matrix'

    # Data column names
    index_col = 'id'
    target_col = 'loan_status'
    drop_cols = []
    task_type = 'classification'
    retrain = True
    perf_metric_direction = 'maximize' # 'maximize' or 'minimize'. Defined by Optuna function study option
    n_trials = 50

    # Load/preprocess
    print('Loading and preprocessing data..')
    X_train, y_train, X_pred = load_data(train_file, pred_file, target_col, drop_cols)
    X_train_full, y_train_full = X_train, y_train   # full data needed for preprocessed eda (before train/test split)

    # train test split
    print('Splitting train and test..')
    X_train, X_test, y_train, y_test = split(X_train, y_train)

    # Train and evaluate models
    model_names = ['RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier', 'RidgeClassifier']
    tune_train_current_models(X_train, y_train, X_test, y_test, model_names, task_type, perf_metric_direction, n_trials, retrain)
    best_models = [load(f'models/{model_name}_best.joblib') for model_name in model_names]
    

    eda_and_insight(X_train_full, y_train_full, best_models, model_names, X_pred.columns, eda_required(col_distr), eda_required(mi), eda_required(corr_matrix), eda_required(feature_imp))

    # predict and submit
    for model, model_name in zip(best_models, model_names):
        if model_name == 'RidgeClassifier':
            predictions = model.decision_function(X_pred)
        else:
            predictions = model.predict_proba(X_pred)[:, 1]
            submission = pd.DataFrame({index_col: X_pred[index_col], target_col: predictions})
            submission.to_csv(f'submissions/submission_{model_name}.csv', index=False)

    log("\nSubmission file(s) created successfully.")

if __name__ == "__main__":
    main()
