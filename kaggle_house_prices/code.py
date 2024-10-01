import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import optuna

def log(msg):
    fname = 'log.txt'
    if isinstance(msg, str):
        with open(fname, 'a') as file:
            file.write(msg + '\n')
    elif isinstance(msg, pd.DataFrame):
        with open(fname, 'a') as file:
            msg.to_csv(fname, index=False)

def save_best_params_to_file(filename, params):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_params_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_best_params(model_name, objective_func, n_trials=10):
    params_file = f"best_params/{model_name}.txt"
    
    # Check if params file exists
    if os.path.exists(params_file):
        log(f"Loading best parameters for {model_name} from {params_file}.")
        best_params = load_best_params_from_file(params_file)
    else:
        log(f"Optimizing parameters for {model_name}.")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_func, n_trials=n_trials)
        best_params = study.best_params
        
        # Save the best params to a file
        save_best_params_to_file(params_file, best_params)
    
    return best_params

def load_data(train_file, pred_file):
    train_data = pd.read_csv(train_file)
    pred_data = pd.read_csv(pred_file)
    return train_data, pred_data

def eda(X_train):
    # Investigate_correlations
    print('EDA: Investigating correlations ..')
    corr_matrix = X_train.corr().where(np.triu(np.abs(corr_matrix) > 0.5, k=1))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Significant correlation matrix heatmap')
    plt.savefig(f'graphs/correlation_matrix.png')
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

    # Calculate skewness for all columns and visualize the column distributions
    print('EDA: Investigating skewness..')
    for col in X_train.columns:
        skewness = round(X_train[col].skew(), 2)
        sns.histplot(X_train[col], kde=True)
        plt.title(f'Distribution of {col} (skew = {skewness})')
        plt.savefig(f'graphs/column_distribution/skew={skewness}_col={col}.png')
        plt.close()

def intelligent_train_test_split(X, y):
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

def prep_data(train_data, pred_data, target_column):
    """Preprocess the data by handling missing values, encoding categorical features and dropping fully duplicated rows."""
    def fill_missing_values(df):
        for col in df.columns:
            if df[col].isna().sum() == 0:
                pass
            elif df[col].dtype in ['object', 'bool']:  # Categorical
                df[col] = df[col].fillna(df[col].mode()[0], inplace=True)
            else:   
                # Numerical
                skewness = round(df[col].skew(), 2)  # Decide whether to use mean or mode
                if abs(skewness) < 0.5:  # Approximately normal distribution
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                    log(f'{col}: Filled missing values with mean {mean_value}')
                else:  
                    # Skewed distribution
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    log(f'{col}: Filled missing values with mode {median_value}')
        return df

    train_data.drop_duplicates(inplace=True)
    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_pred = pred_data

    fill_missing_values(X_train)
    fill_missing_values(X_pred)

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_pred = pd.get_dummies(X_pred, drop_first=True)
    
    # Align the test set with the training set, filling missing columns with 0
    X_train, X_pred = X_train.align(X_pred, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_pred

def tune_train_models(X_train, y_train, X_test, y_test, model_names, task_type):
    """Train models (with hyperparameter tuning) and evaluate final performance"""

    # Objective function for Random Forest
    def objective_rf(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42
        }
        model = RandomForestRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    # Objective function for XGBoost
    def objective_xgb(trial):
        param = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        model = XGBRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    # Objective function for CatBoost
    def objective_cat(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'random_seed': 42,
            'verbose': 0
        }
        
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    # Objective function for LogisticRegressor
    def objective_ridge(trial):
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e2)  # Regularization strength
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    print('Training: Finding best params for RandomForestRegressor..')
    best_rf_params = get_best_params("RandomForestRegressor", objective_rf)
    print('Training: Finding best params for XGB..')
    best_xgb_params = get_best_params("XGBRegressor", objective_xgb)
    print('Training: Finding best params for CatBoost..')
    best_cat_params = get_best_params("CatBoostRegressor", objective_cat)
    print('Training: Finding best params for Ridge regression..')
    best_ridge_params = get_best_params("RidgeRegression", objective_ridge)

    # Choose and train models
    model_rf = RandomForestRegressor(**best_rf_params)
    model_xgb = XGBRegressor(**best_xgb_params)
    model_cat = CatBoostRegressor(**best_cat_params, silent=True)
    model_ridge = Ridge(**best_ridge_params)

    print('Fitting all models and prediction on test set..')
    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)

    # Collect models
    models = [model_rf, model_xgb, model_cat, model_ridge]
    
    # Predict with all models
    y_pred_rf =  model_rf.predict(X_test)
    y_pred_xgb = model_xgb.predict(X_test)
    y_pred_cat = model_cat.predict(X_test)
    y_pred_ridge = model_ridge.predict(X_test)

    print('Checking KFold performance..')
    if task_type == 'classification':
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    
    cross_val_scores = [cross_val_score(model, X_train, y_train, cv=kfold) for model in models]
    perf_data = {
        'Model': model_names,
        'MSE': [mean_squared_error(y_test, y_pred_rf)
            ,mean_squared_error(y_test, y_pred_xgb)
            ,mean_squared_error(y_test, y_pred_cat)
            ,mean_squared_error(y_test, y_pred_ridge)],
        'R²': [r2_score(y_test, y_pred_rf)
            ,r2_score(y_test, y_pred_xgb)
            ,r2_score(y_test, y_pred_cat)
            ,r2_score(y_test, y_pred_ridge)],
        'KFold Validation (mean, std)': ["%.2f%% (%.2f%%)" % (cross_val_scores[0].mean()*100, cross_val_scores[0].std()*100)
                                        ,"%.2f%% (%.2f%%)" % (cross_val_scores[1].mean()*100, cross_val_scores[1].std()*100)
                                        ,"%.2f%% (%.2f%%)" % (cross_val_scores[2].mean()*100, cross_val_scores[2].std()*100)
                                        ,"%.2f%% (%.2f%%)" % (cross_val_scores[3].mean()*100, cross_val_scores[3].std()*100)]}
    log(f"Performance with k = 5 folds.")
    log(pd.DataFrame(perf_data))

    return models[0], models[1], models[2], models[3]

def plot_feature_importance(models, model_names, feature_names, top_n=10):
    """Plot the feature importance from multiple models, showing only the top N features."""
    for model, model_name in zip(models, model_names):
        if model_name == 'RidgeRegression':
            # Access the coefficients
            coefficients = model.coef_

            # Create a DataFrame for better visualization
            feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

            # Sort by absolute value of coefficients
            feature_importance['Importance'] = feature_importance['Coefficient'].abs()
            feature_importance = feature_importance.sort_values(by='Importance', ascending=True)

            # Get the top N feature names and importances
            top_features = feature_importance['Feature'].head(top_n).values
            top_importances = feature_importance['Importance'].head(top_n).values
            
        else:
            importances = model.feature_importances_
            # Get the indices of the top N features or top 30% features
            num_features_to_plot = int(min(top_n, 0.3 * len(importances)))
            top_indices = np.argsort(importances)[-num_features_to_plot:]
            # Get the top feature names and their importances
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]

        log(f"The top features are: {top_features}")
        log(f"The top feature importances are: {top_importances}")

        # Plot feature importances for the current model
        plt.figure(figsize=(10, 5))
        plt.barh(top_features, top_importances)
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.savefig(f'graphs/feature_importance_{model_name}.png')
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
        plt.savefig(f'{model.__class__.__name__}_top_features_distribution.png')  # Save the figure instead of showing
        plt.close()  # Close the figure

def main():
    # Clean up previous logs
    with open('log.txt', 'w') as file:
        pass
    
    # Specify data
    train_file = 'input_data/train.csv'
    pred_file = 'input_data/test.csv'
    target_column = 'SalePrice'
    task_type = 'number'

    # Load/preprocess
    print('Loading data..')
    train_data, pred_data = load_data(train_file, pred_file)
    print('Preprocessing data..')
    X_train, y_train, X_pred = prep_data(train_data, pred_data, target_column)

    # train test split
    print('Splitting data into train and test..')
    X_train, X_test, y_train, y_test = intelligent_train_test_split(X_train, y_train)

    # Train and evaluate models
    model_names = ['Random Forest', 'XGBoost', 'CatBoost',  'RidgeRegression']
    print('Training models..')
    model_rf, model_xgb, model_cat, model_reg = tune_train_models(X_train, y_train, X_test, y_test, model_names, task_type)
    trained_models = [model_rf, model_xgb, model_cat, model_reg]
    
    # Plot feature importance
    print('Plotting feature importance')
    plot_feature_importance(trained_models, model_names, X_pred.columns)

    print('Performing exploratory data analysis..')
    eda(X_train)
    # predict and submit
    for model in trained_models:
        predictions = model.predict(X_pred)
        submission = pd.DataFrame({'Id': X_pred['Id'], 'SalePrice': predictions})
        model_name = model.__class__.__name__
        submission.to_csv(f'submissions/submission_{model_name}.csv', index=False)

    log("Submission file(s) created successfully.")

if __name__ == "__main__":
    main()
