import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
import optuna
import json
import os

def save_best_params_to_file(filename, params):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_params_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_best_params(model_name, objective_func, n_trials=50):
    params_file = f"best_params/{model_name}.txt"
    
    # Check if params file exists
    if os.path.exists(params_file):
        print(f"Loading best parameters for {model_name} from file.")
        best_params = load_best_params_from_file(params_file)
    else:
        print(f"Optimizing parameters for {model_name}.")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_func, n_trials=n_trials)
        best_params = study.best_params
        
        # Save the best params to a file
        save_best_params_to_file(params_file, best_params)
    
    return best_params

def load_data(train_file, pred_file):
    """Load the training and testing data from CSV files."""
    train_data = pd.read_csv(train_file)
    pred_data = pd.read_csv(pred_file)
    return train_data, pred_data

def intelligent_train_test_split(X, y):
    # Determine the size of the dataset
    num_samples = len(X)
    print(f"Number of samples in the dataset: {num_samples}")

    # Decide the train/test split ratio
    if num_samples < 100:
        train_size = 0.5  # 50/50 split for small datasets
        print("Using 50/50 split for small dataset.")
    elif num_samples < 1000:
        train_size = 0.7  # 70/30 split for medium datasets
        print("Using 70/30 split for medium dataset.")
    else:
        train_size = 0.8  # 80/20 split for large datasets
        print("Using 80/20 split for large dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test

def fill_missing_values(df, column):
    # Check for NaN values
    if df[column].isna().sum() == 0:
        print("No missing values to fill.")
        return df

    # Calculate skewness and visualize the column distribution
    skewness = df[column].skew()
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column} (skew={skewness})')
    plt.savefig(f'Distribution of {column} (skew={skewness}).png')
    plt.close()


    # Decide whether to use mean or mode
    if abs(skewness) < 0.5:  # Approximately normal distribution
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
        print(f'Filled missing values with mean: {mean_value}')
    else:  # Skewed distribution
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
        print(f'Filled missing values with mode: {mode_value}')

    return df

def prep_data(train_data, pred_data, target_column):
    """Preprocess the data by handling missing values, encoding categorical features and dropping fully duplicated rows."""
    train_data.drop_duplicates(inplace=True)
    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_pred = pred_data

    # Fill missing values
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Categorical
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        else:  # Numerical
            X_train[col] = X_train[col].fillna(X_train[col].mean())
            X_train = fill_missing_values(X_train, col)

    # Fill missing values
    for col in X_pred.columns:
        if X_pred[col].dtype == 'object':  # Categorical
            X_pred[col] = X_pred[col].fillna(X_pred[col].mode()[0])
        else:  # Numerical
            X_pred = fill_missing_values(X_pred, col)

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_pred = pd.get_dummies(X_pred, drop_first=True)
    
    # Align the test set with the training set, filling missing columns with 0
    X_train, X_pred = X_train.align(X_pred, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_pred

def tune_train_models(X_train, y_train, X_test, y_test, model_names):
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
            'verbose': 0}
        
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    best_rf_params = get_best_params("RandomForestRegressor", objective_rf)
    best_xgb_params = get_best_params("XGBRegressor", objective_xgb)
    best_cat_params = get_best_params("CatBoostRegressor", objective_cat)

    # Choose and train models
    model_rf = RandomForestRegressor(**best_rf_params)
    model_xgb = XGBRegressor(**best_xgb_params)
    model_cat = CatBoostRegressor(**best_cat_params, silent=True)

    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)

    # Collect models
    models = [model_rf, model_xgb, model_cat]
    
    # Predict with all models
    y_pred_rf, y_pred_xgb, y_pred_cat = model_rf.predict(X_test), model_xgb.predict(X_test), model_cat.predict(X_test)

    for kfold_splits in [5, 10]:
        kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=7)
        cross_val_scores = [cross_val_score(model, X_train, y_train, cv=kfold) for model in models]
        perf_data = {
            'Model': model_names,
            'MSE': [mean_squared_error(y_test, y_pred_rf)
                ,mean_squared_error(y_test, y_pred_xgb)
                ,mean_squared_error(y_test, y_pred_cat)],
            'R²': [r2_score(y_test, y_pred_rf)
                ,r2_score(y_test, y_pred_xgb)
                ,r2_score(y_test, y_pred_cat)],
            'KFold Validation (mean, std)': ["%.2f%% (%.2f%%)" % (cross_val_scores[0].mean()*100, cross_val_scores[0].std()*100)
                                            ,"%.2f%% (%.2f%%)" % (cross_val_scores[1].mean()*100, cross_val_scores[1].std()*100)
                                            ,"%.2f%% (%.2f%%)" % (cross_val_scores[2].mean()*100, cross_val_scores[2].std()*100)]}
        print(f"Performance with k = {kfold_splits} folds.")
        print(pd.DataFrame(perf_data))

    return models[0], models[1], models[2]

def plot_feature_importance(models, model_names, feature_names, top_n=10):
    """Plot the feature importance from multiple models, showing only the top N features."""
    for model, model_name in zip(models, model_names):
        # Get feature importances
        importances = model.feature_importances_

        # Get the indices of the top N features or top 30% features
        num_features_to_plot = int(min(top_n, 0.3 * len(importances)))
        top_indices = np.argsort(importances)[-num_features_to_plot:]

        # Get the top feature names and their importances
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]

        # Plot feature importances for the current model
        plt.figure(figsize=(10, 5))
        plt.barh(top_features, top_importances)
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.savefig(f'graphs/feature_importance_{model_name}.png')
        plt.close()

def visualize_top_columns(X_train, models, top_n=10):
    
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
            sns.histplot(X[feature], kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.savefig(f'top_features_distribution.png')  # Save the figure instead of showing
        plt.close()  # Close the figure

def main():
    # Specify your files
    train_file = 'input_data/train.csv'
    pred_file = 'input_data/test.csv'
    target_column = 'SalePrice'

    # Load/preprocess
    train_data, pred_data = load_data(train_file, pred_file)
    X_train, y_train, X_pred = prep_data(train_data, pred_data, target_column)

    # train test split
    X_train, X_test, y_train, y_test = intelligent_train_test_split(X_train, y_train)

    # Train and evaluate models
    model_names = ['Random Forest', 'XGBoost', 'CatBoost']
    model_rf, model_xgb, model_cat = tune_train_models(X_train, y_train, X_test, y_test, model_names)
    trained_models = [model_rf, model_xgb, model_cat]
    # Plot feature importance
    plot_feature_importance(trained_models, model_names, X_pred.columns)

    for model in trained_models:
        # predict and submit
        predictions = model.predict(X_pred)
        submission = pd.DataFrame({'Id': X_pred['Id'], 'SalePrice': predictions})
        model_name = model.__class__.__name__
        submission.to_csv(f'submissions/submission_{model_name}.csv', index=False)

    print("Submission file(s) created successfully.")

if __name__ == "__main__":
    main()
