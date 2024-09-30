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

def load_data(train_file):
    """Load the training and testing data from CSV files."""
    train_data = pd.read_csv(train_file)
    return train_data

def preprocess_data(train_data, target_column):
    """Preprocess the data by handling missing values, encoding categorical features and dropping fully duplicated rows."""
    train_data.drop_duplicates(inplace=True)
    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    # Fill missing values
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Categorical
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        else:  # Numerical
            X_train[col] = X_train[col].fillna(X_train[col].mean())

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    y_train = pd.get_dummies(y_train, drop_first=True)

    return X_train, y_train

def tune_train_models(X_train, y_train, X_valid, y_valid, model_names):
    """Train all models (with hyperparameter tuning) evaluate final performance and make final predictions."""

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
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

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
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

    # Objective function for CatBoost
    def objective_cat(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'random_seed': 42,
            'verbose': 0  # Disable output
        }
        
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

    # Optimize Random Forest Regressor
    study_rf = optuna.create_study(direction='minimize')
    study_rf.optimize(objective_rf, n_trials=5)
    best_rf_params = study_rf.best_params
    print("Best Random Forest Parameters:", best_rf_params)
    
    # Optimize XGBoost
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=5)
    best_xgb_params = study_xgb.best_params
    print("Best XGBoost Parameters:", best_xgb_params)

    # Optimize CatBoost
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_cat, n_trials=5)
    best_cat_params = study_cat.best_params
    print("Best CatBoost Parameters:", best_cat_params)

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
    y_pred_rf, y_pred_xgb, y_pred_cat = model_rf.predict(X_valid), model_xgb.predict(X_valid), model_cat.predict(X_valid)

    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    cross_val_scores = [cross_val_score(model, X_train, y_train, cv=kfold) for model in models]

    perf_data = {
        'Model': model_names,
        'MSE': [mean_squared_error(y_valid, y_pred_rf)
            ,mean_squared_error(y_valid, y_pred_xgb)
            ,mean_squared_error(y_valid, y_pred_cat)],
        'R²': [r2_score(y_valid, y_pred_rf)
            ,r2_score(y_valid, y_pred_xgb)
            ,r2_score(y_valid, y_pred_cat)],
        'KFold Validation (mean, std)': ["%.2f%% (%.2f%%)" % (cross_val_scores[0].mean()*100, cross_val_scores[0].std()*100)
                                        ,"%.2f%% (%.2f%%)" % (cross_val_scores[1].mean()*100, cross_val_scores[1].std()*100)
                                        ,"%.2f%% (%.2f%%)" % (cross_val_scores[2].mean()*100, cross_val_scores[2].std()*100)]}

    perf = pd.DataFrame(perf_data)
    
    print(perf)

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
        plt.savefig(f'feature_importance_{model_name}.png')  # Save the figure for the model
        plt.close()  # Close the figure

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
    train_file = 'train.csv'
    
    # Load your data
    train_data= load_data(train_file)

    # Specify your target column
    target_column = 'Churn'  # Target column

    # Preprocess the data
    X_train, y_train = preprocess_data(train_data, target_column)

    # Split the dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Train and evaluate models
    model_names = ['Random Forest', 'XGBoost', 'CatBoost']  # List of model names (used for saving files)
    model_rf, model_xgb, model_cat = tune_train_models(X_train, y_train, X_valid, y_valid, model_names)
    trained_models = [model_rf, model_xgb, model_cat]  # List of models

    # Plot feature importance
    plot_feature_importance(trained_models, model_names, X_train.columns)


    print("Submission file(s) created successfully.")

if __name__ == "__main__":
    main()
