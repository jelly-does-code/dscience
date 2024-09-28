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

def load_data(train_file, test_file):
    """Load the training and testing data from CSV files."""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def preprocess_data(train_data, test_data, target_column):
    """Preprocess the data by handling missing values and encoding categorical features."""
    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data


    # Fill missing values
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Categorical
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        else:  # Numerical
            X_train[col] = X_train[col].fillna(X_train[col].mean())

    
    # Fill missing values
    for col in X_test.columns:
        if X_test[col].dtype == 'object':  # Categorical
            X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
        else:  # Numerical
            X_test[col] = X_test[col].fillna(X_test[col].mean())

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    
    
    # Align the test set with the training set, filling missing columns with 0
    X_test, _ = X_test.align(X_train, join='right', axis=1, fill_value=0)

    return X_train, y_train, X_test

def run_models(X_train, y_train, X_valid, y_valid):
    """Train Random Forest and XGBoost models and evaluate performance."""
    # Train Random Forest
    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train, y_train)

    # Train XGBoost
    model_xgb = XGBRegressor(random_state=42)
    model_xgb.fit(X_train, y_train)

    # Predictions
    y_pred_rf = model_rf.predict(X_valid)
    y_pred_xgb = model_xgb.predict(X_valid)


    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    rf_cross_val = cross_val_score(model_rf, X_train, y_train, cv=kfold)
    xgb_cross_val = cross_val_score(model_xgb, X_train, y_train, cv=kfold)

    


    perf_data = {
        'Model': ['Random Forest', 'XGB'],
        'MSE': [mean_squared_error(y_valid, y_pred_rf)
            ,mean_squared_error(y_valid, y_pred_xgb)],
        'R²': [r2_score(y_valid, y_pred_rf)
            ,r2_score(y_valid, y_pred_xgb)],
        'KFold Validation (mean, std)': ["%.2f%% (%.2f%%)" % (rf_cross_val.mean()*100, rf_cross_val.std()*100)
                                        ,"%.2f%% (%.2f%%)" % (xgb_cross_val.mean()*100, xgb_cross_val.std()*100)]}

    perf = pd.DataFrame(perf_data)
    
    print(perf)

    return model_rf, model_xgb

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

def visualize_top_columns(X, feature_importances, top_n=10):
    """Visualize distributions of the top N most important features."""
    # Get the indices of the top N features
    top_indices = feature_importances.argsort()[-top_n:][::-1]
    top_features = X.columns[top_indices]
    
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
    test_file = 'test.csv'
    
    # Load your data
    train_data, test_data = load_data(train_file, test_file)

    # Specify your target column
    target_column = 'SalePrice'  # Example target column

    # Preprocess the data
    X_train, y_train, X_test = preprocess_data(train_data, test_data, target_column)

    # Split the dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Run models
    model_rf, model_xgb = run_models(X_train, y_train, X_valid, y_valid)

    models = [model_rf, model_xgb]  # List of models
    model_names = ['Random Forest', 'XGBoost']  # List of model names (used for saving files)

    # Plot feature importance
    plot_feature_importance(models, model_names, X_test.columns)

    # Visualize top columns distributions
    visualize_top_columns(X_train, model_rf.feature_importances_, top_n=10)
    visualize_top_columns(X_train, model_xgb.feature_importances_, top_n=10)

    test_rf_predictions = model_rf.predict(X_test)
    # rf submission file
    submission_rf = pd.DataFrame({'Id': X_test['Id'], 'SalePrice': test_rf_predictions})
    submission_rf.to_csv('submission_rf.csv', index=False)

    test_xgb_predictions = model_xgb.predict(X_test)
    # xgb submission file
    submission_xgb = pd.DataFrame({'Id': X_test['Id'], 'SalePrice': test_xgb_predictions})
    submission_xgb.to_csv('submission_xgb.csv', index=False)

    print("Submission file(s) created successfully.")

if __name__ == "__main__":
    main()
