import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def load_data(train_file, test_file):
    """Load the training and testing data from CSV files."""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def preprocess_data(data, target_column):
    """Preprocess the data by handling missing values and encoding categorical features."""
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':  # Categorical
            X[col] = X[col].fillna(X[col].mode()[0])
        else:  # Numerical
            X[col] = X[col].fillna(X[col].mean())

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

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

    # Calculate performance metrics
    metrics_rf = {
        'MSE': mean_squared_error(y_valid, y_pred_rf),
        'R²': r2_score(y_valid, y_pred_rf)
    }
    
    metrics_xgb = {
        'MSE': mean_squared_error(y_valid, y_pred_xgb),
        'R²': r2_score(y_valid, y_pred_xgb)
    }

    print("Random Forest Metrics:", metrics_rf)
    print("XGBoost Metrics:", metrics_xgb)

    return model_rf, model_xgb


def plot_feature_importance(model_rf, model_xgb, feature_names):
    """Plot the feature importance from both models, showing only the top 10 features."""
    
    # Get feature importances
    rf_importances = model_rf.feature_importances_
    xgb_importances = model_xgb.feature_importances_

    # Get the indices of the top 10 features or top 30% features

    rf_features_plot = -1 * min(10, len(rf_importances))
    xgb_features_plot  = -1 * min(10, len(xgb_importances))

    rf_indices = np.argsort(rf_importances)[rf_features_plot :]
    xgb_indices = np.argsort(xgb_importances)[xgb_features_plot:]

    # Get the top 10 feature names and importances for Random Forest
    rf_top_features = [feature_names[i] for i in rf_indices]
    rf_top_importances = rf_importances[rf_indices]

    # Plot feature importances for Random Forest
    plt.figure(figsize=(10, 5))
    plt.barh(rf_top_features, rf_top_importances)
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.savefig('feature_importance_rf.png')  # Save the figure instead of showing
    plt.close()  # Close the figure

    # Get the top 10 feature names and importances for XGBoost
    xgb_top_features = [feature_names[i] for i in xgb_indices]
    xgb_top_importances = xgb_importances[xgb_indices]

    # Plot feature importances for XGBoost
    plt.figure(figsize=(10, 5))
    plt.barh(xgb_top_features, xgb_top_importances)
    plt.title('Feature Importance - XGBoost')
    plt.xlabel('Importance')
    plt.savefig('feature_importance_xgb.png')  # Save the figure instead of showing
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
    X, y = preprocess_data(train_data, target_column)

    # Split the dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run models
    model_rf, model_xgb = run_models(X_train, y_train, X_valid, y_valid)

    # Plot feature importance
    plot_feature_importance(model_rf, model_xgb, X.columns)

    # Visualize top columns distributions
    visualize_top_columns(X, model_rf.feature_importances_, top_n=10)
    visualize_top_columns(X, model_xgb.feature_importances_, top_n=10)

if __name__ == "__main__":
    main()
