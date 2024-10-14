from pandas import DataFrame, concat, cut
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from aux_functions import log

def remove_outliers_isolation_forest(X_train, X_test, y_train, y_test, contamination=0.000001):
    # Create a copy of the original DataFrames to avoid modifying it
    cleaned_X_train, cleaned_X_test = X_train.copy(), X_test.copy()
    
    # Initialize a list to store removed records
    train_removed_records, test_removed_records = [], []

    # Loop through all numeric columns
    for col in cleaned_X_train.select_dtypes(include=['number']).columns:
        # Fit Isolation Forest on the X_train column and apply fitted model to X_test too
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        train_outliers = iso_forest.fit_predict(cleaned_X_train[[col]])  # Fit and predict on the specified column
        test_outliers = iso_forest.predict(cleaned_X_test[[col]])

        # Identify outlier rows
        train_outlier_mask = train_outliers == -1
        test_outlier_mask = test_outliers == -1

        # Store removed records in the list
        train_removed_records.append(cleaned_X_train[train_outlier_mask])
        test_removed_records.append(cleaned_X_test[test_outlier_mask])

        # Filter out the outliers
        cleaned_X_train = cleaned_X_train[~train_outlier_mask]  # Keep only non-outliers
        cleaned_X_test = cleaned_X_test[~test_outlier_mask]  # Keep only non-outliers

    # Concatenate all removed records into a single DataFrame
    train_removed_df = concat(train_removed_records, ignore_index=True)
    test_removed_df = concat(test_removed_records, ignore_index=True)

    # Print the removed records if any
    for df in [train_removed_df, test_removed_df]:
        if not df.empty:
            log("Removed records:")
            log(df)

    cleaned_y_train = y_train[cleaned_X_train.index]
    cleaned_y_test = y_test[cleaned_X_test.index]

    return cleaned_X_train, cleaned_X_test, cleaned_y_train, cleaned_y_test

def cat_to_ordered_numeric(dfs, mapping, existing_col, replace_existing=False):  
    for i, df in enumerate(dfs):
        if replace_existing:
            # Apply the mapping directly to the existing column
            df[existing_col] = df[existing_col].map(mapping)
        else:
            # Apply the mapping to new column
            df[existing_col + '_mapped'] = df[existing_col].map(mapping)
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def add_interaction_feature_number(dfs, col1, col2, operation, drop1=False, drop2=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but add_interaction_feature function got {}".format(len(dfs)))
    
    for i, df in enumerate(dfs):
        if operation == '+':
            feature = col1 + '_' + col2 + '_sum'
            df[feature] = df[col1] + df[col2]
        elif operation == '-':
            feature = col1 + '_' + col2 + '_diff'
            df[feature] = df[col1] - df[col2]
        elif operation == '*':
            feature = col1 + '_' + col2 + '_product'
            df[feature] = df[col1] * df[col2]
        elif operation == '/':
            feature = col1 + '_' + col2 + '_division'
            # Prevent division by zero
            df[feature] = df[col1] / df[col2].replace(0, np.nan)
        # Drop original columns if requested
        if drop1:
            df.drop(col1, axis=1, inplace=True)
        if drop2:
            df.drop(col2, axis=1, inplace=True)

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def add_interaction_feature_raw(dfs, num_col, cat_col, operation, drop1=False, drop2=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but add_interaction_feature function got {}".format(len(dfs)))
    
    for i, df in enumerate(dfs):
        df[cat_col + '_' + num_col + '_grouped'] = df.groupby(cat_col, observed=True)[num_col].transform('median')

        # Drop original columns if requested
        if drop1:
            df.drop(cat_col, axis=1, inplace=True)
        if drop2:
            df.drop(num_col, axis=1, inplace=True)

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def create_binned_feature(dfs, col, bins, drop=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but create_binned_feature got {}".format(len(dfs)))
    
    bin_name = col + '_binned'
    
    for i in range(len(dfs)):
        # Create binned feature
        dfs[i][bin_name] = cut(dfs[i][col], bins=bins, labels=False)
        
        # Drop the original column if requested
        if drop:
            dfs[i] = dfs[i].drop(col, axis=1)

    return dfs[0], dfs[1], dfs[2]

def boxcox_transform_skewed_features(dfs, xform_cols, threshold=1):
    shift_value = 0
    # First, determine the minimum shift value across all dataframes
    for df in dfs:        
        # Find the minimum value for all numeric features
        for col in xform_cols:
            shift_value = min(df[col].min(), shift_value)
    
    # If the shift value is negative, calculate the shift amount
    shift_amount = 0
    if shift_value < 0:
        shift_amount = abs(shift_value) + 1  # Shift to avoid negative values
    
    # Now apply the shift and Box-Cox transformation to each dataframe
    for i, df in enumerate(dfs):        
        # Iterate over each to be transformed feature
        for col in xform_cols:
            # Shift the values to ensure they are all positive for Box-Cox
            if shift_amount > 0:
                df[col] = df[col] + shift_amount
                print(f"Shifted '{col}' by {shift_amount} to avoid negative values for Box-Cox transformation.")
            
            # Calculate skewness for the current feature
            skewness = df[col].skew()

            # Apply Box-Cox transformation if skewness exceeds the threshold
            if abs(skewness) > threshold:
                print(f"Applying Box-Cox transformation to '{col}' as skewness {skewness} exceeds threshold of {threshold}.")
                df[col], _ = stats.boxcox(df[col])
        
        # Update the dataframe in the list
        dfs[i] = df

    return dfs

def scale_selected_features(dfs, cols_to_scale):
    if len(dfs) != 3:
        raise ValueError(f"Expected exactly 3 DataFrames, but got {len(dfs)}")
    
    # Instantiate the scaler
    scaler = StandardScaler()

    for i, df in enumerate(dfs):
        # Check if all the specified columns exist in the DataFrame
        missing_cols = [col for col in cols_to_scale if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing in DataFrame {i+1}: {missing_cols}")
        
        # Scale the selected columns
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs

def create_group_scaled(dfs, num_col, cat_col):
    # This function groups by the cat_col and scales the num_col within each group
    for i, df in enumerate(dfs):
        df[num_col + '_scaled'] = df.groupby(cat_col)[num_col].transform(lambda x: (x - x.mean()) / x.std())
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def drop_uninteresting(dfs, cols):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but drop_uninteresting got {}".format(len(dfs)))
       
    for i, df in enumerate(dfs):
        for col in cols:
            dfs[i] = df.drop(cols, axis=1)

    return dfs[0], dfs[1], dfs[2]

def compare_models_with_without_engineered_features(model_map, X_train, y_train, X_train_no_engineered, kfold, scoring='roc_auc'):
    print("Comparing model performance with and without engineered features..")
    for name in model_map:
        if name == 'XGBClassifier':
            model = XGBClassifier(enable_categorical = True)
            score_with_engineered = np.mean(cross_val_score(model, X_train, y_train, scoring=scoring, cv=kfold))
            score_without_engineered = np.mean(cross_val_score(model, X_train_no_engineered, y_train, scoring=scoring, cv=kfold))

            # Store the results in a dictionary
            performance_comparison = {
                'with_engineered': score_with_engineered,
                'without_engineered': score_without_engineered
            }

            # Plot the comparison
            categories = ['With Engineered Features', 'Without Engineered Features']
            scores = [performance_comparison['with_engineered'], performance_comparison['without_engineered']]

            plt.figure(figsize=(8, 6))
            plt.bar(categories, scores, color=['skyblue', 'salmon'])
            plt.title('Model Performance Comparison')
            plt.ylabel(scoring.replace('_', ' ').capitalize())  # Dynamic ylabel based on the scoring metric
            plt.ylim(0, 1)  # Assuming scoring metric is ROC AUC or similar (range 0-1)
            
            # Display the score values on the bars
            for i, score in enumerate(scores):
                plt.text(i, score + 0.02, f'{score:.5f}', ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'../eda/model/{name}_performance_comparison_{scoring}.png')
            plt.close()

            # Log the performance comparison
            log(f"Model performance with engineered features: {score_with_engineered}")
            log(f"Model performance without engineered features: {score_without_engineered}")

def onehotencode(X_train, X_test, X_pred):
    # Combine the DataFrames while keeping original indices
    combined = concat([X_train, X_test, X_pred], ignore_index=False)

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')

    # Select only the categorical columns for one-hot encoding
    categorical_cols = combined.select_dtypes(include=['object']).columns
    numeric_cols = combined.select_dtypes(exclude=['object']).columns
    
    # Fit and transform the combined data for categorical columns
    combined_encoded = DataFrame(
        encoder.fit_transform(combined[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=combined.index  # Keep the original index
    )

    # Merge the encoded categorical columns back with the original numeric columns
    combined_encoded = concat([combined[numeric_cols], combined_encoded], axis=1)

    # Split the DataFrames back using the original lengths
    X_train_encoded = combined_encoded.iloc[:len(X_train)]
    X_test_encoded = combined_encoded.iloc[len(X_train):len(X_train) + len(X_test)]
    X_pred_encoded = combined_encoded.iloc[len(X_train) + len(X_test):]

    return X_train_encoded, X_test_encoded, X_pred_encoded

def feature_engineering(X_train, X_test, y_train, y_test, X_pred, data_map, runtime_map):
    # Remove outliers
    #X_train, X_test, y_train, y_test = remove_outliers_isolation_forest(X_train, X_test, y_train, y_test)
    

    # These seem good? 
    X_train, X_test, X_pred = add_interaction_feature_raw([X_train, X_test, X_pred], 'loan_int_rate', 'loan_grade', '*')


    # Experiment

    # Update maps
    data_map['cat_cols_engineered'] = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    data_map['num_cols_engineered']= X_train.select_dtypes(include=[np.number]).columns.tolist()

    return X_train, X_test, y_train, y_test, X_pred, data_map