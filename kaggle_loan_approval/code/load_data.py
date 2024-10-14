from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from aux_functions import log

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() == 0:
            pass
        elif df[col].dtype in ['object', 'bool']:  # Categorical
            df.fillna({col: df[col].mode()}, inplace=True)
        else:   
            # Numerical
            skewness = round(df[col].skew(), 1)  # Decide whether to use mean or mode
            if abs(skewness) < 0.5:  # Approximately normal distribution
                mean_value = df[col].mean()
                df.fillna({col: mean_value}, inplace=True)
                log(f'{col}: Filled missing values with mean {mean_value}')
            else:  
                # Skewed distribution
                median_value = df[col].median()
                df.fillna({col: median_value}, inplace=True)
                log(f'{col}: Filled missing values for {col} with mode {median_value}')
    return df

def split(X, y):
    print('Splitting train and test..')
    num_samples = len(X)
    log(f"Number of samples in the dataset: {num_samples}")

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
    
def custom_filter(df, filter_col, filter_type, filter_amt):
    if filter_type not in ['gt', 'st']:
        raise ValueError("custom_filters function expected gt or st as condition_type, got {}".format(filter_type))
    
    print(f'train data shape before custom_filter: {df.shape}')
    
    if filter_type == 'gt':
        output = df[(df[filter_col] > filter_amt) | (df[filter_col].isna())]
        print(f'train data shape after custom_filter: {output.shape}')
        return output
    
    if filter_type == 'st':
        output = df[(df[filter_col] < filter_amt) | (df[filter_col].isna())]
        print(f'train data shape after custom_filter: {output.shape}')
        return output

def load_data(data_map, runtime_map):
    print('Loading data..')
    target_col, drop_cols = data_map['target_col'], data_map['drop_cols']
    
    train_data = read_csv('../input_data/train.csv', index_col=data_map['index_col'])
    X_pred = read_csv('../input_data/test.csv', index_col=data_map['index_col'])

    # Set some mapping variables
    data_map['cat_cols_raw'] =  [col for col in train_data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_col]
    data_map['num_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
    runtime_map['kfold'] = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) if runtime_map['task_type'] == 'classification' else KFold(n_splits=5, shuffle=True, random_state=7)

    # Cast all object columns to category data type (XGBoost, CatBoost, ...)
    for col in data_map['cat_cols_raw']:
        train_data[col] = train_data[col].astype('category')
        X_pred[col] = X_pred[col].astype('category')

    train_data = custom_filter(train_data, 'person_age', 'st', 110)
    #train_data = custom_filter(train_data, 'person_emp_length', 'st', 100) # Don't filter! It seems to degrade performance.
    train_data = custom_filter(train_data, 'loan_percent_income', 'st', 0.8)
    #train_data = custom_filter(train_data, 'person_income', 'st', 1200001)


    train_data.drop_duplicates(inplace=True)                          # Drop fully duplicated records ..
    train_data.dropna(subset=[target_col], inplace=True)              # Drop training records which don't have a target variable ..
    train_data.drop(drop_cols, axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..

    X_train_data = train_data.drop(columns=[target_col])
    y_train_data = train_data[target_col]

    X_train, X_test, y_train, y_test = split(X_train_data, y_train_data)                                                            # train test split
    #X_train, X_test, X_pred = fill_missing_values(X_train), fill_missing_values(X_test), fill_missing_values(pred_data)             # This will be used for training (no data leakage)

    return X_train, X_test, y_train, y_test, X_pred, data_map, runtime_map
