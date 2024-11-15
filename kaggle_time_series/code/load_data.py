from pandas import merge, read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from functions import log

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() == 0:
            pass
        elif df[col].dtype in ['object', 'bool', 'category']:  # Categorical
            df[col].fillna('na', inplace=True)
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
    
    if filter_type == 'gt':
        output = df[(df[filter_col] > filter_amt) | (df[filter_col].isna())]
    
    if filter_type == 'st':
        output = df[(df[filter_col] < filter_amt) | (df[filter_col].isna())]

    log(f'train data shape change after custom_filter: {df.shape}, {output.shape}')
    return output

def load_data(data_map, runtime_map):
    print('\nLoading data..')
    target_col = data_map['target_col']                                      # For readability
    
    # Load csv files into dataframes
    #train_data = read_csv(data_map['train_file'], index_col=data_map['index_col'])
    train_data = read_csv('../input_data/train_code_sample.csv', index_col=data_map['index_col'])
    oil_data = read_csv('../input_data/oil.csv', index_col='date')
    stores_data = read_csv('../input_data/stores.csv', index_col='store_nbr')
    holidays_data = read_csv('../input_data/holidays_events.csv', index_col='date')
    transactions_data = read_csv('../input_data/transactions.csv', index_col='date')
    X_pred = read_csv(data_map['pred_file'], index_col=data_map['index_col'])


    # Combine data into single train/pred dataframes
  
    train_data = merge(train_data, oil_data, on='date', how='left')             # Add oil price data
    train_data = merge(train_data, stores_data, on='store_nbr', how='left')     # Add store information
    train_data = merge(train_data, holidays_data, on='date', how='left')        # Add holidays
    # Interpret transferred column as boolean
    train_data['transferred'] = train_data['transferred'].replace({'True': True, 'False': False}).astype(bool)
    train_data = train_data[~train_data['transferred']]                         # Transferred data has data point elsewhere, under type: Transferred. So remove

    
    train_data = merge(train_data, transactions_data, on='date', how='left')    # Add transactions_data
    
    X_pred = merge(X_pred, oil_data, on='date', how='left')                     # Add oil price data
    X_pred = merge(X_pred, stores_data, on='store_nbr', how='left')             # Add store information
    X_pred = merge(X_pred, holidays_data, on='date', how='left')                # Add holidays
    
    # Interpret transferred column as boolean
    X_pred['transferred'] = X_pred['transferred'].replace({'True': True, 'False': False}).astype(bool)
    X_pred = X_pred[~X_pred['transferred']]                                     # Transferred data has data point elsewhere, under type: Transferred. So remove
    
    X_pred = merge(X_pred, transactions_data, on='date', how='left')            # Add transactions_data
    

    # Free up memory ..
    del oil_data, stores_data, holidays_data, transactions_data


    # Note different columns, determine kfold type
    data_map['cat_cols_raw'] =  [col for col in train_data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_col]
    data_map['num_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
    data_map['date_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.datetime64]).columns.tolist() if col != target_col]
    runtime_map['kfold'] = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) if runtime_map['task_type'] == 'classification' else KFold(n_splits=5, shuffle=True, random_state=7)

    # Cast all object columns to category data type
    for col in data_map['cat_cols_raw']:
        train_data[col] = train_data[col].astype('category')
        X_pred[col] = X_pred[col].astype('category')

    # Apply filters to training data
    filter_conditions = {}
    for col, (value, method) in filter_conditions.items():
        data_map['train_data'] = custom_filter(data_map['train_data'], col, method, value)

    train_data.drop_duplicates(inplace=True)                                      # Drop fully duplicated records ..
    train_data.dropna(subset=target_col, inplace=True)                            # Drop training records which don't have a target variable ..
    train_data.drop(data_map['drop_cols'], axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..
    X_pred.drop(data_map['drop_cols'], axis=1, inplace=True)                      # Drop irrelevant columns as defined in drop_cols ..

    # Create X, y for training data
    y_train_data = train_data.pop(target_col)

    # Train/test split
    data_map['X_train'], data_map['X_test'], data_map['y_train'], data_map['y_test'] = split(train_data, y_train_data)   # Train test split
    data_map['X_train_no_engineered'] = data_map['X_train']                                                                # Save this for comparing model with/without feature engineering 
    
    # Note index values. This is needed for converting sparse back to dense later on
    data_map['X_train_index_values'] = data_map['X_train'].index.tolist()
    data_map['X_test_index_values'] = data_map['X_test'].index.tolist()
    data_map['X_pred_index_values'] = X_pred.index.tolist()

    #data_map['X_train'], data_map['X_test'], data_map['X_pred'] = fill_missing_values(data_map['X_train']), fill_missing_values(data_map['X_test']), fill_missing_values(X_pred)             # This will be used for training (no data leakage)
    
    del train_data, y_train_data, X_pred

    log(f"X_train shape: {data_map['X_train']}")
    log(f"X_test shape: {data_map['X_test']}")
    log(f"X_pred shape: {data_map['X_pred']}")


    return data_map, runtime_map



