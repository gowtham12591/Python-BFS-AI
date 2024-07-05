
import pandas as pd
import numpy as np
import traceback
# Auto plots
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def missing_value(df):
    # Check for missing values and replace it with appropriate methods
    try:
        # Identify columns with missing values
        missing_value_columns = df.columns[df.isnull().sum() > 0]

        # Loop through the identified columns and replace NaN values with the mean of the column
        for column in missing_value_columns:
            if df[column].dtype in ['float64', 'int64', 'datetime64[ns]']:  # Ensure column is numeric or date time
                # df[column].fillna(df[column].mean(), inplace=True) # Replacing with mean
                df[column] = df[column].fillna(df[column].median()) # Replacing with median

            else:
                df[column] = df[column].fillna(df[column].mode()) # Replacing with mode for object type columns

        return df, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 400
    
def duplicate_value(df):

    try:
        # Check for duplicates in the entire dataframe
        if len(df[df.duplicated()]) > 0:
            df.dropduplicates(inplace=True)
            return df, 200
        
        else:
            return df, 200
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 400
    
def data_visualization(df, data_path):

    try:
        # Automated plots for easy visuzlization
        AV = AutoViz_Class()
        profile_autoviz = AV.AutoViz('data_path', sep=';', depVar='y', dfte='Data', header=0, verbose=1, lowess=False,
                    chart_format='html',save_plot_dir='AutoViz_Plots')
        
        # Another report generating tool to analyze/visualize the reports in pandas
        # Pandas profiling
        profile_pandas = ProfileReport(df)
        profile_pandas.to_file('pandas_profile.html')

    except Exception as e:
        print(f"Exception in received request {traceback.format_exc()}")
    

def encoding(df):

    try:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        #Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)

        # Apply one-hot encoding to the categorical columns
        one_hot_encoded = encoder.fit_transform(df[categorical_columns])

        #Create a DataFrame with the one-hot encoded columns
        #We use get_feature_names_out() to get the column names for the encoded data
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

        # Concatenate the one-hot encoded dataframe with the original dataframe
        df_encoded = pd.concat([df, one_hot_df], axis=1)

        # Drop the original categorical columns
        df_encoded = df_encoded.drop(categorical_columns, axis=1)

        return df_encoded, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 400
    
def feature_splitting(df):

    try:
        # Splitting the independent and target features
        independent_features = df.drop('y', axis=1)
        target_feature = df[['y']]

        return independent_features, target_feature, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 'error', 400
    
def data_splitting(independent_features, target_feature):

    try:

        count = target_feature.value_counts()
        print('Target feature before sampling: \n', count)

        # Target feature balancing (up-sampling)
        smote = SMOTE()
        independent_features_sampled, target_feature_sampled = smote.fit_resample(independent_features, target_feature)

        count_sam = target_feature_sampled.value_counts()
        print('Target feature after sampling: \n', count_sam)

        # Train_test_split (As the dataset is not very large, let us take the test_size to be 15% of the entire dataframe)
        # As we have sampled, startify is not necessary
        X_train, X_val, y_train, y_val = train_test_split(independent_features_sampled, target_feature_sampled, 
                                                        test_size=0.15, 
                                                        random_state=42, 
                                                        shuffle=True, 
                                                        stratify=target_feature_sampled)

        print(X_train.shape, X_val.shape)
        print(y_train.shape, y_val.shape)

        return X_train, X_val, y_train, y_val, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 'error', 'error', 'error', 400

def scaling(X_train, X_val):

    try:

        # Let us scale the features using z-scalar technique
        sc = StandardScaler() 
        X_train_sc = sc.fit_transform(X_train) # Equivalent to X_train_ = (X_train - X_train.mean()) / X_train.std()
        X_val_sc = sc.fit_transform(X_val)

        # Checking the max and min value of the series
        print(X_train_sc.max(), X_train_sc.min())         # Gives the max and min values of all the features combined
        print(X_val_sc.max(), X_val_sc.min())

        return X_train_sc, X_val_sc, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 'error', 400
    
def scaling_test(X_test):

    try:

        # Let us scale the features using z-scalar technique
        sc = StandardScaler() 
        X_test_sc = sc.fit_transform(X_test) # Equivalent to X_train_ = (X_train - X_train.mean()) / X_train.std()

        # Checking the max and min value of the series
        print(X_test_sc.max(), X_test_sc.min())         # Gives the max and min values of all the features combined

        return X_test_sc, 200
    
    except Exception as e:
        return f"Exception in received request {traceback.format_exc()}", 400