# Import required libraries
import pandas as pd
import numpy as np

# Model saving
import pickle

# Import the required the functions for preprocessing, model training and tuning
from src.data_preprocess import missing_value, duplicate_value, data_visualization, encoding, feature_splitting, data_splitting, scaling, scaling_test
from src.model_train import get_metrics, model_build
from src.model_tuning import hyper_parameter_tuning

#### Dataset Link : https://www.kaggle.com/datasets/rashmiranu/banking-dataset-classification

# Read the dataset
df = pd.read_csv('data/train.csv', sep=';')

# Checking for missing values and replace it with appropriate methods
df,status = missing_value(df)
if status != 200:
    print(df)

# Check for duplicate values and replace with respective techniques
df,status = duplicate_value(df)
if status != 200:
    print(df)

# Perform automated data visualization
data_visualization(df, 'data/train')

# Encode the categorical data - One-hot encoding is used, which is better than label-encoding
df, status = encoding(df)
print(df.head())
if status != 200:
    print(df)

# Feature Splitting
independent_features, dependent_feature, status = feature_splitting(df)
if status != 200:
    print(independent_features)

# Data split
X_train, X_val, y_train, y_val, status = data_splitting(independent_features, dependent_feature)
if status != 200:
    print(X_train)

# Scaling
X_train_sc, X_val_sc, status = scaling(X_train, X_val)
if status != 200:
    print(X_train_sc)

# Model Building and Training
# Get the metrics for all the defined models
y_pred_lg, y_pred_nb, y_pred_svm, y_pred_knn, y_pred_dt, y_pred_bg, y_pred_rf, y_pred_gb, y_pred_ab, status = model_build(X_train_sc, X_val_sc, y_train, y_val)

LG_model = get_metrics(y_val, y_pred_lg)
NB_model = get_metrics(y_val, y_pred_nb)
SVM_model = get_metrics(y_val, y_pred_svm)
KNN_model = get_metrics(y_val, y_pred_knn)
DT_model = get_metrics(y_val, y_pred_dt)
BG_model = get_metrics(y_val, y_pred_bg)
RF_model = get_metrics(y_val, y_pred_rf)
GB_model = get_metrics(y_val, y_pred_gb)
AB_model = get_metrics(y_val, y_pred_ab)

model_metrics = {'Model_Name': ['LG_model', 'NB_model', 'SVM_model', 'KNN_model', 'DT_model', 'BG_model', 'RF_model', 'GB_model', 'AB_model'],
                 'Model_Classifier': ['LogisticRegression', 'GaussianNB', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'BaggingClassifier',
                                      'RandomForestClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier'],
                 'Model_Metrics': [LG_model, NB_model, SVM_model, KNN_model, DT_model, BG_model, RF_model, GB_model, AB_model]}

model_df = pd.DataFrame(model_metrics)

print('Model_Performance: ', model_df)

# Identifying the best model name and its respective classifier from the above dataframe
# Extract accuracy values from dictionaries
model_df['accuracy'] = model_df['Model_Metrics'].apply(lambda x: x.get('accuracy'))

# Find the model with the highest accuracy
best_model_index = model_df['accuracy'].idxmax()
best_model_name = model_df.loc[best_model_index, 'Model_Name']
best_model_classifier = model_df.loc[best_model_index, 'Model_Classifier']
best_accuracy = model_df['accuracy'].max()

print(f"Model with highest accuracy: {best_model_name}, {best_model_classifier}, (Accuracy: {best_accuracy:.2f})")

# Hyper paramter tuning
n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap, status = hyper_parameter_tuning(X_train, y_train, 
                                                                                                               best_model_classifier)
if status != 200:
    print(n_estimators)

model_tuned = best_model_classifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                         min_samples_leaf= min_samples_leaf, max_features = max_features,
                                         max_depth= max_depth, bootstrap=bootstrap) 
model_tuned.fit(X_train_sc, y_train)

# Predicting the output on validation data using the fine tuned model
y_pred = model_tuned.predict(X_val_sc)
print(get_metrics(y_val, y_pred))

# Save the model to a file
with open('model/tuned_model.pkl', 'wb') as file:
    pickle.dump(model_tuned, file)

# Load the test dataset and then pre=process it before making the prediction
df_test = pd.read_csv('data/test.csv', sep=';')

# Encode the categorical data - One-hot encoding is used, which is better than label-encoding
df_test, status = encoding(df_test)
if status != 200:
    print(df_test)

# Feature Splitting
X_test, y_test, status = feature_splitting(df_test)
if status != 200:
    print(independent_features)

# Scaling
X_test_sc, status = scaling(X_test)
if status != 200:
    print(X_test_sc)

# Loading the saved model
with open('model/tuned_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Predicting the output on test data using the loaded fine tuned model
y_pred = loaded_model.predict(X_test_sc)
print(get_metrics(y_test, y_pred))