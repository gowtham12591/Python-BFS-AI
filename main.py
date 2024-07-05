# Import required libraries

import pandas as pd
import numpy as np

# Model Selection libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier 
# Model Metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score
# Hyper Parameter Tuning
from sklearn.model_selection import RandomizedSearchCV
# Model saving
import pickle

from src.data_preprocess import missing_value, duplicate_value, data_visualization, encoding, data_splitting, scaling
from src.model_train import get_metrics, model_build

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
if status != 200:
    print(df)

# Data split
X_train, X_val, y_train, y_val, status = data_splitting(df)
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
                 'Model_Metrics': [LG_model, NB_model, SVM_model, KNN_model, DT_model, BG_model, RF_model, GB_model, AB_model]}

model_df = pd.DataFrame(model_metrics)

print('Model_Performance: ', model_df)


model_tuned = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                         min_samples_leaf= min_samples_leaf, max_features = max_features,
                                         max_depth= max_depth, bootstrap=bootstrap) 
model_tuned.fit( X_train, y_train)