# app/utils.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Define paths
CSV_FILE_PATH = '/mount/data/Mall_Customers.csv'
GLOBAL_MODEL_PATH = '/mount/data/global_model.npy'

def preprocess_data(df):
    categorical_cols = ['Genre']
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X = preprocessor.fit_transform(df)
    return X

def train_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return cluster_centers, labels

def aggregate_cluster_centers(centers_list):
    avg_centers = np.mean(np.array(centers_list), axis=0)
    return avg_centers

def update_global_model(avg_centers):
    np.save(GLOBAL_MODEL_PATH, avg_centers)

def read_and_split_csv():
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)
    df1 = df.iloc[:len(df)//2].reset_index(drop=True)
    df2 = df.iloc[len(df)//2:].reset_index(drop=True)
    return df1, df2
