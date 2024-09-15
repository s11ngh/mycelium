import modal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the image with system-level and Python dependencies (include pandas and scikit-learn)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "scikit-learn")
)

# Define the Modal app
app = modal.App("ml-training", image=image)

# Preprocessing function
def preprocess_data(df):
    # Define categorical and numerical columns
    categorical_cols = ['Genre']
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    # Preprocessing for numerical data: Scaling
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data: One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    X = preprocessor.fit_transform(df)
    return X

# Define Modal functions for model training
@app.function(image=image)
def train_kmeans_on_data(dataframe, n_clusters=3):
    X = preprocess_data(dataframe)

    # Train a KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return cluster_centers, labels

# Define a function to aggregate cluster centers
@app.function(image=image)
def aggregate_cluster_centers(centers_list):
    try:
        # Compute average cluster centers
        avg_centers = np.mean(np.array(centers_list), axis=0)
        return avg_centers
    except Exception as e:
        print(f"Error while aggregating: {e}")
        raise

# Define a function to update the global model with averaged cluster centers
@app.function(image=image)
def update_global_model(avg_centers):
    # Save the averaged cluster centers to a file
    np.save('global_cluster_centers.npy', avg_centers)

@app.function(image=image)
def read_and_split_csv_file():
    csv_file_path = 'Mall_Customers.csv'
    df = pd.read_csv(csv_file_path)

    df1 = df.iloc[:len(df)//2]
    df2 = df.iloc[len(df)//2:]

    # Train KMeans models on each dataframe and get their cluster centers
    centers1, labels1 = train_kmeans_on_data.local(df1)
    centers2, labels2 = train_kmeans_on_data.local(df2)

    # Aggregate cluster centers
    avg_centers = aggregate_cluster_centers.local([centers1, centers2])

    # Update global model with averaged cluster centers
    update_global_model.local(avg_centers)

    return df1, df2

@app.local_entrypoint()
def main():
    df1, df2 = read_and_split_csv_file.local()
    print("DataFrames processed and global model updated.")
