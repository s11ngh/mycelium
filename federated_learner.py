import modal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "scikit-learn")
)

app = modal.App("ml-training", image=image)

def preprocess_data(df):
    categorical_cols = ['Genre']
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    X = preprocessor.fit_transform(df)
    return X

# Modal functions for local model training
@app.function(image=image)
def local_trainer(dataframe, n_clusters=3):
    X = preprocess_data(dataframe)

    # Train a KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return cluster_centers, labels

# aggregate cluster centers
@app.function(image=image)
def aggregate_cluster_centers(centers_list):
    try:
        # Compute average cluster centers
        avg_centers = np.mean(np.array(centers_list), axis=0)
        return avg_centers
    except Exception as e:
        print(f"Error while aggregating: {e}")
        raise

# update global model with averaged cluster centers
@app.function(image=image)
def update_global_model(avg_centers):
    np.save('global_cluster_centers.npy', avg_centers)

@app.function(image=image)
def read_and_split_csv_file():
    csv_file_path = 'Mall_Customers.csv'
    df = pd.read_csv(csv_file_path)

    df1 = df.iloc[:len(df)//2]
    df2 = df.iloc[len(df)//2:]

    # train each dataframe locally
    centers1, labels1 = local_trainer.local(df1)
    centers2, labels2 = local_trainer.local(df2)

    # aggregate result
    avg_centers = aggregate_cluster_centers.local([centers1, centers2])

    # update global model
    update_global_model.local(avg_centers)

    return df1, df2

@app.local_entrypoint()
def main():
    df1, df2 = read_and_split_csv_file.local()
    print("DataFrames processed and global model updated.")
