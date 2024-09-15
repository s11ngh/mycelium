# app/main.py

import modal
from modal import Volume  # Import Volume directly
from app.utils import (
    read_and_split_csv,
    preprocess_data,
    train_kmeans,
    aggregate_cluster_centers,
    update_global_model,
    GLOBAL_MODEL_PATH
)
import numpy as np
import os

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "scikit-learn", "numpy", "requests")  # Ensure 'requests' is installed
)

# Initialize the Modal app
app = modal.App("ml-training", image=image)

# Define and persist the Volume using the correct constructor method
data_volume = Volume.from_name("data_volume", create_if_missing=True)

@app.function(image=image, volumes={"/mount/data": data_volume})
def local_trainer(dataframe, n_clusters=3):
    X = preprocess_data(dataframe)
    cluster_centers, labels = train_kmeans(X, n_clusters=n_clusters)
    return cluster_centers, labels

@app.function(image=image)
def aggregate_cluster_centers_func(centers_list):
    try:
        avg_centers = aggregate_cluster_centers(centers_list)
        return avg_centers
    except Exception as e:
        print(f"Error while aggregating: {e}")
        raise

@app.function(image=image)
def update_global_model_func(avg_centers):
    update_global_model(avg_centers)

@app.function(image=image, volumes={"/mount/data": data_volume})
def read_and_split_csv_file():
    csv_path = '/mount/data/Mall_Customers.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    print(f"CSV file found at {csv_path}")
    
    # Read and split the DataFrame into three parts
    df = pd.read_csv(csv_path)
    split1 = len(df) // 3
    split2 = 2 * len(df) // 3
    df1 = df.iloc[:split1]
    df2 = df.iloc[split1:split2]
    df3 = df.iloc[split2:]

    # Train each dataframe locally
    centers1, labels1 = local_trainer.call(df1)
    centers2, labels2 = local_trainer.call(df2)
    centers3, labels3 = local_trainer.call(df3)
    
    # Aggregate result
    avg_centers = aggregate_cluster_centers_func.call([centers1, centers2, centers3])
    
    # Update global model
    update_global_model_func.call(avg_centers)
    
    return df1, df2, df3

@app.function(image=image, volumes={"/mount/data": data_volume})
def upload_data():
    import requests

    remote_csv_path = "/mount/data/Mall_Customers.csv"
    csv_url = "https://raw.githubusercontent.com/s11ngh/mycelium/main/Mall_Customers.csv"  # Updated URL

    response = requests.get(csv_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download CSV: {response.status_code}")
    
    with open(remote_csv_path, "w") as f:
        f.write(response.text)
    
    return "Data uploaded successfully."

# Optional: Local entry point for testing
@app.local_entrypoint()
def main():
    df1, df2, df3 = read_and_split_csv_file.call()
    print("DataFrames processed and global model updated.")
