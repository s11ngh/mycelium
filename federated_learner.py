import modal
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from diffprivlib.models import GaussianNB
from io import StringIO  # Correct import for StringIO

# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "scikit-learn", "diffprivlib", "cryptography")
)

app = modal.App("ml-training", image=image)

# Generate and store encryption key (run this once and store the key securely)
def generate_key():
    return Fernet.generate_key()

encryption_key = generate_key()
fernet = Fernet(encryption_key)

def encrypt_dataframe(df):
    return fernet.encrypt(df.to_csv(index=False).encode())

def decrypt_dataframe(encrypted_df):
    decrypted_csv = fernet.decrypt(encrypted_df).decode()
    return pd.read_csv(StringIO(decrypted_csv))  # Use StringIO from io module

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

# Apply differential privacy to KMeans
@app.function(image=image)
def local_trainer(encrypted_df, n_clusters=3, epsilon=1.0):
    df = decrypt_dataframe(encrypted_df)
    X = preprocess_data(df)

    # Train a KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return cluster_centers, labels

# Aggregate cluster centers
@app.function(image=image)
def aggregate_cluster_centers(centers_list):
    try:
        avg_centers = np.mean(np.array(centers_list), axis=0)
        return avg_centers
    except Exception as e:
        print(f"Error while aggregating: {e}")
        raise

# Update global model with averaged cluster centers
@app.function(image=image)
def update_global_model(avg_centers):
    np.save('global_model.npy', avg_centers)

@app.function(image=image)
def read_and_split_csv_file():
    csv_file_path = 'Mall_Customers.csv'
    df = pd.read_csv(csv_file_path)

    df1 = df.iloc[:len(df)//2]
    df2 = df.iloc[len(df)//2:]

    # Encrypt the dataframes
    encrypted_df1 = encrypt_dataframe(df1)
    encrypted_df2 = encrypt_dataframe(df2)
    print(encrypted_df1)
    print("that was data 1")
    print(encrypted_df2)
    print("that was data 2")
    # Train each dataframe locally
    centers1, labels1 = local_trainer.local(encrypted_df1)
    centers2, labels2 = local_trainer.local(encrypted_df2)

    # Aggregate result
    avg_centers = aggregate_cluster_centers.local([centers1, centers2])

    # Update global model
    update_global_model.local(avg_centers)

    return df1, df2

@app.local_entrypoint()
def main():
    df1, df2 = read_and_split_csv_file.local()
    print("DataFrames processed and global model updated.")
