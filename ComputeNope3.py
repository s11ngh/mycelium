import modal
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define your Modal app and volume
app = modal.App("Compute_Node_1")

# Correct volume name
vol = modal.Volume.from_name("data_volume")

# Define the image with the required dependencies
image = modal.Image.debian_slim().pip_install("pandas", "tensorflow", "numpy", "scikit-learn")

@app.function(volumes={"/data": vol}, image=image, gpu="H100")
def compute():
    # Path to the CSV file inside the volume
    csv_file_path = "/data/Anomaly_part3.csv"
    
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Preprocessing step: Using actual columns from the CSV
    X = df[['Age', 'Income', 'Transaction_amount', 'Number_of_accounts', 'Suspicion_score']]  # Features
    y = df['Is_high_risk']  # Target (binary classification: high-risk indicator)

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert to TensorFlow constants
    X_train_tf = tf.constant(X_train_scaled, dtype=tf.float32)
    y_train_tf = tf.constant(np.array(y_train).reshape(-1, 1), dtype=tf.float32)

    # Define a simple linear regression model using TensorFlow
    class LinearRegressionModel(tf.Module):
        def __init__(self):
            # Initialize the weights and bias
            self.W = tf.Variable(tf.random.normal([5, 1]))  # 5 is the number of features
            self.b = tf.Variable(tf.random.normal([1]))

        def __call__(self, X):
            # Define the forward pass
            return tf.matmul(X, self.W) + self.b

    # Instantiate the model
    model = LinearRegressionModel()

    # Define Huber loss with L2 regularization
    def huber_loss_with_l2(y_true, y_pred, model, delta=1.0, lambda_l2=0.01):
        error = y_true - y_pred
        small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        huber_loss = tf.where(small_error, squared_loss, linear_loss)

        # Add L2 regularization to the weights
        l2_loss = lambda_l2 * tf.reduce_sum(tf.square(model.W))
        total_loss = tf.reduce_mean(huber_loss) + l2_loss
        return total_loss

    # Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Training function
    def train(model, X_train, y_train, epochs=1000, lambda_l2=0.01):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(X_train)
                # Compute the loss
                loss = huber_loss_with_l2(y_train, predictions, model, lambda_l2=lambda_l2)

            # Backward pass
            gradients = tape.gradient(loss, [model.W, model.b])
            # Apply the gradients
            optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    # Train the model
    train(model, X_train_tf, y_train_tf, epochs=1000, lambda_l2=0.01)

    # Save weights and biases to data_volume
    np.save("/data/weights3.npy", model.W.numpy())  # Save weights
    np.save("/data/biases3.npy", model.b.numpy())   # Save biases

    print("Model weights and biases saved to /data")

# Entry point for local execution
@app.local_entrypoint()
def main():
    compute.remote()
