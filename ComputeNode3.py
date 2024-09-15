import modal
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = modal.App("Compute_Node_3")

vol = modal.Volume.from_name("data_volume")

image = modal.Image.debian_slim().pip_install("pandas", "tensorflow", "numpy", "scikit-learn")

@app.function(volumes={"/data": vol}, image=image, gpu="H100")
def compute():

    csv_file_path = "/data/Anomaly_part3.csv"
    
    df = pd.read_csv(csv_file_path)

    X = df[['Age', 'Income', 'Transaction_amount', 'Number_of_accounts', 'Suspicion_score']]  
    y = df['Is_high_risk']  

    X_train, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_tf = tf.constant(X_train_scaled, dtype=tf.float32)
    y_train_tf = tf.constant(np.array(y_train).reshape(-1, 1), dtype=tf.float32)

    class LinearRegressionModel(tf.Module):
        def __init__(self):
  
            self.W = tf.Variable(tf.random.normal([5, 1]))  
            self.b = tf.Variable(tf.random.normal([1]))

        def __call__(self, X):
    
            return tf.matmul(X, self.W) + self.b


    model = LinearRegressionModel()

    def huber_loss_with_l2(y_true, y_pred, model, delta=1.0, lambda_l2=0.01):
        error = y_true - y_pred
        small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        huber_loss = tf.where(small_error, squared_loss, linear_loss)

        l2_loss = lambda_l2 * tf.reduce_sum(tf.square(model.W))
        total_loss = tf.reduce_mean(huber_loss) + l2_loss
        return total_loss

    optimizer = tf.optimizers.Adam(learning_rate=0.01)

 
    def train(model, X_train, y_train, epochs=1000, lambda_l2=0.01):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
     
                predictions = model(X_train)
      
                loss = huber_loss_with_l2(y_train, predictions, model, lambda_l2=lambda_l2)

            gradients = tape.gradient(loss, [model.W, model.b])
  
            optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    train(model, X_train_tf, y_train_tf, epochs=1000, lambda_l2=0.01)

    np.save("/data/weights3.npy", model.W.numpy()) 
    np.save("/data/biases3.npy", model.b.numpy())  

@app.local_entrypoint()
def main():
    compute.remote()
