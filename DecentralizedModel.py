import modal
import numpy as np
import tensorflow as tf


app = modal.App("Federated_Averaging_Node")
vol = modal.Volume.from_name("data_volume")


image = modal.Image.debian_slim().pip_install("tensorflow", "numpy")

@app.function(volumes={"/data": vol}, image=image, gpu="H100")
def federated_averaging():
    
    weights_1 = np.load("/data/weights.npy")   
    weights_2 = np.load("/data/weights1.npy")  
    weights_3 = np.load("/data/weights3.npy")   

# Federated Average
    avg_weights = (weights_1 + weights_2 + weights_3) / 3.0

    class GlobalModel(tf.Module):
        def __init__(self):
          
            self.W = tf.Variable(avg_weights, dtype=tf.float32)

        def __call__(self, X):
            return tf.matmul(X, self.W)

    global_model = GlobalModel()

# Updating Global Model Weights
    np.save("/data/global_weights.npy", global_model.W.numpy()) 

    def make_prediction(X):
        X_tf = tf.constant(X, dtype=tf.float32)
        return global_model(X_tf)

@app.local_entrypoint()
def main():
    federated_averaging.remote()
