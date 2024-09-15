import modal

app = modal.App("Compute_Node_1")

image = modal.Image.debian_slim().pip_install("pandas", "tensorflow", "numpy")

@app.function(image=image, gpu="H100")
def compute():
    import tensorflow as tf

    X_train = tf.constant([[18], [25], [30], [35], [40], [50], [60], [70]], dtype=tf.float32)
    y_train = tf.constant([1000, 2000, 3000, 3500, 4000, 5000, 6000, 7000], dtype=tf.float32)


    class LinearRegressionModel(tf.Module):
        def __init__(self):
            self.W = tf.Variable(0.0)
            self.b = tf.Variable(0.0)

        def __call__(self, X):
            return self.W * X + self.b
        
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

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss: {loss.numpy()}, W: {model.W.numpy()}, b: {model.b.numpy()}")

    train(model, X_train, y_train, epochs=1000, lambda_l2=0.01)

    predictions = model(X_train)
    print("Predicted values:", predictions.numpy())


@app.local_entrypoint()
def main():
    compute.remote()
