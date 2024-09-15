import modal
import torch as torch
import syft as sy

stub = modal.App("pytorch-regression-modal")

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "numpy", "matplotlib", "syft"),
    gpu=modal.gpu.A100(),  
)
def train_model():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    import io

    # Initialize PySyft Worker
    worker = sy.Worker(name="client")

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Data Preparation
    np.random.seed(42)
    X_train = np.random.rand(1000, 3).astype(np.float32)
    y_train = X_train @ np.array([3.5, -2.1, 1.7], dtype=np.float32) + 0.5

    X_test = np.random.rand(200, 3).astype(np.float32)
    y_test = X_test @ np.array([3.5, -2.1, 1.7], dtype=np.float32) + 0.5

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(1).to(device)

    # Model Definition
    class RegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(RegressionModel, self).__init__()
            self.linear1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(64, 64)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(64, output_dim)

        def forward(self, x):
            out = self.linear1(x)
            out = self.relu1(out)
            out = self.linear2(out)
            out = self.relu2(out)
            out = self.output(out)
            return out

    # Instantiate the model
    model = RegressionModel(input_dim=3, output_dim=1).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 50
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

    # Move data back to CPU for visualization
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    # Visualization
    plt.scatter(y_test_np, predictions_np)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Encode the buffer to base64
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Return the base64 string
    return img_base64

if __name__ == "__main__":
    with stub.run():
        img_base64 = train_model()
        # Decode the base64 string and save it as an image
        import base64
        img_bytes = base64.b64decode(img_base64)
        with open('regression_plot.png', 'wb') as f:
            f.write(img_bytes)
        print("Plot saved as regression_plot.png")
