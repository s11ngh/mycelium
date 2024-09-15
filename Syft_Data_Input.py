import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import syft as sy  # Import PySyft for encryption

# Step 1: Set up Virtual Workers for encryption (data holders)
worker_1 = sy.VirtualMachine(name="worker_1")
worker_2 = sy.VirtualMachine(name="worker_2")
worker_3 = sy.VirtualMachine(name="worker_3")

# Get clients to represent the users
client_1 = worker_1.get_root_client()
client_2 = worker_2.get_root_client()
client_3 = worker_3.get_root_client()

# Step 2: Generate synthetic data
# Replace this with your real user data
data = torch.randn(300, 28 * 28)  # 300 samples, 28x28 images
target = torch.randint(0, 10, (300,))  # Labels for 10 classes (0-9)

# Step 3: Encrypt the data using Additive Secret Sharing
# Split and encrypt the data to distribute among workers
encrypted_data = data.share(client_1, client_2, client_3)
encrypted_target = target.share(client_1, client_2, client_3)

# Step 4: Organize the encrypted data into datasets for federated learning
encrypted_dataset = TensorDataset(encrypted_data, encrypted_target)
federated_train_loader = sy.FederatedDataLoader(
    {
        client_1: DataLoader(encrypted_dataset, batch_size=32, shuffle=True),
        client_2: DataLoader(encrypted_dataset, batch_size=32, shuffle=True),
        client_3: DataLoader(encrypted_dataset, batch_size=32, shuffle=True),
    }
)

# Step 5: Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 6: Define the model, loss function, and optimizer
model = SimpleNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 7: Train the model using encrypted data
def train(model, federated_train_loader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(federated_train_loader):
            # Send model to the current worker
            model = model.send(data.location)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update model and get it back from worker
            optimizer.step()
            model = model.get()

            # Accumulate the loss from the encrypted data
            epoch_loss += loss.get().item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(federated_train_loader)}')

# Step 8: Train for a few epochs
train(model, federated_train_loader, optimizer, criterion, epochs=5)
