import torch
import torch.optim as optim
import numpy as np
from quantum_torch_model import QuantumModel  # Replace with your model file
import driving_data  # Import driving_data

# Parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = QuantumModel().to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, driving_data.num_train_images, batch_size):
        # Load a batch of data
        batch_images, batch_angles = driving_data.LoadTrainBatch(batch_size)

        # Convert to a single numpy array and then to a tensor
        inputs = torch.tensor(np.array(batch_images), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        targets = torch.tensor(batch_angles, dtype=torch.float32).view(-1, 1).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 0:  # Adjust print frequency as needed
            print(f'Epoch: {epoch + 1}, Batch: {i // batch_size + 1}, Loss: {running_loss / (i // batch_size + 1):.3f}')

print('Finished Training')
torch.save(model.state_dict(), 'quantum_model.pth')
