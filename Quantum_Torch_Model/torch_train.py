import torch
import torch.optim as optim
import numpy as np
from quantum_torch_model import QuantumModel  # Replace with your model file
import driving_data  # Import driving_data
import argparse

# Parameters
# Argument parser
parser = argparse.ArgumentParser(description='Train a QuantumModel with custom parameters.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--model_name', type=str, default='model1', help='Name of the model to save')
args = parser.parse_args()

# Use arguments
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
model_name = args.model_name
savedir= "models_saved/"+str(model_name)+ "__l"+ str(learning_rate)+ "bs"+ str(batch_size)+ "ne"+str(num_epochs)+ ".pth"

print("Model initiating")
print("Model Name: ", model_name)
print("Saved at: ", savedir)
print("Num Epochs: ", num_epochs)
print("Batch Size: ", batch_size)
print("Learning rate: ", learning_rate)
# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

print("GPU Available : ", torch.cuda.is_available())

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

torch.save(model.state_dict(), savedir)
