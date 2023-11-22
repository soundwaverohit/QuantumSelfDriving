import torch
import numpy as np
from quantum_torch_model import QuantumModel  # Replace with your model file
import driving_data  # Import driving_data or your validation data module

import argparse


parser = argparse.ArgumentParser(description='Run a QuantumModel with custom parameters.')
parser.add_argument('--model_name', type=str, default='quantum_model1.pth', help='Model to run')
parser.add_argument('--threshold', type=float, default=1, help='Accuracy threshold for validation test')
args = parser.parse_args()

model_name= "models_saved/" +args.model_name
threshold= args.threshold

# Load the trained model
model = QuantumModel()
model.load_state_dict(torch.load('models_saved/quantum_model1.pth'))
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and threshold for accuracy
criterion = torch.nn.MSELoss()
accuracy_threshold = threshold  # Define your threshold here

# Assuming driving_data has a method to load the validation batch
num_val_images = driving_data.num_val_images  # replace with actual number of validation images
batch_size = 32  # or any other batch size you want to use

total_loss = 0.0
correct_predictions = 0
num_batches = 0

# Validation loop
with torch.no_grad():
    for i in range(0, num_val_images, batch_size):
        # Load a batch of validation data
        batch_images, batch_angles = driving_data.LoadValBatch(batch_size)

        # Convert to tensor and move to the appropriate device
        inputs = torch.tensor(np.array(batch_images), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        targets = torch.tensor(batch_angles, dtype=torch.float32).view(-1, 1).to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Accumulate loss and calculate accuracy
        total_loss += loss.item()
        correct_predictions += torch.sum(torch.abs(outputs - targets) < accuracy_threshold).item()
        num_batches += 1

# Calculate average loss and accuracy
average_loss = total_loss / num_batches
accuracy = correct_predictions / (num_batches * batch_size)

print("For model Name: ", model_name)
print(f'Average Validation Loss: {average_loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')
