import torch
import cv2
from quantum_torch_model import QuantumModel  # Replace with your actual model file
import driving_data  # Import driving_data

def main():
    # Load the model
    model = QuantumModel()
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess an image
    img_path = 'steering_wheel.jpg'  # Replace with your image path
    img = cv2.imread(img_path)
    img = cv2.resize(img[-150:], (200, 66)) / 255.0  # Resize and normalize

    # Convert to a tensor and add a batch dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Permute to [C, H, W] and add batch dimension

    # Pass the image tensor to the model
    with torch.no_grad():
        output = model(img_tensor)
        print("Model output:", output)

if __name__ == "__main__":
    main()
