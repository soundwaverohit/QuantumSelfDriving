import torch
import cv2
import numpy as np
from quantum_torch_model import QuantumModel  # Replace with your model file
import os
from subprocess import call

# Check if on Windows OS
windows = False
if os.name == 'nt':
    windows = True

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantumModel().to(device)
model.load_state_dict(torch.load('quantum_model.pth'))
model.eval()

img = cv2.imread('../CNN_Model/steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0
i = 0
while(cv2.waitKey(10) != ord('q')):
    print(i)
    full_image = cv2.imread("../Images/driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.tensor(image, dtype=torch.float32).to(device)

    with torch.no_grad():
        degrees = model(image_tensor).cpu().numpy()[0][0] * 180.0 / np.pi
        if not windows:
            call("clear")
        print("Predicted steering angle: " + str(degrees) + " degrees")
        print(i)
        cv2.imshow("frame", full_image)
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        i += 1

cv2.destroyAllWindows()
