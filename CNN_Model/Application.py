import streamlit as st
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
import os
import numpy as np

# Check if on Windows OS
windows = False
if os.name == 'nt':
    windows = True

# Load TensorFlow model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "/app/quantumselfdriving/CNN_Model/save/model.ckpt")

# Load steering wheel image
img = cv2.imread('/app/quantumselfdriving/CNN_Model/steering_wheel_image.jpg', 0)
rows, cols = img.shape

# Initialize variables
smoothed_angle = 0
i = 0

# Streamlit web app
st.title("Self-Driving Car Simulation")
simulation_frame = st.empty()
simulation_frame1= st.empty()
angle_text = st.empty()
# Main simulation loop
while st.button("Start"):
    # Read and preprocess the image
    full_image = cv2.imread("/app/quantumselfdriving/Images/driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    
    # Predict the steering angle
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / np.pi
    
    # Clear the console output if not on Windows
    if not windows:
        _ = os.system('clear')

    # Display the predicted steering angle
    #st.write("Predicted steering angle:", degrees, "degrees")

    # Update the smoothed angle and rotate the steering wheel image
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    
    # Display the simulation frame and the rotated steering wheel image
    simulation_frame.image(full_image, channels="RGB", caption="Simulation")
    simulation_frame1.image(dst, channels="RGB", caption="Steering wheel")
    angle_text.text("Predicted steering angle: " + str(degrees) + " degrees")
    
    # Update the frame index
    i += 1

    # Check if 'q' key is pressed to stop the simulation
    #if cv2.waitKey(10) == ord('q'):
     #   break
