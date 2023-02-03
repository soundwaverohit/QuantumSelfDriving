# Variational Quantum Convolutional Neural Network for Image Recognition to Determine Steering Angle for Autonomous Vehicles 
(Quantum Computing @ UCI)

## Goal
Design and implement a Variational Quantum Algorithm alongside a classical Machine Learning model to determine the steering angle correction for self-driving vehicles

## Motivation
* Existing classical deep neural networks yield good performance
* Aim for more generalized models and higher efficiency while training and testing
* Can we harness the power of quantum computing to help increase generality and/or efficiency?

## Project Approach
1. Dataset: 
    a. Find a dataset which contains front-camera view of the road and corresponding steering wheel angle change to keep the car straight. 
    b. Understand the dimensions and variety of input images
    c. Clean and preprocess the dataset
2. Build the Quantum and Classical Models
    a. Group 1: Create a classical Convolutional Neural Network (CNN) to predict the vehicle's steering angle based on images
    b. Group 2: Simultaneously build and test a Variational Quantum Algorithm (VQA) 
        i. Use a simple existing vanilla neural network in place of the CNN
       ii. Build various ansatze circuits and compare their performances (e.g., convergence, accuracy, loss, runtime, etc.) 
3. After initial development, combine CNN and VQA, and fine-tune the model
4. Import the model and work with vehicle-environment simulations (e.g., CARLA, AirSim) and test hybrid model in the environment. 

## Communication Tools: 
* Check Asana for tasks of the project
* Utilize GitHub for project code updates
* Check Discord for updates
