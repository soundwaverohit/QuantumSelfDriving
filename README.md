# Variational Quantum Convolutional Neural Network for Image Recognition to Determine Steering Angle for Autonomous Vehicles 
(Quantum Computing @ UCI)

## Goal
Implement a Variational Quantum Algorithm alongside a classical Machine Learning model to determine the steering angle correction for self-driving vehicles

## Motivation
* Existing classical deep neural networks yield good performance
* Aim for more generalized models and higher efficiency while training and testing
* Can we harness the power of quantum computing to help increase generality and/or efficiency?

## Project Approach
1. Build a Variatonal Quantum Circut with 4 qubits
2. Take a set of images of cars driving on the road and label them with an appropriate steering angles for the car.
3. Train a classical CNN model with multiple layers to predict the car's steering angle based on images
4. Interchange the weights to a new model architecture where the dense layer changes with the Variatonal Quantum Circuit
5. Train the quantum circuits with correct steering angles to keep the car on the road
6. Build a simulation to show the results


# UCI Quantum Computing Club Members Directions: 
Go to Asana to check tasks regarding the project or email to be added to the Asana for this project
