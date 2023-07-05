# Variational Quantum Convolutional Neural Network for Image Recognition to Determine Steering Angle for Autonomous Vehicles 
### By Quantum Computing @ UCI Club
### Members
* Rohit Ganti
* Diptanshu Sikdar
* Arya Mhaiskar
* Stewart 
* Brendan 

## Goal
Design and implement a Variational Quantum Algorithm alongside a classical Machine Learning model to determine the steering angle correction for self-driving vehicles

## Motivation
* Existing classical deep neural networks yield good performance
* Aim for more generalized models and higher efficiency while training and testing
* Can we harness the power of quantum computing to help increase generality and/or efficiency?

## Project Approach
1. Dataset: 
    * Find a dataset which contains front-camera view of the road and corresponding steering wheel angle change to keep the car straight.
    * Understand the dimensions and variety of input images
    * Clean and preprocess the dataset
2. Build the Quantum and Classical Models
    * Group 1: Create a classical Convolutional Neural Network (CNN) to predict the vehicle's steering angle based on images
    * Group 2: Simultaneously build and test a Variational Quantum Algorithm (VQA) 
        * Use a simple existing vanilla neural network in place of the CNN
        * Build various ansatze circuits and compare their performances (e.g., convergence, accuracy, loss, runtime, etc.) 
3. After initial development, combine CNN and VQA, and fine-tune the model
4. Import the model and work with vehicle-environment simulations (e.g., CARLA, AirSim) and test hybrid model in the environment. 

## Project Architecture
<p align="left">
  <img src="https://user-images.githubusercontent.com/69136009/216572541-905d78ac-8088-46e0-b9a3-590c36a0bd00.png" width="745" title="VQCNN Architecture">
</p>

### Deep Learning Neural Networks + Variational Quantum Circuit 
Training models using 4 qubit HVA circuit + Deep Learnning Neural netwwrk

## Communication Tools: 
* Please go to Asana to check tasks regarding the project or email us to be added to the Asana for this project
* Utilize GitHub for project code updates
* Check Discord for updates
   * 8009f3ad0249a9565410ca6f247697926ea1e7b1


### Environment Activations
* Once you clone the repository make sure to activate the environment with the following command "source env/bin/activate" 
* Once that is done make sure to reinstall the libraries by using the following command in the virstual environment "python3 -m pip install -r requirements.txt" (Make sure your pip is upgraded)
