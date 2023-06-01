import tensorflow.compat.v1 as tf
import pennylane as qml

import numpy as np
import circuits

tf.disable_v2_behavior()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def classical_cnn(x):
    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 2) + b_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    # Third convolutional layer
    W_conv3 = weight_variable([5, 5, 36, 48])
    b_conv3 = bias_variable([48])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

    # Fourth convolutional layer
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

    # Fifth convolutional layer
    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

    # Flatten
    h_conv5_flat = tf.reshape(h_conv5, [-1, 4])   #the shapes must match here 

    return h_conv5_flat

'''
VARIATIONAL QUANTUM CIRCUIT : To change depending on the circuit we are trying
'''

# trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]
def variational_quantum_circuit(inputs, weights):
    return circuits.circuit6(inputs, weights)

x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)

# Classical CNN
classical_output = classical_cnn(x)
print(classical_output.shape)

# Variational Quantum Circuit
num_weights = 4  # Number of weights in the VQC
weights = tf.Variable(tf.random.uniform(shape=[num_weights, 3],
                                        minval=0, maxval=2 * 3.14159, dtype=tf.float32))
print(weights.shape)
quantum_output = variational_quantum_circuit(classical_output, weights)
print("VARIATIONAL QUANTUM CIRUIT ")
#print(qml.draw(quantum_output))
# Reshape quantum_output tensor
quantum_output_reshaped = tf.cast(quantum_output, tf.float32)  # Cast quantum_output to float32
quantum_output_reshaped = tf.reshape(quantum_output_reshaped, [-1, num_weights])

# Fully connected layer
W_fc = weight_variable([num_weights, 1])
b_fc = bias_variable([1])
W_fc = tf.cast(W_fc, tf.float32)  # Cast W_fc to float32
y = tf.matmul(quantum_output_reshaped, W_fc) + b_fc

# Loss and optimizer
loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
