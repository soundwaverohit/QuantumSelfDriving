import tensorflow.compat.v1 as tf
import pennylane as qml

import numpy as np

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



"""
VARIATIONAL QUANTUM CIRCUIT : To change depending on the circuit we are trying

Original template sample_circuit1

def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        for i in range(4):
            qml.Rot(weights[i][0], weights[i][1], weights[i][2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


sample circuit 2

def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Circuit 1
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        
        # Circuit 2
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
        
        # Circuit 3
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


sample circuit 3
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Perception
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)

        # Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
        
        for i in range(4):
            qml.RY(weights[i][1], wires=i)

        # Control
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
        
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


    
sample circuit 4

def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Circuit 1: Quantum feature encoding
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        
        # Circuit 2: Quantum image processing
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
        
        # Circuit 3: Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


sample circuit 5


def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Circuit 1: Perception
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)

        # Circuit 2: Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
        
        # Circuit 3: Control
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
        
        # Circuit 4: Interaction with environment
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


sample circuit 6:


def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Circuit 1: Encode inputs
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        
        # Circuit 2: Learnable weights for decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
            qml.RY(weights[i][1], wires=i)
        
        # Circuit 3: Decision-making and output
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)

circuit 7:
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Circuit 1 - Sensing
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        
        # Circuit 2 - Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
        
        # Circuit 3 - Control
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
        
        # Circuit 4 - Additional functionality for self-driving car
        for i in range(4):
            qml.RZ(weights[i][2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


circuit 8:
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):

        
        # Circuit 2 - Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            #qml.CNOT(wires=[3, 0])
        
        # Circuit 3 - Control
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])
        
        
        # Circuit 4 - Additional functionality for self-driving car
        for i in range(4):
            qml.RZ(weights[i][2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)

"""
trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(trained_rotations[0], wires=0)
        qml.RX(trained_rotations[1], wires=1)
        qml.RX(trained_rotations[2], wires=2)
        qml.RX(trained_rotations[3], wires=3)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.RX(trained_rotations[4], wires=1)
        qml.RX(trained_rotations[5], wires=2)
        qml.CNOT(wires=[3,0])
        qml.RX(trained_rotations[6], wires=0)
        qml.RX(trained_rotations[7], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

        
        # Circuit 2 - Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            #qml.CNOT(wires=[3, 0])
        
        # Circuit 3 - Control
        for i in range(4):
            qml.RY(weights[i][1], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])
        
        
        # Circuit 4 - Additional functionality for self-driving car
        for i in range(4):
            qml.RZ(weights[i][2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)




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
