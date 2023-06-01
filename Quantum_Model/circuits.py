
import pennylane as qml

import numpy as np


dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev)
def circuit1(inputs, weights):
    for i in range(4):
        qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
    for i in range(4):
        qml.Rot(weights[i][0], weights[i][1], weights[i][2], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev)
def circuit2(inputs, weights):
    for i in range(4):
        qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
    
    for i in range(4):
        qml.RX(weights[i][0], wires=i)
    
    for i in range(4):
        qml.RY(weights[i][1], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev)
def circuit3(inputs, weights):
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


@qml.qnode(dev)
def circuit4(inputs, weights):
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


@qml.qnode(dev)
def circuit5(inputs, weights):
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


@qml.qnode(dev)
def circuit6(inputs, weights):
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


@qml.qnode(dev)
def circuit7(inputs, weights):
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


@qml.qnode(dev)
def circuit8(inputs, weights):
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


trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]

@qml.qnode(dev)
def circuit9(inputs, weights):
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