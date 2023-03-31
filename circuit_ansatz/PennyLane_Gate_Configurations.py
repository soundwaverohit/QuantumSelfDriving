# Importing necessary libraries
import pennylane as qml
import numpy as np

'''
## REFERENCE CODE ##
def data_encoding(inputs):
    for index, data_input in enumerate(inputs):
        qml.RX(data_input, wires=index)

def quantum_black_box(weights, n_qubits):
    for idx in range(n_qubits-1):
        qml.CNOT(wires=[idx, idx+1])
    for idx in range(n_qubits): 
        qml.RY(weights[idx], wires=idx)
'''


# Non-parameterized Unary Gate Cascade Arrangements #
# ------------------------------------------------- #

# applies the hadamard gate to every qubit
def build_cascade_hadamard(n_qubits): 
    for idx in range(n_qubits): 
        qml.H(idx)
        
# applies the NOT gate to every qubit
def build_cascade_not(n_qubits): 
    for idx in range(n_qubits): 
        qml.X(idx)
        
        
# Unary Rotation Gate Cascade Arrangements #
# ---------------------------------------- #

# applies the RX gate to every qubit        
def build_cascade_rx(weights, n_qubits):
    for idx in range(n_qubits):
        qml.RX(weights[idx], wires=idx)

# applies the RY gate to every qubit  
def build_cascade_ry(weights, n_qubits):
    for idx in range(n_qubits):
        qml.RY(weights[idx], wires=idx)

# applies the RZ gate to every qubit        
def build_cascade_rz(weights, n_qubits):
    for idx in range(n_qubits):
        qml.RZ(weights[idx], wires=idx)
        
        
# Non-parameterized Controlled Gate Cascade Arrangements #
# ------------------------------------------------------ #

def cx_all_neighbors(n_qubits):
    for i in range(n_qubits):
        control = i
        target = i+1

        # Handles last condition
        if control == n_qubits - 1:
            target = 0

        qml.CNOT(wires=[control, target])

def cx_almost_all_neighbors(n_qubits):
    for idx in range(n_qubits-1):
        control = idx
        target = idx+1                
        qml.CNOT(wires=[control, target])

def cx_one_to_all(n_qubits):       
    index_list = [i for i in range(n_qubits)]

    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.CNOT(wires=[control, target])
            
            
# Controlled Rotation Gate Cascade Arrangements #
# --------------------------------------------- #

def crx_all_neighbors(weights, n_qubits):
    for i in range(n_qubits):
        control = i
        target = i+1
        if control == n_qubits - 1:
            target = 0
        qml.crx(weights[i], control, target)
        

def crx_almost_all_neighbors(weights, n_qubits):
    for i in range(n_qubits - 1):
        control = i
        target = i+1
        qml.crx(weights[i], control, target)
        

def crx_one_to_all(weights, n_qubits):
    index_list = [i for i in range(n_qubits)]
    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.crx(weights[j+target], control, target)
            all_gate_weights['w'+str(gate_id)] = weights[j+target]

def crz_all_neighbors(weights, n_qubits):
    for i in range(n_qubits):
        control = i
        target = i+1
        if control == n_qubits - 1:
            target = 0
        qml.crz(weights[i], control, target)
        

def crz_almost_all_neighbors(weights, n_qubits):
    for i in range(n_qubits - 1):
        control = i
        target = i+1
        qml.crz(weights[i], control, target)
        

def crz_one_to_all(weights, n_qubits):
    index_list = [i for i in range(n_qubits)]
    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.crz(weights[j+target], control, target)