# Importing standard Qiskit libraries
import pennylane as qml

# Importing other libraries
import numpy as np
import random
    
# applies the hadamard gate to every qubit
def build_cascade_hadamard(n_qubits): 
    for idx in range(n_qubits): 
        qml.H(idx)

# applies the rx gate to every qubit        
def build_cascade_rx(n_qubits, params):
    for idx in range(n_qubits):
        qml.RX(params[idx], idx)
        all_gate_params['w'+str(gate_id)] = params[idx]

# applies the rz gate to every qubit        
def build_cascade_rz(n_qubits, params=None):
    for idx in range(n_qubits):
        qml.RZ(params[idx], idx)
        all_gate_params['w'+str(gate_id)] = params[idx]

def cx_all_neighbors(n_qubits):
    for i in range(n_qubits):
        control = i
        target = i+1

        # Handles last condition
        if control == n_qubits - 1:
            target = 0

        qml.cx(control, target)

def cx_almost_all_neighbors(n_qubits):
    for i in range(n_qubits-1):
        control = i
        target = i+1                
        qml.cx(control, target)

def cx_one_to_all(n_qubits):       
    index_list = [i for i in range(n_qubits)]

    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.cx(control, target)

def crx_all_neighbors(n_qubits, params=None):
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits)]

    for i in range(n_qubits):
        control = i
        target = i+1
        if control == n_qubits - 1:
            target = 0
        qml.crx(params[i], control, target)
        all_gate_params['w'+str(gate_id)] = params[i]

def crx_almost_all_neighbors(n_qubits, params=None):
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits)]

    for i in range(n_qubits - 1):
        control = i
        target = i+1
        qml.crx(params[i], control, target)
        all_gate_params['w'+str(gate_id)] = params[i]

def crx_one_to_all(n_qubits, params):
    index_list = [i for i in range(n_qubits)]
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits * (n_qubits-1))]

    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.crx(params[j+target], control, target)
            all_gate_params['w'+str(gate_id)] = params[j+target]

def crz_all_neighbors(n_qubits, params):
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits)]

    for i in range(n_qubits):
        control = i
        target = i+1
        if control == n_qubits - 1:
            target = 0
        qml.crz(params[i], control, target)
        all_gate_params['w'+str(gate_id)] = params[i]

def crz_almost_all_neighbors(n_qubits, params):
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits)]

    for i in range(n_qubits - 1):
        control = i
        target = i+1
        qml.crz(params[i], control, target)
        all_gate_params['w'+str(gate_id)] = params[i]

def crz_one_to_all(n_qubits, params):
    index_list = [i for i in range(n_qubits)]
    if params == None:
        params = [random.uniform(0, 2*np.pi) for i in range(n_qubits * (n_qubits-1))]

    for j in range(n_qubits):
        control = j
        temp_index_list = index_list.copy()
        temp_index_list.remove(control)
        for target in temp_index_list:
            qml.crz(params[j+target], control, target)
            all_gate_params['w'+str(gate_id)] = params[j+target]