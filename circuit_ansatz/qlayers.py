# Importing necessary modules and libraries #
# ========================================= #
import pennylane as qml
import tensorflow as tf
import PennyLane_Gate_Configurations as pgc

# Configuring initial circuit parameters #
# ====================================== #
num_of_qubits = input("Enter number of qubits (default=4): ")
if num_of_qubits == "":
    num_of_qubits = 4
else:
    try:
        num_of_qubits = int(num_of_qubits)
    except Exception as error:
        print(error)

weight_shapes = {"weights": num_of_qubits}

# PennyLane Device using 'default.qubit' #
# ====================================== #
dev = qml.device('default.qubit', wires=num_of_qubits, shots=1000)

# PennyLane circuits functions that take in inputs and weights as parameters #
# ========================================================================== #
@qml.qnode(dev)
def qnode0(inputs, weights): 
    # Data Encoding with the inputs
    pgc.data_encoding(inputs)
    
    # Constructing the rest of the circuit with the weights
    pgc.cx_almost_all_neighbors(num_of_qubits)
    pgc.build_cascade_ry(weights, num_of_qubits)
    
    # Return measurement layer output
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode1(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.build_cascade_rz(weights, num_of_qubits)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode2(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.build_cascade_rz(weights, num_of_qubits)
    pgc.cx_almost_all_neighbors(num_of_qubits)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode3(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.build_cascade_rz(weights, num_of_qubits)
    pgc.crz_almost_all_neighbors(num_of_qubits)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode4(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.cx_almost_all_neighbors(num_of_qubits)
    pgc.crx_almost_all_neighbors(weights, num_of_qubits)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode5(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.build_cascade_rz(weights)
    pgc.crz_one_to_all(weights)
    pgc.build_cascade_rx(weights)
    pgc.build_cascade_rz(weights)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

@qml.qnode(dev)
def qnode6(inputs, weights): 
    pgc.data_encoding(inputs)
    pgc.build_cascade_rz(weights)
    pgc.crz_one_to_all(weights)
    pgc.build_cascade_rx(weights)
    pgc.build_cascade_rz(weights)
    return tuple([qml.expval(qml.PauliZ(i)) for i in range(num_of_qubits)])

# ALlow user to select which models to test #
# ========================================= #
dict_of_qnodes = {0: qnode0, 1: qnode1, 2: qnode2, 3: qnode3, 4: qnode4, 5: qnode5, 6: qnode6, 'A': 'ALL'}
user_input = input("Which circuits do you want to select? Separate each option by 1 space.\n")

list_of_qlayers = []
if user_input == "A":
    list_of_qlayers = [qml.qnn.KerasLayer(dict_of_qnodes[qnode_choice], weight_shapes, output_dim=num_of_qubits, dtype='float64') \
                       for qnode_choice in dict_of_qnodes]
else:
    list_of_choices = [dict_of_qnodes[int(choice)] for choice in user_input.split()]
    list_of_qlayers = [qml.qnn.KerasLayer(qnode_choice, weight_shapes, output_dim=num_of_qubits, dtype='float64') \
                       for qnode_choice in list_of_choices]

# Default Value if no input was given
if list_of_qlayers == []:
    list_of_qlayers = [qml.qnn.KerasLayer(qnode0, weight_shapes, output_dim=num_of_qubits, dtype='float64')]

for index, qlayer in enumerate(list_of_qlayers):
    print(f"{index+1}. {qlayer}")