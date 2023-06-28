import pennylane as qml
from pennylane import numpy as np

# Define the variational quantum circuit
def variational_quantum_circuit(params, wires):
    num_layers = len(wires)
    
    # Apply variational layers
    for l in range(num_layers):
        qml.Rot(params[l][0], params[l][1], params[l][2], wires=wires[l])
        qml.CNOT(wires=[wires[l], wires[(l + 1) % num_layers]])

# Initialize the quantum device
dev = qml.device("default.qubit", wires=3)

# Define the quantum node
@qml.qnode(dev)
def quantum_node(params, x):
    variational_quantum_circuit(params, wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(2))

# Define the cost function
def cost(params, x, y):
    predictions = [quantum_node(params, xi) for xi in x]
    return np.mean((predictions - y) ** 2)

# Initialize the parameters
np.random.seed(0)
num_params = 3  # Parameters per layer
num_layers = 4  # Number of layers
params = np.random.random((num_layers, num_params))

# Generate training data
num_samples = 100
x = np.random.random((num_samples, 2))
y = np.random.uniform(-1, 1, num_samples)

# Optimize the circuit parameters
opt = qml.GradientDescentOptimizer(0.1)
num_iterations = 100

for i in range(num_iterations):
    params = opt.step(lambda v: cost(v, x, y), params)
    if (i + 1) % 10 == 0:
        current_cost = cost(params, x, y)
        print(f"Step {i + 1}, Cost: {current_cost}")

# Predict the steering angles for new data
new_data = np.array([[0.1, 0.2], [0.3, 0.4]])
predictions = [quantum_node(params, xi) for xi in new_data]
print("Predictions:", predictions)
