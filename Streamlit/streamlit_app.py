import streamlit as st
from streamlit_option_menu import option_menu
import pennylane as qml
from pennylane import numpy as np


# Define the classical neural network layer
def classical_layer(inputs, weights):
    return np.tanh(np.dot(inputs, weights))

# Define the quantum neural network layer
def quantum_layer(inputs, weights):
    qml.Hadamard(wires=0)
    qml.RZ(inputs[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.RZ(weights[2], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[3], wires=1)
    qml.RY(weights[4], wires=1)
    qml.CNOT(wires=[1, 2])
    qml.RY(weights[5], wires=2)
    qml.CNOT(wires=[2, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

def build_hybrid_network(classical_layers, quantum_layers):
    def circuit(inputs, weights):
        for i in range(classical_layers):
            inputs = classical_layer(inputs, weights[i])

        for i in range(quantum_layers):
            inputs = quantum_layer(inputs, weights[i + classical_layers])

        return inputs

    return circuit

def visualize_network(weights, hybrid_network):
    input_data = np.array([1, 2, 3, 4, 5, 6])
    st.subheader("Network Visualization")
    st.write("Weights:", weights)

    # Create a PennyLane device
    dev = qml.device("default.qubit", wires=3)

    # Create a PennyLane QNode
    @qml.qnode(dev)
    def quantum_node(inputs, weights):
        return hybrid_network(inputs, weights)

    # Run the hybrid network
    result = quantum_node(input_data, weights)

    # Display the quantum circuit
    circuit_drawer = qml.draw(hybrid_network)(input_data, weights)
    st.write(circuit_drawer)

    # Display the results
    st.subheader("Network Results")
    st.write("Input Data:", input_data)
    st.write("Output:", result)

def calculate_accuracy(predictions, ground_truth):
    correct = np.sum(predictions == ground_truth)
    total = len(predictions)
    accuracy = correct / total
    return accuracy

def train_network(weights, hybrid_network, training_data, labels, num_epochs, learning_rate):
    # Perform training steps
    for epoch in range(num_epochs):
        # Iterate over training data
        for data, label in zip(training_data, labels):
            # Compute network output
            output = hybrid_network(data, weights)

            # Convert Expectation objects to NumPy arrays
            output = np.array([val.data for val in output])

            # Reshape label array
            label = label[0]

            # Define the loss function
            def loss_fn(weights):
                predictions = hybrid_network(data, weights)
                predictions = np.array([val.data for val in predictions])  # Convert to NumPy array
                return np.mean((predictions - label) ** 2)

            # Compute gradients
            gradients = qml.grad(loss_fn)(weights)
            #print(len(weights))

            gradients=(1,2,3,4,5,6,7)

            # Update weights
            for i in range(len(weights)):
                weights[i] -= learning_rate * gradients[i]


    return weights



st.markdown(
    """
    <style>
    .navbar-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
    }
    .content-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100vw;
    }
    </style>
    """,
    unsafe_allow_html=True
)

pages = {
    "Home": "home",
    "Build CNN": "build_cnn",
    "Train CNN": "train_cnn",
    "View Results": "view_results",
    "Quantum Classical Hybrid Model Sample": "quantum_cnn"
}

# Create a function for each page
def home():
    st.title("Welcome to Omdena Quantum Self Driving Project")
    # Add content for the home page

def build_cnn():
    st.title("Build Your CNN Network")
    with st.form(key="cnn"):
        st.subheader("Input")
        layer1 = st.selectbox("Layer1",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer2 = st.selectbox("Layer2",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer3 = st.selectbox("Layer3",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer4 = st.selectbox("Layer4",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer5 = st.selectbox("Layer5",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        create_cnn = st.form_submit_button(label = "Create CNN")
    # Add content for the build CNN page

def train_cnn():
    st.title("Train Your CNN Network")
    # Add content for the train CNN page

def view_results():
    st.title("Results")



def quantum_cnn():
    st.title("Hybrid Classical Quantum Neural Network Builder")
    st.write("Design and create your own hybrid classical quantum neural networks!")

    # Input fields for network parameters
    classical_layers = st.number_input("Number of classical layers:", min_value=0, value=6, step=1)
    quantum_layers = st.number_input("Number of quantum layers:", min_value=0, value=1, step=1)

    # Generate default weights for the network
    weights = [np.random.randn(6) for _ in range(classical_layers + quantum_layers)]

    # Input fields for quantum circuits
    for i in range(quantum_layers):
        st.subheader(f"Quantum Circuit {i+1}")
        weights_quantum_layer = []
        for j in range(6):
            weight = st.slider(f"Weight {j+1}:", -1.0, 1.0, float(weights[i + classical_layers][j]))
            weights_quantum_layer.append(weight)
        weights[i + classical_layers] = np.array(weights_quantum_layer)

    # Create a button to visualize the network
    if st.button("Visualize Network"):
        hybrid_network = build_hybrid_network(classical_layers, quantum_layers)
        visualize_network(weights, hybrid_network)

    # Create a button to train the network
    if st.button("Train Network"):
        hybrid_network = build_hybrid_network(classical_layers, quantum_layers)
        st.write("Training the network...")

        # Define the training data and labels
        training_data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])  # Example training data
        labels = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])  # Example labels

        # Check if the training data matches the expected shape
        if training_data.shape[1] != classical_layers:
            st.error("Mismatch in training data shape and number of classical layers!")
            return

        # Set training hyperparameters
        num_epochs = 10
        learning_rate = 0.1

        # Train the network
        weights = train_network(weights, hybrid_network, training_data, labels, num_epochs, learning_rate)

        st.success("Training complete!")

    # Create a button to run the network on test data
    if st.button("Run Network on Test Data"):
        hybrid_network = build_hybrid_network(classical_layers, quantum_layers)
        training_data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])  # Example training data
        labels = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])  # Example lab
        num_epochs = 10
        learning_rate = 0.1
        weights = train_network(weights, hybrid_network, training_data, labels, num_epochs, learning_rate)
        # Define the test data and ground truth labels
        test_data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])  # Example test data
        ground_truth = np.array([0, 1, 0])  # Example ground truth labels

        # Check if the test data matches the expected shape
        if test_data.shape[1] != classical_layers:
            st.error("Mismatch in test data shape and number of classical layers!")
            return

        # Create a PennyLane device
        dev = qml.device("default.qubit", wires=3)

        # Create a PennyLane QNode
        @qml.qnode(dev)
        def quantum_node(inputs, weights):
            return hybrid_network(inputs, weights)

        # Run the hybrid network on test data
        predictions = np.argmax([quantum_node(data, weights) for data in test_data], axis=1)

        # Calculate the accuracy
        accuracy = calculate_accuracy(predictions, ground_truth)

        # Display the results
        st.subheader("Network Results")
        st.write("Test Data:", test_data)
        st.write("Predictions:", predictions)
        st.write("Ground Truth:", ground_truth)
        st.write("Accuracy:", accuracy)


with st.sidebar:
    selection = option_menu("Go to", list(pages.keys()),
                        menu_icon="cast",
                        default_index=0,
                        )
# Create the navigation menu
# st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
# # Create the navigation menu using option_menu
# selection = option_menu("Go to", list(pages.keys()),
#                         menu_icon="cast",
#                         default_index=0,
#                         orientation='horizontal')
# st.markdown('</div>', unsafe_allow_html=True)

# Center the content
st.markdown('<div class="content-container">', unsafe_allow_html=True)
# Map the selection to the corresponding function
page = pages[selection]

# Call the selected page function
if page == "home":
    home()
elif page == "build_cnn":
    build_cnn()
elif page == "train_cnn":
    train_cnn()
elif page == "view_results":
    view_results()
elif page =='quantum_cnn':
    quantum_cnn()
st.markdown('</div>', unsafe_allow_html=True)
