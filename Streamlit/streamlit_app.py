import streamlit as st
from streamlit_option_menu import option_menu
import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import circuit_drawer, plot_state_city
import matplotlib.pyplot as plt
from streamlit_ws_localstorage import injectWebsocketCode, getOrCreateUID
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.keras import layers, models
import io
from tensorflow.keras.utils import plot_model
import subprocess
from tensorflow.keras.callbacks import Callback
import cv2
import os


class StreamlitCallback(Callback):
    def __init__(self, display):
        super(StreamlitCallback, self).__init__()
        self.display = display

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = f"Epoch {epoch + 1}: Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}"
        self.display.text(message)


# Define the classical neural network layer
def classical_layer(inputs, weights):
    return np.tanh(np.dot(inputs, weights))

def build_model(layers, input_shape):
    model = Sequential()
    first_layer = True
    for layer in layers:
        if layer['type'] == 'CNN':
            if first_layer:
                model.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'], 
                                 strides=layer['strides'], padding=layer['padding'], 
                                 activation=layer['activation'], input_shape=input_shape))
                first_layer = False
            else:
                model.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'], 
                                 strides=layer['strides'], padding=layer['padding'], 
                                 activation=layer['activation']))
        # Handle other layers similarly
    return model

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
    "Build Quantum Circuit": "quantum_composer",
    "Quantum Classical Hybrid Model Sample": "quantum_cnn"
}

# Create a function for each page
def home():
    st.title("Welcome to Omdena Quantum Self Driving Project")
    st.write("""
    ## Introduction
    Quantum Self-Driving Neural Networks represent a groundbreaking intersection 
    of quantum computing and autonomous vehicle technology. This approach leverages 
    the complex computational capabilities of quantum algorithms to enhance the 
    efficiency and performance of neural networks in self-driving cars.
    """)

    st.write("""
    ## What are Quantum Neural Networks?
    Quantum Neural Networks (QNNs) are a novel type of neural networks that operate 
    on the principles of quantum mechanics. They use quantum bits (qubits) for 
    processing, which allows them to handle vast amounts of data and complex 
    computations more efficiently than traditional neural networks.
    """)

    st.write("""
    ## Advancements in Self-Driving Technology
    Self-driving technology has made significant strides in recent years, yet it 
    faces challenges like real-time decision making and processing massive datasets. 
    Quantum computing promises to accelerate data processing, enhance real-time 
    responses, and improve decision-making algorithms.
    """)

    st.write("""
    ## Intersection of Quantum Computing and Self-Driving Cars
    In self-driving cars, quantum neural networks can potentially process 
    environmental data more rapidly, leading to quicker and more accurate 
    decision-making. This can enhance the safety and reliability of autonomous vehicles.
    """)

    st.write("""
    ## Potential Benefits and Challenges
    The integration of quantum computing in self-driving cars could lead to 
    unprecedented advancements in traffic efficiency, safety, and navigation. 
    However, challenges like quantum hardware development and algorithm stability 
    remain significant hurdles.
    """)

    st.write("""
    ## Future Perspectives
    The future of quantum self-driving cars holds immense potential. As quantum 
    technology matures, we can expect more robust, efficient, and safer autonomous 
    vehicles, revolutionizing personal and public transportation.
    """)

    # Add content for the home page

def build_cnn():
    st.title("Build Your CNN Network")
    input_height = st.number_input("Input Height", min_value=1, value=28)
    input_width = st.number_input("Input Width", min_value=1, value=28)
    input_channels = st.number_input("Input Channels", min_value=1, value=3)
    input_shape = (input_height, input_width, input_channels)
    with st.form(key="cnn"):
        st.subheader("Input")

        layers = []
        for i in range(5):
            layer_type = st.selectbox(f"Layer {i+1} Type", ["CNN", "Activation", "MaxPool", "Dense", "Flatten"])
            
            if layer_type == 'CNN':
                filters = st.number_input(f"Layer {i+1} Filters", min_value=1, max_value=512, step=1)
                kernel_size = st.slider(f"Layer {i+1} Kernel Size", 1, 7, 3)
                strides = st.slider(f"Layer {i+1} Strides", 1, 5, 1)
                padding = st.selectbox(f"Layer {i+1} Padding", ["valid", "same"])
                activation = st.selectbox(f"Layer {i+1} Activation", ["relu", "sigmoid", "tanh"])
                layers.append({'type': layer_type, 'filters': filters, 'kernel_size': kernel_size,
                            'strides': strides, 'padding': padding, 'activation': activation})

            elif layer_type == 'MaxPool':
                pool_size = st.slider(f"Layer {i+1} Pool Size", 1, 4, 2)
                layers.append({'type': layer_type, 'pool_size': pool_size})

            elif layer_type == 'Dense':
                units = st.number_input(f"Layer {i+1} Units", min_value=1, max_value=1024, step=1)
                activation = st.selectbox(f"Layer {i+1} Activation", ["relu", "sigmoid", "tanh"])
                layers.append({'type': layer_type, 'units': units, 'activation': activation})

            elif layer_type == 'Flatten' or layer_type == 'Activation':
                layers.append({'type': layer_type, 'activation': 'none' if layer_type == 'Flatten' else 'relu'})

        create_cnn = st.form_submit_button(label="Create CNN")

    if create_cnn:
        model = build_model(layers, input_shape)
        plot_file = 'cnn_model_plot.png'
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)


        string = model.summary()
        st.text("The Classical Neural Network") # Display model summary
        if os.path.isfile(plot_file):
            st.image(plot_file)


    # Add content for the build CNN page

def train_cnn():
    st.title("Train Your CNN Network")
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Reshape and normalize
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    training_display = st.empty()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display model summary
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
    model_summary = summary_buffer.getvalue()
    with st.expander("Model Summary", expanded=False):
        st.text(model_summary)

    # Display model architecture
    st.subheader("Model Architecture:")
    with st.expander("Graph", expanded=True):
        plot_file = 'cnn_model_plot.png'
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)
        st.image(plot_file)

    # Train the model
    model.fit(X_train, y_train, epochs=5, callbacks=[StreamlitCallback(training_display)])



    # Save the model
    model_file = 'mymodel.h5'
    model.save(model_file)

    # Provide a link to download the model
    with open(model_file, "rb") as file:
        st.download_button(label="Download Model", data=file, file_name=model_file, mime='application/octet-stream')

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    st.write("Test Accuracy:", test_accuracy)

def view_results():
    st.title("Results")

def quantum_composer():
    st.title("Build Quantum Cicuit")
    global conn
    conn = injectWebsocketCode(hostPort='linode.liquidco.in', uid=getOrCreateUID())

    def update(circuit, circuit_visualization, **n):
        opr = st.session_state.gate_operations.copy()

        while len(opr) > st.session_state._qubit:
                opr.pop()
                
        for qubit, gate_ops in enumerate(opr):
            for gate in gate_ops:
                if gate == "Hadamard":
                    circuit.h(qubit)
                elif gate == "Pauli-X":
                    circuit.x(qubit)
                elif gate == "Pauli-Y":
                    circuit.y(qubit)
                elif gate == "Pauli-Z":
                    circuit.z(qubit)
                elif gate == "CNOT":
                    control = (qubit + 1) % st.session_state._qubit
                    circuit.cx(qubit, control)

        circuit_visualization.image(update_circuit_visualization(circuit), use_column_width=True)

    def update_circuit_visualization(circuit):
        circuit_drawer_file = "circuit_drawer.png"

        circuit_drawer(circuit, output='mpl', filename=circuit_drawer_file)
        image = plt.imread(circuit_drawer_file)
        plt.close()  # Close the figure to avoid overlapping images
        return image
    

    if "state_number" not in st.session_state:
        st.session_state.state_number = 0
    st.title("Quantum Composer")
    
    # Number of Qubits
    num_qubits = st.number_input("Number of Qubits", min_value=1, max_value=5, value=2, step=1)

    # Initialize gate operations
    gate_operations = [[] for _ in range(num_qubits)]

    # Initialize session-states
    if "_qubit" not in st.session_state:
        st.session_state._qubit = num_qubits

    if st.session_state._qubit != num_qubits:
        st.session_state._qubit = num_qubits
        if len(st.session_state.gate_operations) < st.session_state._qubit:
            st.session_state.gate_operations.append([])

    if "gate_operations" not in st.session_state:
        st.session_state.gate_operations = gate_operations

    # Initialize a Quantum Circuit
    circuit = QuantumCircuit(st.session_state._qubit)

    # Quantum Gates
    gate_options = ["Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z", "CNOT"]

    # Display the Quantum Circuit
    st.subheader("Quantum Circuit")
    circuit_visualization = st.empty()

    # Add Gates to the Circuit
    st.subheader("Add Gates")

    # Save and Load functionality
    if st.button("Save", key="s"):
        # Logic for save functionality
        pass  # Replace with your save logic

    for qubit in range(num_qubits):
        gate_label = f"Gate - Qubit {qubit}"
        selected_gate = st.selectbox(gate_label, gate_options, key=f"gate-{qubit}")

        if st.button(f"Apply Gate {qubit}", key=f"apply-{qubit}"):
            st.session_state.gate_operations[qubit].append(selected_gate)
            update(circuit, circuit_visualization)

    if st.session_state.state_number:
        for i in range(st.session_state.state_number):
            if st.sidebar.button(f"state_${i + 1}", key=i):
                vt= conn.getLocalStorageVal(key=i)
                parsed_arr_1 = vt.split(",")
                r = [[]]

                for t in parsed_arr_1:
                    if t.isdigit():
                        r.append([])
                    else:
                        r[-1].append(t)
            


                st.session_state.gate_operations = r
                update(circuit, circuit_visualization)
    # Execute the Circuit
    st.subheader("Execute Circuit")
    backend_options = ["qasm_simulator", "statevector_simulator", "unitary_simulator"]
    selected_backend = st.selectbox("Select Backend", backend_options)

    if st.button("Run Circuit"):
        # Reset the circuit to remove the previously applied gates
        circuit = QuantumCircuit(num_qubits)        

        # Update the circuit with the selected gates
        for qubit, gate_ops in enumerate(st.session_state.gate_operations):
            for gate in gate_ops:
                if gate == "Hadamard":
                    circuit.h(qubit)
                elif gate == "Pauli-X":
                    circuit.x(qubit)
                elif gate == "Pauli-Y":
                    circuit.y(qubit)
                elif gate == "Pauli-Z":
                    circuit.z(qubit)
                elif gate == "CNOT":
                    control = (qubit + 1) % num_qubits
                    circuit.cx(qubit, control)

        backend = Aer.get_backend(selected_backend)
        job = execute(circuit, backend)
        result = job.result()

        if selected_backend == "qasm_simulator":
            counts = result.get_counts()
            st.subheader("Measurement Results")
            st.text(counts)
        elif selected_backend == "statevector_simulator":
            statevector = result.get_statevector()
            st.subheader("Final Statevector")
            st.text(statevector)
            st.subheader("Statevector Visualization")
            st.pyplot(plot_state_city(statevector))
        elif selected_backend == "unitary_simulator":
            unitary = result.get_unitary()
            st.subheader("Final Unitary")
            st.text(unitary)



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
elif page == "quantum_composer":
    quantum_composer()

elif page == "view_results":
    view_results()
elif page =='quantum_cnn':
    quantum_cnn()
st.markdown('</div>', unsafe_allow_html=True)
